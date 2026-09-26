"""Stage 3 — batched judgement, the first stage that costs real money.

Everything before this point exists to make this stage see as few candidates as
possible. It is also the only stage whose cost is dominated by something the
pipeline controls rather than by the size of the pile.

Two levers, and they are not equal
----------------------------------

**Batching is the big one.** Five candidates share one call, so the static prefix
— system prompt, rubric, job definition — is billed once per five candidates
instead of once per candidate. Measured against the live provider, that prefix
runs to roughly 900-1100 tokens, which is the largest single component of a
call's input; dividing it by five is a larger saving than every prompt-tuning
trick combined.

**Prefix caching is the smaller one, and it constrains how the prompt is built.**
The system message is assembled once per run and reused byte-identically, so the
first call warms the cache and every later batch in the same run hits it. Measured
hit rates run about 80% at this prefix size. That works only if the prefix does
not vary between calls, which is why the date, the batch and any per-candidate
text live in the *user* message, never in the system message.

A cheap model is not a lever
----------------------------

Judgement quality is what the whole project is for, and a batch that is scored
badly has to be re-examined by a human — which costs more than the tokens saved.
Model choice is therefore left at the configured model, with escalation reserved
for the borderline review in stage 4.

Why the tier is computed here rather than asked for
---------------------------------------------------

Asking the model for both a score and a tier invites them to disagree, and the
disagreement is invisible: a score of 82 sitting next to a tier of "qualified"
goes unnoticed because both values look reasonable in isolation. One derived value
cannot contradict itself, and asking for one fewer field is one fewer thing for
the model to get wrong.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from app.core.config import settings
from app.core.log import get_logger
from app.core.prompts import load_prompt, load_system_prompt
from app.models.job import JobDescription, SkillKind, degree_label
from app.models.screening import CallPurpose, CandidateTier
from app.services.llm import ChatModel, LLMError
from app.services.screening.rules import RuleVerdict
from app.services.screening.store import ChunkHit

logger = get_logger(__name__)

# Caps applied to model output before it reaches the database. The prompt asks
# for these limits; the code enforces them, because a prompt is a request.
_MAX_EVIDENCE = 2
_MAX_EVIDENCE_CHARS = 80
_MAX_GAPS = 2
_MAX_GAP_CHARS = 40

# Rule-layer reasons are prompt context, not the verdict. Truncated so one
# verbose candidate cannot inflate the whole batch's input.
_MAX_RULE_REASONS = 4


@dataclass(frozen=True)
class JudgeCandidate:
    """One candidate as presented to the judge.

    Attributes:
        resume_id: Database identifier.
        chunks: Retrieved chunks, best first. This is the evidence the judge
            reads — not the full resume, which would roughly triple the input
            and add material retrieval already decided was irrelevant.
        rule: Stage-1 verdict, if available. Its matched/missing skill lists are
            free signals already computed, so passing them costs about 30 tokens
            and saves the judge from re-deriving them.
    """

    resume_id: int
    chunks: tuple[ChunkHit, ...] = ()
    rule: Optional[RuleVerdict] = None


@dataclass(frozen=True)
class JudgeVerdict:
    """The judge's conclusion for one candidate.

    Attributes:
        resume_id: Database identifier.
        score: 0-100 match score.
        tier: Derived from ``score`` via :func:`tier_for_score`.
        evidence: Short quotes lifted from the material.
        gaps: Short descriptions of shortfalls.
        note: Free-text commentary from the review stage. Kept apart from
            ``gaps`` because it explains a *decision* rather than describing a
            candidate: an earlier version wrote the review's rationale into
            ``gaps``, which both mislabelled it and silently discarded the
            shortfalls the judge had actually identified.
    """

    resume_id: int
    score: float
    tier: str
    evidence: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class JudgeOutcome:
    """Result of judging a shortlist.

    Attributes:
        verdicts: Scored candidates, keyed by resume id.
        failed: Candidates the judge did not return a usable score for. Kept
            separate and explicit: a candidate missing from ``verdicts`` because
            of a malformed reply must never be indistinguishable from one that
            was never in the shortlist.
        calls: Provider calls issued, including failed ones.
    """

    verdicts: dict[int, JudgeVerdict]
    failed: tuple[int, ...] = ()
    calls: int = 0


def tier_for_score(score: float) -> str:
    """Map a 0-100 score onto a tier.

    Args:
        score: Match score.

    Returns:
        A :class:`~app.models.screening.CandidateTier` value.
    """
    pipeline = settings.PIPELINE
    if score >= pipeline.tier_strong:
        return CandidateTier.STRONG.value
    if score >= pipeline.tier_qualified:
        return CandidateTier.QUALIFIED.value
    if score >= pipeline.tier_borderline:
        return CandidateTier.BORDERLINE.value
    return CandidateTier.WEAK.value


def build_job_block(job: JobDescription) -> str:
    """Render the job-specific part of the static prefix.

    Built once per run and reused unchanged, so it must not contain anything that
    varies between calls within a run — no timestamps, no batch numbers.

    Args:
        job: The job description, with ``requirements`` and their ``term`` loaded.

    Returns:
        A markdown block describing the job and its rubric weights.
    """
    lines = ["## 岗位定义", f"- 岗位名称：{job.title}"]

    if job.department:
        lines.append(f"- 所属部门：{job.department}")
    if job.location:
        lines.append(f"- 工作地点：{job.location}")

    lines.append(f"- 学历门槛：{degree_label(job.min_degree)}")
    lines.append(f"- 经验年限要求：{job.min_years:g} 年以上")

    lines.append(
        "- 维度权重："
        f"技能 {job.weight_skills:.0%}、"
        f"经验 {job.weight_experience:.0%}、"
        f"学历 {job.weight_education:.0%}、"
        f"项目 {job.weight_projects:.0%}"
    )

    required = [
        requirement.term.name
        for requirement in (job.requirements or [])
        if requirement.kind == SkillKind.REQUIRED.value and requirement.term is not None
    ]
    preferred = [
        requirement.term.name
        for requirement in (job.requirements or [])
        if requirement.kind == SkillKind.PREFERRED.value and requirement.term is not None
    ]

    if required:
        lines.append(f"- 必备技能（缺失即显著扣分）：{'、'.join(required)}")
    if preferred:
        lines.append(f"- 加分技能：{'、'.join(preferred)}")

    return "\n".join(lines)


def build_judge_system_prompt(job: JobDescription) -> str:
    """Assemble the static prefix shared by every judge call in a run.

    Order is deliberate and must stay stable: the invariant instructions first,
    then the job definition. Provider caching matches the longest common prefix,
    so appending job-specific text last keeps the invariant part cacheable even
    when the same rubric is reused across different jobs.

    Args:
        job: The job description.

    Returns:
        The system message content.
    """
    return "\n\n".join(
        [load_system_prompt(), load_prompt("judge"), build_job_block(job)]
    )


def _render_rule_context(rule: Optional[RuleVerdict]) -> str:
    """Summarise the free stage-1 signals for the prompt.

    Args:
        rule: The stage-1 verdict, if one was computed.

    Returns:
        A one-line summary, or an empty string when there is nothing to say.
    """
    if rule is None:
        return ""
    reasons = [reason for reason in rule.reasons[:_MAX_RULE_REASONS] if reason]
    return "规则层预判：" + "；".join(reasons) if reasons else ""


def build_batch_message(candidates: Sequence[JudgeCandidate]) -> str:
    """Render one batch as the user message.

    Args:
        candidates: The batch, in presentation order.

    Returns:
        The user message content. Local ids (``C1``, ``C2``, ...) are scoped to
        the batch so the model never has to echo a database identifier, which it
        would occasionally mistype.
    """
    blocks: list[str] = []

    for position, candidate in enumerate(candidates, start=1):
        lines = [f"[C{position}]"]

        context = _render_rule_context(candidate.rule)
        if context:
            lines.append(context)

        if candidate.chunks:
            for chunk in candidate.chunks:
                # Chunk text already carries its section heading prefix.
                lines.append(f"- {chunk.text}")
        else:
            lines.append("- （该候选人无可用检索片段）")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def coerce_score(value: Any) -> Optional[float]:
    """Coerce a model-supplied score into ``[0, 100]``.

    Deliberately permissive about form and strict about range: models return
    ``82``, ``"82"``, ``"82/100"`` and occasionally ``82.5``. Rejecting all but a
    bare integer would discard usable verdicts over formatting.

    Args:
        value: The raw field.

    Returns:
        The score, or ``None`` when nothing numeric could be read.
    """
    if isinstance(value, bool):
        # bool is an int subclass; True would otherwise score as 1.
        return None

    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match is None:
            return None
        score = float(match.group())
    else:
        return None

    if not math.isfinite(score):
        return None

    return min(max(score, 0.0), 100.0)


def _as_text_list(value: Any, *, limit: int, max_chars: int) -> tuple[str, ...]:
    """Normalise a model-supplied list of short strings.

    Args:
        value: The raw field, expected to be a list.
        limit: Maximum items to keep.
        max_chars: Maximum characters per item.

    Returns:
        A tuple of cleaned strings. Non-list input yields an empty tuple rather
        than raising, since a missing evidence list is a quality problem, not a
        reason to lose an otherwise usable score.
    """
    if not isinstance(value, (list, tuple)):
        return ()

    out: list[str] = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if not text:
            continue
        out.append(text[:max_chars])
        if len(out) >= limit:
            break

    return tuple(out)


def parse_verdicts(
    payload: Any,
    local_to_resume: Mapping[str, int],
) -> tuple[dict[int, JudgeVerdict], list[int]]:
    """Turn a judge reply into verdicts, reporting anything unusable.

    Split out from the call so the failure handling can be tested without a
    provider. Every candidate in the batch is accounted for in exactly one of the
    two returned collections.

    Args:
        payload: Parsed JSON from the model.
        local_to_resume: Mapping of batch-local id (``"C1"``) to resume id.

    Returns:
        A ``(verdicts, failed)`` pair. ``failed`` lists candidates with no usable
        score, including ones the model simply omitted.
    """
    verdicts: dict[int, JudgeVerdict] = {}
    failed: list[int] = []

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        logger.error("judge reply had no results array: %r", str(payload)[:200])
        return {}, list(local_to_resume.values())

    for entry in payload["results"]:
        if not isinstance(entry, dict):
            continue

        local_id = str(entry.get("id", "")).strip().upper()
        resume_id = local_to_resume.get(local_id)

        if resume_id is None:
            # An id the model invented or mangled. Dropping it is the only safe
            # move: binding a score to a guessed candidate corrupts the ranking
            # with no error anywhere.
            logger.warning("judge returned unknown candidate id %r", local_id)
            continue

        if resume_id in verdicts:
            logger.warning("judge returned candidate %s twice; keeping the first", local_id)
            continue

        score = coerce_score(entry.get("score"))
        if score is None:
            logger.warning("judge returned unusable score for %s: %r", local_id, entry.get("score"))
            failed.append(resume_id)
            continue

        verdicts[resume_id] = JudgeVerdict(
            resume_id=resume_id,
            score=score,
            tier=tier_for_score(score),
            evidence=_as_text_list(
                entry.get("evidence"), limit=_MAX_EVIDENCE, max_chars=_MAX_EVIDENCE_CHARS
            ),
            gaps=_as_text_list(entry.get("gaps"), limit=_MAX_GAPS, max_chars=_MAX_GAP_CHARS),
        )

    # Candidates the model never mentioned. Silently dropping these would shrink
    # the shortlist without anything recording that it happened.
    for resume_id in local_to_resume.values():
        if resume_id not in verdicts and resume_id not in failed:
            failed.append(resume_id)

    return verdicts, failed


def judge(
    client: ChatModel,
    job: JobDescription,
    candidates: Sequence[JudgeCandidate],
    *,
    batch_size: Optional[int] = None,
) -> JudgeOutcome:
    """Score a shortlist in batches.

    A batch that fails outright is recorded as failed and the run continues. One
    bad reply should cost those five candidates their score, not the whole run —
    and reporting them as failed keeps the gap visible for a retry.

    Args:
        client: LLM client, sharing the run's ledger.
        job: The job description, with requirements loaded.
        candidates: The shortlist, best-first from recall.
        batch_size: Candidates per call. Defaults to config.

    Returns:
        The outcome, with verdicts and failures both accounted for.
    """
    pipeline = settings.PIPELINE
    size = pipeline.batch_size if batch_size is None else max(batch_size, 1)

    # Built once and reused verbatim: rebuilding it per batch would change the
    # prefix and turn every cache hit back into a miss.
    system_prompt = build_judge_system_prompt(job)

    verdicts: dict[int, JudgeVerdict] = {}
    failed: list[int] = []
    calls = 0

    for start in range(0, len(candidates), size):
        window = list(candidates[start : start + size])
        if not window:
            continue

        local_to_resume = {
            f"C{position}": candidate.resume_id
            for position, candidate in enumerate(window, start=1)
        }

        calls += 1
        try:
            payload = client.complete_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_batch_message(window)},
                ],
                purpose=CallPurpose.JUDGE.value,
                max_tokens=pipeline.judge_max_tokens,
                # No single resume owns a batch call, so attribution is by batch
                # size — which is also what makes the batching payoff measurable.
                resume_id=None,
                batch_size=len(window),
            )
        except LLMError as exc:
            logger.error("judge batch %d failed: %s", start // size + 1, exc)
            failed.extend(local_to_resume.values())
            continue

        batch_verdicts, batch_failed = parse_verdicts(payload, local_to_resume)
        verdicts.update(batch_verdicts)
        failed.extend(batch_failed)

    logger.info(
        "judge scored %d candidates in %d calls (%d failed)",
        len(verdicts),
        calls,
        len(failed),
    )

    return JudgeOutcome(verdicts=verdicts, failed=tuple(failed), calls=calls)
