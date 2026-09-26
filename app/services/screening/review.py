"""Stage 4 — borderline review, the last and narrowest slice of the funnel.

Why a fourth stage exists at all
--------------------------------

Stage 3 judges in batches of five and shows the model only the chunks retrieval
picked out — roughly 450 tokens of a resume that runs to 1200. That is the right
trade almost everywhere, but it fails in one specific and expensive way: when the
evidence a decision hinges on was not retrieved, the judge is not uncertain, it is
confidently wrong. A score that lands near the pass line is exactly where that
failure costs the most, because it is the case a human will actually argue about.

So stage 4 takes the candidates whose score sits within a band of the pass line
and re-runs them individually on the **full resume text**. It is the only place in
the pipeline where a single candidate gets a call to themselves, and it is
affordable precisely because the band keeps the set small.

What stage 4 is not
-------------------

It is not a general second opinion, and it does not re-score the whole shortlist.
Re-running everything individually would multiply the dominant cost of the
pipeline to re-derive verdicts that were never in doubt. The band is the control:
widen it and cost grows roughly linearly, narrow it and the stage quietly stops
running — which is not hypothetical. The first value of ``borderline_band`` was
``0.15``, carried over from thinking in 0-1 similarity terms while judge scores
run 0-100, so the band selected nothing and this entire module was dead code
without anything failing.

On the direction label
----------------------

The prompt asks the model whether it raised, lowered or confirmed, and the model
will sometimes contradict its own score — reporting ``lower`` alongside a higher
number. The score is the decision quantity and the label is only narration, so the
direction is always derived from the two scores, and a disagreement is counted
rather than obeyed. Silently trusting the label would flip a candidate's outcome
based on a field nothing downstream reads.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from app.core.config import settings
from app.core.log import get_logger
from app.core.prompts import load_prompt, load_system_prompt
from app.models.job import JobDescription
from app.models.screening import CallPurpose
from app.services.llm import ChatModel, LLMError
from app.services.screening.judge import (
    JudgeVerdict,
    build_job_block,
    coerce_score,
    tier_for_score,
)

logger = get_logger(__name__)

_MAX_REASON_CHARS = 160
_MAX_TEXT_CHARS = 8000

_DIRECTIONS = ("confirm", "raise", "lower")


@dataclass(frozen=True)
class ReviewTarget:
    """A candidate escalated to individual review.

    Attributes:
        resume_id: Database identifier.
        full_text: The complete normalised resume text.
        prior: The stage-3 verdict being reconsidered.
    """

    resume_id: int
    full_text: str
    prior: JudgeVerdict


@dataclass(frozen=True)
class ReviewOutcome:
    """Result of reviewing borderline candidates.

    Attributes:
        verdicts: Revised verdicts, keyed by resume id. Candidates whose review
            failed keep their stage-3 verdict and are absent here.
        changes: Direction of change per reviewed candidate.
        disagreements: Candidates where the model's own label contradicted its
            score, kept as a model-quality signal rather than acted upon.
        failed: Candidates whose review call failed.
        calls: Provider calls issued.
    """

    verdicts: dict[int, JudgeVerdict]
    changes: dict[int, str]
    disagreements: tuple[int, ...] = ()
    failed: tuple[int, ...] = ()
    calls: int = 0


def select_borderline(
    verdicts: Sequence[JudgeVerdict],
    *,
    band: Optional[float] = None,
    threshold: Optional[float] = None,
) -> list[int]:
    """Pick the candidates close enough to the pass line to be worth a second look.

    Ordered by distance from the line, nearest first, so that if the set is ever
    capped the most genuinely uncertain candidates are the ones reviewed.

    Args:
        verdicts: Stage-3 verdicts.
        band: Half-width in score points. Defaults to config.
        threshold: The pass line. Defaults to the ``qualified`` tier boundary.

    Returns:
        Resume ids, nearest to the line first.
    """
    pipeline = settings.PIPELINE
    band = pipeline.borderline_band if band is None else band
    threshold = pipeline.tier_qualified if threshold is None else threshold

    distance = {
        verdict.resume_id: abs(verdict.score - threshold)
        for verdict in verdicts
        if abs(verdict.score - threshold) <= band
    }

    return sorted(distance, key=lambda resume_id: distance[resume_id])


def build_review_system_prompt(job: JobDescription) -> str:
    """Assemble the static prefix for review calls.

    Reuses ``judge.md`` as the rubric rather than restating it. Two copies of the
    scoring criteria would drift, and the drift would be invisible: both would
    look reasonable, and the same candidate would be scored against two different
    standards depending on which stage reached them.

    Args:
        job: The job description.

    Returns:
        The system message content.
    """
    return "\n\n".join(
        [
            load_system_prompt(),
            load_prompt("judge"),
            load_prompt("review"),
            build_job_block(job),
        ]
    )


def build_review_message(target: ReviewTarget) -> str:
    """Render one review request as the user message.

    The prior score travels in the user message, never in the system prompt, so
    the prefix stays byte-identical across review calls and keeps hitting the
    provider cache.

    Args:
        target: The candidate to re-examine.

    Returns:
        The user message content.
    """
    text = target.full_text[:_MAX_TEXT_CHARS]
    truncated = "\n（材料过长，以上为截断内容）" if len(target.full_text) > _MAX_TEXT_CHARS else ""

    return "\n".join(
        [
            f"[C1] 第一轮分数：{target.prior.score:g}",
            "",
            "完整材料：",
            text + truncated,
        ]
    )


def parse_review(payload: Any, prior_score: float) -> Optional[tuple[float, str, str]]:
    """Parse a review reply into ``(score, direction, reason)``.

    The direction is derived from the two scores rather than taken from the
    model's label. See the module docstring for why.

    Args:
        payload: Parsed JSON from the model.
        prior_score: The stage-3 score being reconsidered.

    Returns:
        The revised score, derived direction and reason, or ``None`` when the
        reply carries no usable score.
    """
    if not isinstance(payload, dict):
        return None

    score = coerce_score(payload.get("score"))
    if score is None:
        return None

    if score > prior_score:
        direction = "raise"
    elif score < prior_score:
        direction = "lower"
    else:
        direction = "confirm"

    reason = " ".join(str(payload.get("reason", "")).split())[:_MAX_REASON_CHARS]

    return score, direction, reason


def review(
    client: ChatModel,
    job: JobDescription,
    targets: Sequence[ReviewTarget],
    *,
    on_disagreement: Optional[list[int]] = None,
) -> ReviewOutcome:
    """Re-score borderline candidates individually on their full text.

    One call per candidate. That is the expensive shape on purpose: the whole
    point is individual attention on a small set, and batching these would
    reintroduce the context truncation the stage exists to correct.

    Args:
        client: LLM client, sharing the run's ledger.
        job: The job description.
        targets: Candidates to reconsider.
        on_disagreement: Optional list that receives ids where the model's label
            contradicted its score.

    Returns:
        The revised verdicts and the failures, both explicit.
    """
    if not targets:
        return ReviewOutcome(verdicts={}, changes={})

    # Same reason as stage 3: built once, reused verbatim, so the prefix caches.
    system_prompt = build_review_system_prompt(job)

    verdicts: dict[int, JudgeVerdict] = {}
    changes: dict[int, str] = {}
    disagreed: list[int] = []
    failed: list[int] = []
    calls = 0

    for target in targets:
        calls += 1
        try:
            payload = client.complete_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_review_message(target)},
                ],
                purpose=CallPurpose.REVIEW.value,
                max_tokens=300,
                resume_id=target.resume_id,
                batch_size=1,
            )
        except LLMError as exc:
            logger.error("review failed for resume %s: %s", target.resume_id, exc)
            failed.append(target.resume_id)
            continue

        parsed = parse_review(payload, target.prior.score)
        if parsed is None:
            logger.warning("review returned no usable score for resume %s", target.resume_id)
            failed.append(target.resume_id)
            continue

        score, direction, reason = parsed

        label = str(payload.get("verdict", "")).strip().lower()
        if label in _DIRECTIONS and label != direction:
            disagreed.append(target.resume_id)
            logger.warning(
                "review label disagreed with score for resume %s: label=%s derived=%s (%.0f -> %.0f)",
                target.resume_id, label, direction, target.prior.score, score,
            )

        verdicts[target.resume_id] = JudgeVerdict(
            resume_id=target.resume_id,
            score=score,
            tier=tier_for_score(score),
            evidence=target.prior.evidence,
            # The prior gaps are preserved: the review did not re-derive them, and
            # replacing them with the review's rationale lost real information.
            gaps=target.prior.gaps,
            note=reason,
        )
        changes[target.resume_id] = direction

    if on_disagreement is not None:
        on_disagreement.extend(disagreed)

    logger.info(
        "reviewed %d candidates in %d calls (%d failed, %d label disagreements, %d changed)",
        len(verdicts),
        calls,
        len(failed),
        len(disagreed),
        sum(1 for direction in changes.values() if direction != "confirm"),
    )

    return ReviewOutcome(
        verdicts=verdicts,
        changes=changes,
        disagreements=tuple(disagreed),
        failed=tuple(failed),
        calls=calls,
    )
