"""The funnel, end to end.

Stage order is fixed, and the order is the entire token argument:

    L0 parse      0 tokens      every document
    L1 rules      0 tokens      every parsed document
    L2 recall     ~free         survivors of L1
    L3 judge      expensive     shortlist only
    L4 review     most per head borderline only

Each stage hands a strictly smaller set to a more expensive one. Inverting any
two adjacent stages changes nothing about correctness and multiplies the bill,
which is why this ordering lives in one place rather than being re-decided by
callers.

Where the run's numbers come from
---------------------------------

Every paid call is recorded against the run by :class:`~app.services.llm.TokenLedger`,
so the totals are measured facts read back from the provider's own usage payload
rather than estimates. The comparison figure is the opposite: nobody can measure
what a different architecture would have spent, so the baseline is explicitly an
estimate built from the real resume lengths, and it is labelled as one wherever
it is shown. Reporting a made-up baseline next to a measured total, with the same
confidence, would be the most misleading thing this project could do.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from sqlmodel import Session, col, select

from app.core.config import settings
from app.core.log import get_logger
from app.models.base import require_id
from app.models.job import JobDescription
from app.models.screening import (
    CallPurpose,
    CandidateTier,
    LLMCallLog,
    ResumeDocument,
    RunStatus,
    ScreeningResult,
    ScreeningRun,
)
from app.services.database import engine
from app.services.jobs import load_job
from app.services.llm import (
    ChatModel,
    Embedder,
    LLMClient,
    EmbeddingClient,
    TokenLedger,
    estimate_cost,
    estimate_tokens,
)
from app.services.screening.chunker import chunk_resume
from app.services.screening.judge import (
    JudgeCandidate,
    JudgeVerdict,
    build_judge_system_prompt,
    judge,
)
from app.services.screening.recall import build_job_query_text, recall
from app.services.screening.rerank import create_reranker, rerank_or_fallback
from app.services.screening.review import ReviewTarget, review, select_borderline
from app.services.screening.rules import RuleVerdict, evaluate
from app.services.screening.store import embed_pending_chunks, store_chunks

logger = get_logger(__name__)

PIPELINE_VERSION = "v1"

# Tier to next-action mapping. The tier names a bucket; an HR reviewer needs the
# decision. Keeping the wording here rather than in the prompt means it is
# consistent across runs and cannot drift with the model's phrasing.
_TIER_RECOMMENDATIONS = {
    CandidateTier.STRONG.value: "建议优先安排面试",
    CandidateTier.QUALIFIED.value: "建议进入面试流程",
    CandidateTier.BORDERLINE.value: "建议人工复核后再定",
    CandidateTier.WEAK.value: "暂不建议推进",
}


def _recommendation(tier: str) -> str:
    """Translate a tier into the action a reviewer should take.

    Args:
        tier: A :class:`~app.models.screening.CandidateTier` value.

    Returns:
        A short Chinese recommendation, or an empty string for an unknown tier.
    """
    return _TIER_RECOMMENDATIONS.get(tier, "")

# Token estimation for the baseline lives in app.services.llm.estimate_tokens,
# which is calibrated against direct provider measurements. An earlier version
# kept a flat characters-per-token constant here; it over-counted English text by
# 112% and under-counted Chinese by 10% at the same time.


@dataclass
class ScreeningOutcome:
    """Everything a caller needs to know about one run.

    Attributes:
        run_id: Identifier of the persisted :class:`ScreeningRun`.
        status: Terminal status value.
        counts: Funnel counts, stage by stage.
        verdicts: Final ranked verdicts. Candidates that never reached judgement
            are absent rather than present with a zero.
        ledger: The in-memory token ledger.
        baseline_prompt_tokens: Estimated tokens a naive approach would have spent.
        baseline_completion_tokens: Estimated tokens a naive approach would have
            produced.
        error: Failure detail when ``status`` is ``failed``.
    """

    run_id: int
    status: str
    ledger: TokenLedger
    counts: dict[str, int] = field(default_factory=dict)
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    baseline_prompt_tokens: int = 0
    baseline_completion_tokens: int = 0
    error: Optional[str] = None

    @property
    def actual_tokens(self) -> int:
        """Tokens actually billed for this run."""
        return self.ledger.totals().total_tokens

    @property
    def baseline_tokens(self) -> int:
        """Estimated tokens the naive approach would have billed."""
        return self.baseline_prompt_tokens + self.baseline_completion_tokens

    @property
    def token_ratio(self) -> Optional[float]:
        """How many times cheaper this run was, or ``None`` with nothing spent."""
        actual = self.actual_tokens
        if actual <= 0:
            return None
        return self.baseline_tokens / actual


def estimate_baseline(
    documents: Sequence[ResumeDocument],
    *,
    prefix_tokens: Optional[int] = None,
) -> tuple[int, int]:
    """Estimate what a naive per-resume evaluation would have cost.

    The counterfactual is the obvious implementation: one call per resume, full
    text, system prompt and job description resent every time, and a prose
    analysis returned.

    ``prefix_tokens`` is passed in from the *assembled* prompt rather than taken
    from a constant. The first version used a configured guess of 800 while the
    real prefix measured about 1285, which understated the baseline by roughly 40%
    — flattering in the wrong direction, and, worse, not measured. The naive
    approach would use the same prompt this pipeline uses, so its size is
    something that can be known rather than assumed.

    The result is still an estimate: it models an architecture nobody ran. Resume
    lengths and the prefix are real; the per-resume call shape is the assumption,
    and every appearance of the figure says so.

    Args:
        documents: The resumes that entered the run.
        prefix_tokens: Measured size of the shared prompt. Defaults to the
            configured fallback when a caller has no prompt to measure.

    Returns:
        A ``(prompt_tokens, completion_tokens)`` estimate.
    """
    pipeline = settings.PIPELINE
    prefix = pipeline.baseline_prefix_tokens if prefix_tokens is None else prefix_tokens

    prompt_tokens = sum(
        prefix + estimate_tokens(document.raw_text) for document in documents
    )
    completion_tokens = len(documents) * pipeline.baseline_output_tokens

    return prompt_tokens, completion_tokens


def _load_documents(
    session: Session,
    resume_ids: Optional[Iterable[int]],
) -> list[ResumeDocument]:
    """Load the resumes to screen.

    Documents that failed to parse are excluded: they have no text, so nothing
    downstream can score them, and including them would dilute the funnel counts
    with entries no stage could act on.

    Args:
        session: Database session.
        resume_ids: Restrict to these ids. ``None`` means every parsed document.

    Returns:
        Documents in id order.
    """
    statement = select(ResumeDocument).where(col(ResumeDocument.parse_error).is_(None)).order_by(col(ResumeDocument.id))
    if resume_ids is not None:
        ids = list(resume_ids)
        if not ids:
            return []
        statement = statement.where(col(ResumeDocument.id).in_(ids))

    return list(session.exec(statement).all())


def _persist_rule_stage(
    session: Session,
    run_id: int,
    rules: dict[int, RuleVerdict],
) -> dict[int, ScreeningResult]:
    """Write one result row per candidate with its stage-1 verdict.

    Rows are created before any paid stage so that a crash mid-run still leaves
    an auditable record of which candidates were eliminated for free and why.

    Args:
        session: Database session.
        run_id: Owning run.
        rules: Stage-1 verdicts keyed by resume id.

    Returns:
        The persisted rows keyed by resume id.
    """
    rows: dict[int, ScreeningResult] = {}

    for resume_id, verdict in rules.items():
        row = ScreeningResult(
            run_id=run_id,
            resume_id=resume_id,
            rule_passed=verdict.passed,
            rule_score=verdict.score,
            rule_reasons=list(verdict.reasons),
        )
        session.add(row)
        rows[resume_id] = row

    session.commit()
    for row in rows.values():
        session.refresh(row)

    return rows


def run_screening(
    session: Session,
    job: JobDescription,
    *,
    resume_ids: Optional[Iterable[int]] = None,
    llm: Optional[ChatModel] = None,
    embedder: Optional[Embedder] = None,
    batch_size: Optional[int] = None,
    review_borderline: bool = True,
    run: Optional[ScreeningRun] = None,
) -> ScreeningOutcome:
    """Run the full funnel against one job description.

    Args:
        session: Database session. The job must already be persisted.
        job: The job description, with ``requirements`` and their ``term`` loaded.
        resume_ids: Restrict to these resumes. ``None`` screens everything parsed.
        llm: Chat client. One is created if not supplied; pass one to share a
            ledger across runs.
        embedder: Embedding client, likewise.
        batch_size: Candidates per judge call. Defaults to config.
        review_borderline: Whether to run stage 4.
        run: An existing run row to use. Created when omitted.

    Returns:
        The outcome, including funnel counts and both the measured and estimated
        token totals.

    Raises:
        ValueError: If the job has not been persisted.
    """
    if job.id is None:
        raise ValueError("job must be persisted before screening; job.id is None")

    # A caller may pre-create the run so it can hand an id back before any work
    # starts — which is what lets the API answer with 202 instead of holding a
    # request open for the length of a screening pass.
    if run is None:
        run = ScreeningRun(job_id=job.id, pipeline_version=PIPELINE_VERSION)
        session.add(run)
        session.commit()
        session.refresh(run)

    ledger = TokenLedger(session=session, run_id=require_id(run.id, "screening_run"))

    # Separate locals rather than reassigning the parameters: the parameters are
    # Optional, and reusing their names leaves every later call site looking like it
    # might pass None to a stage that requires a client.
    chat: ChatModel = llm or LLMClient(ledger)
    vectors: Embedder = embedder or EmbeddingClient(ledger)

    # A caller-supplied client may carry its own ledger; keep them consistent so
    # usage cannot land in one ledger while the run reports from another.
    if chat.ledger is None:
        chat.ledger = ledger
    if vectors.ledger is None:
        vectors.ledger = ledger

    counts = {
        "documents": 0,
        "rule_passed": 0,
        "rule_rejected": 0,
        "shortlisted": 0,
        "judged": 0,
        "reviewed": 0,
        "failed": 0,
    }

    # Measured from the prompt the judge will actually send, so the baseline
    # compares against the real shared prefix rather than a configured guess.
    prefix_tokens = estimate_tokens(build_judge_system_prompt(job))

    try:
        documents = _load_documents(session, resume_ids)
        counts["documents"] = len(documents)
        run.total_documents = len(documents)
        run.status = RunStatus.RUNNING.value
        session.add(run)
        session.commit()

        if not documents:
            logger.warning("screening run %s found no documents", run.id)
            return _finalise(session, run, counts, [], documents, ledger, prefix_tokens)

        # --- L0/L2 prep: chunk and embed, no generation cost ------------------
        resume_id_of = {
            document.id: require_id(document.id, "resume_document")
            for document in documents
        }
        for document in documents:
            store_chunks(session, resume_id_of[document.id], chunk_resume(document.raw_text))
        embed_pending_chunks(
            session, vectors, resume_ids=sorted(resume_id_of.values())
        )

        # --- L1: free filtering ----------------------------------------------
        rules: dict[int, RuleVerdict] = {
            resume_id_of[document.id]: evaluate(document.raw_text, job)
            for document in documents
        }
        passed = [resume_id for resume_id, verdict in rules.items() if verdict.passed]

        counts["rule_passed"] = len(passed)
        counts["rule_rejected"] = len(documents) - len(passed)
        run.parsed_documents = len(documents)
        run.rule_rejected = counts["rule_rejected"]
        session.add(run)
        session.commit()

        result_rows = _persist_rule_stage(
            session, require_id(run.id, "screening_run"), rules
        )

        # --- L2: retrieval ranking -------------------------------------------
        # The pool is deliberately wider than the shortlist: bi-encoder ordering
        # near the cut is measured to be close to arbitrary, so candidates just
        # outside it are as good as those just inside, and the reranker below
        # needs something to actually reorder.
        pool = recall(
            session,
            job,
            embedder=vectors,
            resume_ids=passed,
            shortlist_size=settings.PIPELINE.recall_pool_size,
            top_chunks=settings.PIPELINE.recall_top_chunks,
        )

        reranked = rerank_or_fallback(
            create_reranker(ledger),
            build_job_query_text(job),
            pool,
            shortlist_size=settings.PIPELINE.shortlist_size,
            top_chunks=settings.PIPELINE.recall_top_chunks,
        )
        shortlist = reranked.hits
        counts["shortlisted"] = len(shortlist)
        counts["reranked"] = reranked.documents_scored
        run.shortlisted = len(shortlist)

        for hit in shortlist:
            row = result_rows.get(hit.resume_id)
            if row is None:
                continue
            row.recall_score = hit.score
            row.recall_rank = hit.rank
            session.add(row)
        session.commit()

        if not shortlist:
            logger.warning("screening run %s produced an empty shortlist", run.id)
            return _finalise(session, run, counts, [], documents, ledger, prefix_tokens)

        # --- L3: batched judgement -------------------------------------------
        candidates = [
            JudgeCandidate(
                resume_id=hit.resume_id,
                chunks=hit.chunks,
                rule=rules.get(hit.resume_id),
            )
            for hit in shortlist
        ]
        outcome = judge(chat, job, candidates, batch_size=batch_size)
        counts["judged"] = len(outcome.verdicts)
        counts["failed"] = len(outcome.failed)
        run.judged = len(outcome.verdicts)
        run.failed = len(outcome.failed)

        verdicts: dict[int, JudgeVerdict] = dict(outcome.verdicts)
        reviewed: set[int] = set()

        # --- L4: borderline review -------------------------------------------
        if review_borderline and verdicts:
            borderline = select_borderline(list(verdicts.values()))
            targets = [
                ReviewTarget(
                    resume_id=resume_id,
                    full_text=next(
                        document.raw_text
                        for document in documents
                        if document.id == resume_id
                    ),
                    prior=verdicts[resume_id],
                )
                for resume_id in borderline
            ]
            if targets:
                revised = review(chat, job, targets)
                verdicts.update(revised.verdicts)
                reviewed = set(revised.verdicts)
                counts["reviewed"] = len(revised.verdicts)
                counts["failed"] += len(revised.failed)
                run.reviewed = len(revised.verdicts)
                run.failed = counts["failed"]

        # --- write paid-stage results and rank --------------------------------
        ranked = sorted(verdicts.values(), key=lambda item: item.score, reverse=True)

        for position, verdict in enumerate(ranked, start=1):
            row = result_rows.get(verdict.resume_id)
            if row is None:
                continue
            row.judge_score = verdict.score
            row.judge_tier = verdict.tier
            row.judge_evidence = list(verdict.evidence)
            row.judge_gaps = list(verdict.gaps)
            row.judge_note = verdict.note or None
            row.judge_model = settings.LLM.model
            row.reviewed = verdict.resume_id in reviewed
            row.final_rank = position
            row.recommendation = _recommendation(verdict.tier)
            session.add(row)

        session.commit()

        logger.info(
            "run %s: %d documents -> %d rule-passed -> %d shortlisted -> %d judged",
            run.id,
            counts["documents"],
            counts["rule_passed"],
            counts["shortlisted"],
            counts["judged"],
        )

        return _finalise(session, run, counts, ranked, documents, ledger, prefix_tokens)

    except Exception as exc:
        # A failed run still gets its ledger written. Discarding it would hide
        # the cost of the failure, which is exactly when the number matters.
        session.rollback()
        run.status = RunStatus.FAILED.value
        run.error = f"{type(exc).__name__}: {exc}"[:1000]
        _apply_ledger(run, ledger)
        session.add(run)
        session.commit()
        logger.exception("screening run %s failed", run.id)
        raise


def _apply_ledger(run: ScreeningRun, ledger: TokenLedger) -> None:
    """Copy aggregated usage from the ledger onto the run row."""
    totals = ledger.totals()
    run.llm_calls = len(ledger.entries)
    run.prompt_tokens = totals.prompt_tokens
    run.cached_tokens = totals.cached_tokens
    run.completion_tokens = totals.completion_tokens
    # Priced per model rather than at the chat rate: embeddings make up most of a
    # run's tokens by volume and cost a fraction as much, so a single rate would
    # overstate the bill substantially.
    run.est_cost = estimate_cost(ledger.calls)


def _finalise(
    session: Session,
    run: ScreeningRun,
    counts: dict[str, int],
    ranked: Sequence[JudgeVerdict],
    documents: Sequence[ResumeDocument],
    ledger: TokenLedger,
    prefix_tokens: Optional[int] = None,
) -> ScreeningOutcome:
    """Close out a run: aggregate usage, compute the baseline, persist.

    Args:
        session: Database session.
        run: The run row.
        counts: Funnel counts.
        ranked: Final ranked verdicts.
        documents: Documents that entered the run.
        ledger: The run's token ledger.
        prefix_tokens: Measured size of the shared prompt, for the baseline.

    Returns:
        The outcome handed back to the caller.
    """
    _apply_ledger(run, ledger)

    baseline_prompt, baseline_completion = estimate_baseline(
        documents, prefix_tokens=prefix_tokens
    )
    run.baseline_prompt_tokens = baseline_prompt
    run.baseline_completion_tokens = baseline_completion
    run.baseline_est_cost = settings.LLM.price.estimate(
        baseline_prompt, 0, baseline_completion
    )

    run.status = RunStatus.SUCCEEDED.value
    session.add(run)
    session.commit()
    session.refresh(run)

    return ScreeningOutcome(
        run_id=require_id(run.id, "screening_run"),
        status=run.status,
        counts=dict(counts),
        verdicts=list(ranked),
        ledger=ledger,
        baseline_prompt_tokens=baseline_prompt,
        baseline_completion_tokens=baseline_completion,
    )


def run_calls(session: Session, run_id: int) -> list[LLMCallLog]:
    """Load every recorded call for a run, for the cost report.

    Args:
        session: Database session.
        run_id: The run to inspect.

    Returns:
        Call rows in id order.
    """
    return list(
        session.exec(
            select(LLMCallLog).where(LLMCallLog.run_id == run_id).order_by(col(LLMCallLog.id))
        ).all()
    )


def execute_run(
    job_id: int,
    run_id: int,
    *,
    resume_ids: Optional[Iterable[int]] = None,
    batch_size: Optional[int] = None,
    review_borderline: bool = True,
) -> Optional[ScreeningOutcome]:
    """Execute a pre-created run in its own database session.

    This is the background entry point. An HTTP handler creates the run row,
    answers with its id, and the work happens after the response has been sent —
    which matters because a screening pass takes tens of seconds, and holding a
    request open that long is how a small service acquires a timeout bug.

    A separate session is opened rather than reusing one: a request-scoped session
    is already closed by the time this runs.

    Failures are recorded on the run row and swallowed. There is no caller left to
    raise to, so raising would only destroy the record of what went wrong; the
    run's ``status`` and ``error`` fields are where a polling client finds it.

    Args:
        job_id: Job to screen against.
        run_id: The pre-created run row.
        resume_ids: Resumes to include. ``None`` means everything parsed.
        batch_size: Candidates per judge call.
        review_borderline: Whether to run stage 4.

    Returns:
        The outcome, or ``None`` when the job or run has vanished.
    """
    with Session(engine) as session:
        run = session.get(ScreeningRun, run_id)
        if run is None:
            logger.error("run %s disappeared before execution", run_id)
            return None

        job = load_job(session, job_id)
        if job is None:
            run.status = RunStatus.FAILED.value
            run.error = f"job {job_id} not found"
            session.add(run)
            session.commit()
            logger.error("run %s: job %s not found", run_id, job_id)
            return None

        try:
            return run_screening(
                session,
                job,
                resume_ids=resume_ids,
                batch_size=batch_size,
                review_borderline=review_borderline,
                run=run,
            )
        except Exception:
            # Already recorded on the run row by run_screening; logging here keeps
            # the traceback, which the row cannot.
            logger.exception("background screening run %s failed", run_id)
            return None
