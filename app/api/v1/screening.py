"""Screening endpoints.

Run creation is asynchronous on purpose. A screening pass makes several provider
calls and takes tens of seconds — measured at 45s for seven candidates on a slow
network day — so holding an HTTP request open for its duration is how a small
service acquires a timeout bug and a client that cannot tell "still working" from
"dead". The handler creates the run row, answers 202 with its id, and the work
happens after the response is sent.

The tradeoff is honest and worth stating: this runs in the API process, so a
restart mid-run leaves a run stuck in ``running``. A worker queue would fix that
and is the right move once runs are longer than a request budget. At this scale
the extra infrastructure would be the larger risk.
"""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlmodel import Session, col, select

from app.core.log import get_logger
from app.models.screening import ResumeDocument, RunStatus, ScreeningResult, ScreeningRun
from app.schemas.screening import (
    CostReportRead,
    ResultRead,
    RunCreate,
    RunCreated,
    RunRead,
    StageCostRead,
)
from app.models.base import require_id
from app.services.database import get_session
from app.services.jobs import load_job
from app.services.screening.pipeline import (
    PIPELINE_VERSION,
    execute_run,
    run_calls,
)
from app.services.screening.report import build_report, format_report

logger = get_logger(__name__)

router = APIRouter()


def _require_run(session: Session, run_id: int) -> ScreeningRun:
    """Load a run or raise 404.

    Args:
        session: Database session.
        run_id: The run to load.

    Returns:
        The run.

    Raises:
        HTTPException: 404 when the run does not exist.
    """
    run = session.get(ScreeningRun, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"run {run_id} not found"
        )
    return run


@router.post(
    "/screening/runs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunCreated,
    summary="Start a screening run",
)
def create_run(
    payload: RunCreate,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
) -> RunCreated:
    """Accept a screening run and execute it in the background.

    The job is validated before the run row is written, so a bad request fails
    immediately rather than asynchronously where the caller cannot see it.

    Args:
        payload: The run request.
        background: FastAPI background task queue.
        session: Injected database session.

    Returns:
        202 with the new run's id.

    Raises:
        HTTPException: 404 when the job does not exist.
    """
    job = load_job(session, payload.job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"job {payload.job_id} not found",
        )

    job_id_value = require_id(job.id, "job_description")

    run = ScreeningRun(
        job_id=job_id_value,
        status=RunStatus.PENDING.value,
        pipeline_version=PIPELINE_VERSION,
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    logger.info(
        "accepted run %s for job %s (batch_size=%s, review=%s)",
        run.id, job.id, payload.batch_size, payload.review_borderline,
    )

    background.add_task(
        execute_run,
        job_id_value,
        require_id(run.id, "screening_run"),
        resume_ids=payload.resume_ids,
        batch_size=payload.batch_size,
        review_borderline=payload.review_borderline,
    )

    return RunCreated(
        run_id=require_id(run.id, "screening_run"), status=run.status
    )


@router.get("/screening/runs", response_model=list[RunRead], summary="List screening runs")
def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> list[RunRead]:
    """List recent screening runs, newest first.

    Args:
        limit: Maximum runs to return.
        session: Injected database session.

    Returns:
        Run summaries with their measured token totals.
    """
    runs = session.exec(
        select(ScreeningRun).order_by(col(ScreeningRun.id).desc()).limit(limit)
    ).all()
    return [RunRead.from_model(run) for run in runs]


@router.get("/screening/runs/{run_id}", response_model=RunRead, summary="Get run status")
def get_run(run_id: int, session: Session = Depends(get_session)) -> RunRead:
    """Report a run's status, funnel counts and measured totals.

    Args:
        run_id: The run to inspect.
        session: Injected database session.

    Returns:
        The run.

    Raises:
        HTTPException: 404 when the run does not exist.
    """
    return RunRead.from_model(_require_run(session, run_id))


@router.get(
    "/screening/runs/{run_id}/results",
    response_model=list[ResultRead],
    summary="Get ranked candidates",
)
def get_results(
    run_id: int,
    include_rejected: bool = Query(
        default=False,
        description=(
            "Include candidates dropped by the rule stage. Off by default because "
            "the shortlist is what a reviewer acts on, but the rejections carry "
            "their reasons and are what makes the filter auditable."
        ),
    ),
    session: Session = Depends(get_session),
) -> list[ResultRead]:
    """Return a run's candidates, ranked.

    Ordering is applied in Python rather than in SQL. ``final_rank`` is null for
    candidates that never reached judgement, and NULL ordering differs between
    SQLite and Postgres — a query that puts rejections first on one and last on the
    other is a bug that only shows up after a deployment.

    Args:
        run_id: The run to inspect.
        include_rejected: Whether to include rule-stage rejections.
        session: Injected database session.

    Returns:
        Ranked results, judged candidates first.

    Raises:
        HTTPException: 404 when the run does not exist.
    """
    _require_run(session, run_id)

    rows = session.exec(
        select(ScreeningResult, ResumeDocument)
        .where(ScreeningResult.run_id == run_id)
        .join(ResumeDocument, col(ScreeningResult.resume_id) == col(ResumeDocument.id))
    ).all()

    results = [
        ResultRead.from_models(result, document.filename) for result, document in rows
    ]

    if not include_rejected:
        results = [result for result in results if result.final_rank is not None]

    results.sort(
        key=lambda item: (
            item.final_rank is None,
            item.final_rank if item.final_rank is not None else 0,
        )
    )

    return results


@router.get(
    "/screening/runs/{run_id}/cost",
    response_model=CostReportRead,
    summary="Get the cost report",
)
def get_cost(run_id: int, session: Session = Depends(get_session)) -> CostReportRead:
    """Return a run's token accounting and the naive-baseline comparison.

    ``measured`` and ``estimated`` are returned as separate objects so a client
    cannot present the counterfactual with the confidence of the observation.

    Args:
        run_id: The run to inspect.
        session: Injected database session.

    Returns:
        The cost report.

    Raises:
        HTTPException: 404 when the run does not exist.
    """
    run = _require_run(session, run_id)
    report = build_report(run, run_calls(session, run_id))

    return CostReportRead(
        run_id=report.run_id,
        currency=report.currency,
        funnel=report.funnel,
        measured={
            "tokens": report.actual_tokens,
            "est_cost": report.actual_est_cost,
        },
        estimated={
            "tokens": report.baseline_tokens,
            "est_cost": report.baseline_est_cost,
        },
        token_ratio=report.ratio,
        cost_ratio=report.cost_ratio,
        cache_hit_rate=report.cache_hit_rate,
        tokens_per_judged=report.tokens_per_judged,
        stages=[
            StageCostRead(
                purpose=stage.purpose,
                label=stage.label,
                calls=stage.calls,
                prompt_tokens=stage.prompt_tokens,
                cached_tokens=stage.cached_tokens,
                completion_tokens=stage.completion_tokens,
                total_tokens=stage.total_tokens,
                cache_hit_rate=stage.cache_hit_rate,
                avg_batch_size=stage.avg_batch_size,
                est_cost=stage.est_cost,
            )
            for stage in report.stages
        ],
        report=format_report(report),
    )
