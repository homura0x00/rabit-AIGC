"""Screening API schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.base import require_id
from app.models.screening import ScreeningResult, ScreeningRun


class RunCreate(BaseModel):
    """Request body for starting a screening run."""

    job_id: int = Field(description="Job description to screen against")
    resume_ids: Optional[list[int]] = Field(
        default=None,
        description=(
            "Resumes to include. Omit to screen every parsed document, which is "
            "the normal case: the funnel is cheaper than choosing."
        ),
    )
    batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Candidates per judge call. Defaults to the configured value. Lowering "
            "it costs more, because the shared prompt is then paid more often."
        ),
    )
    review_borderline: bool = Field(
        default=True,
        description="Whether to re-score candidates near the pass line on their full text.",
    )


class RunCreated(BaseModel):
    """Response to a run being accepted."""

    run_id: int
    status: str
    detail: str = (
        "Screening runs in the background. Poll GET /screening/runs/{run_id} for status."
    )


class RunRead(BaseModel):
    """A screening run's state and measured totals."""

    run_id: int
    job_id: int
    status: str
    pipeline_version: str
    error: Optional[str] = None

    counts: dict[str, int]
    tokens: dict[str, int]
    cache_hit_rate: Optional[float] = None
    est_cost: float = 0.0

    baseline_tokens: int = 0
    baseline_est_cost: float = 0.0
    token_ratio: Optional[float] = None

    created_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, run: ScreeningRun) -> "RunRead":
        """Build from a run row.

        Args:
            run: The run.

        Returns:
            The API representation.
        """
        return cls(
            run_id=require_id(run.id, "screening_run"),
            job_id=run.job_id,
            status=run.status,
            pipeline_version=run.pipeline_version,
            error=run.error,
            counts={
                "documents": run.total_documents,
                "rule_rejected": run.rule_rejected,
                "shortlisted": run.shortlisted,
                "judged": run.judged,
                "reviewed": run.reviewed,
                "failed": run.failed,
            },
            tokens={
                "prompt": run.prompt_tokens,
                "cached": run.cached_tokens,
                "completion": run.completion_tokens,
                "total": run.total_tokens,
            },
            cache_hit_rate=run.cache_hit_rate,
            est_cost=run.est_cost,
            baseline_tokens=run.baseline_total_tokens,
            baseline_est_cost=run.baseline_est_cost,
            token_ratio=run.token_ratio,
            created_at=run.created_at,
        )


class ResultRead(BaseModel):
    """One candidate's outcome within a run."""

    resume_id: int
    filename: str
    final_rank: Optional[int] = None

    score: Optional[float] = None
    tier: Optional[str] = None
    recommendation: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    note: Optional[str] = Field(
        default=None,
        description=(
            "Review-stage commentary explaining why a borderline score was "
            "confirmed or revised. Distinct from gaps, which describe the candidate."
        ),
    )
    reviewed: bool = False

    rule_passed: bool
    rule_score: float
    rule_reasons: list[str] = Field(default_factory=list)
    recall_score: Optional[float] = None
    recall_rank: Optional[int] = None

    @classmethod
    def from_models(
        cls,
        result: ScreeningResult,
        filename: str,
    ) -> "ResultRead":
        """Build from a result row and its resume's filename.

        Args:
            result: The persisted result.
            filename: The originating resume's filename.

        Returns:
            The API representation.
        """
        return cls(
            resume_id=result.resume_id,
            filename=filename,
            final_rank=result.final_rank,
            score=result.judge_score,
            tier=result.judge_tier,
            recommendation=result.recommendation,
            evidence=list(result.judge_evidence or []),
            gaps=list(result.judge_gaps or []),
            note=result.judge_note,
            reviewed=result.reviewed,
            rule_passed=result.rule_passed,
            rule_score=result.rule_score,
            rule_reasons=list(result.rule_reasons or []),
            recall_score=result.recall_score,
            recall_rank=result.recall_rank,
        )


class StageCostRead(BaseModel):
    """One stage's contribution to a run's bill."""

    purpose: str
    label: str
    calls: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_hit_rate: Optional[float] = None
    avg_batch_size: float
    est_cost: float


class CostReportRead(BaseModel):
    """A run's full cost accounting.

    ``measured`` and ``estimated`` are separated as top-level keys rather than
    interleaved, so a client cannot render one with the other's confidence by
    accident.
    """

    run_id: Optional[int]
    currency: str
    funnel: dict[str, int]

    measured: dict[str, float] = Field(
        description="Tokens and cost read from the provider's own usage payload."
    )
    estimated: dict[str, float] = Field(
        description=(
            "The naive-baseline counterfactual. Computed, never observed — an "
            "architecture nobody ran."
        )
    )

    token_ratio: Optional[float] = None
    cost_ratio: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    tokens_per_judged: Optional[float] = None
    stages: list[StageCostRead] = Field(default_factory=list)
    report: str = Field(default="", description="The same figures rendered as text.")
