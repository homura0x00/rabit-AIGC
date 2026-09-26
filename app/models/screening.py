"""Screening pipeline tables.

:class:`LLMCallLog` is the reason this module exists. The project's central claim
is that it reaches a comparable shortlist to a naive per-resume evaluation for a
fraction of the tokens — and a claim like that is only worth anything if every
call is recorded: which stage issued it, against which resume, and what the
provider reported it cost, including the cache-hit split.

The cache-hit split is the part that matters most. A static prefix that is never
reused produces an identical-looking token total while costing several times
more, and without ``cached_tokens`` recorded per call that regression is
invisible. Record it, then the cost report can prove the optimisation works
rather than asserting it.
"""

from enum import Enum
from typing import List, Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Column, Field, JSON, Relationship, SQLModel

from app.models.base import TimestampMixin


class RunStatus(str, Enum):
    """Lifecycle state of a screening run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CandidateTier(str, Enum):
    """Judge verdict bucket, coarse on purpose.

    A 0-100 score invites false precision and makes the output harder to act on.
    Four buckets map directly onto what an HR reviewer does next: fast-track,
    interview, second look, pass.
    """

    STRONG = "strong"
    QUALIFIED = "qualified"
    BORDERLINE = "borderline"
    WEAK = "weak"


class CallPurpose(str, Enum):
    """Which pipeline stage issued an LLM call."""

    EXTRACT = "extract"
    JUDGE = "judge"
    REVIEW = "review"
    EMBED = "embed"
    RERANK = "rerank"
    CHAT = "chat"


class ResumeDocument(TimestampMixin, table=True):
    """A parsed resume, stored before any paid processing.

    ``content_hash`` is unique, which makes de-duplication a database guarantee
    rather than application logic. Re-uploading a resume that has already been
    processed is the cheapest possible screening run: it costs nothing and
    returns the previous verdict.
    """

    __tablename__ = "resume_document"  # pyright: ignore[reportAssignmentType]
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(max_length=255)

    content_hash: str = Field(index=True, unique=True, max_length=64)
    """SHA-256 of the normalised extracted text, not of the PDF bytes.

    Hashing the text rather than the file means a re-export or a re-save of the
    same resume still dedupes, which is the common case when candidates submit
    the same document twice.
    """

    raw_text: str = Field(default="")
    char_count: int = Field(default=0)
    page_count: int = Field(default=0)
    parse_error: Optional[str] = Field(default=None, max_length=500)

    candidate_id: Optional[int] = Field(
        default=None, foreign_key="candidate.id", index=True
    )

    result: Optional["ScreeningResult"] = Relationship(back_populates="resume")
    chunks: List["ResumeChunk"] = Relationship(back_populates="resume")


class ScreeningRun(TimestampMixin, table=True):
    """One execution of the funnel against one job description.

    Carries both the funnel counts (how many candidates each stage removed) and
    the token ledger, so a single row answers "what did this run cost, and where
    did it go".
    """

    __tablename__ = "screening_run"  # pyright: ignore[reportAssignmentType]
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job_description.id", index=True)

    status: str = Field(default=RunStatus.PENDING.value, max_length=16, index=True)
    pipeline_version: str = Field(default="v1", max_length=32)
    error: Optional[str] = Field(default=None, max_length=1000)

    # Funnel counts.
    total_documents: int = Field(default=0)
    parsed_documents: int = Field(default=0)
    rule_rejected: int = Field(default=0)
    shortlisted: int = Field(default=0)
    judged: int = Field(default=0)
    reviewed: int = Field(default=0)
    """Candidates escalated to stage-4 borderline review.

    Stored rather than derived from the call log: the log records how many review
    *calls* were made, and a failed review is still a call, so counting rows would
    overstate how many candidates actually received a second look.
    """
    failed: int = Field(default=0)

    # Actual usage, aggregated from LLMCallLog.
    llm_calls: int = Field(default=0)
    prompt_tokens: int = Field(default=0)
    cached_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    est_cost: float = Field(default=0.0)

    # Counterfactual: the same batch evaluated one-resume-per-call against full
    # text. Estimates, clearly labelled as such in the cost report.
    baseline_prompt_tokens: int = Field(default=0)
    baseline_completion_tokens: int = Field(default=0)
    baseline_est_cost: float = Field(default=0.0)

    results: List["ScreeningResult"] = Relationship(back_populates="run")
    calls: List["LLMCallLog"] = Relationship(back_populates="run")

    @property
    def total_tokens(self) -> int:
        """All tokens billed for this run, input and output."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def baseline_total_tokens(self) -> int:
        """All tokens the naive approach would have billed."""
        return self.baseline_prompt_tokens + self.baseline_completion_tokens

    @property
    def token_ratio(self) -> Optional[float]:
        """How many times cheaper this run was than the naive baseline.

        Returns:
            The ratio, or ``None`` when nothing has been spent yet and the
            comparison would be meaningless (or a division by zero).
        """
        spent = self.total_tokens
        if spent <= 0:
            return None
        return self.baseline_total_tokens / spent

    @property
    def cache_hit_rate(self) -> Optional[float]:
        """Share of input tokens served from the provider's cache.

        This is the number that proves the static prefixes are being reused. It
        should climb across the calls of a single run, because the first call in
        a run is always a miss for that prefix.
        """
        if self.prompt_tokens <= 0:
            return None
        return self.cached_tokens / self.prompt_tokens


class ScreeningResult(TimestampMixin, table=True):
    """One candidate's journey through the funnel for one run.

    Every stage writes into the same row rather than appending its own, so the
    full evidence chain — which regex fired, what recall scored it, what the
    judge concluded — stays readable in one place. HR needs to answer "why was
    this person rejected", and a row is easier to audit than a log.
    """

    __tablename__ = "screening_result"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        UniqueConstraint("run_id", "resume_id", name="uq_screening_result_run_resume"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="screening_run.id", index=True)
    resume_id: int = Field(foreign_key="resume_document.id", index=True)

    # Stage 1 — free regex/dictionary pass.
    rule_passed: bool = Field(default=True)
    rule_score: float = Field(default=0.0)
    rule_reasons: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )
    """Human-readable reasons, in both directions.

    Rejections record why; passes record what matched. Without the positive
    reasons a reviewer cannot tell a strong match from a lucky one.
    """

    # Stage 2 — retrieval ranking.
    recall_score: Optional[float] = Field(default=None)
    recall_rank: Optional[int] = Field(default=None)

    # Stage 3/4 — paid judgement.
    judge_score: Optional[float] = Field(default=None)
    judge_tier: Optional[str] = Field(default=None, max_length=16)
    judge_evidence: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )
    judge_gaps: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )
    """What the candidate is missing, as short phrases.

    Stored as well as the evidence because the two answer different questions an
    HR reviewer asks in sequence: "why is this person worth a look" and "what would
    I be taking on". Keeping only the evidence forces the second question back to
    the model, which is the expensive way to answer something already computed.
    """
    judge_note: Optional[str] = Field(default=None, max_length=500)
    """Review-stage commentary: why a borderline score was confirmed or revised."""
    judge_model: Optional[str] = Field(default=None, max_length=100)
    reviewed: bool = Field(default=False)
    """Whether this candidate was escalated to the stage-4 borderline review."""

    final_rank: Optional[int] = Field(default=None, index=True)
    recommendation: Optional[str] = Field(default=None, max_length=500)

    run: Optional[ScreeningRun] = Relationship(back_populates="results")
    resume: Optional[ResumeDocument] = Relationship(back_populates="result")


class ResumeChunk(TimestampMixin, table=True):
    """One retrievable piece of a resume, with its embedding.

    Chunks are deliberately **job-independent**. Nothing here references a job
    description, which is what makes the reuse story work: a new JD re-embeds one
    short query string and reuses every stored vector, rather than paying to
    re-process the whole candidate pool.

    ``embedding_model`` is not decoration. Vectors from different models are not
    comparable — mixing them produces similarity scores that look plausible and
    are meaningless, with no error raised anywhere. Recording the model that
    produced each vector turns that silent corruption into a detectable
    condition.
    """

    __tablename__ = "resume_chunk"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        UniqueConstraint("resume_id", "index", name="uq_resume_chunk_resume_index"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    resume_id: int = Field(foreign_key="resume_document.id", index=True)

    index: int = Field(default=0)
    """Position within the document, used to keep ordering stable."""

    section: str = Field(default="HEADER", max_length=64)
    text: str = Field(default="")
    char_count: int = Field(default=0)

    # Stored as JSON rather than a native vector column so the same schema works
    # on SQLite. See app/services/screening/store.py for why an in-process index
    # is sufficient at this scale, and what changes if it stops being.
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    embedding_model: Optional[str] = Field(default=None, max_length=100)

    @property
    def is_embedded(self) -> bool:
        """Whether this chunk carries a usable vector."""
        return bool(self.embedding)

    resume: Optional[ResumeDocument] = Relationship(back_populates="chunks")


class LLMCallLog(TimestampMixin, table=True):
    """One provider call, recorded at the moment it is billed.

    Written for every call, including failures — a retry storm is exactly the
    kind of cost regression that should be visible in the report rather than
    inferred from a surprising invoice.
    """

    __tablename__ = "llm_call_log"  # pyright: ignore[reportAssignmentType]
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: Optional[int] = Field(
        default=None, foreign_key="screening_run.id", index=True
    )
    resume_id: Optional[int] = Field(
        default=None, foreign_key="resume_document.id", index=True
    )

    purpose: str = Field(default=CallPurpose.JUDGE.value, max_length=16, index=True)
    model: str = Field(max_length=100)

    prompt_tokens: int = Field(default=0)
    cached_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)

    latency_ms: int = Field(default=0)
    batch_size: int = Field(default=1)
    """How many candidates shared this call. Recorded so the report can show the
    batching payoff directly, rather than deriving it from call counts."""

    ok: bool = Field(default=True)
    error: Optional[str] = Field(default=None, max_length=1000)

    run: Optional[ScreeningRun] = Relationship(back_populates="calls")

    @property
    def total_tokens(self) -> int:
        """All tokens billed for this call."""
        return self.prompt_tokens + self.completion_tokens
