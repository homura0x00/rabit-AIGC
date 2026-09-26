"""Application settings.

The environment (``.env``) is the single source of truth. ``config.yaml`` is no
longer read: it disagreed with ``.env`` on both provider and casing
(``DEEPSEEK_API_KEY`` vs ``deepseek_api_key``), so which value won was decided by
import order rather than by intent.

Cost reporting is deliberately split in two:

* **Tokens are facts.** They come from the provider's ``usage`` payload —
  including the cache-hit breakdown — and they are what this project optimises.
* **Money is an estimate.** It is derived from a price table that providers
  change without notice, so the defaults below are *placeholders*. Verify them
  against current pricing before quoting a figure, and never report a cost
  without the token counts behind it.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load an explicit path rather than letting python-dotenv search from the current
# working directory. The search-based default fails two ways:
#   * it is CWD-dependent, so starting the app from another directory silently
#     loads no configuration and every setting falls back to its default;
#   * find_dotenv() walks the call stack and raises AssertionError when the entry
#     point has no discoverable filename — which is the case for `python -` with a
#     heredoc, and for some test runners.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_FILE)


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


def get_environment() -> Environment:
    """Resolve the active environment.

    Reads ``APP_ENV`` first, then falls back to ``ENV``. This project's ``.env``
    sets ``ENV``, while the original code read only ``APP_ENV`` — so the setting
    was silently ignored and the app always ran as development no matter what was
    configured.

    Returns:
        The matching :class:`Environment`, defaulting to ``DEVELOPMENT``.
    """
    raw = os.getenv("APP_ENV") or os.getenv("ENV") or "development"
    match raw.strip().lower():
        case "production" | "prod":
            return Environment.PRODUCTION
        case "staging" | "stage":
            return Environment.STAGING
        case "test":
            return Environment.TEST
        case _:
            return Environment.DEVELOPMENT


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    try:
        return default if not raw else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    try:
        return default if not raw else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_str(name).lower()
    return default if not raw else raw in ("true", "1", "t", "yes", "on")


class MissingSecret(RuntimeError):
    """Raised when a provider credential is required but not configured."""


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices for one model.

    Prices are quoted per million tokens because that is how providers publish
    them; every calculation here divides by ``1_000_000`` exactly once to avoid
    compounding rounding mistakes across a run.
    """

    input_per_million: float
    cached_input_per_million: float
    output_per_million: float
    currency: str = "CNY"

    def estimate(
        self,
        prompt_tokens: int,
        cached_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Estimate the cost of one call.

        ``cached_tokens`` is treated as a *subset* of ``prompt_tokens`` (which is
        how OpenAI-compatible providers report it), so the full-price portion is
        the difference rather than the total.

        Args:
            prompt_tokens: Total input tokens billed for the call.
            cached_tokens: Portion of ``prompt_tokens`` served from cache.
            completion_tokens: Output tokens generated.

        Returns:
            Estimated cost in :attr:`currency`.
        """
        cached = min(max(cached_tokens, 0), max(prompt_tokens, 0))
        uncached = max(prompt_tokens, 0) - cached
        return (
            uncached / 1_000_000 * self.input_per_million
            + cached / 1_000_000 * self.cached_input_per_million
            + max(completion_tokens, 0) / 1_000_000 * self.output_per_million
        )


@dataclass(frozen=True)
class LLMSettings:
    """Chat-completion provider settings."""

    base_url: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    timeout_seconds: float
    price: ModelPrice

    def require_key(self) -> str:
        """Return the API key, or fail with an actionable message.

        Raises:
            MissingSecret: If no key is configured.
        """
        if not self.api_key:
            raise MissingSecret(
                "LLM_API_KEY / DEEPSEEK_API_KEY is not set. "
                "Add it to .env — the screening pipeline cannot run without it."
            )
        return self.api_key


@dataclass(frozen=True)
class EmbeddingSettings:
    """Embedding provider settings.

    Kept separate from :class:`LLMSettings` because the two rarely share a
    provider: the chat model may be DeepSeek while embeddings come from a
    different vendor, since DeepSeek does not serve an embedding endpoint.
    """

    base_url: str
    api_key: str
    model: str
    dimensions: int
    batch_size: int
    timeout_seconds: float
    price: ModelPrice

    def require_key(self) -> str:
        """Return the API key, or fail with an actionable message.

        Raises:
            MissingSecret: If no key is configured.
        """
        if not self.api_key:
            raise MissingSecret(
                "EMBEDDING_API_KEY / ALIYUN_API_KEY is not set. "
                "Add it to .env — the recall stage cannot run without it."
            )
        return self.api_key


@dataclass(frozen=True)
class RerankSettings:
    """Cross-encoder reranking settings.

    A separate provider block because reranking is served by a different API from
    both chat and embeddings: the OpenAI-compatible endpoint this project uses for
    chat exposes no rerank route, so it goes to the provider's native one.

    ``enabled`` defaults to True but the stage degrades quietly to "no reranking"
    when the endpoint is unreachable. That is deliberate: reranking improves the
    ordering of candidates already selected, so losing it costs precision, not
    correctness, and a screening run that fails entirely because an optional
    refinement was down would be the wrong trade.
    """

    enabled: bool
    base_url: str
    api_key: str
    model: str
    top_n: Optional[int]
    timeout_seconds: float
    price_per_million: float
    currency: str = "CNY"

    def require_key(self) -> str:
        """Return the API key, or fail with an actionable message."""
        if not self.api_key:
            raise MissingSecret(
                "RERANK_API_KEY / ALIYUN_API_KEY is not set. "
                "Add it to .env, or set RERANK_ENABLED=false."
            )
        return self.api_key

    def estimate(self, total_tokens: int) -> float:
        """Estimate the cost of a rerank call."""
        return max(total_tokens, 0) / 1_000_000 * self.price_per_million


@dataclass(frozen=True)
class PipelineSettings:
    """Funnel knobs.

    Every value here trades tokens against recall, so they are configuration
    rather than constants — the right shortlist size depends on how many
    candidates a human will actually read.
    """

    batch_size: int = 5
    """Candidate records packed into a single judge call.

    This is the single largest token lever in the pipeline: the static prefix
    (system prompt, JD, rubric) is billed once per call instead of once per
    candidate, so raising it from 1 to 5 cuts prefix cost by ~80%.
    """

    shortlist_size: int = 25
    """How many candidates survive recall and reach the paid judge stage."""

    recall_pool_size: int = 50
    """Candidates kept by embedding recall for the reranker to reorder.

    Deliberately larger than ``shortlist_size``. Bi-encoder ranking is measured to
    be unable to separate near-ties, so its ordering near the cut is close to
    arbitrary and candidates just outside the shortlist are as good as those just
    inside. Widening the pool is cheap — reranking already retrieved chunks — and
    it is what gives the cross-encoder something to actually fix.
    """

    recall_top_chunks: int = 3
    """Chunks per resume passed to the judge.

    Sending retrieved chunks rather than the full text is worth roughly a 5x
    reduction on the per-candidate payload (1500 -> ~300 tokens).
    """

    chunk_max_chars: int = 600
    chunk_overlap_chars: int = 80

    borderline_band: float = 8.0
    """Half-width, in score points, of the band around the pass line that gets a
    second look.

    The first value here was 0.15, carried over from thinking about 0-1 cosine
    similarity — but judge scores are 0-100, so a band of ±0.15 points selected
    essentially nothing and stage 4 silently never ran. Expressing it in the same
    unit as the score it bands is the whole fix.
    """

    judge_max_tokens: int = 1200
    """Output cap for one judge call.

    Scales with ``batch_size``: five candidates at roughly 70 output tokens each,
    plus JSON scaffolding. Too low truncates the reply mid-object and the whole
    batch is lost, so this is sized for the batch rather than for one candidate.
    """

    rule_min_required_share: float = 0.0
    """Share of required skills below which stage 1 drops a candidate outright.

    Defaults to 0.0 — drop only when *no* required skill matched at all.

    The first version of this filter used 0.34, and measurement showed it was
    destructive. Against a job listing five must-have skills, a Java backend
    engineer and an ML engineer each matched exactly one and were both discarded
    before any stage that could have brought them back. Six of seven test
    candidates were eliminated for a reason that was really a property of how the
    job had been described, not of the candidates.

    Two things make a conservative default the right call here. Stage 1 is a
    pruning layer with no appeal: every candidate it drops is gone for the whole
    run, and nothing downstream can recover them. And what pruning buys is
    embedding, which costs roughly a thousandth of what judging costs per token —
    so aggressive pruning saves very little and risks a great deal.

    Raise this only against a measured reason.
    """

    tier_strong: float = 80.0
    tier_qualified: float = 65.0
    tier_borderline: float = 50.0
    """Score thresholds, in descending order, that map a 0-100 score onto a tier.

    Derived in code rather than asked of the model. Requesting both a score and a
    tier invites the two to disagree, and the disagreement is invisible — the
    score says 82 and the tier says "qualified". One derived value cannot
    contradict itself, and it costs fewer output tokens to ask for one.
    """

    baseline_prefix_tokens: int = 800
    """Naive counterfactual: system prompt + JD, resent per resume.

    Used only to compute the comparison figure in the cost report. It is an
    estimate, and the report labels it as one.
    """

    baseline_output_tokens: int = 500
    """Naive counterfactual: a per-resume prose evaluation."""


class Settings:
    """Aggregate application settings.

    Sub-settings are frozen dataclasses so that a stray assignment at a call site
    cannot silently change pipeline behaviour mid-run.
    """

    def __init__(self) -> None:
        self.ENVIRONMENT = get_environment()
        self.VERSION = "0.1.0"
        self.DEBUG = _env_bool("DEBUG", False)

        # A real Postgres DSN. SUPABASE_URL holds one in this project's .env, so
        # it is the natural fallback — an earlier version did pass it to
        # create_engine but the code was documented as though it were a REST
        # endpoint. SQLite stays as the last resort so a fresh clone boots with no
        # database credentials at all, but it cannot serve pgvector, so retrieval
        # falls back to an in-process index there (see screening/store.py).
        self.DATABASE_URL = _env_str(
            "DATABASE_URL",
            _env_str("SUPABASE_URL", "sqlite:///./rabit_aigc.db"),
        )
        self.DATABASE_ECHO = _env_bool("DATABASE_ECHO", False)

        self.LLM = LLMSettings(
            base_url=_env_str("LLM_BASE_URL", "https://api.deepseek.com"),
            api_key=_env_str("LLM_API_KEY", _env_str("DEEPSEEK_API_KEY")),
            model=_env_str("LLM_MODEL", "deepseek-chat"),
            max_tokens=_env_int("LLM_MAX_TOKENS", 1024),
            temperature=_env_float("LLM_TEMPERATURE", 0.0),
            timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 120.0),
            price=ModelPrice(
                input_per_million=_env_float("PRICE_LLM_INPUT", 2.0),
                cached_input_per_million=_env_float("PRICE_LLM_CACHED_INPUT", 0.5),
                output_per_million=_env_float("PRICE_LLM_OUTPUT", 8.0),
            ),
        )

        self.EMBEDDING = EmbeddingSettings(
            base_url=_env_str(
                "EMBEDDING_BASE_URL",
                _env_str("API_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            ),
            api_key=_env_str("EMBEDDING_API_KEY", _env_str("ALIYUN_API_KEY")),
            model=_env_str("EMBEDDING_MODEL", "text-embedding-v3"),
            dimensions=_env_int("EMBEDDING_DIMENSIONS", 1024),
            # The Aliyun MaaS endpoint rejects more than 10 inputs per request
            # ("batch size is invalid, it should not be larger than 10"), so the
            # default matches the provider's ceiling rather than a guess.
            # EmbeddingClient also splits defensively, so raising this too far
            # degrades to more round trips instead of a 400.
            batch_size=_env_int("EMBEDDING_BATCH_SIZE", 10),
            timeout_seconds=_env_float("EMBEDDING_TIMEOUT_SECONDS", 60.0),
            price=ModelPrice(
                input_per_million=_env_float("PRICE_EMBEDDING_INPUT", 0.5),
                cached_input_per_million=_env_float("PRICE_EMBEDDING_INPUT", 0.5),
                output_per_million=0.0,
            ),
        )

        self.RERANK = RerankSettings(
            enabled=_env_bool("RERANK_ENABLED", True),
            base_url=_env_str(
                "RERANK_BASE_URL",
                "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            ),
            api_key=_env_str(
                "RERANK_API_KEY",
                _env_str("ALIYUN_API_KEY", _env_str("EMBEDDING_API_KEY")),
            ),
            model=_env_str("RERANK_MODEL", "gte-rerank-v2"),
            top_n=None,
            timeout_seconds=_env_float("RERANK_TIMEOUT_SECONDS", 60.0),
            price_per_million=_env_float("PRICE_RERANK_INPUT", 0.8),
        )

        self.PIPELINE = PipelineSettings(
            batch_size=_env_int("SCREEN_BATCH_SIZE", 5),
            shortlist_size=_env_int("SCREEN_SHORTLIST_SIZE", 25),
            recall_pool_size=_env_int("SCREEN_RECALL_POOL_SIZE", 50),
            recall_top_chunks=_env_int("SCREEN_RECALL_TOP_CHUNKS", 3),
            borderline_band=_env_float("SCREEN_BORDERLINE_BAND", 8.0),
            judge_max_tokens=_env_int("SCREEN_JUDGE_MAX_TOKENS", 1200),
            rule_min_required_share=_env_float("SCREEN_RULE_MIN_REQUIRED_SHARE", 0.0),
            tier_strong=_env_float("SCREEN_TIER_STRONG", 80.0),
            tier_qualified=_env_float("SCREEN_TIER_QUALIFIED", 65.0),
            tier_borderline=_env_float("SCREEN_TIER_BORDERLINE", 50.0),
        )

    @property
    def is_test(self) -> bool:
        """Whether the app is running under the test environment."""
        return self.ENVIRONMENT is Environment.TEST

    def price_for(self, model: str) -> ModelPrice:
        """Return the price table that applies to a model name.

        Chat and embedding models are billed at rates that differ by roughly two
        orders of magnitude. Pricing every call at the chat rate would overstate a
        run's cost badly, because the embedding stage is usually its largest token
        consumer by volume even though it is nearly free.

        Args:
            model: The model name recorded on the call.

        Returns:
            The matching price table, defaulting to the chat model's.
        """
        if model and model == self.EMBEDDING.model:
            return self.EMBEDDING.price
        return self.LLM.price


settings = Settings()
