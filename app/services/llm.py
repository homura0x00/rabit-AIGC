"""Provider clients and usage normalisation.

This module exists to make one thing structurally impossible: issuing a paid call
that does not get recorded. Cost accounting is the point of this project, and a
single unrecorded call silently invalidates every number in the report — the
totals would still look plausible, which is what makes the failure dangerous.

Two details of the provider's ``usage`` payload drive the design:

* **Cache hits are reported, in two places.** DeepSeek returns a top-level
  ``prompt_cache_hit_tokens``; the OpenAI-compatible shape is
  ``prompt_tokens_details.cached_tokens``. Both are read, preferring the explicit
  DeepSeek field, because cache-hit rate is what proves the static prefixes are
  being reused rather than merely being long.
* **``prompt_tokens`` includes the cached portion**, so the full-price share is
  the difference. Treating cached tokens as additive would overstate cost by
  roughly the cache-hit rate.

Verified against the live API: a repeated 2172-token prefix reports 1920 cached
tokens on the second call, so the caching this design depends on is real and not
an assumption.

Measured cache behaviour (worth knowing before designing a prompt)
------------------------------------------------------------------

Cache hits are quantised to **128-token blocks**, and roughly the last 128 tokens
of every request always miss — that tail is the user message plus a partial
block. Measured hit rates by prefix size:

    prompt tokens   166   326   646   966  1286  1926  2566
    cache hit rate   0%   39%   79%   80%   90%   93%   95%

Two consequences that shape the pipeline:

* A prefix below ~256 tokens is never cached at all. The planned JD + rubric
  prefix lands near 900 tokens, which measures at ~80% — the assumption holds,
  but only just, and a trimmed-down rubric could silently drop to zero.
* A longer prefix does *not* follow from this. Raising the hit rate from 80% to
  95% means paying for a larger prefix on every call to save a fraction of it.
  The real lever is **batching**: the prefix and its 128-token tail are billed
  once per call, so five candidates per call amortises both across five.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, Union, cast

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat.completion_create_params import ResponseFormat
from openai import (
    NOT_GIVEN,
    APIError,
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from sqlmodel import Session

from app.core.config import settings
from app.core.log import get_logger
from app.models.screening import CallPurpose, LLMCallLog

logger = get_logger(__name__)

# Transient failures worth retrying. A 400 is not in here on purpose: retrying a
# malformed request just bills the same failure five times.
_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 1.5

# Token-estimation coefficients, fitted to the measurements documented on
# estimate_tokens. Kept as named constants so it is obvious they are calibrated
# numbers rather than magic values.
_CJK_TOKENS_PER_CHAR = 0.70
_ASCII_TOKENS_PER_CHAR = 0.24


class LLMError(RuntimeError):
    """Raised when a provider call fails after all retries."""


@dataclass(frozen=True)
class Usage:
    """Normalised token accounting for one call.

    Attributes:
        model: Model that served the call.
        prompt_tokens: Total input tokens billed, cached portion included.
        cached_tokens: Portion of ``prompt_tokens`` served from the provider cache.
        completion_tokens: Output tokens generated.
    """

    model: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """All tokens billed, input and output."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def uncached_prompt_tokens(self) -> int:
        """Input tokens billed at the full rate."""
        return max(self.prompt_tokens - self.cached_tokens, 0)

    @property
    def cache_hit_rate(self) -> Optional[float]:
        """Share of input tokens served from cache, or ``None`` if no input."""
        if self.prompt_tokens <= 0:
            return None
        return self.cached_tokens / self.prompt_tokens

    def __add__(self, other: "Usage") -> "Usage":
        """Sum two usages, for aggregating a run."""
        return Usage(
            model=self.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


def extract_usage(raw: Any, model: str) -> Usage:
    """Normalise a provider ``usage`` object into :class:`Usage`.

    Args:
        raw: The ``usage`` attribute of a completion or embedding response.
        model: Model name to attribute the usage to.

    Returns:
        Normalised usage. Missing fields resolve to zero rather than raising:
        accounting that crashes is worse than accounting that under-reports, and
        a gap is visible as an implausibly round number.
    """
    if raw is None:
        return Usage(model=model)

    prompt_tokens = int(getattr(raw, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(raw, "completion_tokens", 0) or 0)

    cached = getattr(raw, "prompt_cache_hit_tokens", None)
    if cached is None:
        details = getattr(raw, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None

    # Clamp: a provider reporting more cached than prompt tokens would otherwise
    # produce a negative full-price share and a nonsensical cost.
    cached = min(int(cached or 0), prompt_tokens)

    return Usage(
        model=model,
        prompt_tokens=prompt_tokens,
        cached_tokens=max(cached, 0),
        completion_tokens=completion_tokens,
    )


@dataclass(frozen=True)
class LedgerEntry:
    """One recorded call, with the context needed to attribute it.

    The purpose and batch size are kept alongside the usage because the cost
    report has to answer "which stage spent this", and a usage figure alone cannot
    — every call looks like a number of tokens with no owner.
    """

    purpose: str
    usage: Usage
    batch_size: int = 1
    ok: bool = True
    resume_id: Optional[int] = None


def estimate_cost(usages: Iterable[Usage]) -> float:
    """Estimate the cost of a set of calls, each priced at its own model's rate.

    Args:
        usages: Normalised usages.

    Returns:
        Estimated cost in the chat model's currency.
    """
    return sum(
        settings.price_for(usage.model).estimate(
            usage.prompt_tokens, usage.cached_tokens, usage.completion_tokens
        )
        for usage in usages
    )


class TokenLedger:
    """Records every provider call and accumulates its cost.

    A ledger is optional at the call site — pass ``None`` and calls still work —
    but the pipeline always passes one. Tests use :meth:`totals` to assert on
    spend without touching a database.
    """

    def __init__(self, session: Optional[Session] = None, run_id: Optional[int] = None):
        """Initialise the ledger.

        Args:
            session: Session used to persist rows. Without one, usage is only
                accumulated in memory.
            run_id: Screening run to attribute calls to.
        """
        self.session = session
        self.run_id = run_id
        self.entries: list[LedgerEntry] = []

    @property
    def calls(self) -> list[Usage]:
        """The usage of every recorded call, in order."""
        return [entry.usage for entry in self.entries]

    def record(
        self,
        *,
        purpose: str,
        usage: Usage,
        latency_ms: int,
        resume_id: Optional[int] = None,
        batch_size: int = 1,
        ok: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record one provider call.

        Failures are recorded too. A retry storm is exactly the kind of cost
        regression that should be visible in the report rather than inferred from
        a surprising invoice.

        Args:
            purpose: Which stage issued the call.
            usage: Normalised token usage.
            latency_ms: Wall-clock duration.
            resume_id: Candidate the call concerned, when it concerned one.
            batch_size: How many candidates shared this call.
            ok: Whether the call succeeded.
            error: Error text when it did not.
        """
        self.entries.append(
            LedgerEntry(
                purpose=purpose,
                usage=usage,
                batch_size=batch_size,
                ok=ok,
                resume_id=resume_id,
            )
        )

        if self.session is None:
            return

        self.session.add(
            LLMCallLog(
                run_id=self.run_id,
                resume_id=resume_id,
                purpose=purpose,
                model=usage.model,
                prompt_tokens=usage.prompt_tokens,
                cached_tokens=usage.cached_tokens,
                completion_tokens=usage.completion_tokens,
                latency_ms=latency_ms,
                batch_size=batch_size,
                ok=ok,
                error=error[:1000] if error else None,
            )
        )
        self.session.commit()

    def totals(self) -> Usage:
        """Aggregate usage across every recorded call.

        Returns:
            The summed usage, with an empty model name since it spans providers.
        """
        total = Usage(model="")
        for entry in self.entries:
            total = total + entry.usage
        return total

    def by_purpose(self) -> dict[str, list[Usage]]:
        """Group call usage by the stage that issued it.

        Returns:
            Mapping of purpose to the usages recorded under it.
        """
        grouped: dict[str, list[Usage]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.purpose, []).append(entry.usage)
        return grouped

    @property
    def cache_hit_rate(self) -> Optional[float]:
        """Cache-hit rate across the whole ledger.

        A property rather than a method, matching :attr:`Usage.cache_hit_rate`.
        The same name with different access styles is a trap: the first version
        of this class exposed it as a method while ``Usage`` exposed it as a
        property, and formatting the ledger's value as a percentage raised.
        """
        return self.totals().cache_hit_rate


def estimate_tokens(text: str) -> int:
    """Approximate the token count of mixed CJK and ASCII text.

    No tokenizer is bundled, so this is a calibrated heuristic rather than a
    count. It exists because a single characters-per-token ratio cannot serve both
    scripts, and the first version of this project used one: a flat
    ``characters // 2`` over-counted an English resume by 112% and under-counted a
    Chinese-heavy prompt by 10%, in opposite directions, from the same constant.

    Coefficients are fitted to two direct measurements against the configured
    provider:

    ==================  ========  =======  =======
    sample              CJK chars  ASCII   tokens
    ==================  ========  =======  =======
    judge prompt         1063       472      860
    English resume          0      2405      568
    ==================  ========  =======  =======

    Solving those gives roughly 0.70 tokens per CJK character and 0.24 per ASCII
    character — Chinese runs below one token per character because common
    two-character words tokenize as single tokens.

    Fitted to two samples, so treat the output as accurate to within a few tens of
    percent, which is the right precision for a figure explicitly labelled an
    estimate and wrong for anything else. Never present it as a measurement.

    Args:
        text: The text to estimate.

    Returns:
        An approximate token count.
    """
    cjk = 0
    for char in text:
        if (
            "\u4e00" <= char <= "\u9fff"  # CJK unified ideographs
            or "\u3000" <= char <= "\u303f"  # CJK punctuation
            or "\uff00" <= char <= "\uffef"  # full-width forms
        ):
            cjk += 1

    ascii_chars = len(text) - cjk
    return int(cjk * _CJK_TOKENS_PER_CHAR + ascii_chars * _ASCII_TOKENS_PER_CHAR)


class ChatModel(Protocol):
    """The chat surface the pipeline actually depends on.

    Declared as a protocol rather than requiring a concrete :class:`LLMClient`
    because nothing downstream needs a client — it needs something that answers a
    prompt with JSON, which is also exactly what a test double provides. Without
    this every injected fake was a type error, and roughly ninety of those buried
    the handful of genuine ones that mattered.
    """

    ledger: Optional["TokenLedger"]
    """Where the implementation records usage. The pipeline fills this in when a
    caller supplies a client that has none, so accounting cannot be bypassed by
    injection."""

    def complete_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        purpose: str = ...,
        max_tokens: Optional[int] = ...,
        resume_id: Optional[int] = ...,
        batch_size: int = ...,
    ) -> Any:
        """Return a parsed JSON reply for the given messages."""
        ...


class Embedder(Protocol):
    """The embedding surface the pipeline depends on.

    Deliberately just one method: batching and provider limits are the
    implementation's business, and a caller that had to know about them would be
    reaching through the abstraction.
    """

    ledger: Optional["TokenLedger"]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in order."""
        ...


class ToolCallingChatModel(Protocol):
    """A chat model that can choose tools.

    Separate from :class:`ChatModel` because the screening stages never need tool
    calling and the agent never needs plain JSON replies. One combined protocol
    would force every test double to implement methods its stage never calls.
    """

    ledger: Optional["TokenLedger"]

    def complete_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]],
        purpose: str = ...,
        max_tokens: Optional[int] = ...,
    ) -> Any:
        """Return the assistant message, which may carry tool calls."""
        ...


class LLMClient:
    """Chat-completion client that records usage on every call."""

    def __init__(self, ledger: Optional[TokenLedger] = None):
        """Initialise the client.

        Args:
            ledger: Where to record usage. Strongly recommended.
        """
        self.ledger = ledger
        self._client = OpenAI(
            base_url=settings.LLM.base_url,
            api_key=settings.LLM.require_key(),
            timeout=settings.LLM.timeout_seconds,
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        purpose: str = CallPurpose.JUDGE.value,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        resume_id: Optional[int] = None,
        batch_size: int = 1,
        json_mode: bool = False,
    ) -> str:
        """Run one chat completion.

        Args:
            messages: Chat messages. The static prefix must come first and be
                byte-identical between calls for the provider cache to engage.
            purpose: Stage label recorded in the ledger.
            max_tokens: Output cap. Defaults to the configured value; the cap is
                a cost control as much as a safety one.
            temperature: Sampling temperature. Defaults to the configured value,
                which is 0 — screening needs to be reproducible.
            resume_id: Candidate this call concerns, if any.
            batch_size: Candidates sharing this call.
            json_mode: Request a JSON object response.

        Returns:
            The assistant message content.

        Raises:
            LLMError: If the call fails after every retry.
        """
        model = settings.LLM.model
        started = time.monotonic()
        last_error: Optional[Exception] = None

        # Annotated rather than inline: as a bare conditional expression the
        # literal narrows to dict[str, str] and fails against the SDK's
        # ResponseFormat union, which is a typing artefact and not a real mismatch.
        response_format: Union[ResponseFormat, Any] = (
            {"type": "json_object"} if json_mode else NOT_GIVEN
        )

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    # The SDK types ``messages`` as a union of TypedDicts, which a
                    # caller assembling a conversation from model output cannot
                    # satisfy statically. The boundary is asserted once, here, rather
                    # than widening every public signature to ``Any``.
                    messages=cast(
                        "Iterable[ChatCompletionMessageParam]", list(messages)
                    ),
                    max_tokens=max_tokens or settings.LLM.max_tokens,
                    temperature=(
                        settings.LLM.temperature if temperature is None else temperature
                    ),
                    # NOT_GIVEN rather than a **kwargs splat. Unpacking a conditional
                    # dict made the whole call degenerate to `**dict[str, str]` as far
                    # as a type checker was concerned, which produced fifty bogus
                    # argument errors and buried the real ones in this file.
                    response_format=response_format,
                )
            except _RETRYABLE as exc:
                last_error = exc
                logger.warning(
                    "llm call failed (attempt %d/%d) purpose=%s: %s",
                    attempt, _MAX_ATTEMPTS, purpose, exc,
                )
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_BACKOFF_SECONDS * attempt)
                continue
            except APIError as exc:
                # A non-retryable API error. Still recorded, because a failed
                # batch may have been billed.
                self._record_failure(purpose, model, started, resume_id, batch_size, exc)
                raise LLMError(f"{purpose} call failed: {exc}") from exc

            usage = extract_usage(response.usage, model)
            self._record(purpose, usage, started, resume_id, batch_size)

            content = response.choices[0].message.content or ""
            return content.strip()

        self._record_failure(
            purpose, model, started, resume_id, batch_size, last_error
        )
        raise LLMError(f"{purpose} call failed after {_MAX_ATTEMPTS} attempts: {last_error}")

    def complete_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]],
        purpose: str = CallPurpose.CHAT.value,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Run a completion with tool definitions available.

        Returns the raw assistant message rather than its text, because the caller
        has to inspect ``tool_calls`` to decide what happens next. Usage is
        recorded like any other call: the agent loop is allowed to make several
        calls, so it is the one place where an unrecorded call would go unnoticed.

        Args:
            messages: The full conversation so far.
            tools: Tool schemas in OpenAI function-calling form.
            purpose: Stage label recorded in the ledger.
            max_tokens: Output cap.

        Returns:
            The assistant message object.

        Raises:
            LLMError: If the call fails after every retry.
        """
        model = settings.LLM.model
        started = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=cast(
                        "Iterable[ChatCompletionMessageParam]", list(messages)
                    ),
                    tools=cast("Iterable[ChatCompletionToolParam]", list(tools)),
                    max_tokens=max_tokens or settings.LLM.max_tokens,
                    temperature=settings.LLM.temperature,
                )
            except _RETRYABLE as exc:
                last_error = exc
                logger.warning(
                    "tool call failed (attempt %d/%d) purpose=%s: %s",
                    attempt, _MAX_ATTEMPTS, purpose, exc,
                )
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_BACKOFF_SECONDS * attempt)
                continue
            except APIError as exc:
                self._record_failure(purpose, model, started, None, 1, exc)
                raise LLMError(f"{purpose} call failed: {exc}") from exc

            self._record(purpose, extract_usage(response.usage, model), started, None, 1)
            return response.choices[0].message

        self._record_failure(purpose, model, started, None, 1, last_error)
        raise LLMError(f"{purpose} call failed after {_MAX_ATTEMPTS} attempts: {last_error}")

    def complete_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        purpose: str = CallPurpose.JUDGE.value,
        max_tokens: Optional[int] = None,
        resume_id: Optional[int] = None,
        batch_size: int = 1,
    ) -> Any:
        """Run a completion and parse the reply as JSON.

        Args:
            messages: Chat messages; must instruct the model to emit JSON only.
            purpose: Stage label recorded in the ledger.
            max_tokens: Output cap.
            resume_id: Candidate this call concerns, if any.
            batch_size: Candidates sharing this call.

        Returns:
            The parsed JSON value.

        Raises:
            LLMError: If the call fails or the reply is not valid JSON.
        """
        raw = self.complete(
            messages,
            purpose=purpose,
            max_tokens=max_tokens,
            resume_id=resume_id,
            batch_size=batch_size,
            json_mode=True,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            # Truncation is the usual cause, so say so rather than emitting a
            # bare "Expecting value: line 1 column 1".
            raise LLMError(
                f"{purpose} returned non-JSON output "
                f"(likely truncated at max_tokens={max_tokens or settings.LLM.max_tokens}): "
                f"{raw[:200]!r}"
            ) from exc

    def _record(
        self,
        purpose: str,
        usage: Usage,
        started: float,
        resume_id: Optional[int],
        batch_size: int,
    ) -> None:
        if self.ledger is None:
            return
        self.ledger.record(
            purpose=purpose,
            usage=usage,
            latency_ms=int((time.monotonic() - started) * 1000),
            resume_id=resume_id,
            batch_size=batch_size,
        )

    def _record_failure(
        self,
        purpose: str,
        model: str,
        started: float,
        resume_id: Optional[int],
        batch_size: int,
        error: Optional[Exception],
    ) -> None:
        if self.ledger is None:
            return
        self.ledger.record(
            purpose=purpose,
            usage=Usage(model=model),
            latency_ms=int((time.monotonic() - started) * 1000),
            resume_id=resume_id,
            batch_size=batch_size,
            ok=False,
            error=str(error) if error else None,
        )


class EmbeddingClient:
    """Embedding client that records usage on every call.

    Batched deliberately: the provider accepts many inputs per request, so
    embedding 25 candidates costs far fewer round trips than 25 requests. Input
    tokens are billed either way, but latency and rate-limit exposure are not.
    """

    def __init__(self, ledger: Optional[TokenLedger] = None):
        """Initialise the client.

        Args:
            ledger: Where to record usage.
        """
        self.ledger = ledger
        self._client = OpenAI(
            base_url=settings.EMBEDDING.base_url,
            api_key=settings.EMBEDDING.require_key(),
            timeout=settings.EMBEDDING.timeout_seconds,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts, splitting into provider-legal batches.

        The batch split lives here rather than at the call site because the
        provider's ceiling is a property of the provider, not of any one caller:
        the Aliyun MaaS endpoint rejects more than 10 inputs per request with a
        400. Enforcing it in one place means no caller can accidentally exceed it,
        and raising the configured batch size degrades into more round trips
        instead of a hard failure.

        Args:
            texts: Texts to embed.

        Returns:
            One vector per input text, in the same order.

        Raises:
            LLMError: If any batch fails.
        """
        if not texts:
            return []

        cleaned = [text.strip() or "(empty)" for text in texts]
        size = max(settings.EMBEDDING.batch_size, 1)

        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), size):
            vectors.extend(self._embed_batch(cleaned[start : start + size]))

        return vectors

    def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        """Embed one provider-legal batch and record its usage.

        Args:
            batch: Non-empty list of texts, within the provider's size limit.

        Returns:
            One vector per input, in the same order.

        Raises:
            LLMError: If the call fails.
        """
        model = settings.EMBEDDING.model
        started = time.monotonic()

        try:
            response = self._client.embeddings.create(
                model=model,
                input=list(batch),
                dimensions=settings.EMBEDDING.dimensions,
            )
        except APIError as exc:
            if self.ledger is not None:
                self.ledger.record(
                    purpose=CallPurpose.EMBED.value,
                    usage=Usage(model=model),
                    latency_ms=int((time.monotonic() - started) * 1000),
                    batch_size=len(batch),
                    ok=False,
                    error=str(exc),
                )
            raise LLMError(f"embedding call failed: {exc}") from exc

        if self.ledger is not None:
            self.ledger.record(
                purpose=CallPurpose.EMBED.value,
                usage=extract_usage(response.usage, model),
                latency_ms=int((time.monotonic() - started) * 1000),
                batch_size=len(batch),
            )

        # Sort by index: the API does not guarantee response order, and silently
        # misaligning vectors with texts would corrupt every ranking downstream
        # without raising anything.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
