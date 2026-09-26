"""Stage 2.5 — cross-encoder reranking.

Why this stage exists, in measured terms
----------------------------------------

Stage 2 ranks with a bi-encoder, which embeds the job and each chunk
independently and compares vectors. That is what makes it cheap enough to run over
everything, and it is also why its ordering is untrustworthy near the cut.
Measured on seven deliberately dissimilar resumes, the whole similarity band was
0.13 wide, and two candidates with almost nothing in common — a Java backend
engineer and a data engineer — landed **0.0001 apart**. An ordering produced by
differences that small is noise, not a ranking.

A cross-encoder reads the query and the document *together*, so it can weigh
whether a specific chunk actually answers this specific job rather than whether
the two texts live in similar regions of embedding space. It is far too expensive
to run over every chunk of every resume, which is exactly why it sits here: after
the bi-encoder has done the cheap narrowing, over a pool small enough to afford.

Measured effect, on four candidates
----------------------------------

Ranking the same seven resumes both ways and comparing each ordering against the
LLM judge's scores — the closest thing to ground truth available here — the
bi-encoder correlates **-0.20** with the judge while the cross-encoder correlates
**+0.80**, and five of seven positions change, by up to three places. The
reranker also promotes the machine-learning engineer to first, which is where the
judge independently put them, after the bi-encoder had buried them fourth.

Two caveats, stated because the numbers are easy to over-read. Four judged
candidates is a very small sample, so this is suggestive rather than established.
And the reranker's own scores are *more* compressed than the bi-encoder's — a band
of 0.031 against 0.13 — so it reorders reliably while remaining useless as an
absolute threshold. The same rule applies to it as to recall: rank, never cut on
the number.

It also explains the pool size. Because bi-encoder ordering near the cut is close
to arbitrary, candidates just outside the shortlist are as good as those just
inside, so ``recall_pool_size`` is deliberately wider than ``shortlist_size`` —
the reranker needs something to actually fix.

Two implementations
-------------------

:class:`DashScopeReranker` is the default because it works against the configured
provider today and adds no dependency. It exists because the OpenAI-compatible
endpoint this project uses for chat exposes no rerank route — verified 404 — while
the provider's native rerank endpoint serves ``gte-rerank-v2`` and reports token
usage like every other call.

:class:`LocalCrossEncoderReranker` runs ``bge-reranker`` locally via
``sentence-transformers``, behind the optional ``rerank`` extra. It costs nothing
per call and needs no network, at the price of a multi-gigabyte dependency. Both
satisfy :class:`Reranker`, so switching is a configuration change.

A failing reranker is not a failing run
---------------------------------------

When reranking is unavailable — no key, endpoint down, extra not installed — the
stage logs and returns the recall ordering untouched. Reranking improves the order
of candidates that were already selected, so losing it costs precision rather than
correctness, and aborting a screening pass because an optional refinement was
unreachable would trade a real result for a perfect one.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import httpx

from app.core.config import settings
from app.core.log import get_logger
from app.models.screening import CallPurpose
from app.services.llm import TokenLedger, Usage
from app.services.screening.recall import RecallHit

logger = get_logger(__name__)

# Documents per rerank request. Conservative: the provider does not document a
# ceiling, and the RRF-style scoring is O(query x documents) server-side, so a
# single enormous call is slower and harder to retry than a few moderate ones.
_MAX_DOCUMENTS_PER_CALL = 50


class RerankError(RuntimeError):
    """Raised when a reranking backend cannot be used."""


class Reranker(Protocol):
    """A cross-encoder that scores documents against a query."""

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[tuple[int, float]]:
        """Score documents for relevance to a query.

        Args:
            query: The job query text.
            documents: Candidate documents.

        Returns:
            ``(index, score)`` pairs, highest score first. Indexes refer to
            positions in ``documents``.
        """
        ...


@dataclass(frozen=True)
class RerankOutcome:
    """Result of reranking a recall pool.

    Attributes:
        hits: The re-ranked shortlist.
        documents_scored: Chunks sent to the reranker.
        moved: Candidates whose position changed.
        max_delta: Largest single position change, as a magnitude.
        applied: Whether a reranker actually ran. ``False`` means the recall
            ordering was returned unchanged, which a caller should be able to see
            rather than infer from identical output.
    """

    hits: list[RecallHit]
    documents_scored: int = 0
    moved: int = 0
    max_delta: int = 0
    applied: bool = False


class DashScopeReranker:
    """Reranker backed by the provider's native rerank endpoint.

    Uses ``httpx`` directly rather than an SDK: the endpoint is not
    OpenAI-compatible, so no client this project already depends on speaks it, and
    the request is one POST.
    """

    def __init__(self, ledger: Optional[TokenLedger] = None):
        """Initialise the reranker.

        Args:
            ledger: Where to record token usage.
        """
        self.ledger = ledger
        self._settings = settings.RERANK
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._settings.require_key()}",
                "Content-Type": "application/json",
            },
            timeout=self._settings.timeout_seconds,
        )

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[tuple[int, float]]:
        """Score documents via the provider, batching as needed.

        Args:
            query: The job query text.
            documents: Candidate documents.

        Returns:
            ``(index, score)`` pairs across all documents, highest first.

        Raises:
            RerankError: If a request fails.
        """
        if not documents:
            return []

        scored: list[tuple[int, float]] = []

        for start in range(0, len(documents), _MAX_DOCUMENTS_PER_CALL):
            window = list(documents[start : start + _MAX_DOCUMENTS_PER_CALL])
            scored.extend(self._score_batch(query, window, offset=start))

        # The provider already returns each batch sorted, but scores are not
        # guaranteed comparable across batches, so the final sort is what decides
        # the order a caller sees.
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _score_batch(
        self,
        query: str,
        batch: Sequence[str],
        *,
        offset: int,
    ) -> list[tuple[int, float]]:
        """Score one batch and record its usage.

        Args:
            query: The job query text.
            batch: Documents for this request.
            offset: Index of the batch's first document in the full list.

        Returns:
            ``(absolute_index, score)`` pairs.

        Raises:
            RerankError: If the request fails or the reply is malformed.
        """
        payload = {
            "model": self._settings.model,
            "input": {"query": query, "documents": list(batch)},
            "parameters": {"top_n": len(batch), "return_documents": False},
        }

        try:
            response = self._client.post(self._settings.base_url, json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise RerankError(f"rerank request failed: {exc}") from exc
        except ValueError as exc:
            raise RerankError(f"rerank returned non-JSON: {exc}") from exc

        self._record(body, len(batch))

        results = (body.get("output") or {}).get("results")
        if not isinstance(results, list):
            raise RerankError(f"rerank reply had no results array: {str(body)[:200]}")

        pairs: list[tuple[int, float]] = []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            index = entry.get("index")
            score = entry.get("relevance_score")
            if not isinstance(index, int) or not 0 <= index < len(batch):
                # An out-of-range index means the reply is not describing the
                # batch that was sent; binding it to a document would attach a
                # score to the wrong chunk with no error anywhere.
                logger.warning("rerank returned out-of-range index %r", index)
                continue
            if not isinstance(score, (int, float)):
                continue
            pairs.append((offset + index, float(score)))

        return pairs

    def _record(self, body: dict, batch_size: int) -> None:
        """Record the call's token usage, if the provider reported any."""
        if self.ledger is None:
            return

        usage = body.get("usage") or {}
        total = int(usage.get("total_tokens", 0) or 0)

        self.ledger.record(
            purpose=CallPurpose.RERANK.value,
            # Rerank is input-only: the model emits scores, not text.
            usage=Usage(
                model=self._settings.model,
                prompt_tokens=total,
                cached_tokens=0,
                completion_tokens=0,
            ),
            latency_ms=0,
            batch_size=batch_size,
        )


class LocalCrossEncoderReranker:
    """Reranker backed by a locally-run ``bge-reranker`` cross-encoder.

    Requires the optional ``rerank`` extra (``uv sync --extra rerank``). It costs
    nothing per call and needs no network, which makes it the better choice for
    volume; the trade is a multi-gigabyte dependency and model download.

    Kept behind a lazy import so the default installation does not need torch.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        ledger: Optional[TokenLedger] = None,
    ):
        """Load the model.

        Args:
            model_name: A cross-encoder from the sentence-transformers hub.
            device: Torch device string, e.g. ``"mps"`` or ``"cpu"``. ``None``
                lets sentence-transformers choose.
            ledger: Accepted for interface parity. A local model bills no tokens,
                so nothing is recorded.

        Raises:
            RerankError: If the optional dependency is not installed.
        """
        try:
            # Optional extra (``uv sync --extra rerank``); absent by default, which
            # is the whole point of the try/except.
            from sentence_transformers import CrossEncoder  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RerankError(
                "local reranking needs the optional 'rerank' extra: "
                "uv sync --extra rerank (or set RERANK_ENABLED=false)"
            ) from exc

        self.ledger = ledger
        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[tuple[int, float]]:
        """Score documents locally.

        Args:
            query: The job query text.
            documents: Candidate documents.

        Returns:
            ``(index, score)`` pairs, highest first.
        """
        if not documents:
            return []

        pairs = [(query, document) for document in documents]
        # Batch inference matters here: scoring one pair at a time on CPU throws
        # away the vectorisation that makes a local model worth running at all.
        scores = self._model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked


def create_reranker(ledger: Optional[TokenLedger] = None) -> Optional[Reranker]:
    """Build the configured reranker, or ``None`` when it is unavailable.

    Returning ``None`` rather than raising is deliberate: callers treat a missing
    reranker as "skip the stage", and a screening run should not fail because an
    optional refinement is unconfigured.

    Args:
        ledger: Where to record usage.

    Returns:
        A reranker, or ``None`` when reranking is disabled or unconfigured.
    """
    if not settings.RERANK.enabled:
        logger.info("reranking disabled by configuration")
        return None

    try:
        return DashScopeReranker(ledger)
    except Exception as exc:  # MissingSecret, network setup, malformed URL
        logger.warning("reranker unavailable, falling back to recall order: %s", exc)
        return None


def rerank_pool(
    reranker: Reranker,
    query: str,
    pool: Sequence[RecallHit],
    *,
    shortlist_size: int,
    top_chunks: int,
) -> RerankOutcome:
    """Re-score a recall pool with a cross-encoder and cut the shortlist.

    Documents are flattened across the pool and scored in one pass, then
    aggregated back per candidate with the same rule recall uses — best chunk
    wins. Reusing the rule keeps the two stages comparable, so a candidate's
    movement between them is attributable to the reranker rather than to a changed
    aggregation.

    Args:
        reranker: The cross-encoder backend.
        query: The job query text.
        pool: Candidates from recall, each carrying its selected chunks.
        shortlist_size: Candidates to keep after reranking.
        top_chunks: Chunks to retain per candidate.

    Returns:
        The re-ranked shortlist, with movement statistics.

    Raises:
        RerankError: Propagated from the backend. Callers that want a graceful
            fallback should catch it; :func:`rerank_or_fallback` does.
    """
    documents: list[str] = []
    owners: list[int] = []

    for hit in pool:
        for chunk in hit.chunks:
            documents.append(chunk.text)
            owners.append(hit.resume_id)

    if not documents:
        return RerankOutcome(hits=list(pool[:shortlist_size]))

    scored = reranker.score(query, documents)

    # Aggregate per candidate: the best chunk wins, matching recall's rule so a
    # candidate's movement between the two stages is attributable to the reranker
    # rather than to a changed aggregation.
    best_score: dict[int, float] = {}
    for index, score in scored:
        resume_id = owners[index]
        if resume_id not in best_score or score > best_score[resume_id]:
            best_score[resume_id] = score

    ordered = sorted(best_score.items(), key=lambda item: item[1], reverse=True)
    trimmed = ordered[: max(shortlist_size, 0)]

    previous_rank = {hit.resume_id: hit.rank for hit in pool}

    hits: list[RecallHit] = []
    moved = 0
    max_delta = 0

    for position, (resume_id, score) in enumerate(trimmed, start=1):
        source = next(hit for hit in pool if hit.resume_id == resume_id)
        hits.append(
            RecallHit(
                resume_id=resume_id,
                score=score,
                rank=position,
                chunks=source.chunks[: max(top_chunks, 1)],
                spread=source.spread,
            )
        )

        prior = previous_rank.get(resume_id)
        if prior is not None and prior != position:
            moved += 1
            max_delta = max(max_delta, abs(prior - position))

    logger.info(
        "reranked %d documents across %d candidates: %d moved, max delta %d",
        len(documents), len(best_score), moved, max_delta,
    )

    return RerankOutcome(
        hits=hits,
        documents_scored=len(documents),
        moved=moved,
        max_delta=max_delta,
        applied=True,
    )


def rerank_or_fallback(
    reranker: Optional[Reranker],
    query: str,
    pool: Sequence[RecallHit],
    *,
    shortlist_size: int,
    top_chunks: int,
) -> RerankOutcome:
    """Rerank when possible, otherwise return the recall ordering.

    The single place the "optional refinement must not sink the run" policy is
    implemented, so no caller has to remember it.

    Args:
        reranker: The backend, or ``None`` to skip.
        query: The job query text.
        pool: Candidates from recall.
        shortlist_size: Candidates to keep.
        top_chunks: Chunks to retain per candidate.

    Returns:
        A reranked shortlist, or the recall ordering marked ``applied=False``.
    """
    if reranker is None or not pool:
        return RerankOutcome(hits=list(pool[: max(shortlist_size, 0)]), applied=False)

    try:
        return rerank_pool(
            reranker,
            query,
            pool,
            shortlist_size=shortlist_size,
            top_chunks=top_chunks,
        )
    except RerankError as exc:
        logger.warning("reranking failed, using recall order: %s", exc)
        return RerankOutcome(hits=list(pool[: max(shortlist_size, 0)]), applied=False)
