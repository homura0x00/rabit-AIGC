"""Chunk persistence and vector retrieval.

Why an in-process index instead of pgvector
-------------------------------------------

Supabase can enable pgvector, so this is a choice rather than a constraint, and
the choice goes the other way. At this project's scale the corpus is small: 100
resumes at roughly 7 chunks each is about 700 vectors, and a cosine sweep over
700x1024 floats is one matrix multiply that finishes in around a millisecond.
Reaching for a vector database here would buy nothing measurable and cost a
second schema to maintain, a second failure mode to debug, and a test suite that
can no longer run without a live Postgres.

The switch becomes justified by a number, not by fashion: when the sweep stops
being negligible — order 10^5 vectors, or when per-job filtering has to happen in
the database rather than in Python — replace :func:`search` with a ``<=>`` query
and enable the extension. :class:`ChunkHit` is the seam, so no caller changes.

Persisting embeddings is not an optimisation detail
---------------------------------------------------

That part is load-bearing. Because vectors are stored per chunk and chunks carry
no job reference, re-screening the same pool against a *different* job
description re-embeds one short query string and reuses every candidate vector.
The expensive half of retrieval is paid once per resume, not once per resume per
job.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from sqlmodel import Session, col, select

from app.core.config import settings
from app.core.log import get_logger
from app.models.base import require_id
from app.models.screening import ResumeChunk
from app.services.llm import Embedder
from app.services.screening.chunker import Chunk

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChunkHit:
    """One chunk returned by a similarity search.

    Attributes:
        resume_id: Owning resume.
        chunk_id: Row identifier.
        index: Position within the resume.
        section: Canonical section name.
        text: Chunk text, heading included.
        score: Cosine similarity in ``[-1, 1]``.
    """

    resume_id: int
    chunk_id: int
    index: int
    section: str
    text: str
    score: float


def _vector_is_current(row: ResumeChunk, model: str) -> bool:
    """Whether a stored vector is usable for the configured embedding model.

    Three ways a stored vector can be wrong, all of which would otherwise fail
    silently: it may be absent, produced by a different model (in which case
    similarities are meaningless), or the wrong width for the configured output
    dimensions.

    Args:
        row: The chunk row.
        model: Currently configured embedding model name.

    Returns:
        ``True`` if the vector can be used as-is.
    """
    if not row.embedding or row.embedding_model != model:
        return False
    return len(row.embedding) == settings.EMBEDDING.dimensions


def store_chunks(
    session: Session,
    resume_id: int,
    chunks: Sequence[Chunk],
    *,
    model: Optional[str] = None,
) -> list[ResumeChunk]:
    """Persist a resume's chunks, reusing vectors whose text has not changed.

    Re-parsing a resume after a parser change must not invalidate its embeddings,
    so rows are matched by position and the stored vector is kept whenever the
    text at that position is byte-identical. Only genuinely changed chunks lose
    their vector.

    Args:
        session: Database session.
        resume_id: Owning resume.
        chunks: Chunks in document order.
        model: Embedding model name to treat as current. Defaults to config.

    Returns:
        The persisted rows, in document order.
    """
    model = model or settings.EMBEDDING.model

    existing = {
        row.index: row
        for row in session.exec(
            select(ResumeChunk).where(ResumeChunk.resume_id == resume_id)
        ).all()
    }

    rows: list[ResumeChunk] = []

    for chunk in chunks:
        prior = existing.pop(chunk.index, None)

        if prior is None:
            prior = ResumeChunk(
                resume_id=resume_id,
                index=chunk.index,
                section=chunk.section,
                text=chunk.text,
                char_count=len(chunk.text),
            )
        else:
            if prior.text != chunk.text:
                # Text changed, so the old vector describes something else.
                prior.embedding = None
                prior.embedding_model = None
            prior.section = chunk.section
            prior.text = chunk.text
            prior.char_count = len(chunk.text)

        session.add(prior)
        rows.append(prior)

    # Positions beyond the new chunk count are leftovers from a longer previous
    # parse. Leaving them would let stale text compete in the index.
    for orphan in existing.values():
        session.delete(orphan)

    session.commit()
    for row in rows:
        session.refresh(row)

    return rows


def embed_pending_chunks(
    session: Session,
    embedder: Embedder,
    *,
    resume_ids: Optional[Iterable[int]] = None,
    force: bool = False,
) -> int:
    """Embed every chunk that lacks a current vector.

    Texts are grouped into provider-sized batches, so 700 chunks cost a few dozen
    requests rather than 700. Input tokens are billed either way; latency and
    rate-limit exposure are not.

    Args:
        session: Database session.
        embedder: Embedding client.
        resume_ids: Restrict to these resumes. ``None`` means every chunk.
        force: Re-embed even when a current vector exists.

    Returns:
        The number of chunks embedded.
    """
    statement = select(ResumeChunk)
    if resume_ids is not None:
        ids = list(resume_ids)
        if not ids:
            return 0
        statement = statement.where(col(ResumeChunk.resume_id).in_(ids))

    rows = list(session.exec(statement).all())

    model = settings.EMBEDDING.model
    pending = [row for row in rows if force or not _vector_is_current(row, model)]
    if not pending:
        return 0

    batch_size = max(settings.EMBEDDING.batch_size, 1)

    for start in range(0, len(pending), batch_size):
        window = pending[start : start + batch_size]
        vectors = embedder.embed([row.text for row in window])

        if len(vectors) != len(window):
            # Misalignment here would bind vectors to the wrong chunks, which
            # corrupts every ranking downstream while raising nothing.
            raise RuntimeError(
                f"embedding count mismatch: sent {len(window)}, received {len(vectors)}"
            )

        for row, vector in zip(window, vectors):
            row.embedding = vector
            row.embedding_model = model
            session.add(row)

        session.commit()

    logger.info("embedded %d chunks (%d batches)", len(pending), (len(pending) + batch_size - 1) // batch_size)
    return len(pending)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length.

    A zero vector has no direction and cannot be normalised. Dividing by its norm
    yields NaN, and a single NaN propagates through the dot product to make every
    score in the batch NaN — turning one degenerate input into a silently broken
    ranking. Such rows are left as zeros, which score 0 and sort last.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def search(
    session: Session,
    query_vector: Sequence[float],
    *,
    limit: Optional[int] = None,
    resume_ids: Optional[Iterable[int]] = None,
) -> list[ChunkHit]:
    """Find the chunks most similar to a query vector.

    Args:
        session: Database session.
        query_vector: The query embedding.
        limit: Maximum hits to return. ``None`` returns every embedded chunk,
            which is what per-resume aggregation needs: truncating globally would
            drop a candidate's best chunk just because other candidates had
            better ones.
        resume_ids: Restrict the search to these resumes.

    Returns:
        Hits ordered by descending similarity. Empty when nothing is embedded.
    """
    statement = select(ResumeChunk).where(col(ResumeChunk.embedding).is_not(None))
    if resume_ids is not None:
        ids = list(resume_ids)
        if not ids:
            return []
        statement = statement.where(col(ResumeChunk.resume_id).in_(ids))

    rows = list(session.exec(statement).all())
    if not rows:
        return []

    matrix = np.asarray([row.embedding for row in rows], dtype=np.float32)
    query = np.asarray([query_vector], dtype=np.float32)

    if matrix.shape[1] != query.shape[1]:
        # Only reachable if the configured dimensions changed without the stored
        # vectors being refreshed; say so plainly instead of raising a shape error.
        raise ValueError(
            f"embedding width mismatch: stored {matrix.shape[1]}, query {query.shape[1]}. "
            "Re-embed with embed_pending_chunks(force=True)."
        )

    # Vectors are normalised on both sides, so a dot product is the cosine.
    scores = (_normalise(matrix) @ _normalise(query).T).ravel()

    order = np.argsort(-scores)
    if limit is not None:
        order = order[: max(limit, 0)]

    return [
        ChunkHit(
            resume_id=rows[i].resume_id,
            chunk_id=require_id(rows[i].id, "resume_chunk"),
            index=rows[i].index,
            section=rows[i].section,
            text=rows[i].text,
            score=float(scores[i]),
        )
        for i in order
    ]
