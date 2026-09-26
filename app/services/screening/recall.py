"""Stage 2 — embedding recall.

Ranks the survivors of stage 1 by semantic similarity to the job, and selects the
shortlist that paid judgement will look at.

Why this stage exists at all
----------------------------

It is a cost gate, not a search feature. Embeddings cost roughly two orders of
magnitude less per token than generation, so scoring a hundred candidates
semantically is affordable where judging a hundred is not. Recall's job is to make
the paid stage small enough to batch aggressively while staying cheap enough to
run over everything.

Two decisions worth naming
--------------------------

* **A resume scores as its best chunk, not its average.** A resume is one document
  covering many topics, so averaging its similarity to a job punishes a strong
  candidate for having also done unrelated work — the more varied the background,
  the worse the penalty. The question recall answers is "does this resume contain
  anything highly relevant", and the maximum is the statistic that answers it.
* **The judge receives retrieved chunks, not the resume.** Top-N chunks per
  candidate takes the payload from about 1200 tokens to about 450, and because the
  chunks were chosen for relevance, it raises the information density of what the
  judge reads rather than merely shrinking it.

Neither decision saves anything if the shortlist is wrong, so recall is the stage
most worth measuring: a candidate dropped here is never seen by anything that
could bring them back.

Measured behaviour, and what it forbids
---------------------------------------

Ranking seven deliberately dissimilar resumes against one job description — a Go
and AI-agent developer, a Java backend engineer, a data engineer, an ML engineer,
a frontend developer, an HR specialist and a marketing intern — puts the relevant
candidate first and the two irrelevant ones last, so the ordering works. The
scores behind it do not:

    REAL(Go/AI)  0.6582
    java         0.6417
    data         0.6416   <- 0.0001 apart, pure noise
    ml           0.6129
    frontend     0.5664
    hr           0.5569
    marketing    0.5243

The whole band is 0.13 wide across candidates with almost nothing in common. Every
resume is a document about a person with skills and experience, so they all sit at
a similar distance from any job query, and the query's own length dilutes it
further.

Two rules follow, and both are enforced elsewhere in this codebase:

* **Never threshold on an absolute similarity score.** "Above 0.6 is qualified" is
  meaningless here; 0.64 covers both a strong Java backend engineer and a
  mediocre data engineer. Recall selects by *rank*, and the pass/fail decision is
  left to the judge, which reads evidence rather than distances.
* **Near-ties are noise, not ordering.** Scores this close cannot be separated by a
  bi-encoder, which is the measured justification for the cross-encoder rerank
  stage — not a preference for a fancier model, but a threshold on what this one
  demonstrably cannot do.
"""

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence

from sqlmodel import Session

from app.core.config import settings
from app.core.log import get_logger
from app.models.job import JobDescription, SkillKind
from app.services.llm import Embedder
from app.services.screening.store import ChunkHit, search

logger = get_logger(__name__)


@dataclass(frozen=True)
class RecallHit:
    """One candidate that survived recall.

    Attributes:
        resume_id: The candidate's resume.
        score: Best chunk similarity, in ``[-1, 1]``.
        rank: 1-based position in the shortlist.
        chunks: The most relevant chunks, capped by ``top_chunks``. This is what
            the judge stage is given.
        spread: Difference between the best and worst of the selected chunks.
            A candidate whose whole resume is relevant looks different from one
            carried by a single line, and the gap makes that visible.
    """

    resume_id: int
    score: float
    rank: int
    chunks: tuple[ChunkHit, ...] = ()
    spread: float = 0.0


def build_job_query_text(job: JobDescription) -> str:
    """Compose the text embedded as the job's search query.

    Written as a compact structured description rather than a keyword soup: both
    halves of a similarity comparison embed into the same space, and a resume is
    prose, so matching it against prose retrieves better than matching it against
    a comma-separated list.

    Requirements are read from ``job.requirements``, so the caller must have that
    relationship loaded.

    Args:
        job: The job description.

    Returns:
        A single query string.
    """
    lines: list[str] = [f"招聘岗位：{job.title}"]

    if job.department:
        lines.append(f"部门：{job.department}")
    if job.location:
        lines.append(f"工作地点：{job.location}")

    if job.min_degree and job.min_degree != "any":
        lines.append(f"学历要求：{job.min_degree}")
    if job.min_years:
        lines.append(f"工作经验要求：{job.min_years} 年以上")

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
        lines.append(f"必备技能：{'、'.join(required)}")
    if preferred:
        lines.append(f"加分技能：{'、'.join(preferred)}")

    return "\n".join(lines)


def group_by_resume(hits: Sequence[ChunkHit]) -> dict[int, list[ChunkHit]]:
    """Group chunk hits by resume, best chunk first within each.

    Args:
        hits: Flat chunk hits from a search.

    Returns:
        Mapping of resume id to its hits, ordered by descending score.
    """
    grouped: dict[int, list[ChunkHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.resume_id, []).append(hit)

    for chunk_hits in grouped.values():
        chunk_hits.sort(key=lambda item: item.score, reverse=True)

    return grouped


def rank_hits(
    grouped: dict[int, list[ChunkHit]],
    *,
    shortlist_size: int,
    top_chunks: int,
) -> list[RecallHit]:
    """Turn grouped chunk hits into a ranked shortlist.

    Split out from :func:`recall` so the ranking rule can be tested without an
    embedding call or a database.

    Args:
        grouped: Hits grouped by resume, best first.
        shortlist_size: Maximum candidates to keep.
        top_chunks: Chunks to retain per candidate for the judge.

    Returns:
        Candidates in descending score order, ranked from 1.
    """
    candidates: list[RecallHit] = []

    for resume_id, chunk_hits in grouped.items():
        selected = chunk_hits[: max(top_chunks, 1)]
        best = selected[0].score
        worst = selected[-1].score
        candidates.append(
            RecallHit(
                resume_id=resume_id,
                score=best,
                rank=0,
                chunks=tuple(selected),
                spread=best - worst,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    trimmed = candidates[: max(shortlist_size, 0)]

    return [replace(candidate, rank=position) for position, candidate in enumerate(trimmed, 1)]


def recall(
    session: Session,
    job: JobDescription,
    *,
    embedder: Embedder,
    resume_ids: Optional[Iterable[int]] = None,
    shortlist_size: Optional[int] = None,
    top_chunks: Optional[int] = None,
) -> list[RecallHit]:
    """Rank candidates against a job and return the shortlist.

    The only paid work here is embedding the query — one short string. Candidate
    vectors are read from storage, which is what makes running the same pool
    against a new job description nearly free.

    Args:
        session: Database session.
        job: The job description, with ``requirements`` and their ``term`` loaded.
        embedder: Embedding client, ideally sharing the run's ledger.
        resume_ids: Restrict to these resumes. ``None`` searches every embedded
            chunk, which would mix in candidates from other runs.
        shortlist_size: Candidates to keep. Defaults to config.
        top_chunks: Chunks to keep per candidate. Defaults to config.

    Returns:
        The ranked shortlist. Empty when nothing has been embedded yet.
    """
    pipeline = settings.PIPELINE
    shortlist_size = pipeline.shortlist_size if shortlist_size is None else shortlist_size
    top_chunks = pipeline.recall_top_chunks if top_chunks is None else top_chunks

    query_vector = embedder.embed([build_job_query_text(job)])[0]

    hits = search(session, query_vector, limit=None, resume_ids=resume_ids)
    if not hits:
        logger.warning("recall found no embedded chunks; nothing to rank")
        return []

    grouped = group_by_resume(hits)
    shortlist = rank_hits(
        grouped,
        shortlist_size=shortlist_size,
        top_chunks=top_chunks,
    )

    logger.info(
        "recall ranked %d candidates, shortlist %d (best=%.4f worst=%.4f)",
        len(grouped),
        len(shortlist),
        shortlist[0].score if shortlist else 0.0,
        shortlist[-1].score if shortlist else 0.0,
    )

    return shortlist
