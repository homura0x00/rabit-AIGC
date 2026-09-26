"""Screening services — the funnel.

Stage order matters and is enforced by the pipeline, not by this package. The
stages are listed here in the order they run, cheapest first:

0. :mod:`~app.services.resume.parser` — PDF to text, free.
1. :mod:`~app.services.screening.rules` — regex and dictionary filtering, free.
2. :mod:`~app.services.screening.recall` — embedding similarity, very cheap.
3. judge (batch LLM) and borderline review — the only stages that cost real
   money, which is why everything above exists to make them see as few
   candidates as possible.

:mod:`~app.services.screening.chunker` and :mod:`~app.services.screening.store`
support stage 2 rather than being stages themselves.
"""

from app.services.screening.chunker import Chunk, chunk_resume, split_sections
from app.services.screening.judge import (
    JudgeCandidate,
    JudgeOutcome,
    JudgeVerdict,
    build_judge_system_prompt,
    judge,
    tier_for_score,
)
from app.services.screening.pipeline import (
    ScreeningOutcome,
    estimate_baseline,
    run_calls,
    run_screening,
)
from app.services.screening.recall import (
    RecallHit,
    build_job_query_text,
    group_by_resume,
    rank_hits,
    recall,
)
from app.services.screening.report import (
    CostReport,
    StageCost,
    breakdown_by_stage,
    build_report,
    format_report,
)
from app.services.screening.review import (
    ReviewOutcome,
    ReviewTarget,
    review,
    select_borderline,
)
from app.services.screening.rules import (
    DegreeGuess,
    ExperienceGuess,
    RuleVerdict,
    SkillMatch,
    evaluate,
    guess_degree,
    guess_experience_years,
    match_skills,
)
from app.services.screening.store import (
    ChunkHit,
    embed_pending_chunks,
    search,
    store_chunks,
)

__all__ = [
    # stage 1
    "DegreeGuess",
    "ExperienceGuess",
    "RuleVerdict",
    "SkillMatch",
    "evaluate",
    "guess_degree",
    "guess_experience_years",
    "match_skills",
    # chunking
    "Chunk",
    "chunk_resume",
    "split_sections",
    # stage 2
    "RecallHit",
    "build_job_query_text",
    "group_by_resume",
    "rank_hits",
    "recall",
    "ChunkHit",
    "embed_pending_chunks",
    "search",
    "store_chunks",
    # stage 3
    "JudgeCandidate",
    "JudgeOutcome",
    "JudgeVerdict",
    "build_judge_system_prompt",
    "judge",
    "tier_for_score",
    # stage 4
    "ReviewOutcome",
    "ReviewTarget",
    "review",
    "select_borderline",
    # orchestration
    "ScreeningOutcome",
    "estimate_baseline",
    "run_calls",
    "run_screening",
    # reporting
    "CostReport",
    "StageCost",
    "breakdown_by_stage",
    "build_report",
    "format_report",
]

