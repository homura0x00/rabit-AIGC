"""Model registry.

A note on the ``# pyright: ignore[reportAssignmentType]`` comments on every
``__tablename__`` below and in the model modules: SQLModel declares
``__tablename__`` as a ``declared_attr``, so assigning the plain string that
SQLModel's own documentation prescribes is reported as a type error. Annotating
the attribute does not help — it then fails as an incompatible override — and
SQLModel does *not* derive snake_case from the class name, so the explicit names
are required rather than redundant. The directives are per-line and attached to
this one framework limitation instead of switching the rule off project-wide,
which would also hide genuine assignment mistakes.


Importing this package must register every table on ``SQLModel.metadata``:
``create_all`` only creates what has been imported. A model that exists as a file
but is never imported here produces a missing table at runtime, and the failure
surfaces far from the cause — so the imports are explicit and grouped by module
rather than left to whatever a service happens to import first.
"""

from app.models.base import TimestampMixin
from app.models.condidate import Candidate, Education, Experience, Project
from app.models.job import (
    DegreeLevel,
    JobDescription,
    JobSkillRequirement,
    SkillKind,
    SkillTerm,
    degree_label,
    degree_rank,
    meets_degree,
)
from app.models.screening import (
    CallPurpose,
    CandidateTier,
    LLMCallLog,
    ResumeChunk,
    ResumeDocument,
    RunStatus,
    ScreeningResult,
    ScreeningRun,
)

__all__ = [
    # base
    "TimestampMixin",
    # candidate
    "Candidate",
    "Education",
    "Experience",
    "Project",
    # job
    "DegreeLevel",
    "JobDescription",
    "JobSkillRequirement",
    "SkillKind",
    "SkillTerm",
    "degree_label",
    "degree_rank",
    "meets_degree",
    # screening
    "CallPurpose",
    "CandidateTier",
    "LLMCallLog",
    "ResumeChunk",
    "ResumeDocument",
    "RunStatus",
    "ScreeningResult",
    "ScreeningRun",
]
