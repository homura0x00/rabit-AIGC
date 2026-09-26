"""Job description and skill-dictionary models.

The JD is stored as *structure* rather than prose, and that is a token-efficiency
decision as much as a data-modelling one. ``min_degree``, ``min_years`` and the
required-skill set are the inputs to stage 1, which is plain Python and therefore
costs nothing; skill matching is a dictionary lookup against
:attr:`SkillTerm.aliases`. A free-text JD would need a model to parse it before
anything could be filtered, which puts a paid call in front of the cheapest stage
of the funnel — the exact inversion this design is built to avoid.

The aliases are not a nicety. A matcher that only knows ``Go`` drops ``Golang``,
and one that only knows ``Kubernetes`` drops ``K8s``. Each miss is a qualified
candidate silently discarded at zero cost, which is worse than a paid mistake:
nothing in the output reveals that it happened.
"""

from enum import Enum
from typing import List, Optional

from sqlmodel import Column, Field, JSON, Relationship, SQLModel

from app.models.base import TimestampMixin


class DegreeLevel(str, Enum):
    """Canonical education levels, ordered from lowest to highest.

    Stored as plain strings. A native database enum would constrain the column at
    the type level, but it also makes later changes a migration event; at this
    stage the vocabulary is likelier to move than the data is to corrupt.
    """

    ANY = "any"
    COLLEGE = "college"  # 大专
    BACHELOR = "bachelor"  # 本科
    MASTER = "master"  # 硕士
    DOCTOR = "doctor"  # 博士


_DEGREE_RANK: dict[str, int] = {
    DegreeLevel.ANY.value: 0,
    DegreeLevel.COLLEGE.value: 1,
    DegreeLevel.BACHELOR.value: 2,
    DegreeLevel.MASTER.value: 3,
    DegreeLevel.DOCTOR.value: 4,
}


def degree_rank(value: Optional[str]) -> int:
    """Rank a stored degree string.

    Unrecognised or missing values rank as ``ANY`` (0) rather than raising. The
    extraction step is a model output and will eventually return something
    unexpected; ranking it low and letting a human see it beats crashing a run.
    """
    return _DEGREE_RANK.get((value or "").strip().lower(), 0)


def meets_degree(actual: Optional[str], required: Optional[str]) -> bool:
    """Whether ``actual`` satisfies the ``required`` floor.

    Args:
        actual: The candidate's extracted degree level.
        required: The JD's minimum, or ``None``/``"any"`` for no constraint.

    Returns:
        ``True`` when the requirement is unset or met.
    """
    if not required or required == DegreeLevel.ANY.value:
        return True
    return degree_rank(actual) >= degree_rank(required)


_DEGREE_LABELS: dict[str, str] = {
    DegreeLevel.ANY.value: "不限",
    DegreeLevel.COLLEGE.value: "大专",
    DegreeLevel.BACHELOR.value: "本科",
    DegreeLevel.MASTER.value: "硕士",
    DegreeLevel.DOCTOR.value: "博士",
}


def degree_label(level: Optional[str]) -> str:
    """Render a degree value in Chinese for prompts and HR-facing output.

    Lives here rather than in a service because three callers need the same
    wording — the rule reasons, the judge's job block, and the API response — and
    separate copies would drift into describing the same value two ways.

    Args:
        level: A :class:`DegreeLevel` value or ``None``.

    Returns:
        The Chinese label, the unrecognised value unchanged, or ``"未知"``.
    """
    key = (level or "").strip().lower()
    if not key:
        return "未知"
    return _DEGREE_LABELS.get(key, level or "未知")


class SkillKind(str, Enum):
    """Whether a skill is a hard requirement or a scoring bonus."""

    REQUIRED = "required"
    PREFERRED = "preferred"


class SkillTerm(TimestampMixin, table=True):
    """Canonical skill dictionary, shared across all jobs.

    The single source of truth for what counts as "having a skill": stage 1
    matches resume text against :attr:`aliases` and never against the canonical
    name alone.
    """

    __tablename__ = "skill_term"  # pyright: ignore[reportAssignmentType]
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True, max_length=64)
    aliases: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )
    category: Optional[str] = Field(default=None, max_length=32)

    def match_terms(self) -> List[str]:
        """Every surface form that should count as this skill, original casing.

        The canonical name is included, so callers can match against a single
        list and cannot forget it.

        Casing is preserved deliberately, and it is load-bearing. The matcher
        decides case sensitivity from a term's length — short terms like ``Go``
        must be matched case-sensitively or they collide with ordinary English
        words. Lowercasing here would silently defeat that rule and make ``Go``
        match "go to market" while failing to match "Go," in a skill list.

        Returns:
            The canonical name followed by all aliases, de-duplicated
            case-insensitively while preserving first-seen order and casing.
        """
        seen: dict[str, str] = {}
        for term in [self.name, *self.aliases]:
            cleaned = (term or "").strip()
            if cleaned:
                seen.setdefault(cleaned.lower(), cleaned)
        return list(seen.values())


class JobSkillRequirement(TimestampMixin, table=True):
    """Links a job to one skill requirement."""

    __tablename__ = "job_skill_requirement"  # pyright: ignore[reportAssignmentType]
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job_description.id", index=True)
    term_id: int = Field(foreign_key="skill_term.id", index=True)

    kind: str = Field(default=SkillKind.REQUIRED.value, max_length=16)
    weight: float = Field(default=1.0)

    job: Optional["JobDescription"] = Relationship(back_populates="requirements")
    term: Optional[SkillTerm] = Relationship()


class JobDescription(TimestampMixin, table=True):
    """A job opening, expressed as filterable structure plus rubric weights."""

    __tablename__ = "job_description"  # pyright: ignore[reportAssignmentType]
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True, max_length=120)
    department: Optional[str] = Field(default=None, max_length=120)
    location: Optional[str] = Field(default=None, max_length=80)

    min_degree: str = Field(default=DegreeLevel.ANY.value, max_length=16)
    min_years: float = Field(default=0.0)
    accepts_internship: bool = Field(default=True)

    # Rubric weights, consumed by the judge prompt. Kept on the job rather than
    # hardcoded in the prompt so that changing them is a data edit, not a deploy
    # — and so the static prompt prefix stays byte-identical and keeps hitting
    # the provider cache.
    weight_skills: float = Field(default=0.40)
    weight_experience: float = Field(default=0.35)
    weight_education: float = Field(default=0.15)
    weight_projects: float = Field(default=0.10)

    rubric_version: str = Field(default="v1", max_length=32)
    is_active: bool = Field(default=True, index=True)

    requirements: List[JobSkillRequirement] = Relationship(back_populates="job")
