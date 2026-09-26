"""Job description API schemas."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from app.models.base import require_id
from app.models.job import DegreeLevel, JobDescription, SkillKind


class SkillRequirementIn(BaseModel):
    """One skill requirement supplied by a caller."""

    name: str = Field(min_length=1, max_length=64, description="Canonical skill name")
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Alternative surface forms matched against resume text, e.g. "
            "['golang'] for 'Go', ['k8s'] for 'Kubernetes'. Aliases are merged "
            "into a shared dictionary, so adding one here also helps other jobs."
        ),
    )
    kind: SkillKind = SkillKind.REQUIRED
    weight: float = Field(default=1.0, gt=0, le=10)


class JobCreate(BaseModel):
    """Request body for creating a job description."""

    title: str = Field(min_length=1, max_length=120)
    department: Optional[str] = Field(default=None, max_length=120)
    location: Optional[str] = Field(default=None, max_length=80)

    min_degree: DegreeLevel = DegreeLevel.ANY
    min_years: float = Field(default=0.0, ge=0, le=50)
    accepts_internship: bool = True

    # Bounds are deliberately loose: 0.40 and 40 express the same rubric, and the
    # validator below normalises whichever scale the caller used. An earlier
    # version capped these at 1, which rejected every percentage-style request
    # before the normaliser that was written to handle exactly those could run.
    weight_skills: float = Field(default=0.40, ge=0, le=100)
    weight_experience: float = Field(default=0.35, ge=0, le=100)
    weight_education: float = Field(default=0.15, ge=0, le=100)
    weight_projects: float = Field(default=0.10, ge=0, le=100)

    skills: list[SkillRequirementIn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_rubric(self) -> "JobCreate":
        """Normalise the rubric weights and require at least one must-have skill.

        Both checks guard against a job description that looks fine and quietly
        disables part of the pipeline.

        Weights are normalised rather than rejected: a caller writing 40/35/15/10
        is expressing the same rubric as 0.40/0.35/0.15/0.10, and failing them on
        a factor of a hundred teaches nothing.

        At least one ``required`` skill is mandatory. With none, the rule stage has
        no must-haves to match, so its one hard rejection can never fire and the
        free filter silently becomes a no-op — a configuration that costs real
        money on every run and reports nothing wrong.

        Raises:
            ValueError: If all rubric weights are zero, or no required skill is
                supplied.

        Returns:
            The validated model, with weights normalised to sum to 1.
        """
        total = (
            self.weight_skills
            + self.weight_experience
            + self.weight_education
            + self.weight_projects
        )
        if total <= 0:
            raise ValueError("rubric weights must not all be zero")

        if abs(total - 1.0) > 1e-9:
            self.weight_skills /= total
            self.weight_experience /= total
            self.weight_education /= total
            self.weight_projects /= total

        if not any(skill.kind is SkillKind.REQUIRED for skill in self.skills):
            raise ValueError(
                "at least one skill with kind='required' is needed; without a "
                "must-have the rule stage has nothing to filter on and runs as a "
                "no-op"
            )

        return self


class JobSkillRead(BaseModel):
    """One skill requirement as returned by the API."""

    term_id: int
    name: str
    kind: str
    weight: float
    aliases: list[str] = Field(default_factory=list)

    @classmethod
    def from_requirement(cls, requirement) -> "JobSkillRead":
        """Build from a loaded ``JobSkillRequirement``.

        Args:
            requirement: A requirement with its ``term`` relationship resolved.

        Returns:
            The API representation.
        """
        term = requirement.term
        return cls(
            term_id=term.id if term else 0,
            name=term.name if term else "?",
            kind=requirement.kind,
            weight=requirement.weight,
            aliases=list(term.aliases or []) if term else [],
        )


class JobRead(BaseModel):
    """A job description as returned by the API."""

    id: int
    title: str
    department: Optional[str] = None
    location: Optional[str] = None
    min_degree: str
    min_years: float
    accepts_internship: bool
    rubric: dict[str, float]
    skills: list[JobSkillRead] = Field(default_factory=list)

    @classmethod
    def from_model(cls, job: JobDescription) -> "JobRead":
        """Build from a job with its requirements loaded.

        Args:
            job: The job description.

        Returns:
            The API representation.
        """
        return cls(
            id=require_id(job.id, "job_description"),
            title=job.title,
            department=job.department,
            location=job.location,
            min_degree=job.min_degree,
            min_years=job.min_years,
            accepts_internship=job.accepts_internship,
            rubric={
                "skills": job.weight_skills,
                "experience": job.weight_experience,
                "education": job.weight_education,
                "projects": job.weight_projects,
            },
            skills=[
                JobSkillRead.from_requirement(requirement)
                for requirement in (job.requirements or [])
            ],
        )


class JobSummary(BaseModel):
    """A job description without its skill requirements, for listing."""

    id: int
    title: str
    department: Optional[str] = None
    location: Optional[str] = None
    min_degree: str
    min_years: float
    is_active: bool

    @classmethod
    def from_model(cls, job: JobDescription) -> "JobSummary":
        """Build from a job row.

        Args:
            job: The job description.

        Returns:
            The API representation.
        """
        return cls(
            id=require_id(job.id, "job_description"),
            title=job.title,
            department=job.department,
            location=job.location,
            min_degree=job.min_degree,
            min_years=job.min_years,
            is_active=job.is_active,
        )
