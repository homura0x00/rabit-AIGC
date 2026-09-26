"""Candidate profile models — the *extraction target*.

Where these tables sit in the pipeline is a token decision, not a detail. They
are populated only for the handful of candidates that survive the funnel, never
for every uploaded resume.

A per-resume structured extraction costs roughly 1500-2500 tokens. Running it
over all 100 uploaded resumes up front would cost more than every other stage
combined, and most of that spend would land on the ~75 resumes a free regex pass
can already rule out. So raw text and stage-1 verdicts live in
:mod:`app.models.screening`, and these tables hold the enriched profile of
finalists — the data a human actually reads and an agent actually queries.
"""

from typing import List, Optional

from sqlmodel import Column, Field, JSON, Relationship, SQLModel

from app.models.base import TimestampMixin


class Candidate(TimestampMixin, table=True):
    """An enriched candidate profile, built from a screened resume.

    Attributes:
        id: Primary key.
        name: Candidate name as extracted from the resume.
        email: Contact address, used for de-duplication across uploads.
        phone: Contact number.
        highest_degree: Canonical :class:`~app.models.job.DegreeLevel` value.
        total_years: Total professional experience in years, de-overlapped.
        skills: Canonical skill names, resolved against ``SkillTerm``.
        headline: One-line summary from the resume, used in the HR view.
    """

    __tablename__ = "candidate"  # pyright: ignore[reportAssignmentType]
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, max_length=100)
    email: Optional[str] = Field(default=None, index=True, max_length=200)
    phone: Optional[str] = Field(default=None, max_length=40)

    highest_degree: Optional[str] = Field(default=None, max_length=16)
    total_years: Optional[float] = Field(default=None)
    headline: Optional[str] = Field(default=None, max_length=300)

    # A JSON list rather than a link table. Nothing in the pipeline asks SQL
    # "which candidates have skill X" — the agent filters in Python over a
    # shortlist of tens. If that query ever appears at SQL level, promote this
    # to a link table against SkillTerm.
    skills: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )

    educations: List["Education"] = Relationship(back_populates="candidate")
    experiences: List["Experience"] = Relationship(back_populates="candidate")
    projects: List["Project"] = Relationship(back_populates="candidate")


class Education(TimestampMixin, table=True):
    """One education entry belonging to a candidate."""

    __tablename__ = "education"  # pyright: ignore[reportAssignmentType]
    id: Optional[int] = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", index=True)

    school: str = Field(max_length=150)
    major: Optional[str] = Field(default=None, max_length=150)
    degree: Optional[str] = Field(default=None, max_length=50)
    start_date: Optional[str] = Field(default=None, max_length=20)
    end_date: Optional[str] = Field(default=None, max_length=20)
    description: Optional[str] = Field(default=None)  # 主修课程、获奖荣誉、担任职务

    candidate: Optional[Candidate] = Relationship(back_populates="educations")


class Experience(TimestampMixin, table=True):
    """One work-experience entry belonging to a candidate."""

    __tablename__ = "experience"  # pyright: ignore[reportAssignmentType]
    id: Optional[int] = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", index=True)

    company: str = Field(max_length=150)
    title: Optional[str] = Field(default=None, max_length=150)
    employment_type: Optional[str] = Field(default=None, max_length=20)  # 实习/全职
    start_date: Optional[str] = Field(default=None, max_length=20)
    end_date: Optional[str] = Field(default=None, max_length=20)
    description: Optional[str] = Field(default=None)

    candidate: Optional[Candidate] = Relationship(back_populates="experiences")


class Project(TimestampMixin, table=True):
    """One project entry belonging to a candidate."""

    __tablename__ = "project"  # pyright: ignore[reportAssignmentType]
    id: Optional[int] = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", index=True)

    name: str = Field(max_length=150)
    role: Optional[str] = Field(default=None, max_length=150)
    start_date: Optional[str] = Field(default=None, max_length=20)
    end_date: Optional[str] = Field(default=None, max_length=20)
    description: Optional[str] = Field(default=None)

    candidate: Optional[Candidate] = Relationship(back_populates="projects")
