
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel

from app.models.base import BaseModel

# TODO 优化AI提出的Model模板

class Candidate(BaseModel, table=True):
    """Resume model for storing resume.

    Attributes:
        id: The primary key
        profile: User's profile (unique)
        education: User's education
        experience: Optional display experience for the resume
        skills: Optional display skills for the resume
        created_at: When the resume was created
    """
    id: int = Field(default=None, primary_key=True)
    name: str = Field(index=True, max_length=50)
    email: Optional[str] = Field(default=None, index=True, max_length=100)
    phone: Optional[str] = Field(default=None, max_length=30)


    educations: List["Education"] = Relationship(back_populates="candidate")
    experiences: List["Experience"] = Relationship(back_populates="candidate")
    projects: List["Project"] = Relationship(back_populates="candidate")


class Education(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", index=True)

    school: str = Field(max_length=100)
    major: Optional[str] = Field(default=None, max_length=100)
    degree: Optional[str] = Field(default=None, max_length=50)
    start_date: Optional[str] = Field(default=None, max_length=20)
    end_date: Optional[str] = Field(default=None, max_length=20)
    description: Optional[str] = Field(default=None)

    candidate: Optional[Candidate] = Relationship(back_populates="educations")

class Experience(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", index=True)

    company: str = Field(max_length=100)
    title: Optional[str] = Field(default=None, max_length=100)
    employment_type: Optional[str] = Field(default=None, max_length=20)  # 实习/全职
    start_date: Optional[str] = Field(default=None, max_length=20)
    end_date: Optional[str] = Field(default=None, max_length=20)
    description: Optional[str] = Field(default=None)

    candidate: Optional[Candidate] = Relationship(back_populates="works")

class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", index=True)

    name: str = Field(max_length=100)
    title: Optional[str] = Field(default=None, max_length=100)  # 角色
    start_date: Optional[str] = Field(default=None, max_length=20)
    end_date: Optional[str] = Field(default=None, max_length=20)
    description: Optional[str] = Field(default=None)

    candidate: Optional[Candidate] = Relationship(back_populates="projects")