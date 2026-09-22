
from typing import Optional

from sqlmodel import Field

from app.models.base import BaseModel


class Resume(BaseModel, table=True):
    """Resume model for storing resume.

    Attributes:
        id: The primary key
        profile: User's email (unique)
        education: User's education
        experience: Optional display name for the user
        created_at: When the user was created
        sessions: Relationship to user's chat sessions
    """
    id: int = Field(default=None, primary_key=True)
    profile: str = Field(default=None, index=True)
    education: str = Field(default=None)
    workexper: Optional[str] = Field()