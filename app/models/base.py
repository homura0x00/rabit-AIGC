
from sqlmodel import SQLModel, Field
from datetime import datetime, UTC


class BaseModel(SQLModel):
    """Base model with common fields."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))