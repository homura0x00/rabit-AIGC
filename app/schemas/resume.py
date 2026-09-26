"""Resume ingestion schemas."""

from pydantic import BaseModel, Field


class ResumeUploadResponse(BaseModel):
    """Result of storing one uploaded resume."""

    resume_id: int
    filename: str
    content_hash: str
    page_count: int
    char_count: int
    duplicate: bool = Field(
        default=False,
        description=(
            "True when identical normalised text was already stored. Nothing new "
            "was written and no further processing is needed for this file."
        ),
    )
