"""HR follow-up chat schemas."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A question about one screening run."""

    run_id: int = Field(description="The screening run to ask about")
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Natural-language question, in Chinese or English",
    )


class ChatResponse(BaseModel):
    """The agent's answer and what it cost."""

    run_id: int
    answer: str
    steps: int = Field(description="Model calls made, including tool-selection rounds")
    tool_calls: list[str] = Field(
        default_factory=list,
        description=(
            "Tools invoked, in order. Surfaced because it is the difference between "
            "an answer grounded in the run's data and one the model produced from "
            "the question alone."
        ),
    )
    tokens: dict[str, int] = Field(
        default_factory=dict,
        description="Measured usage for this question, including cached input.",
    )
