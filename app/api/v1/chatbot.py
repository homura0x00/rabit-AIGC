"""HR follow-up chat — the one place in this project where an agent is warranted.

Screening itself is a pipeline: a fixed sequence of transforms where each stage
hands a strictly smaller candidate set to the next. Nothing in it needs deciding,
so routing it through a tool-calling loop would only resend context and burn
tokens for no gain.

Conversation is the opposite case. "Which of these have Go experience?" and "why
was candidate 7 rejected?" are different queries over the same small corpus, so
the model genuinely has to choose tools — and because it only ever sees data the
pipeline already derived, never the full pile of uploaded resumes, its context
stays bounded by the shortlist rather than by how many resumes were screened.
That combination is what makes an agent the right tool here and the wrong tool
everywhere else in this codebase.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.core.log import get_logger
from app.models.screening import ScreeningRun
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent import AgentError, ask
from app.services.database import get_session

logger = get_logger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Ask about a screening run")
def chat(
    payload: ChatRequest,
    session: Session = Depends(get_session),
) -> ChatResponse:
    """Answer a follow-up question about a screening run.

    The run is checked before the agent is built so a bad identifier fails with a
    404 rather than surfacing as a provider error several calls later.

    Args:
        payload: The question and the run it concerns.
        session: Injected database session.

    Returns:
        The answer, the tools used, and the measured token cost.

    Raises:
        HTTPException: 404 when the run does not exist, 502 when the model fails.
    """
    if session.get(ScreeningRun, payload.run_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"run {payload.run_id} not found",
        )

    try:
        result = ask(session, payload.run_id, payload.question)
    except AgentError as exc:
        logger.error("chat failed for run %s: %s", payload.run_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"could not answer: {exc}",
        ) from exc

    usage = result.usage

    return ChatResponse(
        run_id=payload.run_id,
        answer=result.answer,
        steps=result.steps,
        tool_calls=list(result.tool_calls),
        tokens={
            "prompt": usage.prompt_tokens,
            "cached": usage.cached_tokens,
            "completion": usage.completion_tokens,
            "total": usage.total_tokens,
        },
    )
