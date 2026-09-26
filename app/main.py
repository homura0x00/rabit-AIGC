"""FastAPI application entry point.

Note what is deliberately absent: no tool-calling loop wraps the screening
pipeline. Screening is a funnel — parse, filter, rank, judge — where each stage
hands a strictly smaller set of candidates to the next, and a fixed sequence of
transforms needs no model to sequence it. Driving it with a ReAct-style loop
would resend the accumulated conversation on every step, which for a few hundred
resumes is the fastest way to burn tokens available. The agent exists only for
the HR follow-up interface, where the context is small and the interaction is
genuinely conversational.
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from app.api.v1 import api_router
from app.core.config import settings
from app.core.log import logger
from app.services.database import health_check as db_health_check
from app.services.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare the schema before serving, and log the shutdown."""
    init_db()
    logger.info(
        "startup env=%s version=%s llm=%s",
        settings.ENVIRONMENT.value,
        settings.VERSION,
        settings.LLM.model,
    )
    yield
    logger.info("shutdown complete")


app = FastAPI(
    title="Rabit AIGC — HR Resume Screening",
    version=settings.VERSION,
    description=(
        "Resume screening for a fixed candidate pool. The funnel is ordered so "
        "the cheapest stage runs first: free parsing and regex filtering, then "
        "embedding recall, then batched LLM judgement over a small shortlist. "
        "Every provider call is recorded so the saving is measurable rather than "
        "asserted."
    ),
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["ops"], summary="Service health")
async def health() -> JSONResponse:
    """Report service health.

    Returns 503 when the database is unreachable, so a load balancer drops the
    instance rather than routing traffic into guaranteed failures.

    Returns:
        A JSON health payload with an appropriate status code.
    """
    database_ok = db_health_check()

    payload = {
        "status": "healthy" if database_ok else "degraded",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "components": {
            "api": "healthy",
            "database": "healthy" if database_ok else "unhealthy",
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }

    code = status.HTTP_200_OK if database_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=payload, status_code=code)
