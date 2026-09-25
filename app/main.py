from datetime import datetime

from fastapi import FastAPI, status
from contextlib import asynccontextmanager

from fastapi.responses import JSONResponse

from app.core.log import logger
from app.core.config import settings
from app.services.database import database_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    yield

    # Cleanup on shutdown
    logger.info("Shutdown successed.")

app = FastAPI(
    title="",
    version=settings.VERSION,
    description="",
    openapi_url="",
    lifespan=lifespan,
)

async def root():
    return 

@app.get("/health")
async def health_check():
    """Health check endpoint with environment-specific information.

    Returns:
        JSONResponse: Health status payload, with HTTP 503 when the
        database is unreachable so load balancers can drop the instance.
    """
    logger.info("health_check_called")

    # Check database connectivity
    db_healthy = await database_service.health_check()

    response = {
        "status": "healthy" if db_healthy else "degraded",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "components": {"api": "healthy", "database": "healthy" if db_healthy else "unhealthy"},
        "timestamp": datetime.now().isoformat(),
    }

    # If DB is unhealthy, set the appropriate status code
    status_code = status.HTTP_200_OK if db_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(content=response, status_code=status_code)