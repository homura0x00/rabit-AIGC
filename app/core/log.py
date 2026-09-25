import logging

from app.core.config import settings

def setup_logging() -> None:
    """Configure structlog with different formatters based on environment.

    In development: pretty console output
    In staging/production: structured JSON logs
    """
    # Determine log level based on DEBUG setting
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    logging.basicConfig(
        filename="app.log", 
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%dT%H:%S.%s+0800',
    )

logger = logging.getLogger()

