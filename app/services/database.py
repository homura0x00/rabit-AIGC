from app.core.config import settings
from app.core.log import logger

from sqlmodel import Session, create_engine
from sqlalchemy.exc import SQLAlchemyError

from app.models.condidate import Resume


class DatabaseService:
    def __init__(self):
        try:
            self.engine = create_engine(settings.DATABASE_URL)

            logger.info("database connected")
        except SQLAlchemyError as e:
            logger.error("database_initialization_error")

    async def create_resume(self):
        with Session(self.engine) as session:
            resume = Resume()


database_service = DatabaseService()