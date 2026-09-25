from typing import Optional

from app.core.config import settings
from app.core.log import logger

from sqlmodel import Session, create_engine, select
from sqlalchemy.exc import SQLAlchemyError

from app.models.condidate import Candidate


class DatabaseService:
    def __init__(self):
        try:
            self.engine = create_engine(settings.DATABASE_URL)

            logger.info("database connected")
        except SQLAlchemyError as e:
            logger.error("database_initialization_error")

    async def create_resume(self, name, email, phone):
        with Session(self.engine) as session:
            candidate = Candidate(name=name, email=email, phone=phone)
            session.add(candidate)
            session.commit()
            session.refresh(candidate)
            logger.info("candidate_created", email)
            return candidate

    async def get_condidate(self, candidate_id: int) -> Optional[Candidate]:
        with Session(self.engine) as session:
            candidate = session.get(Candidate, candidate_id)
            return candidate

    async def health_check(self) -> bool:
        """Check database connection health.

        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            with Session(self.engine) as session:
                # Execute a simple query to check connection
                session.exec(select(1)).first()
                return True
        except Exception as e:
            logger.error("database_health_check_failed", e)
            return False



database_service = DatabaseService()