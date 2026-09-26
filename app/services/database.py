"""Database engine, schema creation, and session management.

Two problems in the previous version are worth naming, because both fail loudly
in production and silently in development:

* ``create_engine(settings.DATABASE_URL)`` was handed the Supabase *REST* URL —
  an ``https://`` endpoint — rather than a Postgres DSN, so it could never have
  connected. The resulting error was swallowed by a ``try/except SQLAlchemyError``
  that logged and continued, leaving ``self.engine`` unset; every later call then
  died with an ``AttributeError`` that pointed nowhere near the cause.
* ``health_check`` caught bare ``Exception`` and reduced everything to a boolean,
  which hides schema and credential problems behind "unhealthy".

Here the engine is built eagerly and a bad URL fails at import, where the
traceback still names the configuration.
"""

from collections.abc import Iterator

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, SQLModel, create_engine, select

from app.core.config import settings
from app.core.log import logger


def _connect_args(url: str) -> dict:
    """Build driver-specific connect arguments.

    ``check_same_thread`` is a SQLite-only flag; passing it to psycopg2 raises.
    FastAPI runs sync dependencies in a threadpool, so the default SQLite
    thread affinity has to be relaxed for development.
    """
    return {"check_same_thread": False} if url.startswith("sqlite") else {}


engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_pre_ping=True,
    connect_args=_connect_args(settings.DATABASE_URL),
)


def init_db() -> None:
    """Create any missing tables.

    Adequate for development and for this project's scale. A real deployment
    would use migrations: ``create_all`` adds missing tables but never alters an
    existing one, so a changed column would go unnoticed until a query failed.
    """
    # Importing the registry is what registers the tables on the metadata.
    # Without it create_all sees an empty MetaData and creates nothing at all.
    import app.models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    logger.info("database schema ready (%d tables)", len(SQLModel.metadata.tables))


def get_session() -> Iterator[Session]:
    """Yield a database session, rolling back on error.

    Intended as a FastAPI dependency.

    Yields:
        An open :class:`sqlmodel.Session`.
    """
    with Session(engine) as session:
        try:
            yield session
        except Exception:
            session.rollback()
            raise


def health_check() -> bool:
    """Check whether the database answers a trivial query.

    Returns:
        ``True`` if the connection works, ``False`` otherwise. The underlying
        error is logged rather than returned, so a failure is diagnosable.
    """
    try:
        with Session(engine) as session:
            # ``select(1)`` rather than ``text("SELECT 1")``. SQLModel's ``exec``
            # is typed for its own statement objects, so a raw TextClause is an
            # argument-type error even though it runs; ``select`` is both typed and
            # one less string of SQL to keep correct.
            session.exec(select(1)).first()
        return True
    except SQLAlchemyError as exc:
        logger.warning("database health check failed: %s", exc)
        return False
