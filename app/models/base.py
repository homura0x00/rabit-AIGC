"""Shared column mixins for SQLModel tables."""

from datetime import UTC, datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class TimestampMixin(SQLModel):
    """Adds a creation timestamp to a table.

    Named ``TimestampMixin`` rather than ``BaseModel`` on purpose: the previous
    name shadowed ``pydantic.BaseModel``, so every ``from ... import BaseModel``
    in the project was ambiguous to read. A mixin name should say what it adds.
    """

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )


def require_id(value: Optional[int], what: str) -> int:
    """Narrow a persisted row's primary key to a non-optional ``int``.

    SQLModel types every ``id`` as ``Optional[int]`` because it is unset before
    the insert, so a bare ``row.id`` is ``int | None`` everywhere it is read. Once
    a row has been committed and refreshed the value is always present, which
    makes the narrowing safe — but only because the caller committed.

    Writing ``row.id`` straight into a response body or a foreign key is what
    makes this worth a function: if a caller forgets to commit, ``None`` travels
    onward silently and surfaces much later as a null identifier in an API
    response or a broken join. Failing here instead turns it into an immediate
    error that names the row.

    Args:
        value: The primary key as SQLModel types it.
        what: Short description used in the error message.

    Returns:
        The identifier.

    Raises:
        RuntimeError: If the value is ``None``, meaning the row was never
            persisted.
    """
    if value is None:
        raise RuntimeError(f"{what} has no id — was it committed before use?")
    return value
