"""Logging configuration.

Standard library ``logging`` only. The previous docstring advertised structlog,
which is neither a dependency nor imported anywhere — promising structured logs
and shipping ``basicConfig`` is worse than doing one thing clearly.

Two fixes that matter in practice:

* **Console output exists.** Logging went exclusively to ``app.log``, so running
  the API printed nothing and a failure was only visible by tailing a file.
* **The timestamp format is valid.** ``datefmt='%Y-%m-%dT%H:%S.%s+0800'`` is not
  a working strftime pattern (``%s`` is not portable), so every line carried a
  malformed timestamp.
"""

import logging
import sys
from pathlib import Path

from app.core.config import settings

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

_configured = False


def setup_logging(log_file: str | Path | None = "app.log") -> None:
    """Configure the root logger. Safe to call repeatedly.

    Args:
        log_file: File to write to in addition to stdout. Pass ``None`` to keep
            output on stdout only, which is what tests want.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        A standard library logger.
    """
    setup_logging()
    return logging.getLogger(name)


# Configured on import so that `from app.core.log import logger` behaves as
# callers already expect. The alternative — a bare module logger with no
# handlers — silently drops every INFO line to `logging.lastResort`.
setup_logging()

logger = logging.getLogger("app")
