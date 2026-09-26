"""Prompt loading.

Prompts are the *static prefix* of every LLM call in this project, which makes
them the single most important thing to keep byte-stable: an unchanged prefix is
exactly what provider-side context caching rewards, and a prefix that shifts by
one character silently turns every cache hit back into a cache miss.

So prompts live in files (never as f-strings at call sites) and are read once per
process. If you must edit a prompt, edit the file — do not interpolate dynamic
text into the prefix at the call site.
"""

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent
_SUFFIX = ".md"


class PromptNotFound(FileNotFoundError):
    """Raised when a requested prompt file does not exist."""


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Read a prompt by name, without the ``.md`` suffix.

    Cached for the lifetime of the process: prompts are re-read on every call
    otherwise, and more importantly a stable object identity keeps the prefix
    bytes identical between calls.

    Args:
        name: Prompt stem, e.g. ``"judge"`` for ``judge.md``.

    Returns:
        The prompt text, stripped of trailing whitespace.

    Raises:
        ValueError: If ``name`` looks like a path rather than a prompt stem.
        PromptNotFound: If the file is missing, listing what is available.
    """
    if not name or name != Path(name).name or name.startswith("."):
        # Reject traversal up front: prompts are config, but they are still read
        # from disk with a caller-supplied name.
        raise ValueError(f"invalid prompt name: {name!r}")

    path = _PROMPTS_DIR / f"{name}{_SUFFIX}"
    if not path.is_file():
        available = sorted(p.stem for p in _PROMPTS_DIR.glob(f"*{_SUFFIX}"))
        raise PromptNotFound(
            f"prompt {name!r} not found in {_PROMPTS_DIR}; available: {available}"
        )

    return path.read_text(encoding="utf-8").strip()


def load_system_prompt() -> str:
    """Return the shared system prompt.

    Kept as a named helper so call sites do not repeat the literal ``"system"``.
    """
    return load_prompt("system")
