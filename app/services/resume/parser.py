"""PDF resume parsing — stage 0 of the funnel, at zero token cost.

Nothing here calls a model, and that is the point. Pulling text out of a PDF is a
deterministic, solved problem; paying a model to do it would put a per-resume
cost in front of every other optimisation in the pipeline, and would make the
cheapest stage the most expensive one.

The content hash is computed over the *normalised text*, not the uploaded bytes.
A resume that is re-saved, re-exported, or submitted twice by the same candidate
therefore resolves to a single document, and a single document means one set of
paid calls rather than two. Hashing the bytes would treat each re-export as a new
candidate.
"""

import hashlib
import re
from dataclasses import dataclass

import pymupdf

# NOTE for anyone running this under a strict warnings filter:
# importing pymupdf emits DeprecationWarnings from its SWIG-generated types
# ("builtin type SwigPyPacked has no __module__ attribute"). Promoting those to
# errors — e.g. `python -W error::DeprecationWarning` — does not raise a catchable
# exception; it segfaults, because the warning is raised inside SWIG's C-extension
# import machinery. The traceback points at this import and looks like a bug in
# this module. It is not. Use a targeted filter (see the pytest config) instead of
# a blanket `-W error`.

# Intra-line runs of spaces and tabs, including the full-width and no-break
# variants that CJK PDFs use in place of ASCII spaces.
_INLINE_SPACE = re.compile(r"[ \t\u00a0\u3000]+")
# Three or more newlines collapse to a paragraph break.
_BLANK_LINES = re.compile(r"\n{3,}")
# A word split across a line by a hyphen, e.g. "machi-\nne learning".
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

# Below this, the text layer is almost certainly missing rather than short.
# A one-page resume runs to roughly 1200 tokens; 50 characters is far below any
# real resume and above the noise a blank or image-only page produces.
MIN_TEXT_CHARS = 50


class ResumeParseError(ValueError):
    """Raised when a document cannot be turned into usable text."""


def _page_text(page: "pymupdf.Page") -> str:
    """Extract one page's text as a plain string.

    pymupdf types ``get_text`` loosely — ``(self, *args, **kwargs)`` — because its
    return type depends on an option argument the signature cannot express, so a
    checker sees ``str | list | dict``. Narrowing with ``str()`` would silently
    stringify a list if the option were ever ignored; failing here means a change
    in extraction behaviour is loud rather than quietly producing a Python repr in
    a resume's text.

    Args:
        page: The PDF page.

    Returns:
        The page's plain text.

    Raises:
        ResumeParseError: If extraction returned something other than text.
    """
    result = page.get_text("text")
    if not isinstance(result, str):
        raise ResumeParseError(
            f"get_text returned {type(result).__name__}, expected str"
        )
    return result


@dataclass(frozen=True)
class ParsedResume:
    """The result of parsing one document.

    Attributes:
        text: Normalised plain text.
        content_hash: SHA-256 of :attr:`text`, used for de-duplication.
        page_count: Number of pages in the source PDF.
        char_count: Length of :attr:`text`.
    """

    text: str
    content_hash: str
    page_count: int
    char_count: int


def normalise_text(raw: str) -> str:
    """Collapse PDF text-extraction noise into a stable form.

    Extraction artifacts vary between exporters, so two identical resumes can
    produce different byte sequences. Normalising before hashing is what makes
    de-duplication reliable rather than lucky.

    Args:
        raw: Text exactly as extracted from the PDF.

    Returns:
        Text with hyphenated line breaks rejoined, intra-line whitespace
        collapsed, trailing spaces stripped, and blank runs reduced.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _INLINE_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    """Hash normalised resume text.

    Args:
        text: Already-normalised text.

    Returns:
        A hex SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_pdf_bytes(data: bytes, *, min_chars: int = MIN_TEXT_CHARS) -> ParsedResume:
    """Parse a PDF held in memory.

    Args:
        data: Raw PDF bytes.
        min_chars: Minimum normalised length to accept.

    Returns:
        The parsed resume.

    Raises:
        ResumeParseError: If the file is not a readable PDF, is password
            protected, or carries no usable text layer (a scan needing OCR).
    """
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # pymupdf raises assorted low-level error types
        raise ResumeParseError(f"not a readable PDF: {exc}") from exc

    try:
        if document.needs_pass:
            raise ResumeParseError("PDF is password protected")
        if document.page_count == 0:
            raise ResumeParseError("PDF has no pages")

        page_count = document.page_count
        pages = [_page_text(page) for page in document]
    finally:
        document.close()

    text = normalise_text("\n".join(pages))

    if len(text) < min_chars:
        # Worth distinguishing from a corrupt file: a scanned resume is a real
        # and common case, and the actionable fix (OCR) is different.
        raise ResumeParseError(
            f"no usable text layer ({len(text)} chars). "
            "The PDF is probably a scan and would need OCR."
        )

    return ParsedResume(
        text=text,
        content_hash=content_hash(text),
        page_count=page_count,
        char_count=len(text),
    )


def parse_pdf(path: str, *, min_chars: int = MIN_TEXT_CHARS) -> ParsedResume:
    """Parse a PDF from disk.

    Args:
        path: Filesystem path to the PDF.
        min_chars: Minimum normalised length to accept.

    Returns:
        The parsed resume.

    Raises:
        ResumeParseError: If the file cannot be read or parsed.
    """
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise ResumeParseError(f"cannot read {path}: {exc}") from exc

    return parse_pdf_bytes(data, min_chars=min_chars)
