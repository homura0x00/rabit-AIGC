"""Section-aware chunking for resume retrieval.

Why not a generic recursive character splitter
----------------------------------------------

Resumes already have structure, written by the person who made them. A
character-count splitter is built for prose with no reliable landmarks, and it
shows on a resume: it will cut a single job entry in half, producing two chunks
that each retrieve poorly and neither of which contains a complete claim a judge
could verify against.

Splitting on the headings that are already in the text costs nothing and keeps
each chunk semantically whole. The heading is then prepended to every chunk, so
the embedder gets section context it would otherwise have to infer — "Built a
ReAct agent loop with function calling" retrieves far better under
"EXPERIENCE: ..." than it does standing alone.

The chunk budget is a token decision, not a formatting one
----------------------------------------------------------

The judge stage receives retrieved chunks rather than whole resumes, which takes
a candidate's payload from roughly 1500 tokens down to about 300. That saving is
only real if the chunks still contain the evidence the judge needs, which is
exactly why section boundaries, not character counts, decide where a chunk ends.
"""

import re
from dataclasses import dataclass

# Canonical section names. Resumes in this project's market mix English and
# Chinese headings freely, and often both in one document, so both vocabularies
# are matched.
_SECTION_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("summary", "SUMMARY"),
    ("profile", "SUMMARY"),
    ("objective", "SUMMARY"),
    ("自我评价", "SUMMARY"),
    ("个人简介", "SUMMARY"),
    ("education", "EDUCATION"),
    ("academic", "EDUCATION"),
    ("教育", "EDUCATION"),
    ("学历", "EDUCATION"),
    ("experience", "EXPERIENCE"),
    ("employment", "EXPERIENCE"),
    ("work history", "EXPERIENCE"),
    ("工作经历", "EXPERIENCE"),
    ("工作经验", "EXPERIENCE"),
    ("实习经历", "EXPERIENCE"),
    ("project", "PROJECTS"),
    ("项目", "PROJECTS"),
    ("skill", "SKILLS"),
    ("technolog", "SKILLS"),
    ("技能", "SKILLS"),
    ("award", "AWARDS"),
    ("honor", "AWARDS"),
    ("荣誉", "AWARDS"),
    ("获奖", "AWARDS"),
    ("certificat", "CERTIFICATIONS"),
    ("证书", "CERTIFICATIONS"),
    ("publication", "PUBLICATIONS"),
    ("论文", "PUBLICATIONS"),
)

# A heading is short. This ceiling keeps a long sentence that happens to contain
# the word "experience" from being read as one.
_MAX_HEADING_CHARS = 40

# Trailing punctuation that templates add to headings: "EXPERIENCE:" or "技能 ——".
_HEADING_TRIM = ":-—–=*· \t\u3000"


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit.

    Attributes:
        text: The chunk body, prefixed with its section heading.
        section: Canonical section name.
        index: Position within the document, used to keep results stable.
    """

    text: str
    section: str
    index: int


def _classify_heading(line: str) -> str | None:
    """Decide whether a line is a section heading, and which section.

    Args:
        line: A single line of resume text.

    Returns:
        The canonical section name, or ``None`` if the line is body text.
    """
    stripped = line.strip().strip(_HEADING_TRIM).strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return None

    tokens = stripped.split()
    words = [t for t in tokens if t.isalpha()]
    all_caps = (
        len(words) >= 1
        and len([c for c in stripped if c.isalpha()]) >= 3
        and all(c.isupper() for c in stripped if c.isalpha())
        and len(tokens) <= 5
    )

    target = stripped.lower()

    # Letter-spaced headings are pervasive in designed resume templates: the PDF
    # stores "PROJECTS" as "PROJ ECTS" and "工作经历" as "工 作 经 历", so a keyword
    # lookup misses completely and the section gets misfiled. The collapsed form
    # is therefore always tried, not just for all-caps ASCII — CJK has no case,
    # so an isupper() test silently never fires for exactly the headings that
    # need it most.
    collapsed = "".join(tokens).lower() if len(tokens) > 1 else None

    for candidate in (target, collapsed):
        if not candidate:
            continue
        for keyword, canonical in _SECTION_KEYWORDS:
            if keyword in candidate:
                return canonical

    # Unknown but clearly a heading — an all-caps short line. Return the author's
    # own casing rather than title-casing it: `.title()` rendered the real
    # "PROJ ECTS" as "Proj Ects", which reads like a bug in the output even when
    # the grouping is correct.
    if all_caps:
        return stripped

    return None


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split resume text into ``(section, body)`` pairs at heading lines.

    Args:
        text: Normalised resume text.

    Returns:
        Sections in document order. Text before the first heading, if any, is
        returned under ``HEADER`` rather than discarded — that block usually
        holds the name, contact details and a headline.
    """
    sections: list[tuple[str, list[str]]] = []
    current_name = "HEADER"
    current_lines: list[str] = []

    for line in text.split("\n"):
        heading = _classify_heading(line)
        if heading is not None:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = heading
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, current_lines))

    return [(name, "\n".join(lines).strip()) for name, lines in sections]


def _split_body(body: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split an oversized section body on line boundaries.

    Lines are kept whole. A line-level split still cuts a paragraph, but on a
    resume a line is usually a bullet or a role title, so the damage is bounded
    and the overlap recovers the seam.

    Args:
        body: Section text, already within one section.
        max_chars: Target maximum chunk length.
        overlap_chars: Characters of the previous chunk repeated at the start of
            the next, so a fact spanning the seam is still retrievable.

    Returns:
        One or more chunk bodies.
    """
    lines = [line for line in body.split("\n") if line.strip()]
    if not lines:
        return []

    pieces: list[str] = []
    buffer: list[str] = []
    length = 0

    for line in lines:
        if buffer and length + len(line) + 1 > max_chars:
            pieces.append("\n".join(buffer))
            # Carry the tail of the previous piece forward as overlap.
            if overlap_chars > 0:
                carried = "\n".join(buffer)[-overlap_chars:]
                buffer = [carried]
                length = len(carried)
            else:
                buffer = []
                length = 0
        buffer.append(line)
        length += len(line) + 1

    if buffer:
        pieces.append("\n".join(buffer))

    return pieces


def chunk_resume(
    text: str,
    *,
    max_chars: int = 600,
    overlap_chars: int = 80,
) -> list[Chunk]:
    """Chunk a resume for retrieval, respecting section boundaries.

    Args:
        text: Normalised resume text.
        max_chars: Target maximum characters per chunk.
        overlap_chars: Overlap carried between chunks within a section.

    Returns:
        Chunks in document order, each prefixed with its section heading. A very
        short document yields a single ``HEADER`` chunk rather than nothing, so a
        caller never has to special-case it.
    """
    chunks: list[Chunk] = []

    for section, body in split_sections(text):
        if not body.strip():
            continue

        pieces = (
            [body]
            if len(body) <= max_chars
            else _split_body(body, max_chars, overlap_chars)
        )

        for piece in pieces:
            if not piece.strip():
                continue
            # The heading prefix is what gives the embedder section context.
            chunks.append(
                Chunk(
                    text=f"{section}: {piece.strip()}",
                    section=section,
                    index=len(chunks),
                )
            )

    if not chunks:
        # Degenerate input: keep the document rather than returning empty, so
        # downstream stages see "one useless chunk" instead of "no document".
        stripped = re.sub(r"\s+", " ", text).strip()
        if stripped:
            chunks.append(Chunk(text=f"HEADER: {stripped}", section="HEADER", index=0))

    return chunks
