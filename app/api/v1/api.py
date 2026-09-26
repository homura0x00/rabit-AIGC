"""Resume upload endpoints.

Parsing happens at upload time rather than at screening time, and that ordering
is deliberate. A corrupt file or a scanned PDF has no usable text, so it can
never be scored — discovering that mid-run means the pipeline has already spent
tokens on everything else before hitting it. Rejecting it while the uploader is
still watching is both cheaper and easier to act on.
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlmodel import Session, select

from app.core.log import logger
from app.models.base import require_id
from app.models.screening import ResumeDocument
from app.schemas.resume import ResumeUploadResponse
from app.services.database import get_session
from app.services.resume import ResumeParseError, parse_pdf_bytes

router = APIRouter()

PDF_CONTENT_TYPES = {"application/pdf"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_READ_CHUNK = 64 * 1024


async def _read_capped(file: UploadFile, limit: int) -> bytes:
    """Read an upload into memory, refusing anything over ``limit``.

    Read in chunks rather than calling ``await file.read()`` once: a single read
    buffers the whole body before the length is known, so a 2 GB upload costs
    2 GB of memory on the way to being rejected.

    Args:
        file: The incoming upload.
        limit: Maximum accepted size in bytes.

    Returns:
        The file contents.

    Raises:
        HTTPException: 413 if the body exceeds ``limit``.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_READ_CHUNK):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"file exceeds the {limit // (1024 * 1024)} MB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/resumes",
    status_code=status.HTTP_201_CREATED,
    response_model=ResumeUploadResponse,
    summary="Upload one PDF resume",
)
async def upload_resume(
    file: UploadFile = File(..., description="A PDF resume"),
    session: Session = Depends(get_session),
) -> ResumeUploadResponse:
    """Parse and store one PDF resume.

    De-duplication is keyed on the hash of the normalised text, so a candidate
    who submits the same resume twice produces one document — and therefore one
    set of paid calls rather than two.

    Args:
        file: The uploaded PDF.
        session: Injected database session.

    Returns:
        The stored document, with ``duplicate`` set when nothing was written.

    Raises:
        HTTPException: 415 for a non-PDF, 413 when oversized, 422 when the PDF
            cannot be parsed into usable text.
    """
    if file.content_type not in PDF_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"expected a PDF, got {file.content_type!r}",
        )

    data = await _read_capped(file, MAX_UPLOAD_BYTES)

    try:
        parsed = parse_pdf_bytes(data)
    except ResumeParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    existing = session.exec(
        select(ResumeDocument).where(
            ResumeDocument.content_hash == parsed.content_hash
        )
    ).first()

    if existing is not None:
        logger.info(
            "resume duplicate hash=%s resume_id=%s", parsed.content_hash[:12], existing.id
        )
        return ResumeUploadResponse(
            resume_id=require_id(existing.id, "resume_document"),
            filename=existing.filename,
            content_hash=existing.content_hash,
            page_count=existing.page_count,
            char_count=existing.char_count,
            duplicate=True,
        )

    document = ResumeDocument(
        filename=file.filename or "unnamed.pdf",
        content_hash=parsed.content_hash,
        raw_text=parsed.text,
        char_count=parsed.char_count,
        page_count=parsed.page_count,
    )
    session.add(document)
    session.commit()
    session.refresh(document)

    logger.info(
        "resume stored id=%s chars=%d pages=%d hash=%s",
        document.id,
        document.char_count,
        document.page_count,
        document.content_hash[:12],
    )

    return ResumeUploadResponse(
        resume_id=require_id(document.id, "resume_document"),
        filename=document.filename,
        content_hash=document.content_hash,
        page_count=document.page_count,
        char_count=document.char_count,
        duplicate=False,
    )
