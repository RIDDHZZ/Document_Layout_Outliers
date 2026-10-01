"""Upload validation. Nothing here trusts the client's filename, MIME type or PDF metadata."""
from __future__ import annotations

import re
from pathlib import Path

from backend.app.errors import AppError

ALLOWED_MIME = {"application/pdf", "application/x-pdf"}


def safe_name(name: str | None) -> str:
    base = Path(name or "document.pdf").name
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)[:80] or "document.pdf"


async def read_upload(file, max_bytes: int) -> bytes:
    """Validate extension, MIME type and magic bytes; enforce the size limit while streaming."""
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise AppError(400, "unsupported_format", "Only PDF files (.pdf) are supported.")
    if (file.content_type or "").lower() not in ALLOWED_MIME:
        raise AppError(400, "unsupported_format", "The uploaded file does not look like a PDF (unexpected MIME type).")
    chunks, size = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise AppError(413, "file_too_large", f"The PDF exceeds the {max_bytes // (1024 * 1024)} MB limit.")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise AppError(400, "empty_file", "The uploaded file is empty.")
    if not data.startswith(b"%PDF-"):
        raise AppError(400, "invalid_pdf", "The uploaded file is not a valid PDF.")
    return data


def inspect_pdf(path: Path, max_pages: int) -> int:
    """Open with PyMuPDF and return the page count, or raise a friendly error."""
    import fitz  # PyMuPDF
    try:
        doc = fitz.open(str(path))
    except Exception:
        raise AppError(400, "corrupted_pdf", "The PDF is corrupted or could not be opened.")
    try:
        if doc.needs_pass:
            raise AppError(400, "encrypted_pdf", "Password-protected PDFs are not supported.")
        n = doc.page_count
    finally:
        doc.close()
    if n == 0:
        raise AppError(400, "empty_pdf", "The PDF has no pages.")
    if n > max_pages:
        raise AppError(413, "too_many_pages", f"The PDF has {n} pages; the limit is {max_pages}.")
    return n
