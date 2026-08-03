"""Stage 1: Fetch URL or local file content."""

import io
import logging
import os
import re

import httpx

from ..db import Database, content_hash

try:
    from prefect.logging import get_run_logger as _get_logger
except ImportError:
    _get_logger = None


def _log():
    if _get_logger:
        try:
            return _get_logger()
        except Exception:
            pass
    return logging.getLogger(__name__)


def _is_local_file(uri: str) -> bool:
    """Check if URI is a local file path or file:// URI."""
    return uri.startswith("file://") or os.path.isfile(uri)


def _resolve_path(uri: str) -> str:
    """Resolve file:// URI or path to absolute path."""
    if uri.startswith("file://"):
        return uri[7:]
    return os.path.abspath(uri)


def _fetch_local(path: str) -> tuple[str, str]:
    """Read local file, return (content, source_type). For PDFs, extract text."""
    if not os.path.isfile(path):
        raise RuntimeError(f"File not found: {path}")

    if path.lower().endswith(".pdf"):
        try:
            import fitz  # pymupdf
            doc = fitz.open(path)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n\n".join(pages), "pdf"
        except ImportError:
            raise RuntimeError("pymupdf not installed — run: uv add pymupdf")
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "text"


# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------

_DRIVE_FILE_RE = re.compile(
    r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)"
)
_DRIVE_OPEN_RE = re.compile(
    r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)"
)
_DOCS_RE = re.compile(
    r"docs\.google\.com/(document|spreadsheets|presentation)/d/([a-zA-Z0-9_-]+)"
)


def _extract_drive_file_id(uri: str) -> str | None:
    """Extract a Google Drive file ID from a URL, or None if not a Drive URL."""
    m = _DRIVE_FILE_RE.search(uri)
    if m:
        return m.group(1)
    m = _DRIVE_OPEN_RE.search(uri)
    if m:
        return m.group(1)
    m = _DOCS_RE.search(uri)
    if m:
        return m.group(2)
    return None


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        raise RuntimeError("pymupdf not installed — run: uv add pymupdf")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def _fetch_drive_file(file_id: str) -> tuple[str, str]:
    """Download a file from Google Drive and return (text_content, source_type).

    Uses the public download endpoint which works for publicly shared files.
    """
    log = _log()
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        log.info("Downloading Drive file %s via public URL", file_id)
        resp = client.get(download_url)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        raw_bytes = resp.content

        # Check if we got a virus scan warning page (large files)
        if b"virus scan warning" in raw_bytes[:2000].lower() or b"confirm=" in raw_bytes[:2000]:
            log.info("Got Drive virus scan warning, extracting confirm token")
            import re as _re
            confirm_match = _re.search(rb'confirm=([a-zA-Z0-9_-]+)', raw_bytes)
            if confirm_match:
                confirm_token = confirm_match.group(1).decode()
                resp = client.get(
                    download_url,
                    params={"confirm": confirm_token},
                )
                resp.raise_for_status()
                raw_bytes = resp.content
                content_type = resp.headers.get("content-type", "")

        # Detect PDF by magic bytes or content type
        if raw_bytes[:5] == b"%PDF-" or "pdf" in content_type:
            log.info("Detected PDF content, extracting text")
            return _extract_pdf_text(raw_bytes), "pdf"

        # Text-like content
        if "text" in content_type or "json" in content_type:
            return resp.text, "text"

        # Try PDF extraction as fallback for octet-stream
        if "octet-stream" in content_type:
            try:
                text = _extract_pdf_text(raw_bytes)
                if text.strip():
                    log.info("Successfully extracted text from octet-stream as PDF")
                    return text, "pdf"
            except Exception:
                pass

        # Last resort — treat as text
        try:
            return raw_bytes.decode("utf-8", errors="replace"), "text"
        except Exception:
            raise RuntimeError(
                f"Could not extract content from Drive file {file_id} (content-type: {content_type})"
            )


def run(db: Database, source: dict) -> bool:
    """Fetch URL or local file, store raw_text, compute content_hash. Returns True on success."""
    uri = source["uri"]
    source_id = source["id"]
    log = _log()
    log.info("Fetching %s", uri)

    if _is_local_file(uri):
        path = _resolve_path(uri)
        raw, source_type = _fetch_local(path)
    else:
        drive_file_id = _extract_drive_file_id(uri)
        if drive_file_id:
            log.info("Detected Google Drive URL, file_id=%s", drive_file_id)
            raw, source_type = _fetch_drive_file(drive_file_id)
        else:
            try:
                with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                    resp = client.get(uri)
                    resp.raise_for_status()
            except httpx.HTTPError as e:
                raise RuntimeError(f"HTTP error fetching {uri}: {e}") from e

            raw = resp.text
            content_type = resp.headers.get("content-type", "")
            source_type = "html" if "html" in content_type else "text"

    new_hash = content_hash(raw)

    # Idempotency: skip if content unchanged
    if source.get("content_hash") == new_hash:
        _log().info("Content unchanged for %s, skipping", uri)
        db.update_source(source_id, status="complete", stage="done")
        return False

    # Content changed — mark old entities from this source as deprecated
    if source.get("content_hash") and source["content_hash"] != new_hash:
        db.deprecate_entities_for_source(source_id)
        _log().info("Content changed for %s, deprecated old entities", uri)

    db.update_source(
        source_id,
        raw_text=raw,
        content_hash=new_hash,
        type=source_type,
        stage="parse",
        status="processing",
    )
    return True
