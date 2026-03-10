"""Stage 1: Fetch URL or local file content."""

import logging
import os
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

    db.update_source(
        source_id,
        raw_text=raw,
        content_hash=new_hash,
        type=source_type,
        stage="parse",
        status="processing",
    )
    return True
