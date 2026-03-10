"""Stage 1: Fetch URL content."""

import logging
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


def run(db: Database, source: dict) -> bool:
    """Fetch URL, store raw_text, compute content_hash. Returns True on success."""
    uri = source["uri"]
    source_id = source["id"]
    log = _log()
    log.info("Fetching %s", uri)

    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            resp = client.get(uri)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error fetching {uri}: {e}") from e

    raw = resp.text
    new_hash = content_hash(raw)

    # Idempotency: skip if content unchanged
    if source.get("content_hash") == new_hash:
        _log().info("Content unchanged for %s, skipping", uri)
        db.update_source(source_id, status="complete", stage="done")
        return False

    content_type = resp.headers.get("content-type", "")
    source_type = "html" if "html" in content_type else "text"

    db.update_source(
        source_id,
        raw_text=raw,
        content_hash=new_hash,
        type=source_type,
        stage="parse",
        status="processing",
    )
    return True
