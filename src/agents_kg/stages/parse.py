"""Stage 2: Parse raw content to clean markdown/text."""

import logging
import re

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


def _html_to_text(html: str) -> str:
    """Convert HTML to readable text using readability + basic cleanup."""
    try:
        from readability import Document
        doc = Document(html)
        summary = doc.summary()
        title = doc.title()
    except Exception:
        _log().warning("readability failed, falling back to basic strip")
        summary = html
        title = ""

    # Strip remaining HTML tags
    text = re.sub(r'<[^>]+>', '\n', summary)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    if title and not text.startswith(title):
        text = f"# {title}\n\n{text}"
    return text


def _is_markdown(text: str) -> bool:
    """Heuristic: if it has markdown headers, treat as markdown."""
    return bool(re.search(r'^#{1,6}\s', text, re.MULTILINE))


def run(db, source: dict) -> bool:
    raw = source.get("raw_text", "")
    if not raw:
        raise RuntimeError("No raw_text to parse")

    source_type = source.get("type", "html")
    if source_type == "pdf":
        # PDF text from pymupdf — clean up whitespace, detect structure
        parsed = re.sub(r'\n{3,}', '\n\n', raw).strip()
    elif source_type == "html" and not _is_markdown(raw):
        parsed = _html_to_text(raw)
    else:
        parsed = raw  # markdown passthrough

    title = None
    m = re.match(r'^#\s+(.+)', parsed)
    if m:
        title = m.group(1).strip()

    db.update_source(source["id"], parsed_text=parsed, title=title, stage="chunk", status="processing")
    return True
