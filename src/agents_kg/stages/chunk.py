"""Stage 3: Chunk parsed text into sections (~500 tokens target)."""

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

TARGET_TOKENS = 500
MAX_TOKENS = 800


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split by markdown headers. Returns (heading, body) pairs."""
    parts = re.split(r'^(#{1,6}\s+.+)$', text, flags=re.MULTILINE)

    sections = []
    current_heading = ""
    current_body = ""

    for part in parts:
        if re.match(r'^#{1,6}\s+', part):
            if current_body.strip():
                sections.append((current_heading, current_body.strip()))
            current_heading = part.strip()
            current_body = ""
        else:
            current_body += part

    if current_body.strip():
        sections.append((current_heading, current_body.strip()))

    return sections if sections else [("", text)]


def _split_long(text: str, heading: str) -> list[tuple[str, str]]:
    """Split a long section into paragraph-based chunks."""
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current = ""

    for para in paragraphs:
        if _estimate_tokens(current + "\n\n" + para) > TARGET_TOKENS and current:
            chunks.append((heading, current.strip()))
            current = para
        else:
            current = (current + "\n\n" + para).strip()

    if current.strip():
        chunks.append((heading, current.strip()))

    return chunks


def run(db, source: dict) -> bool:
    text = source.get("parsed_text") or source.get("raw_text", "")
    if not text:
        raise RuntimeError("No text to chunk")

    source_id = source["id"]
    db.delete_chunks(source_id)

    sections = _split_sections(text)
    position = 0

    for heading, body in sections:
        if _estimate_tokens(body) > MAX_TOKENS:
            sub_chunks = _split_long(body, heading)
        else:
            sub_chunks = [(heading, body)]

        for h, chunk_text in sub_chunks:
            full_text = f"{h}\n\n{chunk_text}".strip() if h else chunk_text
            if not full_text.strip():
                continue
            tokens = _estimate_tokens(full_text)
            db.add_chunk(source_id, full_text, position, section_heading=h or None, token_count=tokens)
            position += 1

    _log().info("Created %d chunks for source %d", position, source_id)
    db.update_source(source_id, stage="embed", status="processing")
    return True
