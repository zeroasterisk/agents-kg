"""Stage 4: Embed chunks using Gemini embedding API."""

import logging
import struct
from ..db import Database

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

EMBEDDING_MODEL = "gemini-embedding-2"
BATCH_SIZE = 100


def _floats_to_bytes(floats: list[float]) -> bytes:
    return struct.pack(f'{len(floats)}f', *floats)


def run(db: Database, source: dict) -> bool:
    source_id = source["id"]
    chunks = db.get_unembedded_chunks(source_id)
    if not chunks:
        _log().info("All chunks already embedded for source %d", source_id)
        db.update_source(source_id, stage="extract", status="processing")
        return True

    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai not installed")

    import os
    kwargs = {}
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        kwargs["enterprise"] = True
        kwargs["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        kwargs["location"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    client = genai.Client(**kwargs)

    for idx, chunk in enumerate(chunks):
        _log().info("Embedding chunk %d/%d (source %d)", idx + 1, len(chunks), source_id)
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk["text"],
        )
        emb_bytes = _floats_to_bytes(result.embeddings[0].values)
        db.update_chunk_embedding(chunk["id"], emb_bytes, EMBEDDING_MODEL)

    db.update_source(source_id, stage="extract", status="processing")
    return True
