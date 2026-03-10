"""Stage 4: Embed chunks using Gemini embedding API."""

import logging
import struct
from ..db import Database

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-004"
BATCH_SIZE = 100


def _floats_to_bytes(floats: list[float]) -> bytes:
    return struct.pack(f'{len(floats)}f', *floats)


def run(db: Database, source: dict) -> bool:
    source_id = source["id"]
    chunks = db.get_unembedded_chunks(source_id)
    if not chunks:
        log.info("All chunks already embedded for source %d", source_id)
        db.update_source(source_id, stage="extract", status="processing")
        return True

    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai not installed")

    import os
    kwargs = {}
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        kwargs["vertexai"] = True
        kwargs["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        kwargs["location"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    client = genai.Client(**kwargs)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        log.info("Embedding batch %d-%d of %d chunks", i, i + len(batch), len(chunks))
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
        )

        for chunk, embedding in zip(batch, result.embeddings):
            emb_bytes = _floats_to_bytes(embedding.values)
            db.update_chunk_embedding(chunk["id"], emb_bytes, EMBEDDING_MODEL)

    db.update_source(source_id, stage="extract", status="processing")
    return True
