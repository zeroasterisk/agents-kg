"""SQLite job queue and pipeline state management."""

import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DB = "pipeline.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uri TEXT NOT NULL UNIQUE,
    title TEXT,
    type TEXT DEFAULT 'url',
    content_hash TEXT,
    raw_text TEXT,
    parsed_text TEXT,
    status TEXT DEFAULT 'pending',
    stage TEXT DEFAULT 'fetch',
    error TEXT,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    text TEXT NOT NULL,
    position INTEGER NOT NULL,
    section_heading TEXT,
    chunk_strategy TEXT DEFAULT 'section',
    token_count INTEGER,
    embedding BLOB,
    embedding_model TEXT,
    embedded_at TEXT,
    UNIQUE(source_id, position)
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    kind TEXT,
    type TEXT NOT NULL,
    description TEXT,
    aliases TEXT DEFAULT '[]',
    embedding BLOB,
    status TEXT DEFAULT 'pending_review',
    merged_into TEXT,
    source_id INTEGER REFERENCES sources(id),
    chunk_id INTEGER REFERENCES chunks(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_id TEXT NOT NULL UNIQUE,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    properties TEXT DEFAULT '{}',
    valid_from TEXT,
    valid_to TEXT,
    confidence REAL DEFAULT 0.5,
    chunk_id INTEGER REFERENCES chunks(id),
    source_id INTEGER REFERENCES sources(id),
    extracted_at TEXT,
    source_type TEXT DEFAULT 'automated',
    status TEXT DEFAULT 'pending_review',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);
CREATE INDEX IF NOT EXISTS idx_sources_stage ON sources(stage);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_entities_status ON entities(status);
CREATE INDEX IF NOT EXISTS idx_edges_status ON edges(status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Database:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # --- Sources ---

    def add_source(self, uri: str, title: Optional[str] = None, source_type: str = "url") -> Optional[int]:
        """Add a source. Returns id or None if duplicate."""
        now = _now()
        try:
            cur = self.conn.execute(
                "INSERT INTO sources (uri, title, type, status, stage, created_at, updated_at) VALUES (?, ?, ?, 'pending', 'fetch', ?, ?)",
                (uri, title, source_type, now, now),
            )
            self.conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_source(self, source_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        return dict(row) if row else None

    def get_source_by_uri(self, uri: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM sources WHERE uri = ?", (uri,)).fetchone()
        return dict(row) if row else None

    def get_sources_by_status(self, status: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM sources WHERE status = ? ORDER BY id", (status,)).fetchall()
        return [dict(r) for r in rows]

    def get_pending_sources(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM sources WHERE status IN ('pending', 'processing') ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_source(self, source_id: int, **kwargs):
        kwargs["updated_at"] = _now()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [source_id]
        self.conn.execute(f"UPDATE sources SET {sets} WHERE id = ?", vals)
        self.conn.commit()

    def fail_source(self, source_id: int, error: str):
        source = self.get_source(source_id)
        if not source:
            return
        attempts = source["attempts"] + 1
        if attempts >= source["max_attempts"]:
            status = "dead_letter"
        else:
            status = "failed"
        self.update_source(source_id, status=status, error=error, attempts=attempts)

    def reset_source(self, source_id: int):
        self.update_source(
            source_id, status="pending", stage="fetch", error=None, attempts=0,
            raw_text=None, parsed_text=None, content_hash=None
        )
        self.conn.execute("DELETE FROM edges WHERE source_id = ?", (source_id,))
        self.conn.execute("DELETE FROM entities WHERE source_id = ?", (source_id,))
        self.conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        self.conn.commit()

    def retry_failed(self) -> int:
        now = _now()
        cur = self.conn.execute(
            "UPDATE sources SET status = 'pending', error = NULL, updated_at = ? WHERE status = 'failed'",
            (now,),
        )
        self.conn.commit()
        return cur.rowcount

    def status_summary(self) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) as count FROM sources GROUP BY status"
        ).fetchall()
        return {r["status"]: r["count"] for r in rows}

    # --- Chunks ---

    def add_chunk(self, source_id: int, text: str, position: int,
                  section_heading: Optional[str] = None, token_count: Optional[int] = None) -> int:
        cur = self.conn.execute(
            "INSERT OR REPLACE INTO chunks (source_id, text, position, section_heading, token_count) VALUES (?, ?, ?, ?, ?)",
            (source_id, text, position, section_heading, token_count),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_chunks(self, source_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE source_id = ? ORDER BY position", (source_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_unembedded_chunks(self, source_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE source_id = ? AND embedding IS NULL ORDER BY position",
            (source_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_chunk_embedding(self, chunk_id: int, embedding: bytes, model: str):
        now = _now()
        self.conn.execute(
            "UPDATE chunks SET embedding = ?, embedding_model = ?, embedded_at = ? WHERE id = ?",
            (embedding, model, now, chunk_id),
        )
        self.conn.commit()

    def delete_chunks(self, source_id: int):
        self.conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        self.conn.commit()

    # --- Entities ---

    def add_entity(self, entity_id: str, name: str, entity_type: str, kind: Optional[str] = None,
                   description: Optional[str] = None, aliases: Optional[list] = None,
                   source_id: Optional[int] = None, chunk_id: Optional[int] = None,
                   embedding: Optional[bytes] = None) -> Optional[int]:
        now = _now()
        try:
            cur = self.conn.execute(
                "INSERT INTO entities (entity_id, name, type, kind, description, aliases, embedding, source_id, chunk_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entity_id, name, entity_type, kind, description, json.dumps(aliases or []), embedding, source_id, chunk_id, now, now),
            )
            self.conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_entities_by_status(self, status: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM entities WHERE status = ? ORDER BY id", (status,)).fetchall()
        return [dict(r) for r in rows]

    def approve_entity(self, entity_id: int):
        self.update_entity(entity_id, status="approved")

    def update_entity(self, entity_id: int, **kwargs):
        kwargs["updated_at"] = _now()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [entity_id]
        self.conn.execute(f"UPDATE entities SET {sets} WHERE id = ?", vals)
        self.conn.commit()

    # --- Edges ---

    def add_edge(self, edge_id: str, source_entity_id: str, target_entity_id: str, edge_type: str,
                 properties: Optional[dict] = None, confidence: float = 0.5,
                 chunk_id: Optional[int] = None, source_id: Optional[int] = None,
                 source_type: str = "automated",
                 valid_from: Optional[str] = None, valid_to: Optional[str] = None) -> Optional[int]:
        now = _now()
        try:
            cur = self.conn.execute(
                "INSERT INTO edges (edge_id, source_entity_id, target_entity_id, edge_type, properties, confidence, chunk_id, source_id, extracted_at, source_type, valid_from, valid_to, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (edge_id, source_entity_id, target_entity_id, edge_type, json.dumps(properties or {}), confidence, chunk_id, source_id, now, source_type, valid_from, valid_to, now, now),
            )
            self.conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_edges_by_status(self, status: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM edges WHERE status = ? ORDER BY id", (status,)).fetchall()
        return [dict(r) for r in rows]

    def approve_edge(self, edge_id: int):
        self.update_edge(edge_id, status="approved")

    def update_edge(self, edge_id: int, **kwargs):
        kwargs["updated_at"] = _now()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [edge_id]
        self.conn.execute(f"UPDATE edges SET {sets} WHERE id = ?", vals)
        self.conn.commit()
