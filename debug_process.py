#!/usr/bin/env python3
"""Direct pipeline processing — bypasses Prefect for debugging."""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
    os.path.expanduser("~/.openclaw/credentials/zaf-admin.json"))

from agents_kg.db import Database
from agents_kg.stages import fetch, parse, chunk, embed, extract, load

STAGES = [
    ("fetch", fetch.run),
    ("parse", parse.run),
    ("chunk", chunk.run),
    ("embed", embed.run),
    ("extract", extract.run),
    # skip load (Neo4j) for now
]

def process_source(db, source_id):
    source = db.get_source(source_id)
    if not source:
        print(f"Source {source_id} not found")
        return

    print(f"Source {source_id}: {source['uri'][:80]}")
    print(f"  Current: stage={source['stage']} status={source['status']}")

    for stage_name, stage_fn in STAGES:
        source = db.get_source(source_id)
        if source["stage"] != stage_name:
            if source["status"] in ("complete", "failed"):
                print(f"  Source is {source['status']}, stopping")
                break
            continue

        print(f"  Running {stage_name}...")
        try:
            result = stage_fn(db, source)
            source = db.get_source(source_id)
            print(f"    -> stage={source['stage']} status={source['status']} (result={result})")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            break

    # Show what was extracted
    entities = db.conn.execute(
        "SELECT entity_id, name, type, kind FROM entities WHERE source_id = ?", (source_id,)
    ).fetchall()
    edges = db.conn.execute(
        "SELECT source_entity_id, target_entity_id, edge_type FROM edges WHERE source_id = ?", (source_id,)
    ).fetchall()

    if entities:
        print(f"\n  Entities ({len(entities)}):")
        for e in entities:
            kind = f"/{e['kind']}" if e['kind'] else ""
            print(f"    {e['entity_id']} — {e['name']} ({e['type']}{kind})")

    if edges:
        print(f"\n  Edges ({len(edges)}):")
        for e in edges:
            print(f"    {e['source_entity_id']} —{e['edge_type']}→ {e['target_entity_id']}")


if __name__ == "__main__":
    source_id = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    db = Database()
    process_source(db, source_id)
    db.close()
