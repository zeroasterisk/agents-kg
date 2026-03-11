#!/usr/bin/env python3
"""Batch process all sources through the pipeline."""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
    os.path.expanduser("~/.openclaw/credentials/zaf-admin.json"))

from agents_kg.db import Database
from agents_kg.stages import fetch, parse, chunk, embed, extract
from agents_kg.stages import resolve

STAGES = [
    ("fetch", fetch.run),
    ("parse", parse.run),
    ("chunk", chunk.run),
    ("embed", embed.run),
    ("extract", extract.run),
]

results = {}  # source_id -> {success, error, entities, edges, duration}

def process_source(db, source_id):
    source = db.get_source(source_id)
    if not source:
        return {"success": False, "error": "Not found"}

    start = time.time()
    uri = source["uri"][:80]
    print(f"\n[{source_id}] {uri}")

    for stage_name, stage_fn in STAGES:
        source = db.get_source(source_id)
        if source["stage"] != stage_name:
            if source["status"] in ("complete", "failed"):
                break
            continue
        
        print(f"  -> {stage_name}...", end="", flush=True)
        try:
            result = stage_fn(db, source)
            source = db.get_source(source_id)
            print(f" ok (stage={source['stage']})", flush=True)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)
            return {"success": False, "error": str(e), "stage": stage_name, "duration": time.time() - start}
    
    # Run resolution
    source = db.get_source(source_id)
    if source["stage"] == "resolve" and source["status"] == "processing":
        print(f"  -> resolve...", end="", flush=True)
        try:
            resolve.run(db, source)
            print(" ok", flush=True)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)

    # Count results
    entities = db.conn.execute(
        "SELECT COUNT(*) FROM entities WHERE source_id = ? AND merged_into IS NULL", (source_id,)
    ).fetchone()[0]
    entities_merged = db.conn.execute(
        "SELECT COUNT(*) FROM entities WHERE source_id = ? AND status = 'merged'", (source_id,)
    ).fetchone()[0]
    entities_rejected = db.conn.execute(
        "SELECT COUNT(*) FROM entities WHERE source_id = ? AND status = 'rejected'", (source_id,)
    ).fetchone()[0]
    edges = db.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE source_id = ?", (source_id,)
    ).fetchone()[0]

    duration = time.time() - start
    print(f"  Done: {entities} entities ({entities_merged} merged, {entities_rejected} rejected), {edges} edges [{duration:.1f}s]")
    return {
        "success": True,
        "entities": entities,
        "merged": entities_merged,
        "rejected": entities_rejected,
        "edges": edges,
        "duration": duration
    }

def main():
    start_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end_id = int(sys.argv[2]) if len(sys.argv) > 2 else 27
    
    db = Database()
    
    print(f"\nBatch processing sources {start_id} to {end_id}")
    print("=" * 60)
    
    successes = 0
    failures = []
    
    for source_id in range(start_id, end_id + 1):
        try:
            result = process_source(db, source_id)
            results[source_id] = result
            if result.get("success"):
                successes += 1
            else:
                failures.append((source_id, result.get("error", "unknown")))
        except Exception as e:
            print(f"\n[{source_id}] CRITICAL ERROR: {e}")
            failures.append((source_id, str(e)))
        
        # Brief pause between sources to avoid rate limits
        time.sleep(1)
    
    db.close()
    
    print("\n" + "=" * 60)
    print(f"\nSummary: {successes} succeeded, {len(failures)} failed")
    if failures:
        print("Failed sources:")
        for sid, err in failures:
            print(f"  [{sid}] {err}")

if __name__ == "__main__":
    main()
