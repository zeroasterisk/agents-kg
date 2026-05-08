"""Temporal model: Event nodes and temporal helpers for the knowledge graph.

Events are first-class graph entities representing discrete occurrences
(launches, donations, acquisitions, spec releases, governance changes).
They connect to entities via PARTICIPATED_IN edges with role properties.
"""

import logging
import re
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DEFAULT_EVENTS_DIR = "kg/events"
DEFAULT_TIMELINE_PATH = "kg/timeline.yaml"


def load_event_yaml(path: Path) -> dict | None:
    """Load and validate a single event YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        log.warning("Empty event file: %s", path)
        return None

    required = ["event_id", "title", "event_type", "date"]
    for field in required:
        if field not in data:
            log.warning("Event file %s missing required field: %s", path, field)
            return None

    return data


def load_events_from_yaml(neo4j_driver, events_dir: str = DEFAULT_EVENTS_DIR) -> dict:
    """Load Event nodes from YAML files into Neo4j.

    Returns {"events": N, "participations": N} counts.
    """
    events_path = Path(events_dir)
    if not events_path.exists():
        log.warning("Events directory not found: %s", events_dir)
        return {"events": 0, "participations": 0}

    yaml_files = sorted(events_path.glob("*.yaml"))
    if not yaml_files:
        log.info("No event YAML files found in %s", events_dir)
        return {"events": 0, "participations": 0}

    events = []
    participations = []

    for path in yaml_files:
        event = load_event_yaml(path)
        if not event:
            continue

        events.append({
            "event_id": event["event_id"],
            "title": event["title"],
            "event_type": event["event_type"],
            "date": str(event["date"]),
            "description": event.get("description", ""),
            "source_url": event.get("source_url", ""),
        })

        for p in event.get("participants", []):
            participations.append({
                "entity_id": p["entity_id"],
                "event_id": event["event_id"],
                "role": p.get("role", "participant"),
            })

    if not neo4j_driver:
        log.info("No Neo4j driver, parsed %d events with %d participations", len(events), len(participations))
        return {"events": len(events), "participations": len(participations)}

    with neo4j_driver.session() as session:
        if events:
            session.run(
                """
                UNWIND $events AS evt
                MERGE (e:Event {event_id: evt.event_id})
                SET e.title = evt.title, e.event_type = evt.event_type,
                    e.date = date(evt.date), e.description = evt.description,
                    e.source_url = evt.source_url
                """,
                {"events": events},
            )

        if participations:
            session.run(
                """
                UNWIND $participations AS p
                MATCH (entity:Entity {entity_id: p.entity_id})
                MATCH (event:Event {event_id: p.event_id})
                MERGE (entity)-[r:PARTICIPATED_IN]->(event)
                SET r.role = p.role
                """,
                {"participations": participations},
            )

    log.info("Loaded %d events with %d participations", len(events), len(participations))
    return {"events": len(events), "participations": len(participations)}


def migrate_timeline_yaml(
    timeline_path: str = DEFAULT_TIMELINE_PATH,
    events_dir: str = DEFAULT_EVENTS_DIR,
) -> int:
    """Convert existing timeline.yaml entries into individual event YAML files.

    Returns count of files created.
    """
    tl_path = Path(timeline_path)
    if not tl_path.exists():
        log.warning("Timeline file not found: %s", timeline_path)
        return 0

    with open(tl_path) as f:
        data = yaml.safe_load(f)

    entries = data.get("events", [])
    if not entries:
        log.info("No events in timeline.yaml")
        return 0

    events_out = Path(events_dir)
    events_out.mkdir(parents=True, exist_ok=True)

    created = 0
    for entry in entries:
        slug = re.sub(r"[^a-z0-9]+", "-", entry["title"].lower()).strip("-")
        slug = slug[:60]
        event_id = f"{slug}-{entry['date']}"

        actors = entry.get("actors", [])
        participants = []
        for actor in actors:
            entity_id = f"organization:{actor}"
            participants.append({"entity_id": entity_id, "role": "participant"})

        event_data = {
            "event_id": event_id,
            "title": entry["title"],
            "event_type": entry.get("type", "event"),
            "date": str(entry["date"]),
            "description": entry.get("description", "").strip(),
        }
        if entry.get("source"):
            event_data["source_url"] = entry["source"]
        if participants:
            event_data["participants"] = participants

        out_path = events_out / f"{slug}.yaml"
        if out_path.exists():
            log.info("Skipping existing: %s", out_path)
            continue

        with open(out_path, "w") as f:
            yaml.dump(event_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        log.info("Created %s", out_path)
        created += 1

    return created


def create_temporal_constraints(neo4j_driver):
    """Create Event node constraints and indexes (subset of schema.py)."""
    stmts = [
        "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
        "CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.date)",
        "CREATE INDEX event_type IF NOT EXISTS FOR (e:Event) ON (e.event_type)",
    ]
    with neo4j_driver.session() as session:
        for stmt in stmts:
            try:
                session.run(stmt)
            except Exception as e:
                log.error("Failed: %s — %s", stmt, e)
