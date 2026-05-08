"""Neo4j schema management: constraints and indexes for the knowledge graph."""

import logging

log = logging.getLogger(__name__)

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE",
    "CREATE CONSTRAINT source_uri_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.uri IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX entity_wikidata IF NOT EXISTS FOR (n:Entity) ON (n.wikidata_id)",
    "CREATE INDEX entity_kind IF NOT EXISTS FOR (n:Entity) ON (n.kind)",
    "CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.date)",
    "CREATE INDEX event_type IF NOT EXISTS FOR (e:Event) ON (e.event_type)",
]


def apply_schema(neo4j_driver) -> dict:
    """Apply all constraints and indexes to Neo4j. Returns counts."""
    results = {"constraints": 0, "indexes": 0, "errors": []}

    with neo4j_driver.session() as session:
        for stmt in CONSTRAINTS:
            try:
                session.run(stmt)
                results["constraints"] += 1
                log.info("Applied: %s", stmt[:60])
            except Exception as e:
                results["errors"].append(f"Constraint error: {e}")
                log.error("Failed: %s — %s", stmt[:60], e)

        for stmt in INDEXES:
            try:
                session.run(stmt)
                results["indexes"] += 1
                log.info("Applied: %s", stmt[:60])
            except Exception as e:
                results["errors"].append(f"Index error: {e}")
                log.error("Failed: %s — %s", stmt[:60], e)

    return results
