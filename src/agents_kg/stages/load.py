"""Stage 7: Load approved entities/edges to Neo4j and export YAML."""

import json
import logging
import yaml
from pathlib import Path
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

YAML_DIR = "kg/entities"


def _entity_to_cypher(entity: dict) -> tuple[str, dict]:
    aliases = json.loads(entity["aliases"]) if isinstance(entity["aliases"], str) else entity["aliases"]
    params = {
        "entity_id": entity["entity_id"],
        "name": entity["name"],
        "type": entity["type"],
        "kind": entity["kind"],
        "description": entity["description"],
        "aliases": aliases,
        "source_id": entity.get("source_id"),
    }
    
    label = entity["type"]
    valid_labels = {"Protocol", "Organization", "Project", "Capability", "Group", "Person", "Concept"}
    if label not in valid_labels:
        label = "Entity"

    query = f"""
    MERGE (n {{entity_id: $entity_id}})
    REMOVE n:Protocol:Organization:Project:Capability:Group:Person:Concept
    SET n:Entity, n:{label}, n.name = $name, n.type = $type, n.kind = $kind,
        n.description = $description, n.aliases = $aliases, n.source_id = $source_id
    """
    return query, params


def _edge_to_cypher(edge: dict) -> tuple[str, dict]:
    props = json.loads(edge["properties"]) if isinstance(edge["properties"], str) else edge["properties"]
    params = {
        "src": edge["source_entity_id"],
        "tgt": edge["target_entity_id"],
        "edge_id": edge["edge_id"],
        "confidence": edge["confidence"],
        "source_type": edge["source_type"],
        "valid_from": edge.get("valid_from"),
        "valid_to": edge.get("valid_to"),
        "chunk_id": edge.get("chunk_id"),
        **{f"prop_{k}": v for k, v in props.items()},
    }
    edge_type = edge["edge_type"].upper()
    prop_sets = ", ".join(f"r.{k} = $prop_{k}" for k in props)
    extra = f", {prop_sets}" if prop_sets else ""
    query = f"""
    MATCH (a {{entity_id: $src}}), (b {{entity_id: $tgt}})
    MERGE (a)-[r:{edge_type} {{edge_id: $edge_id}}]->(b)
    SET r.confidence = $confidence, r.source_type = $source_type,
        r.valid_from = $valid_from, r.valid_to = $valid_to,
        r.chunk_id = $chunk_id{extra}
    """
    return query, params


def _export_yaml(entity: dict, base_dir: str = YAML_DIR):
    """Export entity to YAML file."""
    etype = entity["type"].lower()
    dir_path = Path(base_dir) / f"{etype}s"
    dir_path.mkdir(parents=True, exist_ok=True)

    eid = entity["entity_id"].split(":", 1)[-1] if ":" in entity["entity_id"] else entity["entity_id"]
    eid = eid.replace("/", "_").replace("+", "_").replace(" ", "_")
    file_path = dir_path / f"{eid}.yaml"

    aliases = json.loads(entity["aliases"]) if isinstance(entity["aliases"], str) else entity["aliases"]
    data = {
        "id": entity["entity_id"],
        "name": entity["name"],
        "type": entity["type"],
        "kind": entity["kind"],
        "description": entity["description"],
        "aliases": aliases,
    }
    # Remove None values
    data = {k: v for k, v in data.items() if v is not None}

    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    _log().info("Exported %s", file_path)


def run(db: Database, source: dict, neo4j_driver=None) -> bool:
    source_id = source["id"]

    # Load approved entities
    entities = db.conn.execute(
        "SELECT * FROM entities WHERE source_id = ? AND status = 'approved'", (source_id,)
    ).fetchall()
    edges = db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND status = 'approved'", (source_id,)
    ).fetchall()

    if not entities and not edges:
        _log().info("No approved items to load for source %d", source_id)
        db.update_source(source_id, stage="done", status="complete")
        return True

    # Export YAML (always works)
    for ent in entities:
        _export_yaml(dict(ent))

    # Neo4j load (graceful degradation)
    if neo4j_driver:
        try:
            # Get unique chunk IDs
            chunk_ids = set()
            for ent in entities:
                if ent["chunk_id"]:
                    chunk_ids.add(ent["chunk_id"])
            for edge in edges:
                if edge["chunk_id"]:
                    chunk_ids.add(edge["chunk_id"])

            chunks = []
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                chunks = db.conn.execute(
                    f"SELECT * FROM chunks WHERE id IN ({placeholders})", list(chunk_ids)
                ).fetchall()

            with neo4j_driver.session() as session:
                # Create Source node with provenance
                source_data = db.get_source(source_id)
                session.run(
                    """
                    MERGE (s:Source {uri: $uri})
                    SET s.title = $title, s.source_type = $source_type,
                        s.submitter_email = $submitter_email,
                        s.created_at = $created_at, s.updated_at = $updated_at,
                        s.content_hash = $content_hash
                    """,
                    {
                        "uri": source_data["uri"],
                        "title": source_data.get("title"),
                        "source_type": source_data.get("type"),
                        "submitter_email": source_data.get("submitter_email"),
                        "created_at": source_data.get("created_at"),
                        "updated_at": source_data.get("updated_at"),
                        "content_hash": source_data.get("content_hash"),
                    }
                )

                # Load Chunks
                for chunk in chunks:
                    session.run(
                        """
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.text = $text, c.position = $position, c.source_id = $source_id
                        """,
                        {
                            "chunk_id": chunk["id"],
                            "text": chunk["text"],
                            "position": chunk["position"],
                            "source_id": chunk["source_id"],
                        }
                    )
                
                # Load Entities and link to Chunks
                for ent in entities:
                    q, p = _entity_to_cypher(dict(ent))
                    session.run(q, p)
                    if ent["chunk_id"]:
                        session.run(
                            """
                            MATCH (n {entity_id: $entity_id}), (c:Chunk {chunk_id: $chunk_id})
                            MERGE (n)-[:EXTRACTED_FROM]->(c)
                            """,
                            {"entity_id": ent["entity_id"], "chunk_id": ent["chunk_id"]}
                        )
                
                # Link entities to Source
                for ent in entities:
                    session.run(
                        """
                        MATCH (n {entity_id: $entity_id}), (s:Source {uri: $uri})
                        MERGE (n)-[:FROM_SOURCE]->(s)
                        """,
                        {"entity_id": ent["entity_id"], "uri": source_data["uri"]}
                    )

                # Load Edges
                for edge in edges:
                    q, p = _edge_to_cypher(dict(edge))
                    session.run(q, p)
                    
            _log().info("Loaded %d entities, %d edges, %d chunks to Neo4j", len(entities), len(edges), len(chunks))

            # Invalidate synthesis cache for loaded/updated entities
            try:
                from ..query_router import invalidate_entity_cache

                for ent in entities:
                    invalidate_entity_cache(ent["entity_id"])
            except ImportError:
                pass
        except Exception as e:
            _log().error("Neo4j load failed (data saved in SQLite): %s", e)
    else:
        _log().info("Neo4j not configured, skipping graph load (YAML exported)")

    db.update_source(source_id, stage="done", status="complete")
    return True
