"""Knowledge graph tools for the Agent Standards Agent.

These tools call the agents-kg REST API to query the knowledge graph
containing 17,580+ entities about agentic technology standards.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import httpx

from google.adk.tools.tool_context import ToolContext

log = logging.getLogger("agent_standards.tools")

KG_API_URL = os.environ.get("KG_API_URL", "http://35.202.188.73:8000")
KG_API_KEY = os.environ.get("KG_API_KEY", "")

# Google OAuth audience for the KG API (web app client ID)
KG_API_AUDIENCE = os.environ.get(
    "KG_API_AUDIENCE",
    "160698144102-v9a0shre6ntap7jitla83b82i5akl1j0.apps.googleusercontent.com",
)

# Validate entity_id format: type:slug (e.g., protocol:mcp, person:aaron-parecki)
ENTITY_ID_RE = re.compile(r"^[a-z_]+:[a-z0-9][a-z0-9._-]*$")

# Cached ID token credentials (refreshed automatically)
_id_token_creds = None


def _get_auth_token() -> str:
    """Get a Bearer token for the KG API.

    Tries in order:
    1. Static API key (KG_API_KEY env var) — fastest, no network call.
    2. Google service account ID token — for GCP-authenticated environments.
    """
    if KG_API_KEY:
        return KG_API_KEY

    # Fall back to Google service account ID token
    global _id_token_creds
    try:
        if _id_token_creds is None:
            from google.oauth2 import service_account as sa
            import google.auth

            # Try to find service account credentials
            sa_key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            if sa_key_path and os.path.exists(sa_key_path):
                _id_token_creds = sa.IDTokenCredentials.from_service_account_file(
                    sa_key_path, target_audience=KG_API_AUDIENCE
                )
            else:
                # Use ADC (Application Default Credentials)
                creds, _ = google.auth.default()
                _id_token_creds = creds

        from google.auth.transport import requests as google_requests
        _id_token_creds.refresh(google_requests.Request())
        return _id_token_creds.token
    except Exception as e:
        log.warning("Failed to get auth token: %s", e)
        return ""


async def _kg_request(method: str, path: str, **kwargs) -> dict:
    """Make an authenticated request to the KG API.

    Uses Bearer token auth matching the HTTPBearer scheme in the KG API's
    auth.py. Supports both static API keys and Google service account ID tokens.
    """
    headers = kwargs.pop("headers", {})
    token = _get_auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.request(
            method, f"{KG_API_URL}{path}", headers=headers, **kwargs
        )
        if resp.status_code != 200:
            log.warning("KG API %s %s returned %d: %s", method, path, resp.status_code, resp.text[:200])
            return {"error": f"KG API returned {resp.status_code}", "detail": resp.text[:500]}
        return resp.json()


async def query_kg(question: str, tool_context: ToolContext) -> dict:
    """Query the agent standards knowledge graph with a natural language question.

    Uses a three-tier pipeline: text-to-Cypher against Neo4j, optional Gemini
    synthesis from KG subgraph + source chunks, and a 7-day disk cache.

    Args:
        question: The question to answer about agentic technology standards,
            protocols, organizations, projects, or people.

    Returns:
        Dict with keys:
            - answer: The natural language answer
            - confidence: "high", "medium", or "low"
            - entity_ids: List of entity IDs referenced in the answer
            - sources: List of source URIs
            - source: Where the answer came from ("kg", "synthesis", or "cached")
    """
    return await _kg_request("POST", "/ask", json={"question": question})


async def get_entity(entity_id: str, tool_context: ToolContext) -> dict:
    """Get detailed information about a specific entity from the knowledge graph.

    Retrieves entity properties and its direct relationships (neighbors).

    Args:
        entity_id: The entity identifier in 'type:slug' format
            (e.g., 'protocol:mcp', 'person:aaron-parecki', 'organization:google').

    Returns:
        Dict with entity details: entity_id, name, type, kind, description,
        aliases, and a list of relationships with neighboring entities.
    """
    if not ENTITY_ID_RE.match(entity_id):
        return {"error": f"Invalid entity_id format: '{entity_id}'. Expected 'type:slug' (e.g., 'protocol:mcp')."}

    # Use Cypher via POST /query since no /entities/{id} endpoint exists.
    # The /query endpoint enforces read-only via WRITE_KEYWORDS regex.
    safe_id = entity_id.replace("\\", "\\\\").replace("'", "\\'")

    # Get entity properties
    cypher_entity = (
        f"MATCH (e:Entity {{entity_id: '{safe_id}'}}) "
        "RETURN e.entity_id AS entity_id, e.name AS name, e.type AS type, "
        "e.kind AS kind, e.description AS description, e.aliases AS aliases"
    )
    result = await _kg_request("POST", "/query", json={"cypher": cypher_entity})

    if "error" in result:
        return result

    entities = result.get("results", [])
    if not entities:
        return {"error": f"Entity '{entity_id}' not found in the knowledge graph."}

    entity = entities[0]

    # Get relationships
    cypher_rels = (
        f"MATCH (e:Entity {{entity_id: '{safe_id}'}})-[r]-(other:Entity) "
        "RETURN other.entity_id AS related_entity_id, other.name AS related_name, "
        "other.type AS related_type, TYPE(r) AS relationship, "
        "CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END AS direction "
        "LIMIT 20"
    )
    rels_result = await _kg_request("POST", "/query", json={"cypher": cypher_rels})
    entity["relationships"] = rels_result.get("results", [])

    return entity


async def find_related_people(
    topic: str, exclude_ids: list[str], tool_context: ToolContext
) -> dict:
    """Find people working on a specific topic in the agentic standards ecosystem.

    Traverses the knowledge graph from a topic entity to find connected Person
    nodes via relationships like CONTRIBUTES_TO, CHAIRS, AUTHORED, DEVELOPS, etc.

    Args:
        topic: The topic to find related people for. Can be an entity_id
            (e.g., 'protocol:mcp') or a natural language topic description
            (e.g., 'agent authorization').
        exclude_ids: List of person entity_ids already mentioned, to avoid
            repeating people the user already knows about.

    Returns:
        Dict with 'people' list, each containing entity_id, name, description,
        and their relationship to the topic.
    """
    # If topic looks like an entity_id, do a direct graph traversal
    if ENTITY_ID_RE.match(topic):
        return await _find_people_by_entity(topic, exclude_ids)

    # Otherwise, use query_kg to find relevant entities first, then traverse
    kg_result = await _kg_request(
        "POST", "/ask",
        json={"question": f"Who are the key people working on {topic}? List their names and roles."}
    )

    if "error" in kg_result:
        return kg_result

    # Also try to get entity_ids from the KG answer and traverse from them
    entity_ids = kg_result.get("entity_ids", [])
    topic_ids = [eid for eid in entity_ids if not eid.startswith("person:")]

    all_people = []
    seen_ids = set(exclude_ids) if exclude_ids else set()

    # Extract people mentioned in the KG answer
    person_ids = [eid for eid in entity_ids if eid.startswith("person:")]
    for pid in person_ids:
        if pid not in seen_ids:
            seen_ids.add(pid)
            entity = await _kg_request(
                "POST", "/query",
                json={"cypher": f"MATCH (p:Person {{entity_id: '{pid.replace(chr(39), '')}'}}) "
                      "RETURN p.entity_id AS entity_id, p.name AS name, "
                      "p.description AS description LIMIT 1"}
            )
            results = entity.get("results", [])
            if results:
                all_people.append({**results[0], "relationship": "mentioned in KG answer"})

    # Traverse from topic entities to find more people
    for tid in topic_ids[:3]:
        if ENTITY_ID_RE.match(tid):
            people_result = await _find_people_by_entity(tid, list(seen_ids))
            for person in people_result.get("people", []):
                if person.get("entity_id") and person["entity_id"] not in seen_ids:
                    seen_ids.add(person["entity_id"])
                    all_people.append(person)

    return {
        "topic": topic,
        "kg_answer": kg_result.get("answer", ""),
        "people": all_people[:10],
    }


async def _find_people_by_entity(entity_id: str, exclude_ids: list[str]) -> dict:
    """Find Person nodes connected to a specific entity via graph traversal."""
    safe_id = entity_id.replace("\\", "\\\\").replace("'", "\\'")
    exclude_clause = ""
    if exclude_ids:
        safe_excludes = [eid.replace("\\", "\\\\").replace("'", "\\'") for eid in exclude_ids if ENTITY_ID_RE.match(eid)]
        if safe_excludes:
            exclude_list = ", ".join(f"'{e}'" for e in safe_excludes)
            exclude_clause = f" AND NOT p.entity_id IN [{exclude_list}]"

    # Bidirectional traversal: Person -[r]-> Entity and Entity -[r]-> Person
    cypher = (
        f"MATCH (topic:Entity {{entity_id: '{safe_id}'}})-[r]-(p:Person) "
        f"WHERE p.entity_id IS NOT NULL{exclude_clause} "
        "RETURN DISTINCT p.entity_id AS entity_id, p.name AS name, "
        "p.description AS description, TYPE(r) AS relationship, "
        "CASE WHEN startNode(r) = p THEN 'contributor' ELSE 'subject' END AS role "
        "LIMIT 10"
    )
    result = await _kg_request("POST", "/query", json={"cypher": cypher})
    return {
        "topic": entity_id,
        "people": result.get("results", []),
    }
