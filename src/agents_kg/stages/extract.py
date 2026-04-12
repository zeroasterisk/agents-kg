"""Stage 5: Extract entities and edges using Gemini Flash structured output."""

import json
import logging
import hashlib
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

EXTRACT_MODEL = "gemini-3.1-flash-lite-preview"

SYSTEM_PROMPT_TEMPLATE = """You are a knowledge graph extraction engine for the agentic web ecosystem.

Given a text chunk, extract entities and relationships according to this ontology.

## NODE TYPES (use ONLY these):
- Organization: A legal entity, standards body, or consortium (kind: company, standards_body, foundation, consortium)
- Group: A committee, working group, or team WITHIN an organization (kind: tsc, wg, sig, task_force, team)
- Person: A named individual human (NOT roles like "domain expert" or "human expert")
- Project: Runnable code — has a repo, releases, or deployable artifacts (kind: framework, sdk, library, tool, platform)
- Protocol: A specification document — has a version, authors, formal status (kind: spec, standard, rfc, draft)
- Capability: A feature or ability that something provides — always describe what it DOES, not what it IS

## TYPE DISAMBIGUATION (critical):
- "MCP" the specification → protocol:mcp
- "MCP SDK" the code library → project:mcp-sdk-typescript or project:mcp-sdk-python
- "MCP support" as a feature → capability:tool-use (or a more specific capability)
- "ACP" is overloaded. Distinguish between IBM's ACP (protocol:ibm-acp), OpenAI's payments ACP (protocol:openai-acp), and Zed's local stdio ACP (protocol:zed-acp).
- "Google" the company → organization:google
- "Vertex AI" the platform → project:vertex-ai
- A named technique (ReAct, CoT, RAG) → Project/framework, NOT Capability
- An abstract ability (reasoning, planning, tool use) → Capability
- Example agents in a whitepaper (e.g. "SalesAgent", "MarketingAgent") → DO NOT extract as entities (they are illustrative, not real projects)
- Generic roles ("domain expert", "human expert", "product manager") → DO NOT extract as Person entities
- Headings, section titles, book titles → DO NOT extract as entities

## EDGE TYPES (use ONLY these 14):
MEMBER_OF, GOVERNS, DEVELOPS, IMPLEMENTS, COMPETES_WITH, ADDRESSES, AUTHORED, CHAIRS, SPONSORS, PART_OF, SUPERSEDES, CONTRIBUTES_TO, DEFINES, COMPLEMENTS

## EDGE DIRECTION RULES:
- ADDRESSES: Use when an entity was DESIGNED TO SOLVE a capability (e.g., Protocol or Project ADDRESSES Capability)
- DO NOT use ADDRESSES for Person entities — use AUTHORED instead
- Person —AUTHORED→ Protocol/Project (when person created/wrote the thing)
- Person —CONTRIBUTES_TO→ Organization/Project (when person contributes to)
- Person —MEMBER_OF→ Organization/Group
- Group —MEMBER_OF→ Organization (working groups are members of organizations, not protocols)
- Organization —DEVELOPS→ Project (org creates the project)
- Project —IMPLEMENTS→ Protocol (code implements a spec)
- Protocol —COMPLEMENTS→ Protocol (when a protocol builds on or complements another protocol)
- Protocol —DEFINES→ Capability (spec defines a capability)
- Capability —PART_OF→ Capability (sub-capability)
- Organization —SPONSORS→ Protocol/Project
- Protocol —USES→ Protocol (when a protocol is built on top of another protocol)

## KNOWN ENTITIES (prefer these over creating new ones):
{seed_entities}

## OUTPUT FORMAT:
Respond with valid JSON:
{{
  "entities": [
    {{
      "entity_id": "type:kebab-case-name",
      "name": "Display Name",
      "type": "Organization|Group|Person|Project|Protocol|Capability",
      "kind": "specific kind or null",
      "description": "One sentence description",
      "aliases": ["alt name 1"]
    }}
  ],
  "edges": [
    {{
      "source_entity_id": "type:name",
      "target_entity_id": "type:name",
      "edge_type": "DEVELOPS",
      "confidence": 0.9,
      "valid_from": "YYYY-MM-DD or null",
      "valid_to": "YYYY-MM-DD or null",
      "properties": {{}}
    }}
  ]
}}

## RULES:
- entity_id format: ALWAYS "type:kebab-case-name" — NEVER include kind in the id
- CORRECT: "organization:google", "protocol:a2a", "project:mcp-sdk-python"
- WRONG:   "organization:google/company", "protocol:a2a/spec", "project:mcp-sdk-python/sdk"
- The kind field exists separately — do not embed it in the entity_id
- REUSE known entity_ids from the list above when they match
- Only extract what's explicitly stated or strongly implied
- Extract valid_from and valid_to for edges if explicitly stated (e.g. dates of joining, starting, or superseding)
- Set confidence 0.5-1.0 based on how explicit the relationship is
- DO NOT extract illustrative examples, hypothetical agents, or generic roles
- DO NOT invent edge types — use ONLY the 15 listed above
- If nothing relevant found, return {{"entities": [], "edges": []}}
- Prefer fewer, high-quality extractions over many low-quality ones
"""


VALID_EDGE_TYPES = {
    "MEMBER_OF", "GOVERNS", "DEVELOPS", "IMPLEMENTS", "COMPETES_WITH",
    "ADDRESSES", "AUTHORED", "CHAIRS", "SPONSORS", "PART_OF",
    "SUPERSEDES", "CONTRIBUTES_TO", "DEFINES", "COMPLEMENTS", "USES",
}

VALID_ENTITY_TYPES = {
    "Organization", "Group", "Person", "Project", "Protocol", "Capability",
}


def _make_edge_id(src: str, tgt: str, edge_type: str) -> str:
    raw = f"{src}|{edge_type}|{tgt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_system_prompt() -> str:
    """Build the system prompt with seed entities injected."""
    from ..seed import format_seed_for_prompt
    return SYSTEM_PROMPT_TEMPLATE.format(seed_entities=format_seed_for_prompt())


def run(db: Database, source: dict) -> bool:
    source_id = source["id"]
    chunks = db.get_chunks(source_id)
    if not chunks:
        raise RuntimeError("No chunks to extract from")

    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai not installed")

    import os
    kwargs = {}
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        kwargs["vertexai"] = True
        kwargs["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        kwargs["location"] = "global"
    client = genai.Client(**kwargs)

    system_prompt = _build_system_prompt()

    total_entities = 0
    total_edges = 0

    for chunk in chunks:
        _log().info("Extracting from chunk %d (source %d)", chunk["id"], source_id)

        try:
            response = client.models.generate_content(
                model=EXTRACT_MODEL,
                contents=f"Extract entities and relationships from this text:\n\n{chunk['text']}",
                config={
                    "system_instruction": system_prompt,
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                    "thinking_config": {"thinking_level": "medium"},
                },
            )

            data = json.loads(response.text)
        except (json.JSONDecodeError, Exception) as e:
            _log().warning("Extraction failed for chunk %d: %s", chunk["id"], e)
            continue

        for ent in data.get("entities", []):
            etype = ent.get("type", "")
            if etype not in VALID_ENTITY_TYPES:
                _log().warning("Skipping entity with invalid type %r: %s", etype, ent.get("entity_id"))
                continue
            db.add_entity(
                entity_id=ent["entity_id"],
                name=ent["name"],
                entity_type=etype,
                kind=ent.get("kind"),
                description=ent.get("description"),
                aliases=ent.get("aliases"),
                source_id=source_id,
                chunk_id=chunk["id"],
            )
            total_entities += 1

        for edge in data.get("edges", []):
            edge_type = edge.get("edge_type", "")
            if edge_type not in VALID_EDGE_TYPES:
                _log().warning("Skipping edge with invalid type %r: %s -> %s",
                             edge_type, edge.get("source_entity_id"), edge.get("target_entity_id"))
                continue
            edge_id = _make_edge_id(edge["source_entity_id"], edge["target_entity_id"], edge_type)
            db.add_edge(
                edge_id=edge_id,
                source_entity_id=edge["source_entity_id"],
                target_entity_id=edge["target_entity_id"],
                edge_type=edge_type,
                properties=edge.get("properties"),
                confidence=edge.get("confidence", 0.5),
                chunk_id=chunk["id"],
                source_id=source_id,
                valid_from=edge.get("valid_from"),
                valid_to=edge.get("valid_to"),
            )
            total_edges += 1

    _log().info("Extracted %d entities, %d edges from source %d", total_entities, total_edges, source_id)
    db.update_source(source_id, stage="resolve", status="processing")
    return True
