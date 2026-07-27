"""Agent Standards Agent — root agent definition.

A Google ADK 2.0 agent that answers questions about agentic technology
standards, protocols, organizations, and people using a live knowledge
graph with 17,580+ entities.
"""

from __future__ import annotations

import logging
import os

from google.adk import Agent
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.tools.load_memory_tool import load_memory_tool
from google.adk.tools.preload_memory_tool import preload_memory_tool

from .tools.kg_tools import query_kg, get_entity, find_related_people, _kg_request, ENTITY_ID_RE
from .tools.model_armor import before_model_callback, after_model_callback

log = logging.getLogger("agent_standards.agent")

# ---------------------------------------------------------------------------
# System instruction
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """\
You are the **Agent Standards Agent**, an expert assistant for the agentic
technology standards ecosystem. You have access to a live knowledge graph
containing 17,580+ entities covering protocols, standards, organizations,
people, projects, capabilities, concepts, and working groups.

## Knowledge Graph Schema

**Entity types:** Protocol, Organization, Project, Capability, Group, Person, Concept

**Entity properties:** entity_id (unique, format "type:slug"), name, type, kind, description, aliases

**Relationships:** IMPLEMENTS, CONTRIBUTES_TO, AUTHORED, MEMBER_OF, DEVELOPS,
COMPLEMENTS, GOVERNS, SPONSORS, CHAIRS, PART_OF, USES, ADDRESSES,
COMPETES_WITH, SUPERSEDES, DEFINES, FROM_SOURCE, EXTRACTED_FROM

## How to Answer Questions

1. **ALWAYS use `query_kg` for factual questions** about standards, protocols,
   organizations, or people. Never answer from your training data alone when
   the knowledge graph can provide grounded information.

2. **Use `get_entity`** to look up specific entities by their entity_id when
   you need detailed information or relationships (e.g., "protocol:mcp",
   "person:aaron-parecki", "organization:ietf").

3. **Use `find_related_people`** when the user asks about who works on a topic,
   or when you want to proactively surface relevant contributors.

4. **Cite entity IDs** in your responses so users can follow up. Format them
   naturally, e.g., "The MCP protocol (protocol:mcp) was developed by..."

5. **Acknowledge uncertainty honestly.** When the knowledge graph returns
   low-confidence results or no data, say so explicitly:
   - "The knowledge graph doesn't have complete information about this."
   - "I found limited data on this topic — here's what I know."
   - "This may not be fully up to date."

6. **Stay on topic.** You specialize in agentic technology standards. If a user
   asks about something outside this domain, briefly redirect:
   "I specialize in agentic technology standards — protocols like A2A, MCP,
   ACP, AG-UI, identity standards like WIMSE, and the organizations and
   people behind them. How can I help you with that?"

7. **Proactively suggest related people.** When discussing a protocol or
   standard, consider mentioning key contributors the user might want to
   know about. Use `find_related_people` for this.

## Response Style

- Be concise but thorough. Lead with the answer, then provide supporting detail.
- Use structured formatting (headers, bullet points) for complex answers.
- When synthesizing across multiple entities, organize by theme or relationship.
- Include source URLs when available from the knowledge graph.
"""

# ---------------------------------------------------------------------------
# Proactive people suggestion callback
# ---------------------------------------------------------------------------


async def after_agent_callback(callback_context, agent_response):
    """Proactively suggest related people after the agent responds.

    Examines entity_ids from tool results in the current turn and queries
    for connected Person entities not yet mentioned in the response.
    """
    try:
        # Collect entity_ids from the session state or tool results
        entity_ids = set()

        # Try to extract entity_ids from tool call results in the events
        if hasattr(callback_context, "state") and callback_context.state:
            # ADK stores tool results in state
            pass

        # Look for entity_ids mentioned in the response text
        response_text = ""
        if hasattr(agent_response, "text"):
            response_text = agent_response.text or ""
        elif hasattr(agent_response, "content"):
            if hasattr(agent_response.content, "parts"):
                response_text = " ".join(
                    p.text for p in agent_response.content.parts
                    if hasattr(p, "text") and p.text
                )

        if not response_text:
            return agent_response

        # Extract entity_ids from the response text (format: type:slug)
        import re
        # Match entity_id patterns in the response
        potential_ids = re.findall(r'\b([a-z_]+:[a-z0-9][a-z0-9._-]*)\b', response_text)
        for pid in potential_ids:
            if ENTITY_ID_RE.match(pid):
                entity_ids.add(pid)

        if not entity_ids:
            return agent_response

        # Filter to non-Person topic entities
        topic_ids = [eid for eid in entity_ids if not eid.startswith("person:")]
        if not topic_ids:
            return agent_response

        # Find related people for up to 3 topics
        all_people = []
        seen = set()

        for topic_id in topic_ids[:3]:
            safe_id = topic_id.replace("\\", "\\\\").replace("'", "\\'")
            cypher = (
                f"MATCH (topic:Entity {{entity_id: '{safe_id}'}})-[r]-(p:Person) "
                "WHERE p.entity_id IS NOT NULL "
                "RETURN DISTINCT p.entity_id AS entity_id, p.name AS name, "
                "TYPE(r) AS relationship LIMIT 5"
            )
            try:
                result = await _kg_request("POST", "/query", json={"cypher": cypher})
                for person in result.get("results", []):
                    pid = person.get("entity_id", "")
                    name = person.get("name", "")
                    if pid and pid not in seen and name and name not in response_text:
                        seen.add(pid)
                        all_people.append(person)
            except Exception:
                log.debug("People lookup failed for %s", topic_id)

        if not all_people:
            return agent_response

        # Build suggestion text
        people_lines = []
        for p in all_people[:5]:
            rel = p.get("relationship", "related").replace("_", " ").title()
            people_lines.append(f"  - **{p['name']}** ({p['entity_id']}) — {rel}")

        suggestion = (
            "\n\n---\n"
            "You might also be interested in these people connected to this topic:\n"
            + "\n".join(people_lines)
            + "\n\nWould you like to know more about any of them?"
        )

        # Append to the response
        if hasattr(agent_response, "text") and agent_response.text:
            agent_response.text += suggestion

    except Exception as e:
        log.debug("after_agent_callback error (non-blocking): %s", e)

    return agent_response


# ---------------------------------------------------------------------------
# Memory service
# ---------------------------------------------------------------------------

memory_service = VertexAiMemoryBankService(
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", "data-ingest-demo"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    agent_engine_id=os.environ.get("AGENT_ENGINE_ID", ""),
)

# ---------------------------------------------------------------------------
# Root agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="agent_standards_agent",
    model="gemini-3.6-flash",
    description=(
        "Expert on agentic technology standards, protocols, organizations, "
        "and people. Backed by a live knowledge graph with 17,580+ entities."
    ),
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        query_kg,
        get_entity,
        find_related_people,
        load_memory_tool,
        preload_memory_tool,
    ],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)
