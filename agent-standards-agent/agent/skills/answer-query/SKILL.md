---
name: answer-query
description: Answer factual questions about agentic technology standards using the knowledge graph.
allowed-tools: query_kg get_entity
---

When the user asks about protocols, standards, organizations, projects,
capabilities, concepts, or people in the agentic technology space:

1. **Use `query_kg`** to get a grounded answer from the knowledge graph.
   Pass the user's question directly — the KG pipeline handles text-to-Cypher
   conversion and optional Gemini synthesis.

2. **Check the confidence level** in the response:
   - `high`: Present the answer confidently with citations.
   - `medium`: Present the answer but note it may be incomplete.
   - `low`: Explicitly acknowledge the knowledge graph has limited data.

3. **Cite entity IDs** from the response's `entity_ids` field so users can
   drill deeper. Format: "The MCP protocol (protocol:mcp)..."

4. **Use `get_entity`** when you need deeper detail about a specific entity,
   especially its relationships to other entities.

5. **If the KG returns empty or error results**, do NOT hallucinate an answer.
   Instead say: "The knowledge graph doesn't have data on this topic yet."

6. **Include source URLs** when the KG response provides them in the `sources`
   field.
