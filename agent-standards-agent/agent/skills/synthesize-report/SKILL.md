---
name: synthesize-report
description: Generate a narrative report on a topic across the agentic standards landscape.
allowed-tools: query_kg get_entity find_related_people
---

When the user asks for a report, overview, landscape analysis, or summary
of a topic area in the agentic standards ecosystem:

1. **Break the topic into sub-questions.** A landscape report on "agent identity
   standards" might need queries about:
   - What protocols address identity? (`query_kg`)
   - What organizations govern them? (`query_kg`)
   - Who are the key people? (`find_related_people`)
   - How do they relate to each other? (`get_entity` for key entities)

2. **Make multiple `query_kg` calls** to gather comprehensive information.
   Each call to the 3-tier pipeline returns grounded answers with entity IDs.

3. **Organize the report thematically:**
   - Start with a brief executive summary (2-3 sentences)
   - Group by sub-topic or entity type (protocols, organizations, people)
   - Highlight relationships and connections between entities
   - Note areas where information is incomplete

4. **Cite everything.** Every factual claim should reference an entity_id
   or source URL from the knowledge graph.

5. **Include a "Key People" section** using `find_related_people` results.
   This connects users with the humans behind the standards.

6. **Note confidence levels.** If some parts of the report draw on
   low-confidence data, flag them explicitly.

7. **Keep it concise.** Aim for 500-1000 words unless the user requests
   more detail. Use bullet points and headers for scannability.
