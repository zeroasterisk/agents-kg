---
name: find-people
description: Find people working on a specific protocol, standard, or topic area.
allowed-tools: find_related_people get_entity query_kg
---

When the user wants to know about people involved in a topic:

1. **Use `find_related_people`** with the topic. If you know the entity_id
   (e.g., "protocol:mcp"), pass it directly for graph traversal. Otherwise
   pass the topic as natural language.

2. **Pass `exclude_ids`** to filter out people already discussed in the
   conversation. This avoids repeating information.

3. **Present results with context.** For each person, include:
   - Their name and entity_id
   - Their relationship to the topic (e.g., CHAIRS, CONTRIBUTES_TO, AUTHORED)
   - A brief description if available

4. **Use `get_entity`** to get more detail about a specific person the user
   is interested in — their full description, other relationships, etc.

5. **Organize by relevance.** Lead with people who have the strongest
   connections (chairs, authors) before listing general contributors.

6. **Offer to go deeper.** After listing people, offer to look up more
   detail about any specific person or their other work.
