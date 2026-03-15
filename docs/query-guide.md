# KG Query Guide

The `agents-kg` pipeline stores state in a SQLite database (`pipeline.db`). You can query this database directly to analyze the ecosystem.

## Core Tables
- `entities`: All extracted entities (Organization, Project, Protocol, Capability, Person, Group)
- `edges`: Relationships between entities
- `sources`: Ingested documents/URLs
- `chunks`: Text snippets with embeddings

---

## 1. Ecosystem Analysis

### "What protocols are competing or complementing A2A?"
```sql
SELECT e.edge_type, tgt.name, tgt.entity_id
FROM edges e
JOIN entities tgt ON tgt.entity_id = e.target_entity_id
WHERE e.source_entity_id = 'protocol:a2a'
   OR e.target_entity_id = 'protocol:a2a'
ORDER BY e.edge_type;
```

### "Who is building on A2A?" (Implementors)
```sql
SELECT src.name, src.type, src.kind
FROM edges e
JOIN entities src ON src.entity_id = e.source_entity_id
WHERE e.target_entity_id = 'protocol:a2a'
  AND e.edge_type = 'IMPLEMENTS';
```

### "What capabilities does MCP define?"
```sql
SELECT tgt.name, tgt.entity_id
FROM edges e
JOIN entities tgt ON tgt.entity_id = e.target_entity_id
WHERE e.source_entity_id = 'protocol:mcp'
  AND e.edge_type = 'DEFINES';
```

---

## 2. Organization & Governance

### "Which organizations develop which projects?"
```sql
SELECT org.name as organization, proj.name as project
FROM edges e
JOIN entities org ON org.entity_id = e.source_entity_id
JOIN entities proj ON proj.entity_id = e.target_entity_id
WHERE e.edge_type = 'DEVELOPS'
  AND org.type = 'Organization';
```

### "List all working groups for the MCP organization"
```sql
SELECT name, entity_id
FROM entities
WHERE type = 'Group'
  AND entity_id IN (
    SELECT source_entity_id FROM edges 
    WHERE target_entity_id = 'organization:modelcontextprotocol'
      AND edge_type = 'MEMBER_OF'
  );
```

---

## 3. Discovery & Gaps

### "Most-referenced entities (The 'Hubs')"
```sql
SELECT entity_id, COUNT(*) as edge_count
FROM (
  SELECT source_entity_id as entity_id FROM edges
  UNION ALL
  SELECT target_entity_id FROM edges
) GROUP BY entity_id ORDER BY edge_count DESC LIMIT 20;
```

### "Pre-seeded entities we haven't found in any source yet"
```sql
SELECT entity_id, name FROM entities 
WHERE (source_id IS NULL OR source_id = 0)
  AND status = 'pending_review'; -- Seeds start here
```

---

## 4. Maintenance & Quality

### "Find all merged entities and their targets"
```sql
SELECT name, entity_id, merged_into 
FROM entities 
WHERE merged_into IS NOT NULL;
```

### "Find edges with low confidence (< 0.6)"
```sql
SELECT e.source_entity_id, e.edge_type, e.target_entity_id, e.confidence, s.uri
FROM edges e
JOIN sources s ON s.id = e.source_id
WHERE e.confidence < 0.6;
```
