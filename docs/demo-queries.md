# Demo Cypher Queries

Annotated queries demonstrating the combined Wikidata + agentic-domain knowledge graph.

---

## Query 1: What protocols does Google develop or contribute to?

Shows: Wikidata-grounded organization connected to both Wikidata-sourced and agentic-domain entities.

```cypher
MATCH (org:Organization {name: "Google"})-[:DEVELOPS|CONTRIBUTES_TO]->(p)
WHERE p:Protocol OR p:Project
RETURN p.name, p.type, p.kind, p.wikidata_id
ORDER BY p.name
```

**Expected:** A2A, ADK, Gemini, Cloud Run, plus any Wikidata-sourced protocols or projects linked to Google.

---

## Query 2: Which programming languages influence the agentic ecosystem?

Shows: Cross-domain traversal between Wikidata programming languages and agentic-domain projects.

```cypher
MATCH (lang:Project {kind: "programming_language"})<-[:USES|IMPLEMENTS]-(agent_project:Project)
WHERE agent_project.source_type <> 'wikidata'
RETURN lang.name, lang.wikidata_id, COLLECT(agent_project.name) AS used_by
ORDER BY SIZE(used_by) DESC
LIMIT 20
```

**Expected:** Languages like Python, TypeScript, Go appearing as bridges between the Wikidata corpus and agentic-domain projects.

---

## Query 3: Timeline — What happened in the agentic ecosystem in 2025?

Shows: Temporal model with Event nodes.

```cypher
MATCH (e:Event)
WHERE e.date >= date("2025-01-01") AND e.date < date("2026-01-01")
OPTIONAL MATCH (entity)-[:PARTICIPATED_IN]->(e)
RETURN e.date, e.title, e.event_type, COLLECT(entity.name) AS participants
ORDER BY e.date
```

**Expected:** AGNTCY donation to Linux Foundation (2025-07-29) with AGNTCY and Linux Foundation as participants.

---

## Query 4: Point-in-time snapshot — What relationships were active on 2025-06-01?

Shows: Bitemporal edge queries using valid_from/valid_to properties.

```cypher
MATCH (a)-[r]->(b)
WHERE r.valid_from IS NOT NULL
  AND r.valid_from <= date("2025-06-01")
  AND (r.valid_to IS NULL OR r.valid_to > date("2025-06-01"))
RETURN a.name, TYPE(r) AS relationship, b.name, r.valid_from
ORDER BY r.valid_from DESC
LIMIT 50
```

**Expected:** Relationships with populated temporal data from Wikidata inception dates (e.g., developer→language relationships with valid_from matching the language's creation date).

---

## Query 5: Entity grounding — Agentic entities with Wikidata cross-references

Shows: Wikidata grounding of existing entities via the cross-referencing module.

```cypher
MATCH (n:Entity)
WHERE n.wikidata_id IS NOT NULL
RETURN n.name, n.type, n.kind, n.wikidata_id, n.entity_id
ORDER BY n.type, n.name
```

**Expected:** Organizations (Google Q95, Microsoft Q2283, Anthropic Q113575029, etc.), protocols (OAuth Q220450, OpenAPI Q15057770), and projects (ChatGPT Q115647700, Gemini Q115060924) with their Wikidata identifiers.

---

## Query 6: Graph stats — What do we have?

Shows: High-level overview of entity counts by type and source.

```cypher
MATCH (n:Entity)
WITH n.type AS type, COUNT(*) AS count,
     SUM(CASE WHEN n.wikidata_id IS NOT NULL THEN 1 ELSE 0 END) AS with_wikidata,
     SUM(CASE WHEN n.source_type = 'wikidata' THEN 1 ELSE 0 END) AS from_wikidata
RETURN type, count, with_wikidata, from_wikidata
ORDER BY count DESC
```

**Expected:** Project (largest — programming languages + free software), Organization (software companies + AI orgs), Protocol (communication protocols + standards), plus seed entity types.

---

## Query 7: Wikidata entity deep dive — Software companies by country

Shows: Wikidata-enriched attributes beyond what agentic-domain sources provide.

```cypher
MATCH (org:Organization {kind: "company", source_type: "wikidata"})
WHERE org.created_at IS NOT NULL
RETURN org.name, org.created_at, org.wikidata_id, org.url
ORDER BY org.created_at
LIMIT 25
```

**Expected:** A chronological list of software companies with founding dates from Wikidata.

---

## Query 8: Event participation network

Shows: Event-centric view connecting organizations through shared events.

```cypher
MATCH (e:Event)
OPTIONAL MATCH (entity)-[r:PARTICIPATED_IN]->(e)
RETURN e.title, e.event_type, e.date,
       COLLECT({name: entity.name, role: r.role}) AS participants
ORDER BY e.date DESC
```

**Expected:** Events with their participant organizations and roles (donor, recipient, proposer, forum, etc.).
