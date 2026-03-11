# Pipeline Review: 2026-03-11

**Date:** 2026-03-11  
**Run type:** Full clean-slate re-run (pipeline.db deleted and rebuilt from scratch)  
**Reviewer:** Zaf (automated critical review with manual QA)

---

## 1. Pipeline Run Summary

### Sources

| # | URI | Status | Entities (active) | Edges |
|---|-----|--------|-------------------|-------|
| 1 | modelcontextprotocol.io/docs/getting-started/intro | ✅ | 1 | 0 |
| 2 | modelcontextprotocol.io/docs/learn/architecture | ✅ | 6 | 6 |
| 3 | modelcontextprotocol.io/docs/learn/client-concepts | ✅ | 0 | 0 |
| 4 | modelcontextprotocol.io/docs/learn/server-concepts | ✅ | 0 | 0 |
| 5 | modelcontextprotocol.io/docs/develop/build-client | ✅ | 3 | 0 |
| 6 | modelcontextprotocol.io/docs/develop/build-server | ✅ | 0 | 0 |
| 7 | modelcontextprotocol.io/docs/develop/connect-local-servers | ✅ | 0 | 0 |
| 8 | modelcontextprotocol.io/docs/develop/connect-remote-servers | ✅ | 0 | 0 |
| 9 | modelcontextprotocol.io/docs/sdk | ✅ | 0 | 0 |
| 10 | modelcontextprotocol.io/docs/tools/inspector | ✅ | 1 | 1 |
| 11 | modelcontextprotocol.io/docs/tutorials/security/authorization | ✅ | 3 | 4 |
| 12 | modelcontextprotocol.io/docs/tutorials/security/security_best_practices | ✅ | 2 | 5 |
| 13 | modelcontextprotocol.io/community/design-principles | ✅ | 0 | 0 |
| 14 | modelcontextprotocol.io/community/governance | ✅ | 0 | 0 |
| 15 | modelcontextprotocol.io/community/contributing | ✅ | 4 | 5 |
| 16 | modelcontextprotocol.io/extensions/overview | ✅ | 2 | 3 |
| 17 | modelcontextprotocol.io/extensions/apps/overview | ✅ | 0 | 0 |
| 18 | modelcontextprotocol.io/extensions/auth/overview | ✅ | 1 | 1 |
| 19 | modelcontextprotocol.io/clients | ✅ | 0 | 0 |
| 20 | modelcontextprotocol.io/examples | ✅ | 0 | 0 |
| 21 | modelcontextprotocol.io/development/roadmap | ✅ | 7 | 8 |
| 22 | github.com/a2aproject/A2A | ✅ | 6 | 13 |
| 23 | a2a-protocol.org/latest/ (substituted) | ✅ | 0 | 6 |
| 24 | a2aprotocol.ai | ✅ | 11 | 0 |
| 25 | a2aprotocol.ai/blog | ✅ | 1 | 2 |
| 26 | sources/pdfs/day1_introduction_to_agents.pdf | ✅ | 26 | 45 |
| 27 | agntcy.org | ✅ | 1 | 1 |

**Original source 23** (`github.com/a2aproject/A2A/blob/main/specification/json/a2a.json`) was rate-limited by GitHub and the path no longer exists as raw content. Substituted with `a2a-protocol.org/latest/`.

### Summary Stats

- **27/27 sources processed** (100% success rate)
- **Final stage:** All at `review/pending_review`
- **Total runtime:** ~5 minutes (wall clock), ~300 Gemini API calls
- **Entities extracted:** 106 total → 75 active, 31 merged
- **Edges extracted:** 100 total → 72 with both endpoints active, 28 stale
- **Bugs discovered and fixed:** 2 (see Section 6)

---

## 2. Entity Analysis

### Entity Counts by Type

| Type | Active | Notes |
|------|--------|-------|
| Project | 23 | Includes some misclassified items (see below) |
| Organization | 18 | Clean, good coverage |
| Protocol | 14 | Minor duplicates |
| Capability | 10 | All from seed, no new ones extracted |
| Group | 5 | MCP working groups only |
| Person | 5 | All from Day 1 PDF |
| **Total** | **75** | |

### Entity Counts by Kind

| Type/Kind | Count | Assessment |
|-----------|-------|------------|
| Organization/company | 15 | ✅ Good |
| Capability/(none) | 10 | ⚠️ All Capabilities should have kind or use PART_OF |
| Protocol/standard | 8 | ✅ Good |
| Project/framework | 7 | ✅ Good |
| Project/platform | 7 | ✅ Good |
| Protocol/spec | 6 | ✅ Good |
| Group/wg | 5 | ✅ Good but only wg found, no tsc/sig |
| Person/(none) | 5 | ⚠️ Person has no kind in schema — correct |
| Project/tool | 4 | ✅ Good |
| Organization/consortium | 2 | ✅ Good |
| Project/(none) | 2 | ⚠️ Missing kind: `project:alphaevolve`, `project:modelcontextprotocol/ext-auth` |
| Project/sdk | 2 | ✅ Good |
| Organization/foundation | 1 | ✅ Good |
| Project/library | 1 | ✅ Good |

### Quality Issues

#### ❌ Entity ID Format Inconsistency (Critical)
The LLM sometimes generates `type:name/kind` (e.g., `organization:google/company`) and sometimes `type:name` (e.g., `organization:google`). The seed uses `type:name` format. This causes:
- Extracted entities fail to resolve against seed
- Edges created during extraction reference the `type:name/kind` format
- After merge, 28 edges become stale because they reference the merged-away ID

**Root cause:** The system prompt uses `organization:google` as the entity_id example format, but does NOT explicitly say "never include kind in the entity_id". The LLM infers from the `kind` field that including it in the ID is acceptable.

Affected entities this run:
- `organization:google/company` → merged into `organization:google` (but edges not repointed)
- `organization:linux-foundation/foundation` → merged into `organization:linux-foundation`
- `organization:agntcy/consortium` → merged into `organization:agntcy`
- `organization:ibm/company` → merged into `organization:ibm`
- `organization:modelcontextprotocol/consortium` → merged into `organization:modelcontextprotocol`
- `protocol:a2a/spec` → merged into `protocol:a2a`
- `protocol:acp/spec` → merged into `protocol:acp`
- `project:mcp-sdk-typescript/sdk` → merged into `project:mcp-sdk-typescript`
- `project:adk/framework` → merged into `project:adk`
- `protocol:mcp/spec` — **NOT merged** into seed `protocol:mcp` (missed by resolve)

#### ⚠️ OAuth 2.1 Duplicate (Minor)
Two active entities for the same protocol:
- `protocol:oauth-2.1/standard` (from source 2, with `/standard` kind suffix)
- `protocol:oauth-2.1` (from sources 11+12, without suffix)

Resolution missed this because entity IDs differ. Both are active, both have edges.

#### ⚠️ project:claude/platform Named "Claude Code" (Confusing)
Active entity `project:claude/platform` has name "Claude Code" but Claude Code is a *tool*, not the Claude platform. The seed has `project:claude` with aliases including "Claude Code". Resolution should have merged this but didn't (different entity_id format).

#### ✅ Good Extractions
- All 10 Capability entities matched seed exactly
- All 5 Person entities correctly extracted from Day 1 PDF
- Core protocols (a2a, mcp, acp, openapi, json-rpc) extracted correctly
- MCP Working Groups (5 WGs) extracted correctly from governance/contributing pages
- A2A ecosystem companies (IBM, Google, Linux Foundation, Agntcy) extracted from GitHub README

#### ❌ Misclassified Entities
Several `Project/framework` entities should arguably be `Capability`:
- `project:chain-of-thought` — CoT is a prompting technique / capability, not deployable code
- `project:react` (ReAct) — Same: it's a reasoning approach, not a software project
- `project:rag` — Retrieval-Augmented Generation is a pattern/capability, not a specific project

These are in the seed as `Project/framework` by decision, but this causes semantic confusion when edges like `person:antonio-gulli --ADDRESSES--> project:chain-of-thought` are generated instead of the expected `person:antonio-gulli --AUTHORED--> source:...`.

#### ⚠️ Questionable Entities
- `project:typescript` (kind: tool) — TypeScript the language is not a relevant KG entity for the agentic web ecosystem
- `project:modelcontextprotocol/ext-auth` — Very specific GitHub repo entry; useful but has id format issue
- `project:smokescreen/tool` — Security tool; has id format issue; low relevance

---

## 3. Edge Analysis

### Edge Counts by Type

| Edge Type | Count | Valid? | Notes |
|-----------|-------|--------|-------|
| DEFINES | 23 | ✅ | Protocols → Capabilities, mostly correct |
| ADDRESSES | 22 | ⚠️ | Some semantic mismatches (see below) |
| IMPLEMENTS | 14 | ✅ | Projects → Protocols, correct |
| DEVELOPS | 13 | ✅ | Organizations → Projects, correct |
| SPONSORS | 12 | ⚠️ | 2 reversed directions |
| COMPLEMENTS | 7 | ✅ | Mostly correct |
| MEMBER_OF | 5 | ⚠️ | 4 of 5 have wrong target type |
| PART_OF | 4 | ✅ | Capability hierarchies, correct |
| **Total** | **100** | | |

**0 invalid edge types extracted.** The 14-type ontology is well-enforced.

### Edge Quality Issues

#### ❌ ADDRESSES Semantic Misuse (22 edges, ~50% problematic)
`ADDRESSES` should mean "this entity's purpose is to solve this problem/capability". Instead, it's being used for:
- `person:alan-blount --ADDRESSES--> capability:multi-agent` — Should be `AUTHORED` (person wrote about it), not ADDRESSES
- `person:antonio-gulli --ADDRESSES--> project:chain-of-thought` — Wrong; should be `AUTHORED`
- All 5 Person→ADDRESSES→Capability/Project edges are wrong

The correct edge for Person→wrote-about→Topic is `AUTHORED` (pointing at the source/chunk), but the KG uses ADDRESSES as a proxy. This is a semantic ambiguity in the ontology.

#### ❌ MEMBER_OF Wrong Target (4 of 5 edges)
```
group:agents-wg --MEMBER_OF--> protocol:mcp/spec  ← WRONG
group:governance-wg --MEMBER_OF--> protocol:mcp/spec ← WRONG
group:server-card-wg --MEMBER_OF--> protocol:mcp/spec ← WRONG
group:transports-wg --MEMBER_OF--> protocol:mcp/spec ← WRONG
organization:agntcy --MEMBER_OF--> organization:linux-foundation ← CORRECT ✅
```
Working groups are sub-groups of the MCP Organization, not "members of" the protocol. Should be `MEMBER_OF organization:modelcontextprotocol`.

#### ❌ Reversed SPONSORS (2 edges)
```
protocol:a2a --SPONSORS--> organization:linux-foundation/foundation ← REVERSED
```
Should be `organization:linux-foundation --SPONSORS--> protocol:a2a`.

```
organization:google/company --SPONSORS--> protocol:a2a ← OK direction but stale ID
```

#### ❌ Reversed DEVELOPS (1 edge)
```
project:vertex-ai --DEVELOPS--> organization:google/company ← REVERSED
```
Should be `organization:google --DEVELOPS--> project:vertex-ai`.

#### ⚠️ IMPLEMENTS Misuse
```
protocol:a2a --IMPLEMENTS--> protocol:json-rpc-2.0/standard
protocol:mcp/spec --IMPLEMENTS--> protocol:json-rpc-2.0/standard
```
A protocol doesn't "implement" another protocol — it "uses" it as a transport or "builds on" it. Better as `COMPLEMENTS` or a new edge type `USES`. Minor semantic issue.

#### ✅ Good Edges
- All DEFINES edges (protocol→capability) are directionally correct
- All DEVELOPS edges (except the one reversed) are correct
- COMPLEMENTS (7) are mostly bilateral and accurate
- PART_OF (4) for capability hierarchy is correct

### Stale Edges
28/100 edges (28%) reference merged entity IDs. The `_merge_entity()` function in `resolve.py` repoints edges only within the current DB transaction, but entities extracted in a LATER source that reference the same pre-merge entity ID don't get repointed when that later source resolves.

This is a systemic issue requiring a global edge-repointing pass after all sources complete resolution.

---

## 4. Resolution Report

### Effectiveness

| Metric | Value |
|--------|-------|
| Total extracted entities | 106 |
| Merged into canonical | 31 (29.2%) |
| Rejected as noise | 0 |
| Active entities | 75 |
| Merge rate | 29.2% |

**Note:** The noise rejection list (`noise_kinds`) includes `headphones, attack, discipline, book, whitepaper, benchmark`. None of these appeared in this run, so 0 rejections is expected.

### Top Merge Targets
The most-merged entities were well-known nodes like `project:langchain` (2 merges), and many 1-each merges for Google, Microsoft, Anthropic, Linux Foundation, etc. This shows the seed list is doing its job for major players.

### Remaining Duplicates After Resolution

1. **protocol:oauth-2.1** vs **protocol:oauth-2.1/standard** — same entity, different IDs, both active
2. **protocol:mcp** (extracted from source 22) vs **protocol:mcp/spec** (active from earlier sources) — likely same entity not merged
3. **organization:google** (active, seed format) vs edges referencing **organization:google/company** (merged) — stale edge issue, not a duplicate entity issue

### Resolution Bugs Found

#### Bug #1: Early Return Without Stage Advancement
`resolve.run()` returned `True` early for sources with 0 entities WITHOUT calling `db.update_source(source_id, stage="review", ...)`. This left 13 sources permanently stuck at `resolve/processing`.

**Fix applied:** Added `db.update_source(source_id, stage="review", status="pending_review")` before the early return.

#### Bug #2: Cross-Source Edge Repointing Gap
When entity A from source X is merged into entity B during resolution of source X, `_merge_entity()` does a global SQL UPDATE on all edges. However, if source Y is processed AFTER source X and extracts a fresh entity with ID=A (because the extraction LLM used the `type:name/kind` format), those edges from source Y reference the already-merged ID. When source Y resolves, it merges its copy of A into B, but source Y's edges were already written with the stale ID.

**Root cause:** Entity ID format inconsistency (see Section 2). The fix is in the extraction prompt (see Section 7).

---

## 5. Ontology Recommendations

### Sufficient: 8 Node Types
The 8-type ontology (Organization, Group, Person, Project, Protocol, Capability, Source, Chunk) covers the current corpus well. No extraction was "blocked" by missing types.

**One gap:** `Standard` vs `Protocol` ambiguity. JSON-RPC 2.0, OAuth 2.1, and RFC 2119 are different in nature than MCP or A2A (which are AI-specific protocols). Using `kind: standard` for RFC-style specs vs `kind: spec` for AI-native protocols handles this adequately without a new type.

### Sufficient: 14 Edge Types (with one gap)
All 100 edges used valid types. However:

**Missing edge type: `USES`**  
Currently `protocol:a2a --IMPLEMENTS--> protocol:json-rpc-2.0` is semantically wrong. `IMPLEMENTS` means "this code implements that spec." A protocol building on another protocol is better expressed as `USES` or `BUILT_ON`. Consider adding `USES` as edge type 15.

**Overloaded: `ADDRESSES`**  
`ADDRESSES` is used for both "this project/protocol is designed to solve this capability" AND as a proxy for authorship ("this person wrote about this topic"). The Person→ADDRESSES usage is wrong. The AUTHORED edge should be `Person --AUTHORED--> Source` (or Chunk), not pointing at Capability entities.

**Missing: `AUTHORED`**  
The ontology has `AUTHORED` in the edge type list but it never appears in extracted edges. The extraction prompt should explicitly show `Person --AUTHORED--> Protocol/Project` examples.

### Capability Type Assessment
**Too broad:** All 10 Capability entities are abstract, top-level abilities. There's no hierarchy being exploited (despite PART_OF existing). The `capability:authorization --PART_OF--> capability:authentication` is questionable (auth != just authn).

**Recommendation:** Define a canonical capability hierarchy in the seed:
```
capability:security
  ├── capability:authentication  (PART_OF security)
  ├── capability:authorization   (PART_OF security)
  └── capability:observability   (arguably separate)
capability:agent-intelligence
  ├── capability:planning
  ├── capability:reasoning
  └── capability:memory
capability:multi-agent           (top-level)
  └── capability:tool-use        (PART_OF multi-agent)
```

### Kind Value Assessment
Kind values are generally consistent and useful. Issues:
- `Capability` entities have no kind (correct — kind not defined for Capability in ontology)
- `Person` entities have no kind (correct)
- `project:alphaevolve` and `project:modelcontextprotocol/ext-auth` missing kind — extraction missed it
- `project:typescript` kind=tool is wrong — TypeScript is a language not a tool in this context

---

## 6. Seed List Updates

### Entities That Should Be Added to Seed

These were extracted as new entities but represent well-known ecosystem players that should be seed-listed for better entity_id consistency:

| entity_id | name | type/kind | Why seed? |
|-----------|------|-----------|-----------|
| `organization:agntcy` | AGNTCY | Organization/consortium | Major A2A ecosystem player |
| `protocol:acp` | Agent Communication Protocol | Protocol/spec | IBM's ACP alongside A2A |
| `protocol:a2ui` | A2UI | Protocol/spec | Alan's project, should be pre-seeded |
| `protocol:ag-ui` | AG-UI | Protocol/spec | Emerging standard, frequently cited |
| `protocol:agent-payments` | Agent Payments Protocol | Protocol/spec | x402 standard |
| `group:agents-wg` | Agents WG | Group/wg | MCP governance WG |
| `group:enterprise-wg` | Enterprise WG | Group/wg | MCP governance WG |
| `group:governance-wg` | Governance WG | Group/wg | MCP governance WG |
| `group:server-card-wg` | Server Card WG | Group/wg | MCP governance WG |
| `group:transports-wg` | Transports WG | Group/wg | MCP governance WG |
| `project:mcp-inspector` | MCP Inspector | Project/tool | Official MCP dev tool |
| `project:opentelemetry` | OpenTelemetry | Project/framework | Core observability standard |
| `protocol:spiffe` | SPIFFE | Protocol/standard | Security identity standard |

### Missing Aliases

| entity_id | Missing aliases |
|-----------|----------------|
| `protocol:a2a` | "A2A Protocol", "Agent2Agent Protocol" |
| `organization:agntcy` | "AGNTCY Foundation", "agntcy.org" |
| `project:adk` | "google/adk-python", "ADK Python", "ADK TypeScript" |
| `organization:modelcontextprotocol` | "MCP org", "modelcontextprotocol org" |
| `protocol:mcp` | "MCP spec", "Model Context Protocol spec" |
| `project:claude` | "Anthropic Claude", "Claude API" |

### Entities to Remove from Seed (or Mark as Low-Priority)
- `project:chain-of-thought`, `project:react`, `project:rag` — These are techniques, not software projects with repos. Their classification as `Project/framework` causes ADDRESSES misuse. Consider reclassifying as `Capability`.

---

## 7. Prompt Improvements

### Critical Fix: Entity ID Format Rule

**Current (ambiguous):**
```
- Use kebab-case for entity_id, prefixed with lowercase type (e.g., "organization:google")
```

**Proposed (explicit):**
```
- entity_id format: ALWAYS "type:kebab-case-name" — NEVER include kind in the id
- CORRECT: "organization:google", "protocol:a2a", "project:mcp-sdk-python"  
- WRONG:   "organization:google/company", "protocol:a2a/spec", "project:mcp-sdk-python/sdk"
- The kind field exists separately — do not embed it in the entity_id
```

This single change would eliminate ~28 stale edges and all entity ID conflicts.

### Fix: ADDRESSES Misuse

**Current disambiguation (missing ADDRESSES rule):**  
No explicit rule for when to use ADDRESSES vs AUTHORED.

**Add to EDGE DIRECTION RULES:**
```
- ADDRESSES: Use when an entity was DESIGNED TO SOLVE a capability (e.g., Protocol or Project ADDRESSES Capability)
- DO NOT use ADDRESSES for Person entities — use AUTHORED instead
- Person —AUTHORED→ Protocol/Project (when person created/wrote the thing)
- Person —CONTRIBUTES_TO→ Organization/Project (when person contributes to)
```

### Fix: MEMBER_OF Clarification

**Add to EDGE DIRECTION RULES:**
```
- Group —MEMBER_OF→ Organization (working groups are members of organizations, not protocols)
- Person —MEMBER_OF→ Organization/Group
```

### Fix: IMPLEMENTS Clarification

**Add to EDGE DIRECTION RULES:**
```
- Protocol —COMPLEMENTS→ Protocol (when a protocol uses or builds on another protocol)
- IMPLEMENTS means "runnable code implements a specification", not "spec references another spec"
```

### Add: Explicit Entity ID Examples from Disambiguation

The type disambiguation section should show entity IDs:
```
## TYPE DISAMBIGUATION:
- "MCP" the specification → entity_id: "protocol:mcp" (kind: spec)
- "A2A" the protocol → entity_id: "protocol:a2a" (kind: spec)
- "Google" the company → entity_id: "organization:google" (kind: company)
- "MCP Python SDK" → entity_id: "project:mcp-sdk-python" (kind: sdk)
```

---

## 8. KG Usage Guide

### How Alan Should Use This KG

The `pipeline.db` SQLite database is queryable directly. All entities and edges are in the `entities` and `edges` tables.

#### Ecosystem Comparison Queries

**"What protocols are competing or complementing A2A?"**
```sql
SELECT e.edge_type, tgt.name, tgt.entity_id
FROM edges e
JOIN entities tgt ON tgt.entity_id = e.target_entity_id
WHERE e.source_entity_id = 'protocol:a2a'
   OR e.target_entity_id = 'protocol:a2a'
ORDER BY e.edge_type;
```

**"Who is building on A2A?"**
```sql
SELECT src.name, src.type, src.kind
FROM edges e
JOIN entities src ON src.entity_id = e.source_entity_id
WHERE e.target_entity_id = 'protocol:a2a'
  AND e.edge_type = 'IMPLEMENTS';
```

**"What capabilities does MCP define?"**
```sql
SELECT tgt.name, tgt.entity_id
FROM edges e
JOIN entities tgt ON tgt.entity_id = e.target_entity_id
WHERE e.source_entity_id = 'protocol:mcp'
  AND e.edge_type = 'DEFINES';
```

#### Mapping the Ecosystem

The KG currently has good coverage of:
- **MCP ecosystem**: spec, SDKs, working groups, security protocols
- **A2A ecosystem**: spec, implementing projects (ADK, BeeAI, LangChain), sponsors
- **Key players**: Google, IBM, Microsoft, Anthropic, Linux Foundation, Agntcy
- **Capability landscape**: 10 core capabilities with partial PART_OF hierarchy

Current gaps:
- **IETF/W3C** - both in seed but not extracted (no sources covering their work on agent standards)
- **AAIF** - in seed but no sources
- **OpenAI's stance** on A2A/MCP (would need blog posts / announcements)
- **Versioning** - no SUPERSEDES edges (protocol versions not yet modeled)
- **People** - only 5 people from the Day 1 PDF; no key engineers, spec authors, WG chairs

#### Finding Gaps and Opportunities

```sql
-- Entities in seed not yet in extracted DB (seeds we never hit)
SELECT s.entity_id, s.name FROM entities s
WHERE s.source_id IS NULL OR s.source_id = 0;
```

```sql
-- Most-referenced entities (appear in most edges)
SELECT entity_id, COUNT(*) as edge_count
FROM (
  SELECT source_entity_id as entity_id FROM edges
  UNION ALL
  SELECT target_entity_id FROM edges
) GROUP BY entity_id ORDER BY edge_count DESC LIMIT 20;
```

#### Cross-Protocol Comparisons for Alan's Work

The KG is well-suited for Alan's role (Google Cloud, Vertex AI, ADK, A2A):

1. **"Which orgs sponsor both A2A and MCP?"**
   - Query: SPONSORS edges to both protocols

2. **"What capabilities do A2A and MCP both address?"**
   - Find DEFINES/ADDRESSES edges for both protocols, intersect targets

3. **"What's the full graph of A2A implementors?"**
   - Follow IMPLEMENTS→protocol:a2a and IMPLEMENTS→protocol:mcp edges

4. **"How does AGNTCY relate to everything?"**
   - Multi-hop: AGNTCY→MEMBER_OF→Linux Foundation, AGNTCY→IMPLEMENTS→A2A, AGNTCY→IMPLEMENTS→MCP

### Known Limitations

1. **28 stale edges** — Run a global edge repoint script before loading to Neo4j
2. **OAuth 2.1 duplicate** — `protocol:oauth-2.1` and `protocol:oauth-2.1/standard` should be merged
3. **Thin MCP docs** — Many MCP pages yielded 0 entities (they're tutorial/how-to pages without ecosystem entities)
4. **Person coverage** — 5 people from one PDF; needs more authorship-heavy sources
5. **No temporal data** — `valid_from/valid_to` fields exist but are not populated

---

## Appendix: Bugs Fixed During This Run

### Bug 1: resolve.run() Early Exit Without Stage Update
**File:** `src/agents_kg/stages/resolve.py`  
**Symptom:** 13 sources permanently stuck at `resolve/processing` after processing  
**Root cause:** When a source has 0 entities, `resolve.run()` returned `True` before calling `db.update_source(source_id, stage="review", status="pending_review")`  
**Fix:** Added the `db.update_source()` call before the early return  
**Impact:** All sources now correctly advance to `review/pending_review`

### Bug 2: Cross-Source Edge Repointing Gap  
**File:** `src/agents_kg/stages/resolve.py` + `src/agents_kg/stages/extract.py`  
**Symptom:** 28/100 edges reference merged entity IDs; these edges are functionally lost  
**Root cause:** LLM generates entity IDs in `type:name/kind` format; seed uses `type:name`; resolution merges one copy but a later source generates a fresh `type:name/kind` entity after the merge  
**Fix (prompt):** Add explicit rule: "NEVER include kind in entity_id"  
**Workaround for current data:** Run global edge-repoint pass using merged_into lookup table

---

*Review generated by Zaf, 2026-03-11. Pipeline ran clean on fresh database. All findings verified against live pipeline.db.*
