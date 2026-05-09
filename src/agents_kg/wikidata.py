"""Wikidata SPARQL ingestion module.

Pulls structured entity data from Wikidata's public SPARQL endpoint and
loads it into the knowledge graph. This is a direct structured-data-to-graph
path — no LLM, chunking, or embedding needed.
"""

import logging
import os
import re
import time

import httpx

log = logging.getLogger(__name__)

SPARQL_ENDPOINT = os.environ.get(
    "WIKIDATA_SPARQL_ENDPOINT", "https://query.wikidata.org/sparql"
)
USER_AGENT = os.environ.get(
    "WIKIDATA_USER_AGENT",
    "agents-kg/0.1 (https://github.com/agents-kg; research project) python-httpx",
)
RATE_LIMIT = float(os.environ.get("WIKIDATA_RATE_LIMIT", "2.0"))

_last_request_time = 0.0


def sparql_query(query: str, retries: int = 3) -> list[dict]:
    """Execute SPARQL against query.wikidata.org, return parsed bindings."""
    global _last_request_time

    for attempt in range(retries):
        elapsed = time.time() - _last_request_time
        if elapsed < RATE_LIMIT:
            time.sleep(RATE_LIMIT - elapsed)

        _last_request_time = time.time()

        try:
            resp = httpx.post(
                SPARQL_ENDPOINT,
                data={"query": query},
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": USER_AGENT,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=90,
            )

            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", 30))
                wait = min(wait, 60)
                log.warning("Rate limited, waiting %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                log.warning("Server error %d (attempt %d/%d)", resp.status_code, attempt + 1, retries)
                if attempt < retries - 1:
                    time.sleep(10)
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            data = resp.json()
            return data["results"]["bindings"]

        except httpx.TimeoutException:
            log.warning("Query timeout (attempt %d/%d)", attempt + 1, retries)
            if attempt < retries - 1:
                time.sleep(5)
                continue
            raise

    return []


def _qid(uri: str) -> str:
    """Extract Q-ID from Wikidata URI."""
    return uri.rsplit("/", 1)[-1] if "/" in uri else uri


def _val(binding: dict, key: str) -> str | None:
    """Safely get value from a SPARQL binding."""
    return binding[key]["value"] if key in binding else None


def _to_entity_id(name: str, entity_type: str) -> str:
    """Generate entity_id in the project's type:kebab-case-name format."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = slug[:80]
    type_prefix = entity_type.lower()
    return f"{type_prefix}:{slug}"


# --- SPARQL Queries ---

QUERY_PROGRAMMING_LANGUAGES = """
SELECT ?item ?itemLabel ?itemDescription ?inception ?website
       ?designerLabel ?designer ?developerLabel ?developer
WHERE {
  ?item wdt:P31 wd:Q9143 .
  OPTIONAL { ?item wdt:P571 ?inception . }
  OPTIONAL { ?item wdt:P856 ?website . }
  OPTIONAL { ?item wdt:P287 ?designer .
             ?designer rdfs:label ?designerLabel .
             FILTER(LANG(?designerLabel) = "en") }
  OPTIONAL { ?item wdt:P178 ?developer .
             ?developer rdfs:label ?developerLabel .
             FILTER(LANG(?developerLabel) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

QUERY_PROTOCOLS = """
SELECT ?item ?itemLabel ?itemDescription ?inception ?website
       ?developerLabel ?developer
WHERE {
  ?item wdt:P31/wdt:P279* wd:Q15836568 .
  OPTIONAL { ?item wdt:P571 ?inception . }
  OPTIONAL { ?item wdt:P856 ?website . }
  OPTIONAL { ?item wdt:P178 ?developer .
             ?developer rdfs:label ?developerLabel .
             FILTER(LANG(?developerLabel) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

QUERY_STANDARDS = """
SELECT ?item ?itemLabel ?itemDescription ?inception
       ?maintainerLabel ?maintainer
WHERE {
  ?item wdt:P31 wd:Q317623 .
  OPTIONAL { ?item wdt:P571 ?inception . }
  OPTIONAL { ?item wdt:P126 ?maintainer .
             ?maintainer rdfs:label ?maintainerLabel .
             FILTER(LANG(?maintainerLabel) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

QUERY_SOFTWARE_COMPANIES = """
SELECT ?item ?itemLabel ?itemDescription ?inception ?website ?countryLabel
       ?founded_byLabel ?founded_by
WHERE {
  ?item wdt:P31 wd:Q1058914 .
  OPTIONAL { ?item wdt:P571 ?inception . }
  OPTIONAL { ?item wdt:P856 ?website . }
  OPTIONAL { ?item wdt:P17 ?country .
             ?country rdfs:label ?countryLabel .
             FILTER(LANG(?countryLabel) = "en") }
  OPTIONAL { ?item wdt:P112 ?founded_by .
             ?founded_by rdfs:label ?founded_byLabel .
             FILTER(LANG(?founded_byLabel) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

QUERY_AI_ORGS = """
SELECT ?item ?itemLabel ?itemDescription ?inception ?website
WHERE {
  VALUES ?orgType { wd:Q43229 wd:Q4830453 wd:Q1058914 wd:Q163740 }
  ?item wdt:P31 ?orgType .
  ?item wdt:P101 wd:Q11660 .
  OPTIONAL { ?item wdt:P571 ?inception . }
  OPTIONAL { ?item wdt:P856 ?website . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 500
"""

QUERY_FREE_SOFTWARE = """
SELECT ?item ?itemLabel ?itemDescription ?inception ?website
       ?developerLabel ?developer ?prog_langLabel ?prog_lang ?licenseLabel ?license
WHERE {
  ?item wdt:P31 wd:Q341 .
  ?item wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks > 3)
  OPTIONAL { ?item wdt:P571 ?inception . }
  OPTIONAL { ?item wdt:P856 ?website . }
  OPTIONAL { ?item wdt:P178 ?developer .
             ?developer rdfs:label ?developerLabel .
             FILTER(LANG(?developerLabel) = "en") }
  OPTIONAL { ?item wdt:P277 ?prog_lang .
             ?prog_lang rdfs:label ?prog_langLabel .
             FILTER(LANG(?prog_langLabel) = "en") }
  OPTIONAL { ?item wdt:P275 ?license .
             ?license rdfs:label ?licenseLabel .
             FILTER(LANG(?licenseLabel) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY DESC(?sitelinks)
LIMIT 3000
"""


def _is_real_label(label: str, qid: str) -> bool:
    """Check if a label is a real name vs just echoing the Q-ID."""
    return label != qid and not label.startswith("Q")


def transform_to_entities(
    bindings: list[dict], entity_type: str, kind: str
) -> list[dict]:
    """Transform SPARQL bindings to KG entity dicts, deduplicating by Q-ID."""
    seen = {}

    for row in bindings:
        qid = _qid(_val(row, "item") or "")
        if not qid:
            continue

        label = _val(row, "itemLabel") or ""
        if not _is_real_label(label, qid):
            continue

        if qid in seen:
            continue

        entity_id = _to_entity_id(label, entity_type)
        desc = _val(row, "itemDescription") or ""
        inception = _val(row, "inception") or None
        if inception and len(inception) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}", inception):
            inception = inception[:10]
        else:
            inception = None

        entity = {
            "entity_id": entity_id,
            "name": label,
            "type": entity_type,
            "kind": kind,
            "description": desc[:500] if desc else None,
            "wikidata_id": qid,
            "url": _val(row, "website"),
            "created_at": inception,
            "source_type": "wikidata",
        }
        seen[qid] = entity

    return list(seen.values())


def extract_edges(
    bindings: list[dict], entity_type: str, edge_configs: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Extract relationship edges and implicit entities from SPARQL bindings.

    edge_configs is a list of dicts like:
        {"label_key": "developerLabel", "qid_key": "developer",
         "edge_type": "DEVELOPS", "target_type": "Organization",
         "reverse": True}

    Returns (edges, implicit_entities) — implicit entities are people/orgs
    referenced by edges that may not have been loaded as primary entities.
    """
    edges = []
    seen = set()
    implicit = {}

    for row in bindings:
        source_qid = _qid(_val(row, "item") or "")
        source_label = _val(row, "itemLabel") or ""
        if not source_qid or not _is_real_label(source_label, source_qid):
            continue

        source_eid = _to_entity_id(source_label, entity_type)

        for cfg in edge_configs:
            target_label = _val(row, cfg["label_key"])
            target_qid_val = _val(row, cfg["qid_key"])
            if not target_label or not target_qid_val:
                continue

            target_qid = _qid(target_qid_val)
            if not _is_real_label(target_label, target_qid):
                continue

            target_eid = _to_entity_id(target_label, cfg["target_type"])

            if target_qid not in implicit:
                implicit[target_qid] = {
                    "entity_id": target_eid,
                    "name": target_label,
                    "type": cfg["target_type"],
                    "kind": cfg["target_type"].lower(),
                    "wikidata_id": target_qid,
                    "source_type": "wikidata",
                }

            if cfg.get("reverse"):
                src, tgt = target_eid, source_eid
            else:
                src, tgt = source_eid, target_eid

            edge_key = (src, cfg["edge_type"], tgt)
            if edge_key in seen:
                continue
            seen.add(edge_key)

            inception = _val(row, "inception")
            valid_from = inception[:10] if inception and len(inception) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}", inception) else None

            edges.append({
                "source_entity_id": src,
                "target_entity_id": tgt,
                "edge_type": cfg["edge_type"],
                "valid_from": valid_from,
                "source_type": "wikidata",
                "confidence": 0.9,
            })

    return edges, list(implicit.values())


def _merge_implicit(entities: list[dict], implicit: list[dict]):
    """Merge implicit entities into the entity list, skipping duplicates."""
    seen = {e["entity_id"] for e in entities}
    for ent in implicit:
        if ent["entity_id"] not in seen:
            entities.append(ent)
            seen.add(ent["entity_id"])


# --- Pull functions for each entity type ---

def pull_programming_languages() -> tuple[list[dict], list[dict]]:
    """Pull programming languages from Wikidata."""
    log.info("Pulling programming languages...")
    bindings = sparql_query(QUERY_PROGRAMMING_LANGUAGES)
    entities = transform_to_entities(bindings, "Project", "programming_language")
    edges, implicit = extract_edges(bindings, "Project", [
        {"label_key": "developerLabel", "qid_key": "developer",
         "edge_type": "DEVELOPS", "target_type": "Organization", "reverse": True},
        {"label_key": "designerLabel", "qid_key": "designer",
         "edge_type": "DEVELOPS", "target_type": "Person", "reverse": True},
    ])
    _merge_implicit(entities, implicit)
    log.info("Got %d language entities, %d edges", len(entities), len(edges))
    return entities, edges


def pull_protocols() -> tuple[list[dict], list[dict]]:
    """Pull communication protocols and technical standards."""
    log.info("Pulling protocols...")
    bindings = sparql_query(QUERY_PROTOCOLS)
    entities = transform_to_entities(bindings, "Protocol", "standard")

    log.info("Pulling technical standards...")
    std_bindings = sparql_query(QUERY_STANDARDS)
    std_entities = transform_to_entities(std_bindings, "Protocol", "standard")

    seen_qids = {e["wikidata_id"] for e in entities}
    for e in std_entities:
        if e["wikidata_id"] not in seen_qids:
            entities.append(e)
            seen_qids.add(e["wikidata_id"])

    edges, implicit = extract_edges(bindings, "Protocol", [
        {"label_key": "developerLabel", "qid_key": "developer",
         "edge_type": "DEVELOPS", "target_type": "Organization", "reverse": True},
    ])
    std_edges, std_implicit = extract_edges(std_bindings, "Protocol", [
        {"label_key": "maintainerLabel", "qid_key": "maintainer",
         "edge_type": "DEVELOPS", "target_type": "Organization", "reverse": True},
    ])
    edges.extend(std_edges)
    _merge_implicit(entities, implicit + std_implicit)

    log.info("Got %d protocol/standard entities, %d edges", len(entities), len(edges))
    return entities, edges


def pull_organizations() -> tuple[list[dict], list[dict]]:
    """Pull software companies and AI organizations."""
    log.info("Pulling software companies...")
    bindings = sparql_query(QUERY_SOFTWARE_COMPANIES)
    entities = transform_to_entities(bindings, "Organization", "company")

    log.info("Pulling AI organizations...")
    ai_bindings = sparql_query(QUERY_AI_ORGS)
    ai_entities = transform_to_entities(ai_bindings, "Organization", "research_org")

    seen_qids = {e["wikidata_id"] for e in entities}
    for e in ai_entities:
        if e["wikidata_id"] not in seen_qids:
            entities.append(e)
            seen_qids.add(e["wikidata_id"])

    edges, implicit = extract_edges(bindings, "Organization", [
        {"label_key": "founded_byLabel", "qid_key": "founded_by",
         "edge_type": "FOUNDED_BY", "target_type": "Person"},
    ])
    _merge_implicit(entities, implicit)

    log.info("Got %d org entities, %d edges", len(entities), len(edges))
    return entities, edges


def pull_software_projects() -> tuple[list[dict], list[dict]]:
    """Pull notable free/open-source software."""
    log.info("Pulling free software projects...")
    bindings = sparql_query(QUERY_FREE_SOFTWARE)
    entities = transform_to_entities(bindings, "Project", "software")

    edges, implicit = extract_edges(bindings, "Project", [
        {"label_key": "developerLabel", "qid_key": "developer",
         "edge_type": "DEVELOPS", "target_type": "Organization", "reverse": True},
    ])
    _merge_implicit(entities, implicit)

    log.info("Got %d software entities, %d edges", len(entities), len(edges))
    return entities, edges


ALL_PULL_TYPES = {
    "languages": pull_programming_languages,
    "protocols": pull_protocols,
    "orgs": pull_organizations,
    "software": pull_software_projects,
}


def load_wikidata_entities(neo4j_driver, entities: list[dict]):
    """Batch-load Wikidata entities to Neo4j using UNWIND."""
    if not entities:
        return

    type_groups: dict[str, list[dict]] = {}
    for ent in entities:
        type_groups.setdefault(ent["type"], []).append(ent)

    with neo4j_driver.session() as session:
        for entity_type, group in type_groups.items():
            label = entity_type if entity_type in {
                "Protocol", "Organization", "Project", "Capability", "Group", "Person"
            } else "Entity"

            batch_size = 500
            for i in range(0, len(group), batch_size):
                batch = group[i : i + batch_size]
                session.run(
                    f"""
                    UNWIND $entities AS ent
                    MERGE (n:Entity {{entity_id: ent.entity_id}})
                    SET n:{label}, n.name = ent.name, n.description = ent.description,
                        n.wikidata_id = ent.wikidata_id, n.kind = ent.kind,
                        n.url = ent.url,
                        n.created_at = CASE WHEN ent.created_at IS NOT NULL THEN date(ent.created_at) ELSE null END,
                        n.type = ent.type, n.source_type = 'wikidata'
                    """,
                    {"entities": batch},
                )
                log.info("Loaded batch of %d %s entities", len(batch), entity_type)


def load_wikidata_edges(neo4j_driver, edges: list[dict]):
    """Batch-load relationship edges. Groups by edge_type for UNWIND."""
    if not edges:
        return

    type_groups: dict[str, list[dict]] = {}
    for edge in edges:
        type_groups.setdefault(edge["edge_type"], []).append(edge)

    with neo4j_driver.session() as session:
        for edge_type, group in type_groups.items():
            batch_size = 500
            for i in range(0, len(group), batch_size):
                batch = group[i : i + batch_size]
                session.run(
                    f"""
                    UNWIND $edges AS e
                    MATCH (a:Entity {{entity_id: e.source_entity_id}})
                    MATCH (b:Entity {{entity_id: e.target_entity_id}})
                    MERGE (a)-[r:{edge_type}]->(b)
                    SET r.source_type = e.source_type, r.confidence = e.confidence,
                        r.valid_from = CASE WHEN e.valid_from IS NOT NULL THEN date(e.valid_from) ELSE null END
                    """,
                    {"edges": batch},
                )

        total_attempted = len(edges)
        result = session.run(
            "MATCH ()-[r]->() WHERE r.source_type = 'wikidata' RETURN count(r) AS c"
        )
        total_created = result.single()["c"]
        log.info(
            "Edge loading summary: %d attempted, %d in graph from wikidata",
            total_attempted, total_created,
        )


def pull_and_load(neo4j_driver, entity_type: str | None = None) -> dict:
    """Pull from Wikidata and load to Neo4j. Returns counts."""
    if entity_type:
        pull_fn = ALL_PULL_TYPES.get(entity_type)
        if not pull_fn:
            raise ValueError(f"Unknown type: {entity_type}. Valid: {list(ALL_PULL_TYPES)}")
        pull_fns = {entity_type: pull_fn}
    else:
        pull_fns = ALL_PULL_TYPES

    total_entities = 0
    total_edges = 0

    for name, fn in pull_fns.items():
        entities, edges = fn()
        total_entities += len(entities)
        total_edges += len(edges)

        if neo4j_driver:
            load_wikidata_entities(neo4j_driver, entities)
            load_wikidata_edges(neo4j_driver, edges)
            log.info("Loaded %s: %d entities, %d edges", name, len(entities), len(edges))
        else:
            log.info("Pulled %s: %d entities, %d edges (no Neo4j)", name, len(entities), len(edges))

    return {"entities": total_entities, "edges": total_edges}
