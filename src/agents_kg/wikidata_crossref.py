"""Cross-reference existing KG entities with Wikidata Q-IDs.

Reads mappings from kg/wikidata_mappings.yaml and applies wikidata_id
properties to matching entities in Neo4j and seed.py data.
"""

import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DEFAULT_MAPPINGS_PATH = "kg/wikidata_mappings.yaml"


def load_mappings(path: str = DEFAULT_MAPPINGS_PATH) -> dict[str, str | None]:
    """Load entity_id → wikidata Q-ID mappings from YAML."""
    mappings_file = Path(path)
    if not mappings_file.exists():
        log.warning("Mappings file not found: %s", path)
        return {}

    with open(mappings_file) as f:
        data = yaml.safe_load(f)

    return data.get("mappings", {})


def apply_crossref(neo4j_driver=None, mappings_path: str = DEFAULT_MAPPINGS_PATH) -> dict:
    """Apply Wikidata cross-references to existing entities.

    Returns {"applied": N, "skipped": N} counts.
    """
    mappings = load_mappings(mappings_path)
    applied = 0
    skipped = 0

    for entity_id, wikidata_id in mappings.items():
        if not wikidata_id:
            skipped += 1
            continue

        wikidata_id_str = str(wikidata_id)
        if not wikidata_id_str.startswith("Q"):
            wikidata_id_str = f"Q{wikidata_id_str}"

        if neo4j_driver:
            with neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (n {entity_id: $entity_id}) "
                    "SET n.wikidata_id = $wikidata_id "
                    "RETURN count(n) as updated",
                    {"entity_id": entity_id, "wikidata_id": wikidata_id_str},
                )
                record = result.single()
                if record and record["updated"] > 0:
                    log.info("Set wikidata_id=%s on %s", wikidata_id_str, entity_id)

        applied += 1

    log.info("Cross-ref complete: %d applied, %d skipped", applied, skipped)
    return {"applied": applied, "skipped": skipped}
