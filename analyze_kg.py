#!/usr/bin/env python3
"""Full KG analysis for the pipeline review report."""

import os
import json
import sqlite3
from collections import defaultdict, Counter

os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
    os.path.expanduser("~/.openclaw/credentials/zaf-admin.json"))

DB_PATH = os.path.expanduser("~/.openclaw/projects/agents-kg/pipeline.db")

# Valid ontology
VALID_TYPES = {"Organization", "Group", "Person", "Project", "Protocol", "Capability", "Source", "Chunk"}
VALID_EDGE_TYPES = {
    "MEMBER_OF", "GOVERNS", "DEVELOPS", "IMPLEMENTS", "COMPETES_WITH",
    "ADDRESSES", "PART_OF", "SUPERSEDES", "SPONSORS", "CHAIRS",
    "AUTHORED", "CONTRIBUTES_TO", "DEFINES", "COMPLEMENTS"
}

VALID_KINDS = {
    "Organization": {"company", "standards_body", "foundation", "consortium"},
    "Project": {"framework", "sdk", "library", "tool", "platform"},
    "Protocol": {"spec", "standard", "rfc", "draft"},
    "Group": {"tsc", "wg", "sig", "task_force", "team", "division"},
}

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("=" * 70)
    print("AGENTS-KG CRITICAL REVIEW ANALYSIS")
    print("=" * 70)

    # ====== SOURCES ======
    print("\n### SOURCES ###")
    sources = conn.execute("SELECT id, uri, stage, status FROM sources").fetchall()
    print(f"Total sources: {len(sources)}")
    by_status = Counter((s['stage'], s['status']) for s in sources)
    for (stage, status), cnt in sorted(by_status.items()):
        print(f"  {stage}/{status}: {cnt}")

    # ====== ENTITY COUNTS ======
    print("\n### ENTITY COUNTS ###")
    all_entities = conn.execute("SELECT * FROM entities").fetchall()
    active = [e for e in all_entities if e['merged_into'] is None and e['status'] not in ('merged', 'rejected')]
    merged = [e for e in all_entities if e['status'] == 'merged']
    rejected = [e for e in all_entities if e['status'] == 'rejected']

    print(f"Total extracted: {len(all_entities)}")
    print(f"  Active (kept): {len(active)}")
    print(f"  Merged:        {len(merged)}")
    print(f"  Rejected:      {len(rejected)}")

    # By type
    print("\n  By type:")
    type_counts = Counter(e['type'] for e in active)
    for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {cnt}")

    # By type+kind
    print("\n  By type/kind:")
    typekind = Counter((e['type'], e['kind'] or '(none)') for e in active)
    for (t, k), cnt in sorted(typekind.items(), key=lambda x: (-x[1], x[0])):
        print(f"    {t}/{k}: {cnt}")

    # ====== ENTITY QUALITY CHECKS ======
    print("\n### ENTITY QUALITY CHECKS ###")

    # Invalid types
    invalid_types = [e for e in active if e['type'] not in VALID_TYPES]
    print(f"\n  Invalid types ({len(invalid_types)}):")
    for e in invalid_types[:20]:
        print(f"    [{e['entity_id']}] {e['name']} type={e['type']}")

    # Invalid kind values
    print("\n  Invalid kind values:")
    kind_issues = []
    for e in active:
        valid_kinds = VALID_KINDS.get(e['type'])
        if valid_kinds and e['kind'] and e['kind'] not in valid_kinds:
            kind_issues.append(e)
    print(f"  Count: {len(kind_issues)}")
    for e in kind_issues[:20]:
        print(f"    [{e['entity_id']}] {e['name']} type={e['type']} kind={e['kind']}")

    # entity_id format check (should be type:name/kind or similar)
    print("\n  entity_id format issues (no ':'):")
    id_format_issues = [e for e in active if ':' not in (e['entity_id'] or '')]
    print(f"  Count: {len(id_format_issues)}")
    for e in id_format_issues[:20]:
        print(f"    [{e['entity_id']}] {e['name']} type={e['type']}")

    # Remaining duplicates (same name+type, different entity_id, both active)
    print("\n  Remaining duplicates (same name+type, different entity_id):")
    name_type_map = defaultdict(list)
    for e in active:
        key = (e['name'].lower().strip(), e['type'])
        name_type_map[key].append(e)
    dups = {k: v for k, v in name_type_map.items() if len(v) > 1}
    print(f"  Count: {len(dups)}")
    for (name, t), group in list(dups.items())[:15]:
        ids = [e['entity_id'] for e in group]
        print(f"    '{name}' ({t}): {', '.join(ids)}")

    # ====== EDGE COUNTS ======
    print("\n\n### EDGE COUNTS ###")
    all_edges = conn.execute("SELECT * FROM edges").fetchall()
    print(f"Total edges: {len(all_edges)}")

    # By type
    edge_type_counts = Counter(e['edge_type'] for e in all_edges)
    print("\n  By edge_type:")
    for t, cnt in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        valid = "✓" if t in VALID_EDGE_TYPES else "✗ INVALID"
        print(f"    {t}: {cnt} {valid}")

    # Invalid edge types
    invalid_edges = [e for e in all_edges if e['edge_type'] not in VALID_EDGE_TYPES]
    if invalid_edges:
        print(f"\n  Invalid edge types found ({len(invalid_edges)}):")
        for e in invalid_edges[:20]:
            print(f"    {e['source_entity_id']} --{e['edge_type']}--> {e['target_entity_id']}")

    # Orphan edges (source or target entity doesn't exist in active entities)
    active_ids = {e['entity_id'] for e in active}
    all_entity_ids = {e['entity_id'] for e in all_entities}
    orphan_edges = [e for e in all_edges 
                    if e['source_entity_id'] not in all_entity_ids 
                    or e['target_entity_id'] not in all_entity_ids]
    print(f"\n  Orphan edges (entity not in DB): {len(orphan_edges)}")
    for e in orphan_edges[:10]:
        print(f"    {e['source_entity_id']} --{e['edge_type']}--> {e['target_entity_id']}")

    # Edges pointing to merged/rejected entities
    merged_rejected_ids = {e['entity_id'] for e in all_entities if e['status'] in ('merged', 'rejected')}
    dangling_edges = [e for e in all_edges 
                      if e['source_entity_id'] in merged_rejected_ids
                      or e['target_entity_id'] in merged_rejected_ids]
    print(f"\n  Edges pointing to merged/rejected entities: {len(dangling_edges)}")
    for e in dangling_edges[:10]:
        print(f"    {e['source_entity_id']} --{e['edge_type']}--> {e['target_entity_id']}")

    # ====== RESOLUTION EFFECTIVENESS ======
    print("\n\n### RESOLUTION EFFECTIVENESS ###")
    print(f"  Merged entities: {len(merged)}")
    print(f"  Rejected (noise) entities: {len(rejected)}")
    print(f"  Reduction rate: {(len(merged) + len(rejected)) / max(1, len(all_entities)) * 100:.1f}%")

    print("\n  Merged entity breakdown (top 20 merge targets):")
    merge_targets = Counter(e['merged_into'] for e in merged if e['merged_into'])
    for target, cnt in merge_targets.most_common(20):
        print(f"    {target}: {cnt} merged in")

    print("\n  Rejected entities sample:")
    for e in rejected[:10]:
        print(f"    [{e['entity_id']}] {e['name']} type={e['type']} kind={e['kind']}")

    # ====== SPECIFIC ENTITY SAMPLES ======
    print("\n\n### ALL ACTIVE ENTITIES ###")
    for e in sorted(active, key=lambda x: (x['type'], x['name'])):
        aliases_raw = e['aliases']
        try:
            aliases = json.loads(aliases_raw) if aliases_raw else []
        except:
            aliases = []
        alias_str = f" [{', '.join(aliases[:3])}]" if aliases else ""
        print(f"  {e['type']}/{e['kind'] or '-'}: {e['entity_id']} — {e['name']}{alias_str}")

    # ====== SPECIFIC EDGE SAMPLES ======
    print("\n\n### ALL EDGES (sorted by type) ###")
    for e in sorted(all_edges, key=lambda x: (x['edge_type'], x['source_entity_id'])):
        print(f"  {e['source_entity_id']} --{e['edge_type']}--> {e['target_entity_id']}")

    # ====== SEED VS EXTRACTED ENTITIES ======
    print("\n\n### ENTITIES BY SOURCE COUNT ###")
    entity_sources = conn.execute("""
        SELECT entity_id, name, type, kind, COUNT(DISTINCT source_id) as src_count
        FROM entities 
        WHERE merged_into IS NULL AND status NOT IN ('merged', 'rejected')
        GROUP BY entity_id
        ORDER BY src_count DESC, name
    """).fetchall()
    print("  (Top entities by source count)")
    for e in entity_sources[:30]:
        print(f"    [{e['src_count']} sources] {e['entity_id']} — {e['name']}")

    conn.close()

if __name__ == "__main__":
    main()
