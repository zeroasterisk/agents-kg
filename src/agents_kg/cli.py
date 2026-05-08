"""CLI entry point for the agents-kg pipeline."""

import json
import logging
import os
import sys
import click
from .db import Database, DEFAULT_DB
from .pipeline import process_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def get_db_path() -> str:
    return os.environ.get("KG_DB_PATH", DEFAULT_DB)


def get_db():
    return Database(get_db_path())


def get_neo4j_config() -> tuple[str | None, tuple[str, str] | None]:
    """Return (uri, auth) from env vars or defaults. Returns (None, None) if not configured."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "agents-kg-2026")
    return uri, (user, password)


@click.group()
def cli():
    """Knowledge graph ingestion pipeline."""
    pass


@cli.command()
@click.argument("url", required=False)
@click.option("--from", "from_file", type=click.Path(exists=True), help="File with one URL per line")
@click.option("--file", "local_file", type=click.Path(exists=True), help="Local file (PDF, markdown, text)")
@click.option("--submitter-email", default=None, help="Email of the person submitting this source")
def ingest(url, from_file, local_file, submitter_email):
    """Add source(s) to the ingestion queue."""
    db = get_db()
    urls = []

    if url:
        urls.append(url)
    if local_file:
        import os
        urls.append(os.path.abspath(local_file))
    if from_file:
        with open(from_file) as f:
            urls.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))

    if not urls:
        click.echo("Error: provide a URL, --file, or --from file", err=True)
        sys.exit(1)

    added = 0
    skipped = 0
    for u in urls:
        result = db.add_source(u, submitter_email=submitter_email)
        if result:
            click.echo(f"  + {u} (id={result})")
            added += 1
        else:
            click.echo(f"  ~ {u} (already queued)")
            skipped += 1

    click.echo(f"\nAdded {added}, skipped {skipped} duplicate(s)")
    db.close()


@cli.command()
def process():
    """Process all pending sources through the pipeline."""
    db = get_db()
    neo4j_uri, neo4j_auth = get_neo4j_config()

    neo4j_driver = None
    if neo4j_uri:
        try:
            from neo4j import GraphDatabase
            neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
            neo4j_driver.verify_connectivity()
            click.echo(f"Connected to Neo4j at {neo4j_uri}")

        except Exception as e:
            click.echo(f"Neo4j not available ({e}), will export YAML only")
            neo4j_driver = None

    click.echo("Processing pipeline...")
    try:
        stats = process_all(db=db, neo4j_driver=neo4j_driver)
        click.echo(f"Done: {stats['processed']} processed, {stats['failed']} failed, {stats['skipped']} skipped")
    finally:
        db.close()
        if neo4j_driver:
            neo4j_driver.close()


@cli.command()
def status():
    """Show queue status."""
    db = get_db()
    summary = db.status_summary()
    if not summary:
        click.echo("No sources in queue.")
    else:
        click.echo("Pipeline status:")
        for status_name, count in sorted(summary.items()):
            click.echo(f"  {status_name}: {count}")
        total = sum(summary.values())
        click.echo(f"  total: {total}")

    # Also show stage breakdown for processing items
    sources = db.get_pending_sources()
    if sources:
        click.echo("\nIn progress:")
        for s in sources:
            click.echo(f"  [{s['id']}] {s['uri'][:60]}... stage={s['stage']} attempts={s['attempts']}")
    db.close()


@cli.command()
@click.option("--approve", type=int, help="Approve entity or edge by ID")
@click.option("--approve-all", is_flag=True, help="Approve all pending items")
@click.option("--type", "item_type", type=click.Choice(["entity", "edge", "all"]), default="all")
def review(approve, approve_all, item_type):
    """Show or approve items pending review."""
    db = get_db()

    if approve:
        # Try entity first, then edge
        ent = db.conn.execute("SELECT * FROM entities WHERE id = ?", (approve,)).fetchone()
        if ent:
            db.approve_entity(approve)
            click.echo(f"Approved entity {approve}: {ent['name']}")
        else:
            edge = db.conn.execute("SELECT * FROM edges WHERE id = ?", (approve,)).fetchone()
            if edge:
                db.approve_edge(approve)
                click.echo(f"Approved edge {approve}: {edge['source_entity_id']} -> {edge['target_entity_id']}")
            else:
                click.echo(f"No entity or edge with id {approve}", err=True)
        db.close()
        return

    if approve_all:
        now_str = db.conn.execute("SELECT datetime('now')").fetchone()[0]
        if item_type in ("entity", "all"):
            c = db.conn.execute("UPDATE entities SET status = 'approved', updated_at = ? WHERE status = 'pending_review'", (now_str,))
            click.echo(f"Approved {c.rowcount} entities")
        if item_type in ("edge", "all"):
            c = db.conn.execute("UPDATE edges SET status = 'approved', updated_at = ? WHERE status = 'pending_review'", (now_str,))
            click.echo(f"Approved {c.rowcount} edges")
        db.conn.commit()

        # Now check if any sources can advance past review
        review_sources = db.get_sources_by_status("pending_review")
        for s in review_sources:
            db.update_source(s["id"], status="processing", stage="load")
        if review_sources:
            click.echo(f"Advanced {len(review_sources)} sources to load stage")
        db.close()
        return

    # Show pending items
    if item_type in ("entity", "all"):
        entities = db.get_entities_by_status("pending_review")
        if entities:
            click.echo(f"\nPending entities ({len(entities)}):")
            for e in entities:
                aliases = json.loads(e["aliases"]) if isinstance(e["aliases"], str) else e["aliases"]
                alias_str = f' (aka: {", ".join(aliases)})' if aliases else ""
                click.echo(f"  [{e['id']}] {e['type']}/{e['kind'] or '?'}: {e['name']}{alias_str}")
                if e["description"]:
                    click.echo(f"       {e['description'][:80]}")

    if item_type in ("edge", "all"):
        edges = db.get_edges_by_status("pending_review")
        if edges:
            click.echo(f"\nPending edges ({len(edges)}):")
            for e in edges:
                click.echo(f"  [{e['id']}] {e['source_entity_id']} --{e['edge_type']}--> {e['target_entity_id']} (conf={e['confidence']})")

    if item_type in ("entity", "all") and not db.get_entities_by_status("pending_review"):
        if item_type in ("edge", "all") and not db.get_edges_by_status("pending_review"):
            click.echo("No items pending review.")

    db.close()


@cli.command()
def retry():
    """Retry all failed sources."""
    db = get_db()
    count = db.retry_failed()
    click.echo(f"Retried {count} failed source(s)")
    db.close()


@cli.command()
@click.argument("source_id", type=int)
def reset(source_id):
    """Reset a source to re-process from scratch."""
    db = get_db()
    source = db.get_source(source_id)
    if not source:
        click.echo(f"Source {source_id} not found", err=True)
        sys.exit(1)
    db.reset_source(source_id)
    click.echo(f"Reset source {source_id}: {source['uri']}")
    db.close()


@cli.command("schema")
def schema_cmd():
    """Apply Neo4j schema constraints and indexes."""
    from .schema import apply_schema
    neo4j_uri, neo4j_auth = get_neo4j_config()

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        driver.verify_connectivity()
    except Exception as e:
        click.echo(f"Cannot connect to Neo4j at {neo4j_uri}: {e}", err=True)
        sys.exit(1)

    click.echo(f"Applying schema to {neo4j_uri}...")
    try:
        results = apply_schema(driver)
        click.echo(f"Applied {results['constraints']} constraints, {results['indexes']} indexes")
        if results["errors"]:
            for err in results["errors"]:
                click.echo(f"  ERROR: {err}", err=True)
    finally:
        driver.close()


@cli.group()
def wikidata():
    """Wikidata SPARQL integration commands."""
    pass


@wikidata.command("pull")
@click.option("--type", "entity_type", type=click.Choice(["languages", "protocols", "orgs", "software"]),
              help="Pull specific entity type (default: all)")
@click.option("--dry-run", is_flag=True, help="Pull from SPARQL but don't load to Neo4j")
def wikidata_pull(entity_type, dry_run):
    """Pull entities from Wikidata SPARQL endpoint."""
    from .wikidata import pull_and_load

    neo4j_driver = None
    if not dry_run:
        neo4j_uri, neo4j_auth = get_neo4j_config()
        try:
            from neo4j import GraphDatabase
            neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
            neo4j_driver.verify_connectivity()
            click.echo(f"Connected to Neo4j at {neo4j_uri}")
        except Exception as e:
            click.echo(f"Neo4j not available ({e}), pulling without loading")

    try:
        scope = entity_type or "all types"
        click.echo(f"Pulling {scope} from Wikidata...")
        results = pull_and_load(neo4j_driver, entity_type)
        click.echo(f"Done: {results['entities']} entities, {results['edges']} edges")
    finally:
        if neo4j_driver:
            neo4j_driver.close()


@wikidata.command("crossref")
def wikidata_crossref():
    """Cross-reference existing entities with Wikidata IDs."""
    from .wikidata_crossref import apply_crossref

    neo4j_uri, neo4j_auth = get_neo4j_config()
    neo4j_driver = None
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        driver.verify_connectivity()
        neo4j_driver = driver
    except Exception as e:
        click.echo(f"Neo4j not available ({e}), applying mappings without graph update")

    try:
        results = apply_crossref(neo4j_driver)
        click.echo(f"Cross-referenced {results['applied']} entities ({results['skipped']} skipped, no Wikidata match)")
    finally:
        if neo4j_driver:
            neo4j_driver.close()


@cli.group()
def events():
    """Event timeline management commands."""
    pass


@events.command("load")
@click.option("--dir", "events_dir", default="kg/events", help="Directory with event YAML files")
def events_load(events_dir):
    """Load event YAML files into Neo4j."""
    from .temporal import load_events_from_yaml

    neo4j_uri, neo4j_auth = get_neo4j_config()
    neo4j_driver = None
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        driver.verify_connectivity()
        neo4j_driver = driver
    except Exception as e:
        click.echo(f"Neo4j not available ({e}), validating YAML only")

    try:
        results = load_events_from_yaml(neo4j_driver, events_dir)
        click.echo(f"Loaded {results['events']} events with {results['participations']} participations")
    finally:
        if neo4j_driver:
            neo4j_driver.close()


@events.command("migrate")
@click.option("--timeline", default="kg/timeline.yaml", help="Path to timeline.yaml")
@click.option("--dir", "events_dir", default="kg/events", help="Output directory")
def events_migrate(timeline, events_dir):
    """Migrate timeline.yaml entries to individual event YAML files."""
    from .temporal import migrate_timeline_yaml

    created = migrate_timeline_yaml(timeline, events_dir)
    click.echo(f"Created {created} event file(s)")


if __name__ == "__main__":
    cli()
