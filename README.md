# Agentic Web Knowledge Graph

A structured knowledge graph tracking the agentic web ecosystem — protocols, standards, organizations, people, projects, and their relationships.

## Structure

```
kg/
  entities/           # One YAML per entity
    organizations/    # Companies, foundations, working groups
    people/           # Named individuals
    projects/         # Software projects, repos, specs
    protocols/        # Standards and protocols
    concepts/         # Key ideas, patterns, architectures
  relations.yaml      # Explicit relationships between entities
  timeline.yaml       # Dated events (posts, launches, donations, etc.)
  sources/            # Raw source material, posts, quotes
```

## Entity Schema

Each entity is a YAML file with:
```yaml
id: unique-slug
type: organization | person | project | protocol | concept
name: Human Name
aliases: [alt names]
description: Short description
url: primary URL
metadata: {}        # type-specific fields
tags: [relevant, tags]
created: YYYY-MM-DD
updated: YYYY-MM-DD
```

## Why YAML?

- Human-readable and editable
- Git-diffable
- Easy to query with scripts
- Can export to Neo4j/GraphQL/JSON-LD later

## Configuration

The project uses environment variables for configuration. Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

### Environment Variables:
*   `NEO4J_URI`: The URI for the Neo4j database (default: `bolt://localhost:7687`).
*   `NEO4J_USER`: Neo4j username (default: `neo4j`).
*   `NEO4J_PASSWORD`: Neo4j password (default: `agents-kg-2026`).
*   `GOOGLE_CLOUD_PROJECT`: Set this if you are using Vertex AI on Google Cloud.
*   `GEMINI_API_KEY`: Set this if you are using the Gemini API directly.

## Usage

The project uses a `Makefile` to manage the pipeline and services.

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Then use `make`:
```bash
# Show available commands
make help

# Start Neo4j
make start-neo4j

# Ingest default sources (sources.txt)
make ingest

# Process sources and load into graph
make process

# Reset everything (Neo4j and SQLite)
make reset
```


