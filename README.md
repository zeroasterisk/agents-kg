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

## Usage

Query with `scripts/query.py` (coming soon).
