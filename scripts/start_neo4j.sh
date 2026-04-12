#!/bin/bash
# Script to start or ensure Neo4j is running with Podman

REPO_DIR="/usr/local/google/home/alanblount/Code/agents-kg"
DATA_DIR="$REPO_DIR/.neo4j_data"

mkdir -p "$DATA_DIR"

if podman ps -a | grep -q agents-kg-neo4j; then
    echo "Container agents-kg-neo4j already exists."
    if podman ps | grep -q agents-kg-neo4j; then
        echo "Container is already running."
    else
        echo "Starting existing container..."
        podman start agents-kg-neo4j
    fi
else
    echo "Creating and starting new Neo4j container..."
    podman run -d \
      --name agents-kg-neo4j \
      -p 7474:7474 -p 7687:7687 \
      -v "$DATA_DIR":/data \
      -e NEO4J_AUTH=neo4j/agents-kg-2026 \
      neo4j:community
fi

echo "Neo4j should be available at bolt://localhost:7687"
echo "Browser UI at http://localhost:7474"
echo "Auth: neo4j / agents-kg-2026"
