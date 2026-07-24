#!/bin/bash
# Start or ensure Neo4j is running with Docker
set -euo pipefail

CONTAINER_NAME="agents-kg-neo4j"
DATA_DIR="$(pwd)/.neo4j_data"
NEO4J_AUTH="neo4j/agents-kg-2026"

mkdir -p "$DATA_DIR"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} already exists."
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is already running."
    else
        echo "Starting existing container..."
        docker start "$CONTAINER_NAME"
    fi
else
    echo "Creating and starting new Neo4j container..."
    docker run -d \
      --name "$CONTAINER_NAME" \
      -p 7474:7474 -p 7687:7687 \
      -v "$DATA_DIR":/data \
      -e NEO4J_AUTH="$NEO4J_AUTH" \
      neo4j:community
fi

echo ""
echo "Waiting for Neo4j to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:7474 > /dev/null 2>&1; then
        echo "Neo4j is ready!"
        echo ""
        echo "  Bolt:    bolt://localhost:7687"
        echo "  Browser: http://localhost:7474"
        echo "  Auth:    neo4j / agents-kg-2026"
        exit 0
    fi
    sleep 2
done

echo "WARNING: Neo4j did not become ready within 60 seconds."
echo "Check: docker logs ${CONTAINER_NAME}"
exit 1
