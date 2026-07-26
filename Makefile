-include .env
export

.PHONY: start-neo4j status process purge backup restore reset ingest help

help:
	@echo "Available commands:"
	@echo "  make start-neo4j  - Starts or ensures Neo4j is running with Podman"
	@echo "  make status       - Checks the pipeline status (SQLite queue)"
	@echo "  make process      - Runs the extraction pipeline"
	@echo "  make purge        - Purges all data from Neo4j (Cypher command)"
	@echo "  make backup       - Backs up the Neo4j data directory"
	@echo "  make restore FILE=backup.tar.gz - Restores data from a backup file"
	@echo "  make ingest FILE=sources.txt    - Ingests sources from a file (defaults to sources.txt)"
	@echo "  make reset FILE=sources.txt     - Hard reset of Neo4j and SQLite, then ingests from FILE"

start-neo4j:
	./scripts/start_neo4j.sh

status:
	.venv/bin/python3 -m src.agents_kg.cli status

process:
	PREFECT_SERVER_ANALYTICS_ENABLED=false DO_NOT_TRACK=1 .venv/bin/python3 -m src.agents_kg.cli process

# ⚠️  WARNING: purge deletes ALL nodes and edges from Neo4j.
# This is a DESTRUCTIVE operation — it will wipe production data.
# Requires: CONFIRM_WIPE=yes  (e.g. make purge CONFIRM_WIPE=yes)
purge:
	@if [ "$(CONFIRM_WIPE)" != "yes" ]; then \
		echo "ERROR: This will DELETE ALL DATA in Neo4j."; \
		echo "  If you really mean it, run:  make purge CONFIRM_WIPE=yes"; \
		exit 1; \
	fi
	./scripts/neo4j_purge.sh

backup:
	./scripts/neo4j_export_backup.sh

restore:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make restore FILE=backup_file.tar.gz"; \
		exit 1; \
	fi
	./scripts/neo4j_import_backup.sh $(FILE)

ingest:
	@if [ -z "$(FILE)" ]; then \
		if [ -f sources.txt ]; then \
			echo "No FILE specified. Defaulting to sources.txt..."; \
			.venv/bin/python3 -m src.agents_kg.cli ingest --from sources.txt; \
		else \
			echo "Usage: make ingest FILE=sources_file.txt"; \
			exit 1; \
		fi \
	else \
		if [ -f "$(FILE)" ]; then \
			echo "Ingesting from $(FILE)..."; \
			.venv/bin/python3 -m src.agents_kg.cli ingest --from $(FILE); \
		else \
			echo "Error: File $(FILE) not found."; \
			exit 1; \
		fi \
	fi

# ⚠️  WARNING: reset DESTROYS both Neo4j data and SQLite pipeline.db.
# This is irreversible — all extracted entities, edges, and source records are lost.
# NEVER run this during a deploy. Only use for dev/test bootstrapping.
# Requires: CONFIRM_WIPE=yes  (e.g. make reset CONFIRM_WIPE=yes FILE=sources.txt)
reset:
	@if [ "$(CONFIRM_WIPE)" != "yes" ]; then \
		echo "ERROR: This will DESTROY all Neo4j data AND delete pipeline.db."; \
		echo "  If you really mean it, run:  make reset CONFIRM_WIPE=yes FILE=$(FILE)"; \
		exit 1; \
	fi
	@echo "Resetting everything (Hard Reset)..."
	podman stop agents-kg-neo4j || true
	podman unshare rm -rf .neo4j_data
	./scripts/start_neo4j.sh
	rm -f pipeline.db
	@make ingest FILE=$(FILE)
	@echo "Reset complete. Run 'make process' to start extraction."
