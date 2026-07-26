"""Purge all nodes and edges from Neo4j.

⚠️  DESTRUCTIVE: This deletes ALL production data from the graph.
If the graph has more than SAFETY_THRESHOLD entities, this script
will refuse to run unless CONFIRM_WIPE=yes is set in the environment.
"""

import os
import sys
from neo4j import GraphDatabase

SAFETY_THRESHOLD = 100  # refuse to purge if entity count exceeds this


def purge():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "agents-kg-2026"),
    )
    driver = GraphDatabase.driver(uri, auth=auth)

    with driver.session() as session:
        count = session.run("MATCH (n:Entity) RETURN count(n)").single()[0]

        if count > SAFETY_THRESHOLD:
            confirm = os.environ.get("CONFIRM_WIPE", "").lower()
            if confirm != "yes":
                print(
                    f"ABORT: Neo4j has {count:,} entities (threshold={SAFETY_THRESHOLD}).\n"
                    f"This looks like production data. Set CONFIRM_WIPE=yes to proceed.\n"
                    f"  CONFIRM_WIPE=yes python3 scripts/purge_graph.py"
                )
                driver.close()
                sys.exit(1)

            print(f"WARNING: Purging {count:,} entities (CONFIRM_WIPE=yes set).")

        session.run("MATCH (n) DETACH DELETE n")
        print("Purged all nodes and edges from Neo4j")

    driver.close()


if __name__ == "__main__":
    purge()
