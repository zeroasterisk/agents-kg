from neo4j import GraphDatabase
import os

def purge():
    uri = "bolt://localhost:7687"
    auth = ("neo4j", "agents-kg-2026")
    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("Purged all nodes and edges from Neo4j")
    driver.close()

if __name__ == "__main__":
    purge()
