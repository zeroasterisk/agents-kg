from src.agents_kg.db import Database
from src.agents_kg.stages import load
from neo4j import GraphDatabase
import os

def main():
    db = Database("pipeline.db")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "agents-kg-2026")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"Connected to Neo4j at {uri}")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return
        
    sources = db.conn.execute("SELECT * FROM sources WHERE stage = 'load'").fetchall()
    print(f"Found {len(sources)} sources to load")
    
    for source in sources:
        source = dict(source)
        print(f"Loading source {source['id']} ({source['uri'][:50]})...")
        load.run(db, source, neo4j_driver=driver)
        
    driver.close()
    db.close()

if __name__ == "__main__":
    main()
