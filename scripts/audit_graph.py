import os
import json
from src.agents_kg.db import Database
from neo4j import GraphDatabase

def audit():
    # 1. Check SQLite
    db = Database("pipeline.db")
    print("=== SQLite Status ===")
    print(db.status_summary())
    
    entities = db.conn.execute("SELECT status, COUNT(*) FROM entities GROUP BY status").fetchall()
    print("\nEntities by status:")
    for r in entities:
        print(f"  {r[0]}: {r[1]}")
        
    edges = db.conn.execute("SELECT status, COUNT(*) FROM edges GROUP BY status").fetchall()
    print("\nEdges by status:")
    for r in edges:
        print(f"  {r[0]}: {r[1]}")
        
    db.close()
    
    # 2. Check Neo4j
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "agents-kg-2026")
    
    print("\n=== Neo4j Status ===")
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            # Count nodes
            res = session.run("MATCH (n) RETURN labels(n)[0] as label, count(*) as count")
            print("Nodes by label:")
            for r in res:
                print(f"  {r['label']}: {r['count']}")
                
            # Count relationships
            res = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
            print("\nRelationships by type:")
            for r in res:
                print(f"  {r['type']}: {r['count']}")
                
        driver.close()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")

if __name__ == "__main__":
    audit()
