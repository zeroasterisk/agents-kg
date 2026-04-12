from src.agents_kg.db import Database

def approve_and_load():
    db = Database("pipeline.db")
    
    # Approve entities
    cur = db.conn.execute("UPDATE entities SET status = 'approved' WHERE status = 'pending_review'")
    print(f"Approved {cur.rowcount} entities")
    
    # Approve edges
    cur = db.conn.execute("UPDATE edges SET status = 'approved' WHERE status = 'pending_review'")
    print(f"Approved {cur.rowcount} edges")
    
    # Set stage to 'load' for sources in 'review' stage
    cur = db.conn.execute("UPDATE sources SET stage = 'load', status = 'pending' WHERE stage = 'review'")
    print(f"Set {cur.rowcount} sources to 'load' stage")
    
    db.conn.commit()
    db.close()

if __name__ == "__main__":
    approve_and_load()
