# agents-kg REST API

FastAPI REST layer on top of the agents-kg ingestion pipeline.

## Quick Start

```bash
# Install API dependencies (run from repo root)
pip install -r requirements-api.txt

# Set environment variables
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=agents-kg-2026

# For Google OAuth validation
export GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com

# For agent/static API key access
export AGENT_API_KEYS=key1,key2,key3

# Optional: service account email
export SERVICE_ACCOUNT_EMAIL=agents-kg-bot@project.iam.gserviceaccount.com

# Run the server
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

### GET /health
Unauthenticated. Check system status.
```bash
curl http://localhost:8000/health
```

### POST /ingest
Submit URLs for ingestion.
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/doc"], "source_type": "authoritative"}'
```

### GET /ingest/status/{job_id}
Check status of a specific ingestion job.
```bash
curl http://localhost:8000/ingest/status/1 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### GET /ingest/history
List recent ingestion submissions. Supports `?limit=50&status=ingested`.
```bash
curl "http://localhost:8000/ingest/history?limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### POST /query
Run a read-only Cypher query against the knowledge graph.
```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"cypher": "MATCH (n:Protocol) RETURN n.name LIMIT 10"}'
```

## Authentication

### Static API Key (for agents)
Set `AGENT_API_KEYS` env var with comma-separated keys. Send as Bearer token:
```
Authorization: Bearer your-api-key
```

### Google OAuth (for humans)
1. Get a Google OAuth token from your frontend or via the OAuth playground
2. Set `GOOGLE_CLIENT_ID` to your Google OAuth client ID
3. Send the ID token as Bearer token:
```
Authorization: Bearer eyJhbGci...
```

**Getting a test token:**
1. Go to https://developers.google.com/oauthplayground/
2. Select "Google OAuth2 API v2" > "openid" and "email"
3. Authorize with your Google account
4. Copy the `id_token` from the response

### Service Account (for trusted agents)
Set `SERVICE_ACCOUNT_EMAIL` to the service account's email. The service account
authenticates with a Google-issued ID token (standard service account JWT flow).

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEO4J_URI` | No | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | No | `agents-kg-2026` | Neo4j password |
| `KG_DB_PATH` | No | `pipeline.db` | Path to SQLite database |
| `GOOGLE_CLIENT_ID` | No* | — | Google OAuth client ID |
| `AGENT_API_KEYS` | No* | — | Comma-separated static API keys |
| `SERVICE_ACCOUNT_EMAIL` | No | — | Trusted service account email |

\* At least one auth method must be configured for authenticated endpoints.
