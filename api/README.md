# agents-kg REST API

FastAPI REST layer on top of the agents-kg ingestion pipeline.

## Quick Start

```bash
# Install API dependencies (run from repo root)
pip install -r requirements-api.txt

# Set environment variables (see .env.example for full list)
export NEO4J_URI=bolt://35.202.188.73:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=<your-password>

# For browser-based OAuth token validation
export GOOGLE_CLIENT_ID=160698144102-v9a0shre6ntap7jitla83b82i5akl1j0.apps.googleusercontent.com

# For device flow (headless agent auth)
export GOOGLE_DEVICE_CLIENT_ID=160698144102-n08j36n5orda34qe0hekmunp9pl8hrjm.apps.googleusercontent.com
export GOOGLE_DEVICE_CLIENT_SECRET=<your-secret>

# For agent/static API key access
export AGENT_API_KEYS=key1,key2,key3

# Run the server
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## Authentication

### Device Flow (for humans via headless agents)

Agents and coordinators don't have browser access. Use the Device Authorization
Grant (RFC 8628) to let a human approve access from their own browser:

**Step 1: Start the flow**
```bash
curl http://localhost:8000/auth/device
```
Response:
```json
{
  "device_code": "AH-1Ng...",
  "user_code": "WDJB-MJHT",
  "verification_url": "https://www.google.com/device",
  "expires_in": 1800,
  "interval": 5
}
```

**Step 2: Human visits the URL**

The agent relays the `verification_url` and `user_code` to the human (e.g. via
chat message). The human visits `https://www.google.com/device`, enters the
`user_code`, and approves the request.

**Step 3: Poll for the token**
```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"device_code": "AH-1Ng..."}'
```
While waiting: `{"status": "pending"}`
After approval:
```json
{
  "access_token": "ya29...",
  "id_token": "eyJhbG...",
  "token_type": "Bearer",
  "expires_in": 3599
}
```
If the device code expired: `{"status": "expired"}`

**Step 4: Use the token**
```bash
curl http://localhost:8000/ingest/history \
  -H "Authorization: Bearer <id_token>"
```

### Static API Key (for agents)

Set `AGENT_API_KEYS` env var with comma-separated keys. Send as Bearer token:
```bash
curl http://localhost:8000/ingest/history \
  -H "Authorization: Bearer your-api-key"
```

### Google OAuth (browser-based)

For users who already have a Google ID token (e.g. from a web frontend):
1. Set `GOOGLE_CLIENT_ID` to the web app OAuth client ID
2. Send the ID token as Bearer token:
```bash
curl http://localhost:8000/ingest/history \
  -H "Authorization: Bearer eyJhbGci..."
```

### Service Account (for trusted agents)

Set `SERVICE_ACCOUNT_EMAIL` to the service account's email. The service account
authenticates with a Google-issued ID token (standard service account JWT flow).

## Endpoints

### GET /health
Unauthenticated. Check system status.
```bash
curl http://localhost:8000/health
```

### GET /auth/device
Unauthenticated. Start a device authorization flow.
```bash
curl http://localhost:8000/auth/device
```

### POST /auth/token
Unauthenticated. Poll for token after device code approval.
```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"device_code": "AH-1Ng..."}'
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

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEO4J_URI` | No | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | No | `agents-kg-2026` | Neo4j password |
| `KG_DB_PATH` | No | `pipeline.db` | Path to SQLite database |
| `GOOGLE_CLIENT_ID` | No* | — | Web app OAuth client ID (token validation) |
| `GOOGLE_DEVICE_CLIENT_ID` | No* | — | Device flow OAuth client ID |
| `GOOGLE_DEVICE_CLIENT_SECRET` | No* | — | Device flow OAuth client secret |
| `AGENT_API_KEYS` | No* | — | Comma-separated static API keys |
| `SERVICE_ACCOUNT_EMAIL` | No | — | Trusted service account email |
| `PIPELINE_SA_KEY_PATH` | No | — | Path to pipeline service account key JSON |

\* At least one auth method must be configured for authenticated endpoints.
