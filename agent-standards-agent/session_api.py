"""Anonymous session token API for the Agent Standards Agent.

Provides rate-limited session creation and per-session turn tracking.
No OAuth — session tokens are the only auth mechanism for now.

Usage:
    uvicorn session_api:app --port 8081

    # Create a session
    curl -X POST http://localhost:8081/session

    # Use the token in subsequent agent requests
    curl -X POST http://localhost:8080/run \
      -H "X-Session-Token: <token>" \
      -d '{"message": "What is MCP?"}'
"""

from __future__ import annotations

import logging
import os
import secrets
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

log = logging.getLogger("agent_standards.session_api")

SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", "86400"))  # 24h
MAX_TURNS_PER_HOUR = int(os.environ.get("MAX_TURNS_PER_HOUR", "30"))
MAX_SESSIONS_PER_HOUR = int(os.environ.get("MAX_SESSIONS_PER_HOUR", "5"))

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}


def _prune_expired() -> int:
    """Remove expired sessions. Returns count of pruned sessions."""
    now = time.time()
    expired = [k for k, v in _sessions.items() if now > v["expires_at"]]
    for k in expired:
        del _sessions[k]
    return len(expired)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Agent Standards Agent — Session API",
    description="Anonymous session token management with rate limiting.",
    version="0.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SessionResponse(BaseModel):
    token: str
    expires_in: int
    rate_limit: str


class SessionInfo(BaseModel):
    user_id: str
    created_at: float
    expires_at: float
    turn_count: int
    turns_this_hour: int
    rate_limit: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/session", response_model=SessionResponse)
@limiter.limit(f"{MAX_SESSIONS_PER_HOUR}/hour")
async def create_session(request: Request):
    """Create a new anonymous session token.

    Returns a token valid for 24 hours with a rate limit of 30 turns/hour.
    Session creation itself is limited to 5 per hour per IP address.
    """
    token = secrets.token_urlsafe(32)
    now = time.time()

    _sessions[token] = {
        "user_id": f"anon:{token[:8]}",
        "created_at": now,
        "expires_at": now + SESSION_TTL,
        "turn_count": 0,
        "last_turn_hour": int(now / 3600),
        "turns_this_hour": 0,
    }

    # Prune expired sessions on each creation
    pruned = _prune_expired()
    if pruned:
        log.info("Pruned %d expired sessions", pruned)

    return SessionResponse(
        token=token,
        expires_in=SESSION_TTL,
        rate_limit=f"{MAX_TURNS_PER_HOUR} turns/hour",
    )


@app.get("/session/{token}", response_model=SessionInfo)
async def get_session(token: str):
    """Get information about an existing session (for debugging)."""
    session = _sessions.get(token)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if time.time() > session["expires_at"]:
        del _sessions[token]
        raise HTTPException(status_code=404, detail="Session expired")

    return SessionInfo(
        user_id=session["user_id"],
        created_at=session["created_at"],
        expires_at=session["expires_at"],
        turn_count=session["turn_count"],
        turns_this_hour=session["turns_this_hour"],
        rate_limit=f"{MAX_TURNS_PER_HOUR} turns/hour",
    )


# ---------------------------------------------------------------------------
# Session validation (called by the ADK serving layer)
# ---------------------------------------------------------------------------


def validate_session(token: str) -> dict:
    """Validate a session token and increment the turn counter.

    Called before each agent turn to enforce rate limits.

    Args:
        token: The session token from X-Session-Token header.

    Returns:
        The session dict with user_id and updated turn counts.

    Raises:
        HTTPException: If the token is invalid, expired, or rate-limited.
    """
    session = _sessions.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session token")

    if time.time() > session["expires_at"]:
        del _sessions[token]
        raise HTTPException(status_code=401, detail="Session expired")

    # Rate limit: MAX_TURNS_PER_HOUR turns per hour
    current_hour = int(time.time() / 3600)
    if session["last_turn_hour"] != current_hour:
        session["last_turn_hour"] = current_hour
        session["turns_this_hour"] = 0

    if session["turns_this_hour"] >= MAX_TURNS_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail=f"Turn limit exceeded ({MAX_TURNS_PER_HOUR}/hour). "
            "Try again in the next hour.",
        )

    session["turns_this_hour"] += 1
    session["turn_count"] += 1
    return session


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    _prune_expired()
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
    }
