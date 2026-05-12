"""Authentication dependencies for the agents-kg REST API.

Supports three authentication strategies (tried in order):
1. Static API key from AGENT_API_KEYS env var
2. Google service account JWT
3. Google OAuth2 ID token (human users)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()


@dataclass
class User:
    user_id: str
    is_agent: bool = False


def _get_api_keys() -> set[str]:
    raw = os.environ.get("AGENT_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def _check_static_key(token: str) -> User | None:
    keys = _get_api_keys()
    if token in keys:
        prefix = token[:8] if len(token) >= 8 else token
        return User(user_id=f"agent:{prefix}", is_agent=True)
    return None


def _check_google_token(token: str) -> User | None:
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    service_account_email = os.environ.get("SERVICE_ACCOUNT_EMAIL")
    if not client_id and not service_account_email:
        return None

    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        request = google_requests.Request()
        payload = id_token.verify_oauth2_token(token, request, audience=client_id)

        email = payload.get("email", "")
        if not email:
            return None

        is_agent = False
        if service_account_email and email == service_account_email:
            is_agent = True
        elif email.endswith(".iam.gserviceaccount.com"):
            is_agent = True

        return User(user_id=email, is_agent=is_agent)

    except Exception:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    token = credentials.credentials

    user = _check_static_key(token)
    if user:
        return user

    user = _check_google_token(token)
    if user:
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
    )
