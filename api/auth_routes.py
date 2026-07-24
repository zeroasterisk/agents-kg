"""Device Authorization Grant (RFC 8628) endpoints for agents-kg API.

Allows headless agents to initiate OAuth and have a human complete
the browser-based approval step.
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_DEVICE_CODE_URL = "https://oauth2.googleapis.com/device/code"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEVICE_FLOW_SCOPE = "openid email profile"


class DeviceCodeResponse(BaseModel):
    device_code: str
    user_code: str
    verification_url: str
    expires_in: int
    interval: int


class TokenRequest(BaseModel):
    device_code: str


class TokenResponse(BaseModel):
    access_token: str | None = None
    id_token: str | None = None
    token_type: str | None = None
    expires_in: int | None = None
    status: str | None = None


def _get_device_client_id() -> str:
    client_id = os.environ.get("GOOGLE_DEVICE_CLIENT_ID", "")
    if not client_id:
        raise HTTPException(
            status_code=503, detail="GOOGLE_DEVICE_CLIENT_ID not configured"
        )
    return client_id


def _get_device_client_secret() -> str:
    secret = os.environ.get("GOOGLE_DEVICE_CLIENT_SECRET", "")
    if not secret:
        raise HTTPException(
            status_code=503, detail="GOOGLE_DEVICE_CLIENT_SECRET not configured"
        )
    return secret


@router.get("/device", response_model=DeviceCodeResponse)
async def device_code():
    """Start device authorization flow. Returns a user_code and verification_url
    for the human to visit in their browser."""
    client_id = _get_device_client_id()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GOOGLE_DEVICE_CODE_URL,
            data={"client_id": client_id, "scope": DEVICE_FLOW_SCOPE},
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Google device code request failed: {resp.text}",
        )

    data = resp.json()
    return DeviceCodeResponse(
        device_code=data["device_code"],
        user_code=data["user_code"],
        verification_url=data["verification_url"],
        expires_in=data["expires_in"],
        interval=data["interval"],
    )


@router.post("/token", response_model=TokenResponse)
async def poll_token(body: TokenRequest):
    """Poll for token after user has approved the device code."""
    client_id = _get_device_client_id()
    client_secret = _get_device_client_secret()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "device_code": body.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )

    data = resp.json()

    if resp.status_code == 200:
        return TokenResponse(
            access_token=data.get("access_token"),
            id_token=data.get("id_token"),
            token_type=data.get("token_type"),
            expires_in=data.get("expires_in"),
        )

    error = data.get("error", "")
    if error == "authorization_pending":
        return TokenResponse(status="pending")
    if error in ("slow_down", "authorization_pending"):
        return TokenResponse(status="pending")
    if error == "expired_token":
        return TokenResponse(status="expired")

    raise HTTPException(
        status_code=502,
        detail=f"Google token exchange failed: {data.get('error_description', error)}",
    )
