"""Model Armor callbacks for the Agent Standards Agent.

Observe-only mode: scans input and output via the Model Armor REST API
but only logs findings — never blocks or modifies content.

Model Armor template must be created in the GCP console:
  Project: data-ingest-demo
  Location: us-central1
  Input template: aggressive (blocks injection, jailbreaks, hate)
  Output template: moderate (blocks dangerous content, PII exfil)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

log = logging.getLogger("agent_standards.model_armor")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "data-ingest-demo")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_ARMOR_TEMPLATE = os.environ.get(
    "MODEL_ARMOR_TEMPLATE",
    f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/agent-standards-observe",
)

# Timeout for Model Armor API calls — short to avoid slowing the agent
_TIMEOUT = 5.0


def _get_access_token() -> str:
    """Get a Google access token for Model Armor API calls.

    Uses Application Default Credentials (ADC).
    """
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    return credentials.token


async def scan_input(text: str) -> Optional[dict]:
    """Scan user input with Model Armor. Returns findings dict or None.

    This is observe-only: results are logged but never used to block requests.
    """
    try:
        token = _get_access_token()
        url = (
            f"https://modelarmor.{LOCATION}.rep.googleapis.com/v1/"
            f"{MODEL_ARMOR_TEMPLATE}:sanitizeUserPrompt"
        )
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={"user_prompt_data": {"text": text}},
            )
            if resp.status_code == 200:
                return resp.json()
            log.debug("Model Armor input scan returned %d", resp.status_code)
    except Exception as e:
        log.debug("Model Armor input scan failed (observe-only): %s", e)
    return None


async def scan_output(text: str) -> Optional[dict]:
    """Scan model output with Model Armor. Returns findings dict or None.

    This is observe-only: results are logged but never used to modify output.
    """
    try:
        token = _get_access_token()
        url = (
            f"https://modelarmor.{LOCATION}.rep.googleapis.com/v1/"
            f"{MODEL_ARMOR_TEMPLATE}:sanitizeModelResponse"
        )
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={"model_response_data": {"text": text}},
            )
            if resp.status_code == 200:
                return resp.json()
            log.debug("Model Armor output scan returned %d", resp.status_code)
    except Exception as e:
        log.debug("Model Armor output scan failed (observe-only): %s", e)
    return None


async def before_model_callback(callback_context, llm_request):
    """ADK before_model_callback: scan user input with Model Armor.

    Observe-only — logs findings but never blocks the LLM call.
    Returns None to allow normal processing.
    """
    # Extract the latest user message from the LLM request
    try:
        user_text = None
        if hasattr(llm_request, "contents") and llm_request.contents:
            for content in reversed(llm_request.contents):
                if hasattr(content, "role") and content.role == "user":
                    if hasattr(content, "parts") and content.parts:
                        texts = [p.text for p in content.parts if hasattr(p, "text") and p.text]
                        if texts:
                            user_text = " ".join(texts)
                            break

        if user_text:
            result = await scan_input(user_text)
            if result and result.get("sanitizationResult", {}).get("filterMatchState") == "MATCH_FOUND":
                log.warning(
                    "Model Armor INPUT findings (observe-only): %s",
                    result.get("sanitizationResult", {}),
                )
    except Exception as e:
        log.debug("before_model_callback error (non-blocking): %s", e)

    # Always return None — never block
    return None


async def after_model_callback(callback_context, llm_response):
    """ADK after_model_callback: scan model output with Model Armor.

    Observe-only — logs findings but never modifies the response.
    Returns None to allow normal response delivery.
    """
    try:
        response_text = None
        if hasattr(llm_response, "content") and llm_response.content:
            if hasattr(llm_response.content, "parts") and llm_response.content.parts:
                texts = [p.text for p in llm_response.content.parts if hasattr(p, "text") and p.text]
                if texts:
                    response_text = " ".join(texts)

        if response_text:
            result = await scan_output(response_text)
            if result and result.get("sanitizationResult", {}).get("filterMatchState") == "MATCH_FOUND":
                log.warning(
                    "Model Armor OUTPUT findings (observe-only): %s",
                    result.get("sanitizationResult", {}),
                )
    except Exception as e:
        log.debug("after_model_callback error (non-blocking): %s", e)

    # Always return None — never block or modify
    return None
