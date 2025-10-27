from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Mapping, Optional

import httpx

logger = logging.getLogger(__name__)

_ENDPOINT = os.getenv("PERSONA_ANALYTICS_ENDPOINT")
_API_KEY = os.getenv("PERSONA_ANALYTICS_API_KEY")
_CLIENT: Optional[httpx.AsyncClient] = None
_CLIENT_LOCK = asyncio.Lock()


async def _client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT:
        return _CLIENT
    async with _CLIENT_LOCK:
        if _CLIENT:
            return _CLIENT
        timeout = httpx.Timeout(2.5, connect=1.0)
        _CLIENT = httpx.AsyncClient(timeout=timeout)
        return _CLIENT


def _headers() -> Mapping[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _API_KEY:
        headers["Authorization"] = f"Bearer {_API_KEY}"
    return headers


async def track_persona_event(
    *,
    persona: str,
    overrides: Mapping[str, float],
    disease_query: str,
    normalized_disease: Optional[str],
    cached: bool,
    source: str = "backend",
) -> None:
    """
    Emit a persona usage event to the configured analytics sink.

    The call is best-effort and silently ignored if the endpoint is not configured.
    """

    if not _ENDPOINT:
        return

    payload: dict[str, Any] = {
        "event": "persona_selection",
        "timestamp": time.time(),
        "persona": persona,
        "override_keys": sorted(overrides.keys()),
        "override_weights": {key: float(value) for key, value in overrides.items()},
        "disease_query": disease_query,
        "normalized_disease": normalized_disease,
        "cached": cached,
        "source": source,
    }

    try:
        client = await _client()
        await client.post(_ENDPOINT, headers=_headers(), content=json.dumps(payload))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to send persona analytics event: %s", exc)
