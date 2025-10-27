from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import httpx

from app.cache import TTLCache
from app.models import RankWarning

DRUGCENTRAL_ENDPOINT = os.getenv(
    "DRUGCENTRAL_ENDPOINT", "https://drugcentral.org/api/active_ingredient"
)
DRUGCENTRAL_ENABLED = os.getenv("DRUGCENTRAL_ENABLED", "false").lower() in {"1", "true", "yes"}
DRUGCENTRAL_TIMEOUT = float(os.getenv("DRUGCENTRAL_TIMEOUT_SECONDS", "20"))

_CACHE = TTLCache(ttl_seconds=float(os.getenv("DRUGCENTRAL_CACHE_SECONDS", "3600")), maxsize=256)


async def fetch_drug_moa(drug_name: str) -> Tuple[List[Dict[str, Any]], List[RankWarning]]:
    warnings: List[RankWarning] = []
    if not DRUGCENTRAL_ENABLED:
        warnings.append(
            RankWarning(source="drugcentral", detail="DrugCentral integration disabled via config")
        )
        return [], warnings

    key = ("drugcentral", drug_name.lower())

    async def _query() -> Tuple[List[Dict[str, Any]], List[RankWarning]]:
        try:
            async with httpx.AsyncClient(timeout=DRUGCENTRAL_TIMEOUT) as client:
                response = await client.get(
                    DRUGCENTRAL_ENDPOINT,
                    params={"name": drug_name},
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001
            return [], [RankWarning(source="drugcentral", detail=str(exc))]

        if not payload:
            return [], [
                RankWarning(
                    source="drugcentral", detail=f"No DrugCentral data returned for {drug_name}"
                )
            ]

        return payload if isinstance(payload, list) else [payload], []

    cached = await _CACHE.get_or_set(key, lambda: _query())
    return cached

