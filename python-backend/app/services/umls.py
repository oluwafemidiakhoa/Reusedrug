from __future__ import annotations

import os
from typing import Dict, List, Tuple

from app.cache import TTLCache
from app.models import ConceptMapping, RankWarning
from app.services.http import request_with_retry

UMLS_ENABLED = os.getenv("UMLS_ENABLED", "false").lower() in {"1", "true", "yes"}
UMLS_SEARCH_ENDPOINT = os.getenv(
    "UMLS_SEARCH_ENDPOINT", "https://uts-ws.nlm.nih.gov/rest/search/current"
)
UMLS_RELATED_ENDPOINT = os.getenv(
    "UMLS_RELATED_ENDPOINT", "https://uts-ws.nlm.nih.gov/rest/content/current/CUI"
)
UMLS_API_KEY = os.getenv("UMLS_API_KEY")
UMLS_TIMEOUT = float(os.getenv("UMLS_TIMEOUT_SECONDS", "15"))
UMLS_CACHE_SECONDS = float(os.getenv("UMLS_CACHE_SECONDS", "1800"))
UMLS_PAGE_SIZE = int(os.getenv("UMLS_PAGE_SIZE", "5"))

_CACHE = TTLCache(ttl_seconds=UMLS_CACHE_SECONDS, maxsize=128)


async def lookup_concepts(term: str) -> Tuple[List[ConceptMapping], List[RankWarning]]:
    warnings: List[RankWarning] = []
    if not UMLS_ENABLED:
        warnings.append(RankWarning(source="umls", detail="UMLS integration disabled via config"))
        return [], warnings

    query = term.strip()
    if not query:
        warnings.append(RankWarning(source="umls", detail="Empty query provided"))
        return [], warnings

    cache_key = query.lower()
    cached = await _CACHE.get(cache_key)
    if cached:
        return cached

    params = {
        "string": query,
        "searchType": "exact",
        "returnIdType": "concept",
        "pageSize": str(UMLS_PAGE_SIZE),
    }
    if UMLS_API_KEY:
        params["apiKey"] = UMLS_API_KEY

    try:
        response = await request_with_retry(
            "GET",
            UMLS_SEARCH_ENDPOINT,
            params=params,
            timeout=UMLS_TIMEOUT,
        )
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        warning = RankWarning(source="umls", detail=f"Search failed: {exc}")
        warnings.append(warning)
        await _CACHE.set(cache_key, ([], warnings))
        return [], warnings

    result_block = payload.get("result", {}) if isinstance(payload, dict) else {}
    result_items = result_block.get("results", []) or []
    if not result_items:
        warnings.append(
            RankWarning(source="umls", detail="No UMLS concepts returned for query")
        )
        await _CACHE.set(cache_key, ([], warnings))
        return [], warnings

    concepts: List[ConceptMapping] = []
    for item in result_items:
        cui = item.get("ui")
        if not cui or cui.upper() == "NONE":
            continue
        name = item.get("name") or query
        preferred = item.get("preferredName") or name
        semantic_types = item.get("semanticTypes") or []
        synonyms = item.get("synonyms") or item.get("atoms") or []
        concept = ConceptMapping(
            cui=cui,
            name=name,
            preferred_name=preferred,
            semantic_types=[
                st.get("name") if isinstance(st, dict) else str(st) for st in semantic_types
            ],
            synonyms=[syn.get("name") if isinstance(syn, dict) else str(syn) for syn in synonyms],
        )
        concepts.append(concept)

    await _CACHE.set(cache_key, (concepts, warnings))
    return concepts, warnings


async def clear_cache() -> None:
    await _CACHE.clear()
