from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from app.cache import TTLCache
from app.models import RankWarning
from app.services.http import request_with_retry
from app.utils.identifiers import normalize_disease_id, normalize_drug_id

TRANSLATOR_ENDPOINT = os.getenv("TRANSLATOR_ENDPOINT", "https://api.bte.ncats.io/v1/query")
TRANSLATOR_ENABLED = os.getenv("TRANSLATOR_ENABLED", "false").lower() in {"1", "true", "yes"}
TRANSLATOR_TIMEOUT = float(os.getenv("TRANSLATOR_TIMEOUT_SECONDS", "120"))
TRANSLATOR_CACHE_SECONDS = float(os.getenv("TRANSLATOR_CACHE_SECONDS", "900"))

BIO_ENTITY_CHEMICAL = "biolink:ChemicalEntity"
BIO_PREDICATE_TREATS = "biolink:treats"

_CACHE = TTLCache(ttl_seconds=TRANSLATOR_CACHE_SECONDS, maxsize=64)


def _build_query(disease_ids: List[str]) -> Dict[str, Any]:
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "disease": {"ids": disease_ids},
                    "chemical": {"categories": [BIO_ENTITY_CHEMICAL]},
                },
                "edges": {
                    "edge": {
                        "subject": "chemical",
                        "object": "disease",
                        "predicates": [BIO_PREDICATE_TREATS],
                    }
                },
            }
        }
    }


def _normalize_drug_id(node: Dict[str, Any]) -> Optional[str]:
    ids = node.get("ids") or []
    for identifier in ids:
        normalized = normalize_drug_id(identifier)
        if normalized:
            return normalized
    if ids:
        return ids[0]
    return None


async def fetch_disease_treatments(
    disease_ids: List[str],
) -> Tuple[List[Dict[str, Any]], List[RankWarning]]:
    warnings: List[RankWarning] = []
    if not TRANSLATOR_ENABLED:
        warnings.append(
            RankWarning(source="translator", detail="Translator integration disabled via config")
        )
        return [], warnings

    if not disease_ids:
        return [], warnings

    cache_key = tuple(sorted(disease_ids))
    cached = await _CACHE.get(cache_key)
    if cached:
        return cached

    payload = _build_query(disease_ids)
    try:
        response = await request_with_retry(
            "POST",
            TRANSLATOR_ENDPOINT,
            json=payload,
            timeout=TRANSLATOR_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        warnings.append(RankWarning(source="translator", detail=str(exc)))
        return [], warnings

    results = data.get("message", {}).get("results", []) or []
    if not results:
        warnings.append(
            RankWarning(source="translator", detail="No Translator results returned for query")
        )
        await _CACHE.set(cache_key, ([], warnings))
        return [], warnings

    candidates: List[Dict[str, Any]] = []
    for result in results:
        node_bindings = result.get("node_bindings") or {}
        chem_nodes = node_bindings.get("chemical") or []
        disease_nodes = node_bindings.get("disease") or []

        for chem in chem_nodes:
            chem_id = _normalize_drug_id(chem)
            if not chem_id:
                continue
            chem_label = chem.get("name") or chem.get("label") or chem_id
            path = result.get("analyses", [{}])[0].get("edge_bindings", {})
            disease_ids = []
            for disease_node in disease_nodes:
                raw_id = disease_node.get("id")
                normalized = normalize_disease_id(raw_id) if raw_id else None
                disease_ids.append(normalized or raw_id)

            candidates.append(
                {
                    "drug_id": chem_id,
                    "drug_name": chem_label,
                    "disease_ids": [value for value in disease_ids if value],
                    "evidence": path,
                }
            )

    await _CACHE.set(cache_key, (candidates, warnings))
    return candidates, warnings
