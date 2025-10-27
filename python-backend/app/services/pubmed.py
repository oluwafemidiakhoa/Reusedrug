from __future__ import annotations

import os
from typing import Dict, List, Tuple

from app.cache import TTLCache
from app.models import RankWarning
from app.services.http import request_with_retry

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
ESUMMARY_URL = f"{EUTILS_BASE}/esummary.fcgi"

PUBMED_ENABLED = os.getenv("PUBMED_ENABLED", "true").lower() in {"1", "true", "yes"}
PUBMED_CACHE_SECONDS = float(os.getenv("PUBMED_CACHE_SECONDS", "900"))
PUBMED_RESULT_LIMIT = int(os.getenv("PUBMED_RESULT_LIMIT", "3"))
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")

_CACHE = TTLCache(ttl_seconds=PUBMED_CACHE_SECONDS, maxsize=128)


def _query_term(disease: str, drug: str) -> str:
    tokens = []
    if drug:
        tokens.append(f'("{drug}"[Title/Abstract])')
    if disease:
        tokens.append(f'("{disease}"[Title/Abstract])')
    if not tokens:
        return ""
    return " AND ".join(tokens)


def _summary_to_record(entry: Dict) -> Dict:
    pmid = entry.get("uid")
    return {
        "pmid": pmid,
        "title": entry.get("title"),
        "journal": entry.get("fulljournalname"),
        "pub_date": entry.get("pubdate"),
        "authors": [author.get("name") for author in entry.get("authors", []) if author.get("name")],
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
    }


async def fetch_pubmed_summaries(
    disease: str,
    drug: str,
    *,
    limit: int | None = None,
) -> Tuple[List[Dict], List[RankWarning]]:
    warnings: List[RankWarning] = []
    if not PUBMED_ENABLED:
        warnings.append(RankWarning(source="pubmed", detail="PubMed integration disabled via config"))
        return [], warnings

    term = _query_term(disease, drug)
    if not term:
        warnings.append(
            RankWarning(
                source="pubmed",
                detail="Insufficient search context for PubMed query",
            )
        )
        return [], warnings

    retmax = limit or PUBMED_RESULT_LIMIT
    cache_key = (term.lower(), retmax)
    cached = await _CACHE.get(cache_key)
    if cached:
        return cached

    params = {
        "db": "pubmed",
        "retmode": "json",
        "term": term,
        "retmax": str(retmax),
        "sort": "relevance",
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    try:
        search_response = await request_with_retry(
            "GET",
            ESEARCH_URL,
            params=params,
            timeout=10.0,
        )
        search_payload = search_response.json()
    except Exception as exc:  # noqa: BLE001
        warnings.append(RankWarning(source="pubmed", detail=f"esearch failed: {exc}"))
        await _CACHE.set(cache_key, ([], warnings))
        return [], warnings

    id_list = search_payload.get("esearchresult", {}).get("idlist", []) or []
    if not id_list:
        warnings.append(
            RankWarning(
                source="pubmed",
                detail="No PubMed publications found for disease/drug query",
            )
        )
        await _CACHE.set(cache_key, ([], warnings))
        return [], warnings

    summary_params = {
        "db": "pubmed",
        "retmode": "json",
        "id": ",".join(id_list),
    }
    if PUBMED_API_KEY:
        summary_params["api_key"] = PUBMED_API_KEY

    try:
        summary_response = await request_with_retry(
            "GET",
            ESUMMARY_URL,
            params=summary_params,
            timeout=10.0,
        )
        summary_payload = summary_response.json()
    except Exception as exc:  # noqa: BLE001
        warnings.append(RankWarning(source="pubmed", detail=f"esummary failed: {exc}"))
        await _CACHE.set(cache_key, ([], warnings))
        return [], warnings

    result_entries = summary_payload.get("result", {}) or {}
    uids = [pmid for pmid in id_list if pmid in result_entries]
    records = []
    for pmid in uids:
        entry = result_entries.get(pmid)
        if not entry:
            continue
        records.append(_summary_to_record(entry))

    if not records:
        warnings.append(
            RankWarning(
                source="pubmed",
                detail="PubMed summaries missing expected result entries",
            )
        )

    await _CACHE.set(cache_key, (records, warnings))
    return records, warnings


async def clear_cache() -> None:
    await _CACHE.clear()
