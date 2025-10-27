from __future__ import annotations

from typing import List, Tuple

from app.models import RankWarning
from app.services.http import request_with_retry

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


async def fetch_trials(condition: str, intervention: str) -> Tuple[List[dict], List[RankWarning]]:
    search_terms = [term for term in (condition, intervention) if term]
    query_term = " ".join(search_terms)
    if not query_term:
        return [], [
            RankWarning(
                source="clinicaltrials",
                detail="Insufficient data to query ClinicalTrials.gov",
            )
        ]
    params = {
        "format": "json",
        "pageSize": 10,
        "fields": "NCTId,BriefTitle,Phase,OverallStatus",
    }
    if query_term:
        params["query.term"] = query_term
    warnings: List[RankWarning] = []
    try:
        response = await request_with_retry("GET", BASE_URL, params=params, timeout=10.0)
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        warnings.append(RankWarning(source="clinicaltrials", detail=str(exc)))
        return [], warnings

    studies = data.get("studies", []) or []
    summaries: List[dict] = []
    for study in studies:
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        design_module = protocol.get("designModule", {})
        summaries.append(
            {
                "nct_id": identification.get("nctId"),
                "title": identification.get("briefTitle"),
                "phase": (design_module.get("phases") or ["Unknown"])[0],
                "status": status_module.get("overallStatus"),
            }
        )

    if not summaries:
        warnings.append(
            RankWarning(
                source="clinicaltrials",
                detail=f"No clinical trials found for condition={condition} intervention={intervention}",
            )
        )

    return summaries, warnings
