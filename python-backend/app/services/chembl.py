from __future__ import annotations

from typing import Dict, List, Tuple

from app.models import RankWarning
from app.services.http import request_with_retry

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"


async def fetch_strongest_activity(
    molecule_id: str, target_id: str | None
) -> Tuple[List[Dict], List[RankWarning]]:
    warnings: List[RankWarning] = []
    params = {
        "molecule_chembl_id": molecule_id.upper(),
        "limit": 1,
        "offset": 0,
        "orderby": "potency",
    }
    if target_id:
        normalized_target = target_id.upper()
        if normalized_target.startswith("CHEMBL"):
            params["target_chembl_id"] = normalized_target

    try:
        response = await request_with_retry("GET", BASE_URL, params=params, timeout=10.0)
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        warnings.append(RankWarning(source="chembl", detail=str(exc)))
        return [], warnings

    activities = data.get("activities", []) or []
    if not activities:
        warnings.append(
            RankWarning(
                source="chembl",
                detail=f"No ChEMBL activities found for drug={molecule_id} target={target_id or 'any'}",
            )
        )
        return [], warnings

    return activities, warnings
