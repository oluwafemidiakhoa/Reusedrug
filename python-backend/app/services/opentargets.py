from __future__ import annotations

from typing import List, Tuple

from app.models import RankWarning
from app.services.http import request_with_retry

GRAPHQL_ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"

QUERY = """
query DiseaseKnownDrugs($id: String!) {
  disease(efoId: $id) {
    id
    name
    knownDrugs {
      rows {
        target {
          id
          approvedSymbol
        }
        drug {
          id
          name
        }
        disease {
          id
          name
        }
        phase
        status
      }
    }
  }
}
"""


async def fetch_known_drugs(disease_id: str) -> Tuple[List[dict], List[RankWarning]]:
    warnings: List[RankWarning] = []
    payload = {"query": QUERY, "variables": {"id": disease_id}}
    try:
        response = await request_with_retry("POST", GRAPHQL_ENDPOINT, json=payload, timeout=15.0)
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        warnings.append(RankWarning(source="opentargets", detail=str(exc)))
        return [], warnings

    if "errors" in data:
        warnings.append(
            RankWarning(source="opentargets", detail=str(data["errors"][0].get("message", "error")))
        )
        return [], warnings

    disease = data.get("data", {}).get("disease")
    if not disease:
        warnings.append(
            RankWarning(source="opentargets", detail="No disease record returned from Open Targets")
        )
        return [], warnings

    rows = disease.get("knownDrugs", {}).get("rows", []) or []
    results: List[dict] = []
    for row in rows:
        drug = row.get("drug") or {}
        target = row.get("target") or {}
        results.append(
            {
                "drug_id": drug.get("id"),
                "drug_name": drug.get("name"),
                "target_id": target.get("id"),
                "target_symbol": target.get("approvedSymbol"),
                "phase": row.get("phase"),
                "status": row.get("status"),
            }
        )

    return results, warnings
