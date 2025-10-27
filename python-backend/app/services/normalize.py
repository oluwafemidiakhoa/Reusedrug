from __future__ import annotations

from typing import Optional, Tuple

from app.models import RankWarning
from app.services.http import request_with_retry
from app.utils.identifiers import normalize_disease_id, harmonize_identifier

OLS_URL = "https://www.ebi.ac.uk/ols4/api/search"
ONTOLOGIES = ("efo", "mondo")


async def normalize_disease(disease: str) -> Tuple[Optional[str], Optional[str], list[RankWarning]]:
    warnings: list[RankWarning] = []

    for ontology in ONTOLOGIES:
        params = {
            "q": disease,
            "ontology": ontology,
            "type": "class",
            "rows": 1,
            "exact": "false",
        }

        try:
            response = await request_with_retry("GET", OLS_URL, params=params, timeout=8.0)
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            warnings.append(RankWarning(source="ebi_ols", detail=f"{ontology}: {exc}"))
            continue

        docs = payload.get("response", {}).get("docs", [])
        if not docs:
            warnings.append(
                RankWarning(source="ebi_ols", detail=f"{ontology}: no results; trying fallback")
            )
            continue

        top = docs[0]
        disease_id_raw = (
            top.get("obo_id")
            or top.get("short_form")
            or top.get("iri")
            or top.get("curie")
        )
        label = top.get("label") or disease

        normalized_id = normalize_disease_id(disease_id_raw or "")
        if not normalized_id:
            harmonized = harmonize_identifier(disease_id_raw or "")
            normalized_id = harmonized.normalized

        return normalized_id, label, warnings

    warnings.append(
        RankWarning(source="ebi_ols", detail="No normalization result found; using raw input")
    )
    return None, None, warnings
