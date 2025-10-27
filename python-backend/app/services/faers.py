from __future__ import annotations

from typing import List, Tuple

from app.models import RankWarning


async def fetch_top_reactions(drug_name: str) -> Tuple[List[dict], List[RankWarning]]:
    warnings = [
        RankWarning(
            source="openfda_faers",
            detail="TODO: integrate with openFDA FAERS public API for safety signals",
        )
    ]
    return [], warnings

