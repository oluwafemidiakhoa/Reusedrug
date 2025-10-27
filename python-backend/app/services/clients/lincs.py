from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from app.models import RankWarning

LINCS_ENABLED = os.getenv("LINCS_ENABLED", "false").lower() in {"1", "true", "yes"}
CLUE_ENABLED = os.getenv("CLUE_ENABLED", "false").lower() in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def _get_clue_client():
    from cmapPy.clue_api_client.clue_api_client import CMapApiClient  # type: ignore

    return CMapApiClient()


async def fetch_signature_scores(disease_signature: str, limit: int = 25) -> Tuple[List[Dict[str, Any]], List[RankWarning]]:
    warnings: List[RankWarning] = []
    if not LINCS_ENABLED and not CLUE_ENABLED:
        warnings.append(
            RankWarning(source="lincs", detail="LINCS/CLUE integrations disabled via config")
        )
        return [], warnings

    if CLUE_ENABLED:
        try:
            client = _get_clue_client()
        except Exception as exc:  # noqa: BLE001
            warnings.append(RankWarning(source="clue", detail=f"Failed to init client: {exc}"))
        else:
            try:
                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(
                    None,
                    lambda: client.sig_gutc({"key": disease_signature, "limit": limit}),
                )
                return data.get("data", []), warnings
            except Exception as exc:  # noqa: BLE001
                warnings.append(RankWarning(source="clue", detail=str(exc)))

    warnings.append(
        RankWarning(
            source="lincs",
            detail="Signature reversal data unavailable; enable CLUE or LINCS integration",
        )
    )
    return [], warnings

