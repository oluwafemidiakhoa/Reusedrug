from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import csv

from app.cache import TTLCache
from app.models import RankWarning

SIDER_ENABLED = os.getenv("SIDER_ENABLED", "false").lower() in {"1", "true", "yes"}
SIDER_DATA_PATH = Path(os.getenv("SIDER_DATA_PATH", "data/sider/meddra_all_se.tsv"))
SIDER_CACHE_SECONDS = float(os.getenv("SIDER_CACHE_SECONDS", "3600"))

_CACHE = TTLCache(ttl_seconds=SIDER_CACHE_SECONDS, maxsize=256)


def _read_sider() -> Dict[str, List[Dict[str, str]]]:
    records: Dict[str, List[Dict[str, str]]] = {}
    if not SIDER_DATA_PATH.exists():
        return records
    with SIDER_DATA_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            drug_id, reaction, _, _, seriousness = row[:5]
            records.setdefault(drug_id, []).append(
                {"reaction": reaction, "seriousness": seriousness}
            )
    return records


async def fetch_safety(drug_id: str) -> Tuple[List[Dict[str, str]], List[RankWarning]]:
    warnings: List[RankWarning] = []
    if not SIDER_ENABLED:
        warnings.append(RankWarning(source="sider", detail="SIDER integration disabled via config"))
        return [], warnings

    if not SIDER_DATA_PATH.exists():
        warnings.append(
            RankWarning(
                source="sider",
                detail=f"SIDER dataset not found at {SIDER_DATA_PATH}. Download TSV first.",
            )
        )
        return [], warnings

    cache_key = ("sider", SIDER_DATA_PATH.stat().st_mtime)
    records = await _CACHE.get_or_set(cache_key, _read_sider)
    safety = records.get(drug_id) or []
    if not safety:
        warnings.append(RankWarning(source="sider", detail=f"No SIDER matches for {drug_id}"))
    return safety, warnings

