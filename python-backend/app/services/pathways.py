from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_PATHWAYS_PATH = _DATA_ROOT / "pathways.json"


@lru_cache(maxsize=1)
def _load_pathways() -> Dict[str, List[dict]]:
    if not _PATHWAYS_PATH.exists():
        return {}
    with _PATHWAYS_PATH.open("r", encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError:
            return {}
    normalized: Dict[str, List[dict]] = {}
    for key, entries in payload.items():
        normalized[key.upper()] = entries
    return normalized


def related_pathways(gene_symbols: List[str]) -> List[dict]:
    if not gene_symbols:
        return []
    pathways = _load_pathways()
    seen: dict[str, dict] = {}
    for symbol in gene_symbols:
        records = pathways.get(symbol.upper())
        if not records:
            continue
        for record in records:
            name = record.get("name")
            if not name:
                continue
            seen.setdefault(
                name,
                {
                    "name": name,
                    "source": record.get("source", "Pathway"),
                    "url": record.get("url"),
                    "genes": set(),
                },
            )
            seen[name]["genes"].add(symbol.upper())
    results: List[dict] = []
    for entry in seen.values():
        entry["genes"] = sorted(entry["genes"])
        results.append(entry)
    return sorted(results, key=lambda item: item["name"])
