from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_ANNOTATIONS_PATH = _DATA_ROOT / "drug_annotations.json"


@lru_cache(maxsize=1)
def _load_annotations() -> Dict[str, Dict[str, Any]]:
    if not _ANNOTATIONS_PATH.exists():
        return {}
    with _ANNOTATIONS_PATH.open("r", encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError:
            return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, record in payload.items():
        normalized[key.upper()] = record
    return normalized


def drug_annotations(drug_id: str) -> Optional[Dict[str, Any]]:
    annotations = _load_annotations()
    return annotations.get(drug_id.upper())
