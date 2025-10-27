import pytest
import respx

from app.cache import TTLCache
from app.models import RankWarning
from app.services import umls


@pytest.mark.asyncio
async def test_lookup_concepts_success(monkeypatch):
    monkeypatch.setattr(umls, "UMLS_ENABLED", True)
    monkeypatch.setattr(umls, "UMLS_SEARCH_ENDPOINT", "https://umls.test/search")
    monkeypatch.setattr(umls, "_CACHE", TTLCache(ttl_seconds=60, maxsize=16))  # reset cache

    payload = {
        "result": {
            "results": [
                {
                    "ui": "C000000",
                    "name": "Test Disease",
                    "preferredName": "Test Disease Preferred",
                    "semanticTypes": [{"name": "Disease or Syndrome"}],
                    "synonyms": [{"name": "TD"}],
                }
            ]
        }
    }

    with respx.mock() as mock:
        mock.get("https://umls.test/search").respond(200, json=payload)
        concepts, warnings = await umls.lookup_concepts("Test Disease")

    assert len(concepts) == 1
    concept = concepts[0]
    assert concept.cui == "C000000"
    assert concept.preferred_name == "Test Disease Preferred"
    assert "TD" in concept.synonyms
    assert warnings == []


@pytest.mark.asyncio
async def test_lookup_concepts_disabled(monkeypatch):
    monkeypatch.setattr(umls, "UMLS_ENABLED", False)
    concepts, warnings = await umls.lookup_concepts("anything")
    assert concepts == []
    assert warnings and isinstance(warnings[0], RankWarning)
