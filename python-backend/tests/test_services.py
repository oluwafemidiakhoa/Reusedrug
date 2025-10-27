import pytest
import respx

from app.models import RankWarning
from app.services import normalize, opentargets


@pytest.mark.asyncio
async def test_normalize_handles_failure():
    with respx.mock() as mock:
        mock.get(normalize.OLS_URL).respond(500)
        disease_id, label, warnings = await normalize.normalize_disease("foo")
    assert disease_id is None
    assert label is None
    assert warnings and isinstance(warnings[0], RankWarning)


@pytest.mark.asyncio
async def test_opentargets_empty():
    with respx.mock() as mock:
        mock.post(opentargets.GRAPHQL_ENDPOINT).respond(200, json={"data": {"disease": None}})
        rows, warnings = await opentargets.fetch_known_drugs("EFO:1")
    assert rows == []
    assert warnings and warnings[0].source == "opentargets"

