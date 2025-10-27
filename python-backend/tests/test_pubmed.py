import pytest
import respx

from app.models import RankWarning
from app.services import pubmed


@pytest.mark.asyncio
async def test_fetch_pubmed_summaries_success():
    await pubmed.clear_cache()

    search_payload = {
        "esearchresult": {
            "idlist": ["12345678", "87654321"],
        }
    }
    summary_payload = {
        "result": {
            "uids": ["12345678", "87654321"],
            "12345678": {
                "uid": "12345678",
                "title": "Repurposing success story",
                "fulljournalname": "Journal of Testing",
                "pubdate": "2024 Jan",
                "authors": [{"name": "Doe J"}],
            },
            "87654321": {
                "uid": "87654321",
                "title": "Another insight",
                "fulljournalname": "Science of Examples",
                "pubdate": "2023 Oct",
                "authors": [],
            },
        }
    }

    with respx.mock(assert_all_called=False) as mock:
        mock.get(pubmed.ESEARCH_URL).respond(
            200,
            json=search_payload,
        )
        mock.get(pubmed.ESUMMARY_URL).respond(
            200,
            json=summary_payload,
        )

        records, warnings = await pubmed.fetch_pubmed_summaries("diabetes", "metformin", limit=2)

    assert len(records) == 2
    assert not warnings
    assert records[0]["pmid"] == "12345678"
    assert records[0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


@pytest.mark.asyncio
async def test_fetch_pubmed_summaries_no_results():
    await pubmed.clear_cache()

    with respx.mock(assert_all_called=False) as mock:
        mock.get(pubmed.ESEARCH_URL).respond(
            200,
            json={"esearchresult": {"idlist": []}},
        )

        records, warnings = await pubmed.fetch_pubmed_summaries("rare disease", "unknown")

    assert records == []
    assert warnings and isinstance(warnings[0], RankWarning)
