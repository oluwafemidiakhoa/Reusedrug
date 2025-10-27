import pytest
import respx
from httpx import Response
from unittest.mock import AsyncMock

from app.services import chembl, clinicaltrials, normalize, opentargets
from app.db import init_db


def _register_rank_mocks(mock) -> None:
    disease_payload = {
        "data": {
            "disease": {
                "id": "EFO:0000270",
                "name": "Asthma",
                "knownDrugs": {
                    "rows": [
                        {
                            "target": {"id": "ENSG000001", "approvedSymbol": "IL5"},
                            "drug": {"id": "CHEMBL123", "name": "Mepolizumab"},
                            "disease": {"id": "EFO:0000270", "name": "Asthma"},
                            "phase": "Phase 3",
                            "status": "Approved",
                        }
                    ]
                },
            }
        }
    }

    chembl_payload = {
        "activities": [
            {
                "potency": "200",
                "standard_type": "IC50",
            }
        ]
    }

    clinical_payload = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT123", "briefTitle": "Asthma Study"},
                    "statusModule": {"overallStatus": "Completed"},
                    "designModule": {"phases": ["Phase 3"]},
                }
            }
        ]
    }

    mock.get(normalize.OLS_URL).respond(
        200,
        json={
            "response": {
                "docs": [
                    {
                        "obo_id": "EFO:0000270",
                        "label": "Asthma",
                    }
                ]
            }
        },
    )
    mock.post(opentargets.GRAPHQL_ENDPOINT).respond(200, json=disease_payload)
    mock.get(chembl.BASE_URL).respond(200, json=chembl_payload)
    mock.get(clinicaltrials.BASE_URL).respond(200, json=clinical_payload)


@pytest.mark.asyncio
async def test_healthz(test_client):
    response = await test_client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_rank_candidates_success(test_client, monkeypatch):
    analytics_mock = AsyncMock()
    monkeypatch.setattr("app.analytics.track_persona_event", analytics_mock)

    with respx.mock(assert_all_called=False) as mock:
        _register_rank_mocks(mock)
        response = await test_client.post("/v1/rank", json={"disease": "asthma"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_disease"] == "Asthma"
    assert payload["candidates"]
    candidate = payload["candidates"][0]
    assert candidate["drug_id"] == "CHEMBL123"
    assert candidate["score"]["final_score"] >= 0
    assert candidate.get("narrative")
    assert candidate["narrative"]["summary"]
    assert candidate.get("confidence")
    assert candidate["confidence"]["tier"] in {"exploratory", "hypothesis-ready", "decision-grade"}
    assert candidate["confidence"]["score"] >= 0
    assert candidate["confidence"]["signals"]
    assert "Asthma" in candidate["contraindications"]
    assert candidate["annotation_sources"]
    assert payload["pathway_summary"]
    assert payload["pathway_summary"][0]["count"] >= 1
    assert payload["cached"] is False
    assert any(w["source"] == "openfda_faers" for w in payload["warnings"])
    assert any(w["source"] == "drug_annotations" for w in payload["warnings"])
    assert "graph_overview" in payload
    cf = payload["counterfactuals"]
    assert cf
    assert cf[0]["label"] == "Safety -20%"
    scoring = payload["scoring"]
    assert scoring["persona"] == "balanced"
    assert scoring["overrides"] == {}
    assert all(abs(delta) < 1e-6 for delta in scoring["delta_vs_default"].values())
    analytics_mock.assert_awaited()


@pytest.mark.asyncio
async def test_rank_candidates_exclude_contraindicated(test_client, monkeypatch):
    analytics_mock = AsyncMock()
    monkeypatch.setattr("app.analytics.track_persona_event", analytics_mock)

    with respx.mock(assert_all_called=False) as mock:
        _register_rank_mocks(mock)
        response = await test_client.post(
            "/v1/rank",
            json={"disease": "asthma", "exclude_contraindicated": True},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidates"] == []
    assert any(w["source"] == "drug_annotations" for w in payload["warnings"])
    analytics_mock.assert_awaited()


@pytest.mark.asyncio
async def test_workspace_requires_api_key(test_client):
    response = await test_client.get("/v1/workspaces/queries")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_workspace_save_and_list(test_client, monkeypatch):
    monkeypatch.setenv("WORKSPACE_API_KEY", "secret")
    monkeypatch.setenv("WORKSPACE_USER_ID", "tester")

    payload = {
        "disease": "influenza",
        "response": {
            "query": "influenza",
            "normalized_disease": "Influenza",
            "candidates": [],
            "warnings": [],
            "cached": False,
        },
        "note": "Clinical follow-up needed.",
    }

    response = await test_client.post(
        "/v1/workspaces/queries",
        json=payload,
        headers={"x-api-key": "secret"},
    )
    assert response.status_code == 201

    list_response = await test_client.get(
        "/v1/workspaces/queries", headers={"x-api-key": "secret"}
    )
    assert list_response.status_code == 200
    saved = list_response.json()
    assert saved
    assert saved[0]["disease"] == "influenza"
    assert saved[0]["note"] == payload["note"]

@pytest.mark.asyncio
async def test_workspace_update_and_delete(test_client, monkeypatch):
    monkeypatch.setenv("WORKSPACE_API_KEY", "secret")
    monkeypatch.setenv("WORKSPACE_USER_ID", "tester")

    payload = {
        "disease": "influenza",
        "response": {
            "query": "influenza",
            "normalized_disease": "Influenza",
            "candidates": [],
            "warnings": [],
            "cached": False,
        },
        "note": "Initial note",
    }

    create_response = await test_client.post(
        "/v1/workspaces/queries",
        json=payload,
        headers={"x-api-key": "secret"},
    )
    assert create_response.status_code == 201
    created = create_response.json()
    query_id = created["id"]

    patch_response = await test_client.patch(
        f"/v1/workspaces/queries/{query_id}",
        json={"note": "Updated note"},
        headers={"x-api-key": "secret"},
    )
    assert patch_response.status_code == 200
    updated = patch_response.json()
    assert updated["note"] == "Updated note"

    delete_response = await test_client.delete(
        f"/v1/workspaces/queries/{query_id}",
        headers={"x-api-key": "secret"},
    )
    assert delete_response.status_code == 204

    list_response = await test_client.get(
        "/v1/workspaces/queries",
        headers={"x-api-key": "secret"},
    )
    assert list_response.status_code == 200
    assert list_response.json() == []


@pytest.mark.asyncio
async def test_rank_candidates_persona_override(test_client, monkeypatch):
    analytics_mock = AsyncMock()
    monkeypatch.setattr("app.analytics.track_persona_event", analytics_mock)
    with respx.mock(assert_all_called=False) as mock:
        _register_rank_mocks(mock)
        response = await test_client.post(
            "/v1/rank",
            json={"disease": "asthma", "persona": "mechanism-first"},
        )

    assert response.status_code == 200
    payload = response.json()
    scoring = payload["scoring"]
    assert scoring["persona"] == "mechanism-first"
    assert scoring["overrides"] == {}
    assert scoring["weights"]["mechanism"] > scoring["weights"]["clinical"]
    assert payload["candidates"][0]["narrative"]["summary"]
    analytics_mock.assert_awaited()


@pytest.mark.asyncio
async def test_rank_candidates_custom_weights(test_client, monkeypatch):
    analytics_mock = AsyncMock()
    monkeypatch.setattr("app.analytics.track_persona_event", analytics_mock)
    with respx.mock(assert_all_called=False) as mock:
        _register_rank_mocks(mock)
        response = await test_client.post(
            "/v1/rank",
            json={
                "disease": "asthma",
                "weights": {"mechanism": 0.8, "safety": 0.05},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    scoring = payload["scoring"]
    assert scoring["persona"] == "custom"
    assert "mechanism" in scoring["overrides"]
    assert scoring["weights"]["mechanism"] > scoring["weights"]["signature"]
    assert payload["candidates"][0]["narrative"]["summary"]
    analytics_mock.assert_awaited()


@pytest.mark.asyncio
async def test_scoring_metadata_endpoint(test_client):
    response = await test_client.get("/v1/metadata/scoring")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_persona"] == "balanced"
    assert payload["personas"]
    persona_names = {item["name"] for item in payload["personas"]}
    assert {"balanced", "mechanism-first", "clinical-first"} <= persona_names





