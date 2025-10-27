import pytest

from app.services.graph import GraphInsights
from app.services.scoring import score_candidate


def test_score_candidate_is_clamped():
    score = score_candidate(
        "Example",
        activity={"potency": "150"},
        target_count=5,
        trials=[{"phase": "Phase 4", "status": "Completed"}],
        adverse_events=[{"reaction": "Headache", "count": 1000}],
    )
    assert 0.0 <= score.final_score <= 1.0
    assert score.safety_penalty <= 0
    assert score.confidence_low is not None
    assert score.confidence_high is not None


@pytest.mark.parametrize(
    "weights",
    [
        {"SCORE_WEIGHT_MECHANISM": "1.0", "SCORE_WEIGHT_NETWORK": "0", "SCORE_WEIGHT_SIGNATURE": "0", "SCORE_WEIGHT_CLINICAL": "0", "SCORE_WEIGHT_SAFETY": "0"},
        {"SCORE_WEIGHT_MECHANISM": "0.2", "SCORE_WEIGHT_NETWORK": "0.4", "SCORE_WEIGHT_SIGNATURE": "0.2", "SCORE_WEIGHT_CLINICAL": "0.1", "SCORE_WEIGHT_SAFETY": "0.1"},
    ],
)
def test_score_candidate_respects_weight_env(monkeypatch, weights):
    for key, value in weights.items():
        monkeypatch.setenv(key, value)

    score = score_candidate(
        "Configurable",
        activity={"potency": "100"},
        target_count=3,
        trials=[{"phase": "Phase 3", "status": "Completed"}],
        translator_paths=[{"edge_bindings": {"e1": [{"subject": "A", "object": "B"}]}}],
    )
    assert 0.0 <= score.final_score <= 1.0
    assert score.confidence_low <= score.final_score <= score.confidence_high


def test_score_candidate_uses_graph_metrics(monkeypatch):
    metrics = GraphInsights(
        node_count=4,
        edge_count=5,
        density=0.6,
        average_shortest_path=2.0,
        top_central_nodes=["NODE1"],
        nodes=[
            {"id": "NODE1", "label": "Node 1", "category": "gene"},
            {"id": "NODE2", "label": "Node 2", "category": "disease"},
        ],
        edges=[
            {"id": "edge-1", "source": "NODE1", "target": "NODE2", "predicate": "rel"},
        ],
    )
    baseline = score_candidate(
        "Graphy",
        target_count=1,
        translator_paths=[{"edge_bindings": {"e1": [{"subject": "A", "object": "B"}]}}],
    )
    boosted = score_candidate(
        "Graphy",
        target_count=1,
        translator_paths=[{"edge_bindings": {"e1": [{"subject": "A", "object": "B"}]}}],
        graph_metrics=metrics,
    )
    assert boosted.network_proximity >= baseline.network_proximity
