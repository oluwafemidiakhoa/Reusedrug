import pytest

try:
    import networkx as nx  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    nx = None

from app.services.graph import analyze_translator_paths


def test_analyze_translator_paths_returns_none_without_data():
    assert analyze_translator_paths(None) is None
    assert analyze_translator_paths([]) is None


@pytest.mark.skipif(nx is None, reason="networkx required")
def test_analyze_translator_paths_builds_graph():
    paths = [
        {
            "edge_bindings": {
                "edge1": [
                    {"subject": {"id": "CHEMBL:1"}, "object": {"id": "MONDO:1"}},
                    {"subject": {"id": "MONDO:1"}, "object": {"id": "HGNC:1"}},
                ]
            }
        }
    ]
    insights = analyze_translator_paths(paths)
    assert insights is not None
    assert insights.node_count >= 2
    assert insights.graph is not None
    assert len(insights.graph.get("nodes", [])) >= 2
    assert len(insights.graph.get("edges", [])) >= 1
    summary = insights.summary()
    assert "nodes" in summary and "edges" in summary
