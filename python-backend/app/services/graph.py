from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, List

try:
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    nx = None  # type: ignore[assignment]


@dataclass
class GraphInsights:
    node_count: int
    edge_count: int
    density: float
    average_shortest_path: Optional[float]
    top_central_nodes: list[str]
    nodes: list[dict[str, Optional[str]]]
    edges: list[dict[str, Optional[str]]]

    def summary(self) -> str:
        central = ", ".join(self.top_central_nodes[:3])
        avg = f", avg path {self.average_shortest_path:.2f}" if self.average_shortest_path else ""
        return (
            f"{self.node_count} nodes/{self.edge_count} edges; "
            f"density {self.density:.2f}{avg}; central nodes: {central or 'n/a'}"
        )


def _normalize_entity(entity: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if isinstance(entity, dict):
        identifier = (
            entity.get("id")
            or entity.get("identifier")
            or entity.get("curie")
            or entity.get("name")
        )
        label = entity.get("name") or entity.get("label") or identifier
        category = (
            entity.get("type")
            or entity.get("category")
            or entity.get("categories")
        )
        if isinstance(category, list):
            category = category[0]
        return identifier, label, category
    if entity:
        identifier = str(entity)
        return identifier, identifier, None
    return None, None, None


def _build_graph(paths: Iterable[Dict[str, Any]]):
    if nx is None:
        return None, [], []
    graph = nx.MultiDiGraph()
    node_meta: Dict[str, Dict[str, Optional[str]]] = {}
    edge_records: List[Dict[str, Optional[str]]] = []
    for path in paths or []:
        if not isinstance(path, dict):
            continue
        for edge_key, bindings in path.items():
            edges = bindings if isinstance(bindings, list) else []
            for binding in edges:
                subj_info = binding.get("subject")
                obj_info = binding.get("object")
                subj_id, subj_label, subj_category = _normalize_entity(subj_info)
                obj_id, obj_label, obj_category = _normalize_entity(obj_info)
                if subj_id and obj_id:
                    graph.add_edge(subj_id, obj_id, predicate=edge_key)
                    if subj_id not in node_meta:
                        node_meta[subj_id] = {
                            "id": subj_id,
                            "label": subj_label or subj_id,
                            "category": subj_category,
                        }
                    if obj_id not in node_meta:
                        node_meta[obj_id] = {
                            "id": obj_id,
                            "label": obj_label or obj_id,
                            "category": obj_category,
                        }
                    edge_records.append(
                        {
                            "id": binding.get("id") or f"{subj_id}-{obj_id}-{edge_key}",
                            "source": subj_id,
                            "target": obj_id,
                            "predicate": binding.get("predicate") or binding.get("predicate_id") or edge_key,
                        }
                    )
    return graph, list(node_meta.values()), edge_records


def analyze_translator_paths(paths: Iterable[Dict[str, Any]] | None) -> Optional[GraphInsights]:
    if not paths:
        return None
    graph, nodes, edges = _build_graph(paths)
    if graph is None or graph.number_of_nodes() == 0:
        return None

    undirected = graph.to_undirected()
    node_count = undirected.number_of_nodes()
    edge_count = undirected.number_of_edges()
    density = nx.density(undirected) if node_count > 1 else 0.0
    avg_shortest = None
    try:
        if nx.is_connected(undirected):
            avg_shortest = nx.average_shortest_path_length(undirected)
    except Exception:  # pragma: no cover - safety
        avg_shortest = None

    try:
        centrality = nx.degree_centrality(undirected)
    except Exception:  # pragma: no cover
        centrality = {}
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:5]

    return GraphInsights(
        node_count=node_count,
        edge_count=edge_count,
        density=density,
        average_shortest_path=avg_shortest,
        top_central_nodes=top_nodes,
        nodes=nodes,
        edges=edges,
    )
