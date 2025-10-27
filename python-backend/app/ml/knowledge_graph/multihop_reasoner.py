"""Multi-hop reasoning for knowledge graph path finding and explanation."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReasoningPath(BaseModel):
    """A reasoning path through the knowledge graph."""

    nodes: List[str] = Field(..., description="Sequence of nodes")
    relations: List[str] = Field(..., description="Sequence of relations")
    score: float = Field(..., description="Path score/confidence")
    length: int = Field(..., description="Number of hops")
    explanation: str = Field(default="", description="Human-readable explanation")


class PathPattern(BaseModel):
    """A common path pattern in the knowledge graph."""

    pattern: List[str] = Field(..., description="Sequence of relation types")
    frequency: int = Field(..., description="Number of occurrences")
    examples: List[ReasoningPath] = Field(default_factory=list)


class MultiHopReasoner:
    """Multi-hop reasoning engine for knowledge graph traversal."""

    def __init__(self, graph: Optional[nx.Graph] = None):
        """Initialize multi-hop reasoner.

        Args:
            graph: NetworkX graph to reason over
        """
        self.graph = graph or nx.Graph()
        self.path_cache: Dict[Tuple[str, str], List[ReasoningPath]] = {}

    def set_graph(self, graph: nx.Graph) -> None:
        """Set the graph to reason over.

        Args:
            graph: NetworkX graph
        """
        self.graph = graph
        self.path_cache.clear()
        logger.info(f"Set graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    def find_all_paths(
        self,
        source: str,
        target: str,
        max_length: int = 4,
        cutoff: Optional[int] = None,
    ) -> List[ReasoningPath]:
        """Find all paths between source and target.

        Args:
            source: Source node
            target: Target node
            max_length: Maximum path length (number of hops)
            cutoff: Maximum number of paths to return

        Returns:
            List of reasoning paths
        """
        cache_key = (source, target)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]

        if source not in self.graph or target not in self.graph:
            return []

        reasoning_paths = []

        try:
            # Find all simple paths
            all_paths = nx.all_simple_paths(
                self.graph,
                source,
                target,
                cutoff=max_length,
            )

            for path_nodes in all_paths:
                if len(path_nodes) - 1 > max_length:
                    continue

                # Extract relations
                relations = []
                for i in range(len(path_nodes) - 1):
                    edge_data = self.graph.get_edge_data(path_nodes[i], path_nodes[i+1])
                    if edge_data:
                        relation = edge_data.get("relation", "unknown")
                        relations.append(relation)
                    else:
                        relations.append("unknown")

                # Compute path score
                score = self._score_path(path_nodes, relations)

                # Generate explanation
                explanation = self._generate_path_explanation(path_nodes, relations)

                reasoning_path = ReasoningPath(
                    nodes=path_nodes,
                    relations=relations,
                    score=score,
                    length=len(path_nodes) - 1,
                    explanation=explanation,
                )

                reasoning_paths.append(reasoning_path)

                # Check cutoff
                if cutoff and len(reasoning_paths) >= cutoff:
                    break

        except nx.NetworkXNoPath:
            pass
        except Exception as e:
            logger.error(f"Error finding paths: {e}")

        # Sort by score descending
        reasoning_paths.sort(key=lambda p: p.score, reverse=True)

        # Cache results
        self.path_cache[cache_key] = reasoning_paths

        return reasoning_paths

    def _score_path(
        self,
        nodes: List[str],
        relations: List[str],
    ) -> float:
        """Compute score for a reasoning path.

        Scoring factors:
        - Shorter paths are better
        - Paths with higher edge weights are better
        - Prefer certain relation types

        Args:
            nodes: Path nodes
            relations: Path relations

        Returns:
            Path score (higher is better)
        """
        # Length penalty (prefer shorter paths)
        length_score = 1.0 / len(nodes)

        # Edge confidence score
        confidence_scores = []
        for i in range(len(nodes) - 1):
            edge_data = self.graph.get_edge_data(nodes[i], nodes[i+1])
            if edge_data:
                confidence = edge_data.get("confidence", 1.0)
                confidence_scores.append(confidence)

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

        # Relation type bonus
        relation_bonus = 0.0
        preferred_relations = {"treats", "indicates", "targets", "associated_with"}
        for rel in relations:
            if rel in preferred_relations:
                relation_bonus += 0.1

        # Combined score
        score = length_score * 0.4 + avg_confidence * 0.4 + relation_bonus * 0.2

        return float(score)

    def _generate_path_explanation(
        self,
        nodes: List[str],
        relations: List[str],
    ) -> str:
        """Generate human-readable explanation for a path.

        Args:
            nodes: Path nodes
            relations: Path relations

        Returns:
            Explanation string
        """
        parts = []
        for i in range(len(relations)):
            source = nodes[i]
            relation = relations[i]
            target = nodes[i+1]

            # Clean up node IDs for display
            source_display = source.split(":")[-1] if ":" in source else source
            target_display = target.split(":")[-1] if ":" in target else target

            parts.append(f"{source_display} --[{relation}]--> {target_display}")

        return " -> ".join(parts)

    def find_shortest_path(
        self,
        source: str,
        target: str,
    ) -> Optional[ReasoningPath]:
        """Find shortest path between source and target.

        Args:
            source: Source node
            target: Target node

        Returns:
            Shortest reasoning path or None
        """
        if source not in self.graph or target not in self.graph:
            return None

        try:
            path_nodes = nx.shortest_path(self.graph, source, target)

            # Extract relations
            relations = []
            for i in range(len(path_nodes) - 1):
                edge_data = self.graph.get_edge_data(path_nodes[i], path_nodes[i+1])
                if edge_data:
                    relation = edge_data.get("relation", "unknown")
                    relations.append(relation)
                else:
                    relations.append("unknown")

            score = self._score_path(path_nodes, relations)
            explanation = self._generate_path_explanation(path_nodes, relations)

            return ReasoningPath(
                nodes=path_nodes,
                relations=relations,
                score=score,
                length=len(path_nodes) - 1,
                explanation=explanation,
            )

        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            logger.error(f"Error finding shortest path: {e}")
            return None

    def find_connecting_entities(
        self,
        source: str,
        target: str,
        max_paths: int = 5,
    ) -> List[str]:
        """Find entities that connect source and target.

        Args:
            source: Source node
            target: Target node
            max_paths: Maximum number of paths to consider

        Returns:
            List of connecting entity IDs
        """
        paths = self.find_all_paths(source, target, cutoff=max_paths)

        # Extract intermediate nodes
        connecting_nodes: Set[str] = set()
        for path in paths:
            # Exclude source and target
            intermediate = path.nodes[1:-1]
            connecting_nodes.update(intermediate)

        return sorted(connecting_nodes)

    def discover_path_patterns(
        self,
        min_frequency: int = 2,
        max_length: int = 3,
    ) -> List[PathPattern]:
        """Discover common path patterns in the graph.

        Args:
            min_frequency: Minimum pattern frequency
            max_length: Maximum pattern length

        Returns:
            List of discovered patterns
        """
        pattern_counts: Dict[Tuple[str, ...], List[List[str]]] = {}

        # Sample node pairs
        nodes = list(self.graph.nodes())
        sample_size = min(100, len(nodes))

        import random
        random.seed(42)
        sampled_pairs = random.sample(
            [(s, t) for s in nodes for t in nodes if s != t],
            min(sample_size * 10, len(nodes) ** 2),
        )

        # Find paths and extract patterns
        for source, target in sampled_pairs:
            paths = self.find_all_paths(source, target, max_length=max_length, cutoff=3)

            for path in paths:
                pattern = tuple(path.relations)
                if pattern not in pattern_counts:
                    pattern_counts[pattern] = []
                pattern_counts[pattern].append(path.nodes)

        # Filter by frequency and create PathPattern objects
        patterns = []
        for pattern_tuple, node_lists in pattern_counts.items():
            if len(node_lists) >= min_frequency:
                # Create example paths
                examples = []
                for nodes in node_lists[:3]:  # Top 3 examples
                    relations = list(pattern_tuple)
                    score = self._score_path(nodes, relations)
                    explanation = self._generate_path_explanation(nodes, relations)

                    examples.append(
                        ReasoningPath(
                            nodes=nodes,
                            relations=relations,
                            score=score,
                            length=len(nodes) - 1,
                            explanation=explanation,
                        )
                    )

                patterns.append(
                    PathPattern(
                        pattern=list(pattern_tuple),
                        frequency=len(node_lists),
                        examples=examples,
                    )
                )

        # Sort by frequency descending
        patterns.sort(key=lambda p: p.frequency, reverse=True)

        logger.info(f"Discovered {len(patterns)} path patterns")
        return patterns

    def explain_prediction(
        self,
        drug_id: str,
        disease_id: str,
        max_paths: int = 3,
    ) -> Dict[str, any]:
        """Explain a drug-disease prediction using paths.

        Args:
            drug_id: Drug identifier
            disease_id: Disease identifier
            max_paths: Number of paths to include

        Returns:
            Explanation dictionary
        """
        paths = self.find_all_paths(drug_id, disease_id, max_length=4, cutoff=max_paths)

        if not paths:
            return {
                "drug_id": drug_id,
                "disease_id": disease_id,
                "has_paths": False,
                "message": "No connecting paths found",
            }

        # Get connecting entities
        connecting = self.find_connecting_entities(drug_id, disease_id, max_paths=max_paths)

        # Analyze path types
        relation_counts: Dict[str, int] = {}
        for path in paths:
            for rel in path.relations:
                relation_counts[rel] = relation_counts.get(rel, 0) + 1

        return {
            "drug_id": drug_id,
            "disease_id": disease_id,
            "has_paths": True,
            "num_paths": len(paths),
            "shortest_path_length": min(p.length for p in paths),
            "best_path_score": max(p.score for p in paths),
            "paths": [
                {
                    "nodes": p.nodes,
                    "relations": p.relations,
                    "score": p.score,
                    "explanation": p.explanation,
                }
                for p in paths[:max_paths]
            ],
            "connecting_entities": connecting,
            "relation_usage": relation_counts,
        }

    def compute_node_centrality(
        self,
        method: str = "betweenness",
    ) -> Dict[str, float]:
        """Compute centrality scores for all nodes.

        Args:
            method: Centrality method ('betweenness', 'closeness', 'degree', 'pagerank')

        Returns:
            Dictionary mapping node_id -> centrality score
        """
        if method == "betweenness":
            return nx.betweenness_centrality(self.graph)
        elif method == "closeness":
            return nx.closeness_centrality(self.graph)
        elif method == "degree":
            return dict(nx.degree_centrality(self.graph))
        elif method == "pagerank":
            return nx.pagerank(self.graph)
        else:
            raise ValueError(f"Unknown centrality method: {method}")


# Singleton instance
_multihop_reasoner: Optional[MultiHopReasoner] = None


def get_multihop_reasoner(graph: Optional[nx.Graph] = None) -> MultiHopReasoner:
    """Get or create singleton MultiHopReasoner instance."""
    global _multihop_reasoner
    if _multihop_reasoner is None:
        _multihop_reasoner = MultiHopReasoner(graph)
    elif graph is not None:
        _multihop_reasoner.set_graph(graph)
    return _multihop_reasoner
