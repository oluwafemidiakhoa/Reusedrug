"""Node2Vec embeddings for knowledge graph nodes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Optional node2vec import
try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    logger.warning("node2vec package not available. Using fallback implementation.")
    NODE2VEC_AVAILABLE = False


class Node2VecConfig(BaseModel):
    """Configuration for Node2Vec embeddings."""

    dimensions: int = Field(default=128, description="Embedding dimensionality")
    walk_length: int = Field(default=80, description="Length of each random walk")
    num_walks: int = Field(default=10, description="Number of walks per node")
    p: float = Field(default=1.0, description="Return parameter")
    q: float = Field(default=1.0, description="In-out parameter")
    window: int = Field(default=10, description="Context window size")
    min_count: int = Field(default=1, description="Minimum word count")
    workers: int = Field(default=4, description="Number of worker threads")


class NodeEmbedding(BaseModel):
    """Embedding for a single node."""

    node_id: str
    node_type: str  # 'drug', 'disease', 'gene', 'pathway'
    embedding: List[float]
    neighbors: List[str] = Field(default_factory=list)


class Node2VecEmbeddings:
    """Generate and manage Node2Vec embeddings for knowledge graph."""

    def __init__(self, config: Optional[Node2VecConfig] = None):
        """Initialize Node2Vec embeddings.

        Args:
            config: Node2Vec configuration
        """
        self.config = config or Node2VecConfig()
        self.graph: Optional[nx.Graph] = None
        self.embeddings: Dict[str, np.ndarray] = {}
        self.node_types: Dict[str, str] = {}
        self.enabled = NODE2VEC_AVAILABLE

    def build_graph_from_edges(
        self,
        edges: List[Tuple[str, str, str]],
        node_types: Optional[Dict[str, str]] = None,
    ) -> nx.Graph:
        """Build NetworkX graph from edge list.

        Args:
            edges: List of (source, relation, target) tuples
            node_types: Optional mapping of node_id -> node_type

        Returns:
            NetworkX graph
        """
        self.graph = nx.Graph()
        self.node_types = node_types or {}

        for source, relation, target in edges:
            self.graph.add_edge(source, target, relation=relation)

            # Infer node types if not provided
            if source not in self.node_types:
                self.node_types[source] = self._infer_node_type(source)
            if target not in self.node_types:
                self.node_types[target] = self._infer_node_type(target)

        logger.info(
            f"Built graph with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )

        return self.graph

    def _infer_node_type(self, node_id: str) -> str:
        """Infer node type from ID prefix.

        Args:
            node_id: Node identifier

        Returns:
            Node type string
        """
        if node_id.startswith("CHEMBL") or node_id.startswith("DB"):
            return "drug"
        elif node_id.startswith("MONDO") or node_id.startswith("DOID"):
            return "disease"
        elif node_id.startswith("HGNC") or len(node_id) < 10:
            return "gene"
        elif node_id.startswith("KEGG") or node_id.startswith("GO"):
            return "pathway"
        else:
            return "unknown"

    def train_embeddings(self) -> Dict[str, np.ndarray]:
        """Train Node2Vec embeddings on the graph.

        Returns:
            Dictionary mapping node_id -> embedding vector
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph_from_edges first.")

        if not self.enabled:
            logger.warning("Node2Vec not available, using random embeddings")
            return self._generate_random_embeddings()

        logger.info("Training Node2Vec embeddings...")

        try:
            # Initialize Node2Vec
            node2vec = Node2Vec(
                self.graph,
                dimensions=self.config.dimensions,
                walk_length=self.config.walk_length,
                num_walks=self.config.num_walks,
                p=self.config.p,
                q=self.config.q,
                workers=self.config.workers,
                quiet=True,
            )

            # Train embeddings
            model = node2vec.fit(
                window=self.config.window,
                min_count=self.config.min_count,
                batch_words=4,
            )

            # Extract embeddings
            for node in self.graph.nodes():
                self.embeddings[node] = model.wv[node]

            logger.info(f"Trained embeddings for {len(self.embeddings)} nodes")

        except Exception as e:
            logger.error(f"Node2Vec training failed: {e}. Using random embeddings.")
            return self._generate_random_embeddings()

        return self.embeddings

    def _generate_random_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate random embeddings as fallback.

        Returns:
            Dictionary of random embeddings
        """
        if self.graph is None:
            return {}

        np.random.seed(42)
        for node in self.graph.nodes():
            self.embeddings[node] = np.random.randn(self.config.dimensions)

        return self.embeddings

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific node.

        Args:
            node_id: Node identifier

        Returns:
            Embedding vector or None if not found
        """
        return self.embeddings.get(node_id)

    def get_node_embedding_obj(self, node_id: str) -> Optional[NodeEmbedding]:
        """Get NodeEmbedding object for a node.

        Args:
            node_id: Node identifier

        Returns:
            NodeEmbedding object or None
        """
        embedding = self.get_embedding(node_id)
        if embedding is None:
            return None

        neighbors = []
        if self.graph and node_id in self.graph:
            neighbors = list(self.graph.neighbors(node_id))

        return NodeEmbedding(
            node_id=node_id,
            node_type=self.node_types.get(node_id, "unknown"),
            embedding=embedding.tolist(),
            neighbors=neighbors,
        )

    def compute_similarity(
        self,
        node1: str,
        node2: str,
        method: str = "cosine",
    ) -> float:
        """Compute similarity between two nodes using embeddings.

        Args:
            node1: First node ID
            node2: Second node ID
            method: Similarity method ('cosine', 'euclidean', 'dot')

        Returns:
            Similarity score
        """
        emb1 = self.get_embedding(node1)
        emb2 = self.get_embedding(node2)

        if emb1 is None or emb2 is None:
            return 0.0

        if method == "cosine":
            # Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            return float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0

        elif method == "euclidean":
            # Euclidean distance (inverted and normalized)
            distance = np.linalg.norm(emb1 - emb2)
            return float(1.0 / (1.0 + distance))

        elif method == "dot":
            # Dot product
            return float(np.dot(emb1, emb2))

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def find_similar_nodes(
        self,
        node_id: str,
        top_k: int = 10,
        node_type_filter: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Find most similar nodes to a query node.

        Args:
            node_id: Query node ID
            top_k: Number of results
            node_type_filter: Optional filter by node type

        Returns:
            List of (node_id, similarity_score) tuples
        """
        query_emb = self.get_embedding(node_id)
        if query_emb is None:
            return []

        similarities = []
        for other_id, other_emb in self.embeddings.items():
            if other_id == node_id:
                continue

            # Apply type filter
            if node_type_filter:
                other_type = self.node_types.get(other_id, "unknown")
                if other_type != node_type_filter:
                    continue

            sim = self.compute_similarity(node_id, other_id)
            similarities.append((other_id, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def save_embeddings(self, path: Path) -> None:
        """Save embeddings to disk.

        Args:
            path: Path to save file
        """
        np.savez(
            path,
            embeddings=self.embeddings,
            node_types=self.node_types,
            config=self.config.dict(),
        )
        logger.info(f"Saved embeddings to {path}")

    def load_embeddings(self, path: Path) -> None:
        """Load embeddings from disk.

        Args:
            path: Path to load file
        """
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"].item()
        self.node_types = data["node_types"].item()
        logger.info(f"Loaded {len(self.embeddings)} embeddings from {path}")

    def visualize_embeddings_2d(
        self,
        method: str = "tsne",
        perplexity: int = 30,
    ) -> Dict[str, Tuple[float, float]]:
        """Reduce embeddings to 2D for visualization.

        Args:
            method: Dimensionality reduction method ('tsne', 'pca', 'umap')
            perplexity: t-SNE perplexity parameter

        Returns:
            Dictionary mapping node_id -> (x, y) coordinates
        """
        if not self.embeddings:
            return {}

        # Stack embeddings into matrix
        node_ids = list(self.embeddings.keys())
        X = np.vstack([self.embeddings[nid] for nid in node_ids])

        # Reduce dimensionality
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        elif method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                logger.warning("UMAP not available, falling back to PCA")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        X_2d = reducer.fit_transform(X)

        # Map back to node IDs
        coordinates = {}
        for i, node_id in enumerate(node_ids):
            coordinates[node_id] = (float(X_2d[i, 0]), float(X_2d[i, 1]))

        return coordinates


# Singleton instance
_node2vec_embeddings: Optional[Node2VecEmbeddings] = None


def get_node2vec_embeddings(
    config: Optional[Node2VecConfig] = None,
) -> Node2VecEmbeddings:
    """Get or create singleton Node2VecEmbeddings instance."""
    global _node2vec_embeddings
    if _node2vec_embeddings is None:
        _node2vec_embeddings = Node2VecEmbeddings(config)
    return _node2vec_embeddings
