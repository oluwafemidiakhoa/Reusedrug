"""Graph Neural Network for drug-disease link prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.ml.config import MLConfig
from app.ml.models.base import BasePredictor, PredictionResult

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch and PyTorch Geometric not available. GNN predictor disabled.")


if TORCH_AVAILABLE:

    class GNNEncoder(nn.Module):
        """Graph Neural Network encoder using Graph Convolutional layers."""

        def __init__(
            self,
            num_nodes: int,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            num_layers: int = 3,
            dropout: float = 0.2,
            conv_type: str = "gcn",
        ):
            super().__init__()
            self.num_nodes = num_nodes
            self.embedding_dim = embedding_dim
            self.dropout = dropout

            # Node embeddings
            self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

            # Graph convolution layers
            self.convs = nn.ModuleList()
            in_channels = embedding_dim

            for i in range(num_layers):
                out_channels = hidden_dim if i < num_layers - 1 else embedding_dim
                if conv_type == "gcn":
                    self.convs.append(GCNConv(in_channels, out_channels))
                elif conv_type == "sage":
                    self.convs.append(SAGEConv(in_channels, out_channels))
                elif conv_type == "gat":
                    self.convs.append(GATConv(in_channels, out_channels))
                else:
                    raise ValueError(f"Unknown conv_type: {conv_type}")
                in_channels = out_channels

            self.reset_parameters()

        def reset_parameters(self):
            """Initialize parameters."""
            nn.init.xavier_uniform_(self.node_embedding.weight)
            for conv in self.convs:
                conv.reset_parameters()

        def forward(self, node_ids, edge_index):
            """Forward pass.

            Args:
                node_ids: Tensor of node IDs
                edge_index: Edge index tensor [2, num_edges]

            Returns:
                Node embeddings
            """
            x = self.node_embedding(node_ids)

            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

            return x


    class LinkPredictor(nn.Module):
        """Link prediction head for drug-disease associations."""

        def __init__(self, embedding_dim: int = 128):
            super().__init__()
            self.embedding_dim = embedding_dim

            # MLP for link prediction
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, drug_emb, disease_emb):
            """Predict link probability.

            Args:
                drug_emb: Drug embeddings [batch_size, embedding_dim]
                disease_emb: Disease embeddings [batch_size, embedding_dim]

            Returns:
                Link probabilities [batch_size, 1]
            """
            # Concatenate drug and disease embeddings
            x = torch.cat([drug_emb, disease_emb], dim=-1)
            return self.mlp(x)


class GNNPredictor(BasePredictor):
    """GNN-based predictor for drug-disease associations."""

    def __init__(
        self,
        model_name: str = "gnn_predictor",
        device: Optional[str] = None,
    ):
        super().__init__(model_name)

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and PyTorch Geometric required for GNN predictor. "
                "Install with: pip install torch torch-geometric"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder: Optional[GNNEncoder] = None
        self.link_predictor: Optional[LinkPredictor] = None
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        self.edge_index: Optional[torch.Tensor] = None

        logger.info(f"GNN Predictor initialized on device: {self.device}")

    def initialize_model(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        node_to_idx: Dict[str, int],
    ) -> None:
        """Initialize the GNN model.

        Args:
            num_nodes: Total number of nodes in the graph
            edge_index: Edge connectivity [2, num_edges]
            node_to_idx: Mapping from node IDs to indices
        """
        self.node_to_idx = node_to_idx
        self.idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}
        self.edge_index = edge_index.to(self.device)

        # Create encoder
        self.encoder = GNNEncoder(
            num_nodes=num_nodes,
            embedding_dim=MLConfig.GNN_EMBEDDING_DIM,
            hidden_dim=MLConfig.GNN_HIDDEN_DIM,
            num_layers=MLConfig.GNN_NUM_LAYERS,
            dropout=MLConfig.GNN_DROPOUT,
        ).to(self.device)

        # Create link predictor
        self.link_predictor = LinkPredictor(
            embedding_dim=MLConfig.GNN_EMBEDDING_DIM
        ).to(self.device)

        self.is_trained = False
        logger.info(f"GNN model initialized with {num_nodes} nodes")

    def predict(
        self,
        drug_id: str,
        disease_id: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """Make a prediction for a drug-disease pair."""
        if not self.is_trained:
            logger.warning("Model not trained yet, returning default prediction")
            return PredictionResult(
                drug_id=drug_id,
                disease_id=disease_id,
                score=0.5,
                model_name=self.model_name,
                metadata={"status": "untrained"},
            )

        # Get node indices
        drug_idx = self.node_to_idx.get(drug_id)
        disease_idx = self.node_to_idx.get(disease_id)

        if drug_idx is None or disease_idx is None:
            logger.warning(f"Node not found: drug={drug_id}, disease={disease_id}")
            return PredictionResult(
                drug_id=drug_id,
                disease_id=disease_id,
                score=0.0,
                model_name=self.model_name,
                metadata={"status": "node_not_found"},
            )

        self.encoder.eval()
        self.link_predictor.eval()

        with torch.no_grad():
            # Get all node embeddings
            all_node_ids = torch.arange(len(self.node_to_idx), device=self.device)
            embeddings = self.encoder(all_node_ids, self.edge_index)

            # Get specific embeddings
            drug_emb = embeddings[drug_idx].unsqueeze(0)
            disease_emb = embeddings[disease_idx].unsqueeze(0)

            # Predict
            score = self.link_predictor(drug_emb, disease_emb).item()

        return PredictionResult(
            drug_id=drug_id,
            disease_id=disease_id,
            score=score,
            confidence_low=max(0.0, score - 0.1),
            confidence_high=min(1.0, score + 0.1),
            model_name=self.model_name,
            features_used=["graph_structure", "node_embeddings"],
            metadata={"device": self.device},
        )

    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
        features: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PredictionResult]:
        """Make predictions for multiple drug-disease pairs."""
        results = []
        for drug_id, disease_id in pairs:
            result = self.predict(drug_id, disease_id)
            results.append(result)
        return results

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if not self.is_trained:
            logger.warning("Attempting to save untrained model")

        save_dict = {
            "encoder_state": self.encoder.state_dict() if self.encoder else None,
            "link_predictor_state": (
                self.link_predictor.state_dict() if self.link_predictor else None
            ),
            "node_to_idx": self.node_to_idx,
            "is_trained": self.is_trained,
            "config": {
                "embedding_dim": MLConfig.GNN_EMBEDDING_DIM,
                "hidden_dim": MLConfig.GNN_HIDDEN_DIM,
                "num_layers": MLConfig.GNN_NUM_LAYERS,
                "dropout": MLConfig.GNN_DROPOUT,
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """Load model from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.node_to_idx = checkpoint["node_to_idx"]
        self.idx_to_node = {idx: node_id for node_id, idx in self.node_to_idx.items()}
        self.is_trained = checkpoint["is_trained"]

        # Recreate models
        num_nodes = len(self.node_to_idx)
        config = checkpoint.get("config", {})

        self.encoder = GNNEncoder(
            num_nodes=num_nodes,
            embedding_dim=config.get("embedding_dim", MLConfig.GNN_EMBEDDING_DIM),
            hidden_dim=config.get("hidden_dim", MLConfig.GNN_HIDDEN_DIM),
            num_layers=config.get("num_layers", MLConfig.GNN_NUM_LAYERS),
            dropout=config.get("dropout", MLConfig.GNN_DROPOUT),
        ).to(self.device)

        self.link_predictor = LinkPredictor(
            embedding_dim=config.get("embedding_dim", MLConfig.GNN_EMBEDDING_DIM)
        ).to(self.device)

        # Load state
        if checkpoint["encoder_state"]:
            self.encoder.load_state_dict(checkpoint["encoder_state"])
        if checkpoint["link_predictor_state"]:
            self.link_predictor.load_state_dict(checkpoint["link_predictor_state"])

        logger.info(f"Model loaded from {path}")


# Fallback predictor when PyTorch is not available
class DummyGNNPredictor(BasePredictor):
    """Dummy predictor when PyTorch is not available."""

    def __init__(self, model_name: str = "gnn_predictor"):
        super().__init__(model_name)
        logger.warning("Using dummy GNN predictor - PyTorch not available")

    def predict(
        self, drug_id: str, disease_id: str, features: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        return PredictionResult(
            drug_id=drug_id,
            disease_id=disease_id,
            score=0.5,
            model_name=self.model_name,
            metadata={"status": "pytorch_not_available"},
        )

    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
        features: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PredictionResult]:
        return [self.predict(d, dis) for d, dis in pairs]

    def save(self, path: Path) -> None:
        logger.warning("Cannot save dummy predictor")

    def load(self, path: Path) -> None:
        logger.warning("Cannot load dummy predictor")


# Export the appropriate predictor
if TORCH_AVAILABLE:
    __all__ = ["GNNPredictor", "GNNEncoder", "LinkPredictor"]
else:
    GNNPredictor = DummyGNNPredictor
    __all__ = ["GNNPredictor"]
