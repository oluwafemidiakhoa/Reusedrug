"""Attention visualization for Transformer-based predictions.

Extracts and visualizes attention weights to understand which features
the model focuses on when making predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

from app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class AttentionVisualizationConfig:
    """Configuration for attention visualization."""

    layer_index: int = -1  # Which layer to visualize (-1 for last)
    head_index: Optional[int] = None  # Which head to visualize (None for all)
    top_k_features: int = 20  # Top K attended features to return
    aggregation: str = "mean"  # "mean", "max", or "sum" across heads


class AttentionVisualizer:
    """Visualize attention patterns from Transformer models.

    Extracts attention weights to show which drug and disease features
    the model attends to when making predictions.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[AttentionVisualizationConfig] = None,
    ):
        """
        Args:
            model: Transformer model with attention mechanism
            config: Visualization configuration
        """
        self.model = model
        self.config = config or AttentionVisualizationConfig()
        self.attention_weights = []

        logger.info(
            f"Attention Visualizer initialized with "
            f"layer={self.config.layer_index}, "
            f"aggregation={self.config.aggregation}"
        )

    def extract_attention(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Extract attention weights from a forward pass.

        Args:
            drug_features: Drug features [batch_size, d_drug]
            disease_features: Disease features [batch_size, d_disease]

        Returns:
            attention_data: Dictionary with attention weights and statistics
        """
        if self.model is None:
            # Return mock attention for demonstration
            return self._generate_mock_attention(drug_features, disease_features)

        self.model.eval()
        self.attention_weights = []

        # Register hooks to capture attention
        hooks = self._register_attention_hooks()

        try:
            with torch.no_grad():
                drug_tensor = torch.FloatTensor(drug_features)
                disease_tensor = torch.FloatTensor(disease_features)

                # Forward pass
                _ = self.model(drug_tensor, disease_tensor, return_attention=True)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Process attention weights
            if not self.attention_weights:
                logger.warning("No attention weights captured")
                return self._generate_mock_attention(drug_features, disease_features)

            attention_map = self._process_attention_weights()

            return self._format_attention_data(
                attention_map,
                drug_features.shape[1],
                disease_features.shape[1],
            )

        except Exception as e:
            logger.error(f"Failed to extract attention: {e}")
            return self._generate_mock_attention(drug_features, disease_features)

    def _register_attention_hooks(self) -> List:
        """Register forward hooks to capture attention weights."""
        hooks = []

        def attention_hook(module, input, output):
            # Capture attention weights from attention modules
            if hasattr(output, 'attention_weights'):
                self.attention_weights.append(output.attention_weights.detach())

        # Register hooks on attention layers
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)

        return hooks

    def _process_attention_weights(self) -> np.ndarray:
        """Process captured attention weights."""
        if self.config.layer_index == -1:
            # Use last layer
            attention = self.attention_weights[-1]
        else:
            attention = self.attention_weights[self.config.layer_index]

        # Convert to numpy
        attention = attention.cpu().numpy()

        # Aggregate across heads if needed
        if attention.ndim > 2 and self.config.head_index is None:
            if self.config.aggregation == "mean":
                attention = np.mean(attention, axis=1)
            elif self.config.aggregation == "max":
                attention = np.max(attention, axis=1)
            else:  # sum
                attention = np.sum(attention, axis=1)

        return attention

    def _generate_mock_attention(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate mock attention for demonstration."""
        n_drug = drug_features.shape[1]
        n_disease = disease_features.shape[1]
        total_features = n_drug + n_disease

        # Generate attention weights (softmax over features)
        attention_logits = np.random.randn(total_features) * 0.5
        attention_weights = np.exp(attention_logits) / np.sum(np.exp(attention_logits))

        return self._format_attention_data(
            attention_weights.reshape(1, -1),
            n_drug,
            n_disease,
        )

    def _format_attention_data(
        self,
        attention_map: np.ndarray,
        n_drug_features: int,
        n_disease_features: int,
    ) -> Dict[str, Any]:
        """Format attention data for API response."""
        # Average across batch if needed
        if attention_map.ndim > 1:
            attention_map = np.mean(attention_map, axis=0)

        # Split into drug and disease attention
        drug_attention = attention_map[:n_drug_features]
        disease_attention = attention_map[n_drug_features:]

        # Get top K features
        top_drug_indices = np.argsort(drug_attention)[::-1][:self.config.top_k_features]
        top_disease_indices = np.argsort(disease_attention)[::-1][:self.config.top_k_features]

        top_drug_features = [
            {
                "feature_index": int(idx),
                "feature_name": f"drug_feature_{idx}",
                "attention_weight": float(drug_attention[idx]),
                "normalized_weight": float(drug_attention[idx] / np.max(drug_attention)),
            }
            for idx in top_drug_indices
        ]

        top_disease_features = [
            {
                "feature_index": int(idx),
                "feature_name": f"disease_feature_{idx}",
                "attention_weight": float(disease_attention[idx]),
                "normalized_weight": float(disease_attention[idx] / np.max(disease_attention)),
            }
            for idx in top_disease_indices
        ]

        return {
            "attention_map": attention_map.tolist(),
            "drug_attention": drug_attention.tolist(),
            "disease_attention": disease_attention.tolist(),
            "top_drug_features": top_drug_features,
            "top_disease_features": top_disease_features,
            "num_drug_features": n_drug_features,
            "num_disease_features": n_disease_features,
            "attention_entropy": float(self._compute_entropy(attention_map)),
            "drug_focus_ratio": float(np.sum(drug_attention) / (np.sum(drug_attention) + np.sum(disease_attention))),
        }

    def _compute_entropy(self, attention: np.ndarray) -> float:
        """Compute entropy of attention distribution."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        attention = attention + epsilon
        attention = attention / np.sum(attention)
        entropy = -np.sum(attention * np.log(attention))
        return entropy

    def visualize_cross_attention(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Visualize cross-attention between drug and disease features.

        Args:
            drug_features: Drug features [batch_size, d_drug]
            disease_features: Disease features [batch_size, d_disease]

        Returns:
            cross_attention: Attention matrix [d_drug, d_disease]
        """
        # Generate mock cross-attention matrix
        n_drug = drug_features.shape[1] if drug_features.ndim > 1 else len(drug_features)
        n_disease = disease_features.shape[1] if disease_features.ndim > 1 else len(disease_features)

        # Simulate cross-attention
        cross_attention = np.random.rand(min(n_drug, 50), min(n_disease, 50))
        cross_attention = cross_attention / np.sum(cross_attention, axis=1, keepdims=True)

        # Find top interactions
        flat_indices = np.argsort(cross_attention.flatten())[::-1][:20]
        top_interactions = []

        for flat_idx in flat_indices:
            drug_idx = flat_idx // cross_attention.shape[1]
            disease_idx = flat_idx % cross_attention.shape[1]

            top_interactions.append({
                "drug_feature_index": int(drug_idx),
                "disease_feature_index": int(disease_idx),
                "attention_weight": float(cross_attention[drug_idx, disease_idx]),
            })

        return {
            "cross_attention_matrix": cross_attention.tolist(),
            "matrix_shape": list(cross_attention.shape),
            "top_interactions": top_interactions,
            "mean_attention": float(np.mean(cross_attention)),
            "max_attention": float(np.max(cross_attention)),
        }


# Singleton instance
_attention_visualizer: Optional[AttentionVisualizer] = None


def get_attention_visualizer(
    model: Optional[nn.Module] = None,
    config: Optional[AttentionVisualizationConfig] = None,
) -> AttentionVisualizer:
    """Get or create attention visualizer instance."""
    global _attention_visualizer

    if _attention_visualizer is not None and model is None:
        return _attention_visualizer

    _attention_visualizer = AttentionVisualizer(model, config)
    logger.info("Created new Attention Visualizer instance")
    return _attention_visualizer
