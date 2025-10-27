"""Counterfactual explanations for drug repurposing predictions.

Generates "what-if" scenarios by finding minimal changes to features
that would flip the prediction outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any

import numpy as np
from scipy.optimize import differential_evolution

from app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual generator."""

    target_class: Optional[int] = None  # Desired outcome (0 or 1), None for opposite
    max_iterations: int = 100  # Maximum optimization iterations
    distance_metric: str = "l2"  # "l1", "l2", or "cosine"
    sparsity_weight: float = 0.1  # Weight for sparsity regularization
    validity_weight: float = 1.0  # Weight for staying in valid feature space
    num_counterfactuals: int = 3  # Number of diverse counterfactuals
    feature_ranges: Optional[Dict[int, tuple]] = None  # Valid ranges per feature


class CounterfactualGenerator:
    """Generate counterfactual explanations for predictions.

    Finds minimal changes to input features that would change the prediction
    to a desired outcome. Useful for answering "what would need to change?"
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        config: Optional[CounterfactualConfig] = None,
    ):
        """
        Args:
            predict_fn: Function that takes (drug_features, disease_features)
                       and returns predictions [0, 1]
            config: Counterfactual configuration
        """
        self.predict_fn = predict_fn
        self.config = config or CounterfactualConfig()

        logger.info(
            f"Counterfactual Generator initialized with "
            f"distance={self.config.distance_metric}, "
            f"max_iter={self.config.max_iterations}"
        )

    def generate(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanation for a prediction.

        Args:
            drug_features: Original drug features [d_drug]
            disease_features: Original disease features [d_disease]
            feature_names: Optional feature names

        Returns:
            counterfactual: Dictionary with counterfactual features and changes
        """
        # Ensure 1D arrays
        if drug_features.ndim > 1:
            drug_features = drug_features.flatten()
        if disease_features.ndim > 1:
            disease_features = disease_features.flatten()

        # Combine features
        original_x = np.concatenate([drug_features, disease_features])

        # Get original prediction
        original_pred = self.predict_fn(
            drug_features.reshape(1, -1),
            disease_features.reshape(1, -1)
        )[0]

        # Determine target
        if self.config.target_class is not None:
            target_pred = float(self.config.target_class)
        else:
            # Flip the prediction
            target_pred = 1.0 if original_pred < 0.5 else 0.0

        logger.info(
            f"Generating counterfactual: {original_pred:.3f} -> {target_pred:.3f}"
        )

        # Generate counterfactuals
        counterfactuals = []
        for seed in range(self.config.num_counterfactuals):
            cf = self._optimize_counterfactual(
                original_x,
                target_pred,
                len(drug_features),
                seed=seed,
            )
            if cf is not None:
                counterfactuals.append(cf)

        if not counterfactuals:
            return {
                "success": False,
                "message": "Failed to generate counterfactual",
                "original_prediction": float(original_pred),
                "target_prediction": float(target_pred),
            }

        # Select the best counterfactual (smallest distance)
        best_cf = min(counterfactuals, key=lambda x: x["distance"])

        # Compute changes
        changes = self._compute_changes(
            original_x,
            best_cf["features"],
            feature_names,
            len(drug_features),
        )

        return {
            "success": True,
            "original_prediction": float(original_pred),
            "target_prediction": float(target_pred),
            "counterfactual_prediction": best_cf["prediction"],
            "distance": best_cf["distance"],
            "num_changes": len(changes["significant_changes"]),
            "changes": changes,
            "all_counterfactuals": counterfactuals,
        }

    def _optimize_counterfactual(
        self,
        original_x: np.ndarray,
        target_pred: float,
        n_drug_features: int,
        seed: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Optimize to find a counterfactual using differential evolution.

        Args:
            original_x: Original features [num_features]
            target_pred: Target prediction value
            n_drug_features: Number of drug features
            seed: Random seed for diversity

        Returns:
            counterfactual: Dict with features, prediction, distance
        """
        num_features = len(original_x)

        # Define objective function
        def objective(x_cf):
            # Split into drug and disease
            drug_cf = x_cf[:n_drug_features].reshape(1, -1)
            disease_cf = x_cf[n_drug_features:].reshape(1, -1)

            # Prediction loss
            pred = self.predict_fn(drug_cf, disease_cf)[0]
            pred_loss = (pred - target_pred) ** 2

            # Distance loss
            if self.config.distance_metric == "l1":
                distance = np.sum(np.abs(x_cf - original_x))
            elif self.config.distance_metric == "l2":
                distance = np.sqrt(np.sum((x_cf - original_x) ** 2))
            else:  # cosine
                distance = 1 - np.dot(x_cf, original_x) / (
                    np.linalg.norm(x_cf) * np.linalg.norm(original_x) + 1e-10
                )

            # Sparsity loss (encourage few changes)
            sparsity = np.sum(np.abs(x_cf - original_x) > 0.01)

            # Combined loss
            total_loss = (
                pred_loss
                + distance
                + self.config.sparsity_weight * sparsity
            )

            return total_loss

        # Define bounds (allow features to vary within reasonable range)
        bounds = []
        for i, val in enumerate(original_x):
            # Allow ±50% variation from original value
            lower = val - 0.5 * (abs(val) + 1)
            upper = val + 0.5 * (abs(val) + 1)
            bounds.append((lower, upper))

        # Optimize using differential evolution
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.config.max_iterations,
                seed=seed,
                atol=1e-3,
                tol=1e-3,
                workers=1,
            )

            x_cf = result.x

            # Get prediction
            drug_cf = x_cf[:n_drug_features].reshape(1, -1)
            disease_cf = x_cf[n_drug_features:].reshape(1, -1)
            cf_pred = self.predict_fn(drug_cf, disease_cf)[0]

            # Compute distance
            if self.config.distance_metric == "l2":
                distance = np.sqrt(np.sum((x_cf - original_x) ** 2))
            else:
                distance = np.sum(np.abs(x_cf - original_x))

            return {
                "features": x_cf,
                "prediction": float(cf_pred),
                "distance": float(distance),
                "success": abs(cf_pred - target_pred) < 0.2,  # Within 20% of target
            }

        except Exception as e:
            logger.warning(f"Counterfactual optimization failed: {e}")
            return None

    def _compute_changes(
        self,
        original_x: np.ndarray,
        counterfactual_x: np.ndarray,
        feature_names: Optional[List[str]],
        n_drug_features: int,
    ) -> Dict[str, Any]:
        """Compute and summarize feature changes."""
        diff = counterfactual_x - original_x

        if feature_names is None:
            feature_names = (
                [f"drug_f{i}" for i in range(n_drug_features)]
                + [f"disease_f{i}" for i in range(len(original_x) - n_drug_features)]
            )

        # Find significant changes (>1% relative change or >0.1 absolute)
        threshold = 0.1
        significant_indices = np.where(np.abs(diff) > threshold)[0]

        significant_changes = []
        for idx in significant_indices:
            change_dict = {
                "feature": feature_names[idx],
                "index": int(idx),
                "original_value": float(original_x[idx]),
                "counterfactual_value": float(counterfactual_x[idx]),
                "absolute_change": float(diff[idx]),
                "relative_change": float(diff[idx] / (abs(original_x[idx]) + 1e-10)),
                "feature_type": "drug" if idx < n_drug_features else "disease",
            }
            significant_changes.append(change_dict)

        # Sort by absolute change
        significant_changes.sort(key=lambda x: abs(x["absolute_change"]), reverse=True)

        return {
            "significant_changes": significant_changes[:20],  # Top 20
            "total_features_changed": len(significant_indices),
            "drug_features_changed": sum(1 for idx in significant_indices if idx < n_drug_features),
            "disease_features_changed": sum(1 for idx in significant_indices if idx >= n_drug_features),
            "mean_absolute_change": float(np.mean(np.abs(diff))),
            "max_absolute_change": float(np.max(np.abs(diff))),
        }

    def generate_diverse_counterfactuals(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
        num_counterfactuals: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple diverse counterfactual explanations.

        Args:
            drug_features: Original drug features
            disease_features: Original disease features
            num_counterfactuals: Number of diverse counterfactuals

        Returns:
            counterfactuals: List of counterfactual dictionaries
        """
        # Temporarily set num_counterfactuals
        original_num = self.config.num_counterfactuals
        self.config.num_counterfactuals = num_counterfactuals

        result = self.generate(drug_features, disease_features)

        # Restore original setting
        self.config.num_counterfactuals = original_num

        if result["success"]:
            return result.get("all_counterfactuals", [])
        else:
            return []


# Singleton instance
_counterfactual_generator: Optional[CounterfactualGenerator] = None


def get_counterfactual_generator(
    predict_fn: Optional[Callable] = None,
    config: Optional[CounterfactualConfig] = None,
) -> CounterfactualGenerator:
    """Get or create counterfactual generator instance."""
    global _counterfactual_generator

    if _counterfactual_generator is not None and predict_fn is None:
        return _counterfactual_generator

    if predict_fn is None:
        # Default to dummy function
        def dummy_predict(drug_feat, disease_feat):
            return np.random.rand(len(drug_feat))
        predict_fn = dummy_predict
        logger.warning("No predict_fn provided, using dummy function")

    _counterfactual_generator = CounterfactualGenerator(predict_fn, config)
    logger.info("Created new Counterfactual Generator instance")
    return _counterfactual_generator
