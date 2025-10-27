"""SHAP (SHapley Additive exPlanations) for drug repurposing models.

Provides model-agnostic feature importance using Shapley values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any

import numpy as np
from app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class SHAPConfig:
    """Configuration for SHAP explainer."""

    n_samples: int = 100  # Number of background samples
    max_evals: int = 1000  # Maximum model evaluations
    method: str = "kernel"  # "kernel", "sampling", or "exact"


class SHAPExplainer:
    """SHAP explainer for drug-disease prediction models.

    Computes Shapley values to explain feature importance for predictions.
    Uses model-agnostic approach suitable for any black-box model.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        background_drug_features: Optional[np.ndarray] = None,
        background_disease_features: Optional[np.ndarray] = None,
        config: Optional[SHAPConfig] = None,
    ):
        """
        Args:
            predict_fn: Function that takes (drug_features, disease_features)
                       and returns predictions
            background_drug_features: Background dataset for drug features [n, d_drug]
            background_disease_features: Background dataset for disease features [n, d_disease]
            config: SHAP configuration
        """
        self.predict_fn = predict_fn
        self.background_drug = background_drug_features
        self.background_disease = background_disease_features
        self.config = config or SHAPConfig()

        # Create default background if not provided
        if self.background_drug is None or self.background_disease is None:
            self._create_default_background()

        logger.info(
            f"SHAP Explainer initialized with {len(self.background_drug)} "
            f"background samples, method={self.config.method}"
        )

    def _create_default_background(self):
        """Create default background dataset from random samples."""
        # Use small random background as placeholder
        self.background_drug = np.random.randn(self.config.n_samples, 2048) * 0.1
        self.background_disease = np.random.randn(self.config.n_samples, 768) * 0.1
        logger.warning("Using random background dataset - provide real data for better results")

    def explain(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Explain a prediction using SHAP values.

        Args:
            drug_features: Drug features [1, d_drug] or [d_drug]
            disease_features: Disease features [1, d_disease] or [d_disease]
            feature_names: Optional names for features

        Returns:
            explanation: Dictionary with SHAP values and metadata
        """
        # Ensure 2D arrays
        if drug_features.ndim == 1:
            drug_features = drug_features.reshape(1, -1)
        if disease_features.ndim == 1:
            disease_features = disease_features.reshape(1, -1)

        # Concatenate features
        x = np.concatenate([drug_features, disease_features], axis=1)[0]

        # Compute base prediction (expected value over background)
        background_x = np.concatenate(
            [self.background_drug, self.background_disease], axis=1
        )
        base_value = np.mean(
            self.predict_fn(self.background_drug, self.background_disease)
        )

        # Compute actual prediction
        prediction = self.predict_fn(drug_features, disease_features)[0]

        # Compute SHAP values using Kernel SHAP approximation
        shap_values = self._compute_shap_values(x, background_x)

        # Create feature names if not provided
        if feature_names is None:
            feature_names = (
                [f"drug_f{i}" for i in range(drug_features.shape[1])]
                + [f"disease_f{i}" for i in range(disease_features.shape[1])]
            )

        # Get top contributing features
        top_features = self._get_top_features(shap_values, feature_names, top_k=20)

        return {
            "prediction": float(prediction),
            "base_value": float(base_value),
            "shap_values": shap_values.tolist(),
            "feature_names": feature_names,
            "top_positive_features": top_features["positive"],
            "top_negative_features": top_features["negative"],
            "num_drug_features": drug_features.shape[1],
            "num_disease_features": disease_features.shape[1],
        }

    def _compute_shap_values(
        self,
        x: np.ndarray,
        background: np.ndarray,
    ) -> np.ndarray:
        """
        Compute SHAP values using Kernel SHAP approximation.

        This is a simplified implementation. For production, use the shap library.

        Args:
            x: Instance to explain [num_features]
            background: Background dataset [n_background, num_features]

        Returns:
            shap_values: SHAP values for each feature [num_features]
        """
        num_features = len(x)
        shap_values = np.zeros(num_features)

        # Base prediction
        base_pred = self._predict_combined(background)

        # Compute marginal contributions (simplified SHAP approximation)
        n_samples = min(self.config.max_evals // (2 * num_features), 50)

        for i in range(num_features):
            # Sample coalitions with and without feature i
            contributions = []

            for _ in range(n_samples):
                # Random coalition
                coalition = np.random.rand(num_features) > 0.5

                # With feature i
                x_with = background.copy()
                for j, include in enumerate(coalition):
                    if include or j == i:
                        x_with[:, j] = x[j]
                pred_with = self._predict_combined(x_with)

                # Without feature i
                x_without = background.copy()
                for j, include in enumerate(coalition):
                    if include and j != i:
                        x_without[:, j] = x[j]
                pred_without = self._predict_combined(x_without)

                # Marginal contribution
                contributions.append(pred_with - pred_without)

            shap_values[i] = np.mean(contributions)

        return shap_values

    def _predict_combined(self, combined_features: np.ndarray) -> float:
        """Predict using combined drug+disease features."""
        n_drug_features = self.background_drug.shape[1]

        drug_feat = combined_features[:, :n_drug_features]
        disease_feat = combined_features[:, n_drug_features:]

        predictions = self.predict_fn(drug_feat, disease_feat)
        return np.mean(predictions)

    def _get_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 20,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get top positive and negative contributing features."""
        # Sort by absolute value
        abs_values = np.abs(shap_values)
        top_indices = np.argsort(abs_values)[::-1][:top_k]

        positive_features = []
        negative_features = []

        for idx in top_indices:
            feature_dict = {
                "feature": feature_names[idx],
                "shap_value": float(shap_values[idx]),
                "abs_importance": float(abs_values[idx]),
                "index": int(idx),
            }

            if shap_values[idx] > 0:
                positive_features.append(feature_dict)
            else:
                negative_features.append(feature_dict)

        return {
            "positive": positive_features,
            "negative": negative_features,
        }

    def batch_explain(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions.

        Args:
            drug_features: [batch_size, d_drug]
            disease_features: [batch_size, d_disease]
            feature_names: Optional feature names

        Returns:
            explanations: List of explanation dictionaries
        """
        explanations = []

        for i in range(len(drug_features)):
            explanation = self.explain(
                drug_features[i:i+1],
                disease_features[i:i+1],
                feature_names=feature_names,
            )
            explanations.append(explanation)

        return explanations

    def global_importance(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute global feature importance across dataset.

        Args:
            drug_features: [n_samples, d_drug]
            disease_features: [n_samples, d_disease]
            feature_names: Optional feature names

        Returns:
            global_importance: Mean absolute SHAP values per feature
        """
        explanations = self.batch_explain(drug_features, disease_features, feature_names)

        # Aggregate SHAP values
        all_shap_values = np.array([exp["shap_values"] for exp in explanations])
        mean_abs_shap = np.mean(np.abs(all_shap_values), axis=0)

        if feature_names is None:
            feature_names = explanations[0]["feature_names"]

        # Sort by importance
        sorted_indices = np.argsort(mean_abs_shap)[::-1]

        global_features = []
        for idx in sorted_indices[:50]:  # Top 50
            global_features.append({
                "feature": feature_names[idx],
                "mean_abs_shap": float(mean_abs_shap[idx]),
                "index": int(idx),
            })

        return {
            "global_importance": global_features,
            "num_samples": len(explanations),
            "mean_prediction": float(np.mean([exp["prediction"] for exp in explanations])),
        }


# Singleton instance
_shap_explainer: Optional[SHAPExplainer] = None


def get_shap_explainer(
    predict_fn: Optional[Callable] = None,
    background_drug: Optional[np.ndarray] = None,
    background_disease: Optional[np.ndarray] = None,
    config: Optional[SHAPConfig] = None,
) -> SHAPExplainer:
    """Get or create SHAP explainer instance."""
    global _shap_explainer

    # If explainer exists and no new predict_fn provided, return existing
    if _shap_explainer is not None and predict_fn is None:
        return _shap_explainer

    # Create new explainer
    if predict_fn is None:
        # Default to a dummy function
        def dummy_predict(drug_feat, disease_feat):
            return np.random.rand(len(drug_feat))
        predict_fn = dummy_predict
        logger.warning("No predict_fn provided, using dummy function")

    _shap_explainer = SHAPExplainer(
        predict_fn,
        background_drug,
        background_disease,
        config,
    )
    logger.info("Created new SHAP explainer instance")
    return _shap_explainer
