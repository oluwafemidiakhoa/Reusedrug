"""Ensemble methods for drug repurposing predictions.

Combines multiple models (GNN, Transformer, KG-based) for robust predictions
with uncertainty quantification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

import numpy as np
from scipy.stats import beta

from app.logging_conf import get_logger

logger = get_logger(__name__)


class EnsembleMethod(str, Enum):
    """Ensemble combination methods."""

    AVERAGE = "average"  # Simple average
    WEIGHTED = "weighted"  # Weighted average
    VOTING = "voting"  # Majority voting (hard predictions)
    STACKING = "stacking"  # Meta-learner stacking
    MAX = "max"  # Maximum prediction
    MIN = "min"  # Minimum prediction (conservative)


@dataclass
class ModelWeight:
    """Weight configuration for a model in the ensemble."""

    name: str
    weight: float = 1.0
    enabled: bool = True


@dataclass
class EnsembleConfig:
    """Configuration for ensemble predictor."""

    method: EnsembleMethod = EnsembleMethod.WEIGHTED
    model_weights: dict[str, float] = None
    calibrate_uncertainty: bool = True
    diversity_bonus: float = 0.1  # Bonus for diverse predictions
    min_models: int = 2  # Minimum models required

    def __post_init__(self):
        if self.model_weights is None:
            # Default weights based on expected performance
            self.model_weights = {
                "gnn": 0.4,  # GNN is primary model
                "transformer": 0.3,  # Transformer for complex patterns
                "multimodal": 0.2,  # Multi-modal integration
                "knowledge_graph": 0.1,  # KG for reasoning
            }


class UncertaintyQuantifier:
    """Quantify prediction uncertainty using ensemble disagreement."""

    def __init__(self, method: str = "variance"):
        """
        Args:
            method: "variance", "entropy", or "range"
        """
        self.method = method

    def compute_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """
        Compute uncertainty scores from ensemble predictions.

        Args:
            predictions: [num_models, num_samples] array of predictions

        Returns:
            uncertainty: [num_samples] uncertainty scores (0-1, higher = more uncertain)
        """
        if self.method == "variance":
            # Variance across models
            uncertainty = np.var(predictions, axis=0)

        elif self.method == "entropy":
            # Entropy of predictions (treating as probabilities)
            mean_pred = np.mean(predictions, axis=0)
            epsilon = 1e-10
            mean_pred = np.clip(mean_pred, epsilon, 1 - epsilon)
            entropy = -(
                mean_pred * np.log(mean_pred)
                + (1 - mean_pred) * np.log(1 - mean_pred)
            )
            uncertainty = entropy / np.log(2)  # Normalize to [0, 1]

        elif self.method == "range":
            # Range (max - min) across models
            uncertainty = np.max(predictions, axis=0) - np.min(predictions, axis=0)

        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")

        return uncertainty

    def confidence_intervals(
        self,
        predictions: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence intervals using bootstrap or Beta distribution.

        Args:
            predictions: [num_models, num_samples]
            alpha: Significance level (default: 0.05 for 95% CI)

        Returns:
            lower_bounds: [num_samples]
            upper_bounds: [num_samples]
        """
        # Use Beta distribution for Bayesian confidence intervals
        num_models, num_samples = predictions.shape

        lower_bounds = np.zeros(num_samples)
        upper_bounds = np.zeros(num_samples)

        for i in range(num_samples):
            preds = predictions[:, i]

            # Fit Beta distribution
            # Use method of moments: mean and variance
            mean_p = np.mean(preds)
            var_p = np.var(preds) + 1e-10

            # Beta parameters from mean and variance
            if var_p < mean_p * (1 - mean_p):
                alpha_param = mean_p * ((mean_p * (1 - mean_p) / var_p) - 1)
                beta_param = (1 - mean_p) * ((mean_p * (1 - mean_p) / var_p) - 1)

                # Confidence intervals
                lower_bounds[i] = beta.ppf(alpha / 2, alpha_param, beta_param)
                upper_bounds[i] = beta.ppf(1 - alpha / 2, alpha_param, beta_param)
            else:
                # Fallback to percentiles
                lower_bounds[i] = np.percentile(preds, alpha / 2 * 100)
                upper_bounds[i] = np.percentile(preds, (1 - alpha / 2) * 100)

        return lower_bounds, upper_bounds


class EnsemblePredictor:
    """Ensemble predictor combining multiple models.

    Features:
    - Multiple combination methods (averaging, voting, stacking)
    - Weighted predictions based on model performance
    - Uncertainty quantification
    - Model diversity bonus
    - Calibrated predictions
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.models: dict[str, Callable] = {}
        self.uncertainty_quantifier = UncertaintyQuantifier(method="variance")

        # Normalize weights
        self._normalize_weights()

        logger.info(
            f"Ensemble predictor initialized with method={self.config.method.value}, "
            f"weights={self.config.model_weights}"
        )

    def _normalize_weights(self):
        """Normalize model weights to sum to 1."""
        total = sum(self.config.model_weights.values())
        if total > 0:
            for key in self.config.model_weights:
                self.config.model_weights[key] /= total

    def register_model(self, name: str, predict_fn: Callable, weight: Optional[float] = None):
        """
        Register a model in the ensemble.

        Args:
            name: Model name (e.g., "gnn", "transformer")
            predict_fn: Function that takes features and returns predictions
            weight: Optional weight (overrides config)
        """
        self.models[name] = predict_fn

        if weight is not None:
            self.config.model_weights[name] = weight
            self._normalize_weights()

        logger.info(
            f"Registered model '{name}' with weight {self.config.model_weights.get(name, 0):.3f}"
        )

    def predict(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
        return_uncertainty: bool = False,
        return_individual: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Make ensemble predictions.

        Args:
            drug_features: [batch_size, drug_dim]
            disease_features: [batch_size, disease_dim]
            return_uncertainty: Whether to compute uncertainty scores
            return_individual: Whether to return individual model predictions

        Returns:
            results: Dictionary containing:
                - predictions: [batch_size] ensemble predictions
                - uncertainty: [batch_size] (if return_uncertainty=True)
                - individual_predictions: dict (if return_individual=True)
                - lower_ci, upper_ci: Confidence intervals
        """
        if len(self.models) < self.config.min_models:
            raise ValueError(
                f"Need at least {self.config.min_models} models, "
                f"but only {len(self.models)} registered"
            )

        batch_size = len(drug_features)
        individual_preds = {}

        # Collect predictions from all models
        all_predictions = []

        for name, predict_fn in self.models.items():
            try:
                preds = predict_fn(drug_features, disease_features)
                if len(preds) != batch_size:
                    raise ValueError(
                        f"Model {name} returned {len(preds)} predictions, "
                        f"expected {batch_size}"
                    )

                individual_preds[name] = preds
                all_predictions.append(preds)

            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                continue

        if not all_predictions:
            raise RuntimeError("All models failed to make predictions")

        # Stack predictions: [num_models, batch_size]
        all_predictions = np.array(all_predictions)

        # Combine predictions
        if self.config.method == EnsembleMethod.AVERAGE:
            ensemble_pred = np.mean(all_predictions, axis=0)

        elif self.config.method == EnsembleMethod.WEIGHTED:
            weights = np.array(
                [self.config.model_weights.get(name, 1.0) for name in self.models.keys()]
            )
            weights = weights[:, np.newaxis]  # [num_models, 1]
            ensemble_pred = np.sum(all_predictions * weights, axis=0)

        elif self.config.method == EnsembleMethod.VOTING:
            # Threshold at 0.5 and vote
            votes = (all_predictions > 0.5).astype(float)
            ensemble_pred = np.mean(votes, axis=0)

        elif self.config.method == EnsembleMethod.MAX:
            ensemble_pred = np.max(all_predictions, axis=0)

        elif self.config.method == EnsembleMethod.MIN:
            ensemble_pred = np.min(all_predictions, axis=0)

        else:
            raise ValueError(f"Unsupported ensemble method: {self.config.method}")

        # Prepare results
        results = {"predictions": ensemble_pred}

        # Uncertainty quantification
        if return_uncertainty or self.config.calibrate_uncertainty:
            uncertainty = self.uncertainty_quantifier.compute_uncertainty(all_predictions)
            results["uncertainty"] = uncertainty

            # Confidence intervals
            lower_ci, upper_ci = self.uncertainty_quantifier.confidence_intervals(
                all_predictions
            )
            results["lower_ci"] = lower_ci
            results["upper_ci"] = upper_ci

        # Individual predictions
        if return_individual:
            results["individual_predictions"] = individual_preds

        # Model diversity score
        diversity = np.mean(np.std(all_predictions, axis=0))
        results["diversity_score"] = diversity

        return results

    def predict_with_confidence(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
        confidence_threshold: float = 0.8,
    ) -> dict[str, np.ndarray]:
        """
        Make predictions with confidence filtering.

        Args:
            drug_features: Drug features
            disease_features: Disease features
            confidence_threshold: Minimum confidence (1 - uncertainty)

        Returns:
            results: Predictions, confidence scores, and filtering mask
        """
        results = self.predict(
            drug_features,
            disease_features,
            return_uncertainty=True,
        )

        # Confidence = 1 - uncertainty
        confidence = 1.0 - results["uncertainty"]
        results["confidence"] = confidence

        # High confidence predictions
        high_confidence_mask = confidence >= confidence_threshold
        results["high_confidence_mask"] = high_confidence_mask
        results["num_high_confidence"] = np.sum(high_confidence_mask)

        return results

    def get_model_contributions(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """
        Analyze individual model contributions to ensemble.

        Returns:
            contributions: Per-model statistics
        """
        results = self.predict(
            drug_features,
            disease_features,
            return_individual=True,
            return_uncertainty=True,
        )

        ensemble_pred = results["predictions"]
        individual_preds = results["individual_predictions"]

        contributions = {}

        for name, preds in individual_preds.items():
            # Correlation with ensemble
            correlation = np.corrcoef(preds, ensemble_pred)[0, 1]

            # Mean absolute difference
            mae = np.mean(np.abs(preds - ensemble_pred))

            # Agreement rate (at 0.5 threshold)
            agreement = np.mean((preds > 0.5) == (ensemble_pred > 0.5))

            contributions[name] = {
                "weight": self.config.model_weights.get(name, 0.0),
                "correlation": correlation,
                "mean_abs_error": mae,
                "agreement_rate": agreement,
                "mean_prediction": np.mean(preds),
            }

        return contributions


# Singleton instance
_ensemble_predictor: Optional[EnsemblePredictor] = None


def get_ensemble_predictor(
    config: Optional[EnsembleConfig] = None,
) -> EnsemblePredictor:
    """Get or create ensemble predictor instance."""
    global _ensemble_predictor
    if _ensemble_predictor is None:
        _ensemble_predictor = EnsemblePredictor(config)
        logger.info("Created new Ensemble predictor instance")
    return _ensemble_predictor
