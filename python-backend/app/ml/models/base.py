"""Base classes for ML models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel


class PredictionResult(BaseModel):
    """Result of a prediction."""

    drug_id: str
    disease_id: str
    score: float
    confidence_low: Optional[float] = None
    confidence_high: Optional[float] = None
    model_name: str
    features_used: List[str] = []
    metadata: Dict[str, Any] = {}


class BasePredictor(ABC):
    """Base class for all prediction models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False

    @abstractmethod
    def predict(
        self,
        drug_id: str,
        disease_id: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """Make a prediction for a drug-disease pair.

        Args:
            drug_id: Drug identifier
            disease_id: Disease identifier
            features: Optional additional features

        Returns:
            PredictionResult with score and metadata
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
        features: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PredictionResult]:
        """Make predictions for multiple drug-disease pairs.

        Args:
            pairs: List of (drug_id, disease_id) tuples
            features: Optional features for each pair

        Returns:
            List of PredictionResults
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        pass

    def get_top_k(
        self,
        disease_id: str,
        drug_ids: List[str],
        k: int = 10,
    ) -> List[PredictionResult]:
        """Get top K drug predictions for a disease.

        Args:
            disease_id: Disease identifier
            drug_ids: List of candidate drug identifiers
            k: Number of top predictions to return

        Returns:
            List of top K predictions sorted by score
        """
        pairs = [(drug_id, disease_id) for drug_id in drug_ids]
        predictions = self.predict_batch(pairs)
        sorted_predictions = sorted(
            predictions, key=lambda p: p.score, reverse=True
        )
        return sorted_predictions[:k]
