"""ML configuration and settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class MLConfig:
    """Configuration for ML models and training."""

    # Model paths
    MODEL_DIR: Path = Path(os.getenv("ML_MODEL_DIR", "data/models"))
    EMBEDDINGS_DIR: Path = Path(os.getenv("ML_EMBEDDINGS_DIR", "data/embeddings"))

    # MLflow settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "drug-repurposing")
    MLFLOW_ENABLED: bool = os.getenv("MLFLOW_ENABLED", "false").lower() == "true"

    # GNN Model settings
    GNN_EMBEDDING_DIM: int = int(os.getenv("GNN_EMBEDDING_DIM", "128"))
    GNN_HIDDEN_DIM: int = int(os.getenv("GNN_HIDDEN_DIM", "256"))
    GNN_NUM_LAYERS: int = int(os.getenv("GNN_NUM_LAYERS", "3"))
    GNN_DROPOUT: float = float(os.getenv("GNN_DROPOUT", "0.2"))

    # Training settings
    BATCH_SIZE: int = int(os.getenv("ML_BATCH_SIZE", "32"))
    LEARNING_RATE: float = float(os.getenv("ML_LEARNING_RATE", "0.001"))
    MAX_EPOCHS: int = int(os.getenv("ML_MAX_EPOCHS", "100"))
    EARLY_STOPPING_PATIENCE: int = int(os.getenv("ML_EARLY_STOPPING_PATIENCE", "10"))

    # Inference settings
    PREDICTION_THRESHOLD: float = float(os.getenv("ML_PREDICTION_THRESHOLD", "0.5"))
    TOP_K_PREDICTIONS: int = int(os.getenv("ML_TOP_K_PREDICTIONS", "50"))

    # Feature settings
    USE_GNN: bool = os.getenv("ML_USE_GNN", "true").lower() == "true"
    USE_EMBEDDINGS: bool = os.getenv("ML_USE_EMBEDDINGS", "true").lower() == "true"

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


# Ensure directories exist
MLConfig.ensure_directories()
