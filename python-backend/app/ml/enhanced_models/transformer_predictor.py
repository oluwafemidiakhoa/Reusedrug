"""Transformer-based drug-disease prediction model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for Transformer predictor."""

    d_model: int = 256  # Model dimension
    nhead: int = 8  # Number of attention heads
    num_encoder_layers: int = 6  # Number of transformer layers
    dim_feedforward: int = 1024  # FFN dimension
    dropout: float = 0.1
    max_seq_length: int = 512
    num_drug_features: int = 2048  # Drug feature dimension
    num_disease_features: int = 768  # Disease feature dimension
    vocab_size: int = 10000  # For tokenized inputs


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling for sequence aggregation."""

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=False)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]

        Returns:
            pooled: [batch_size, d_model]
        """
        batch_size = x.size(1)
        query = self.query.expand(-1, batch_size, -1)

        # Attend to the sequence
        output, _ = self.attention(query, x, x)

        # Return the pooled representation
        return output.squeeze(0)  # [batch_size, d_model]


class TransformerPredictor(nn.Module):
    """Transformer-based drug-disease interaction predictor.

    Uses multi-head attention to model complex relationships between
    drug molecular features and disease phenotypes.
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()

        # Input projections
        self.drug_projection = nn.Linear(
            self.config.num_drug_features, self.config.d_model
        )
        self.disease_projection = nn.Linear(
            self.config.num_disease_features, self.config.d_model
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            self.config.d_model,
            max_len=self.config.max_seq_length,
            dropout=self.config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=False,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_encoder_layers,
        )

        # Attention pooling
        self.attention_pool = MultiHeadAttentionPooling(
            self.config.d_model, self.config.nhead
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.d_model // 4),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 4, 1),
        )

        # Initialize weights
        self._init_weights()

        logger.info(
            f"Transformer predictor initialized with d_model={self.config.d_model}, "
            f"nhead={self.config.nhead}, layers={self.config.num_encoder_layers}"
        )

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        drug_features: torch.Tensor,
        disease_features: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            drug_features: [batch_size, num_drug_features]
            disease_features: [batch_size, num_disease_features]
            return_attention: If True, return attention weights

        Returns:
            predictions: [batch_size, 1] - interaction probabilities
            attention_weights: Optional attention weights if return_attention=True
        """
        # Project to model dimension
        drug_emb = self.drug_projection(drug_features)  # [batch, d_model]
        disease_emb = self.disease_projection(disease_features)  # [batch, d_model]

        # Concatenate as sequence: [drug, disease]
        # Shape: [2, batch_size, d_model]
        sequence = torch.stack([drug_emb, disease_emb], dim=0)

        # Add positional encoding
        sequence = self.pos_encoder(sequence)

        # Transformer encoding
        encoded = self.transformer_encoder(sequence)  # [2, batch, d_model]

        # Attention pooling
        pooled = self.attention_pool(encoded)  # [batch, d_model]

        # Classification
        logits = self.classifier(pooled)  # [batch, 1]
        predictions = torch.sigmoid(logits)

        if return_attention:
            # Extract attention from last layer (simplified)
            return predictions, None

        return predictions

    def predict_proba(
        self, drug_features: np.ndarray, disease_features: np.ndarray
    ) -> np.ndarray:
        """
        Predict interaction probabilities.

        Args:
            drug_features: [batch_size, num_drug_features]
            disease_features: [batch_size, num_disease_features]

        Returns:
            probabilities: [batch_size]
        """
        self.eval()
        with torch.no_grad():
            drug_tensor = torch.FloatTensor(drug_features)
            disease_tensor = torch.FloatTensor(disease_features)

            predictions = self.forward(drug_tensor, disease_tensor)

            return predictions.squeeze(-1).numpy()


class DrugDiseaseDataset(Dataset):
    """Dataset for drug-disease pairs."""

    def __init__(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
        labels: np.ndarray,
    ):
        self.drug_features = torch.FloatTensor(drug_features)
        self.disease_features = torch.FloatTensor(disease_features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.drug_features[idx],
            self.disease_features[idx],
            self.labels[idx],
        )


class TransformerTrainer:
    """Trainer for Transformer predictor."""

    def __init__(
        self,
        model: TransformerPredictor,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.criterion = nn.BCELoss()

        logger.info(f"Trainer initialized on device: {device}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for drug_feat, disease_feat, labels in dataloader:
            drug_feat = drug_feat.to(self.device)
            disease_feat = disease_feat.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            predictions = self.model(drug_feat, disease_feat).squeeze(-1)

            # Compute loss
            loss = self.criterion(predictions, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for drug_feat, disease_feat, labels in dataloader:
                drug_feat = drug_feat.to(self.device)
                disease_feat = disease_feat.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(drug_feat, disease_feat).squeeze(-1)

                loss = self.criterion(predictions, labels)
                total_loss += loss.item()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Compute metrics
        from sklearn.metrics import roc_auc_score, average_precision_score

        try:
            auroc = roc_auc_score(all_labels, all_predictions)
            auprc = average_precision_score(all_labels, all_predictions)
        except Exception:
            auroc = 0.5
            auprc = 0.0

        return {
            "loss": total_loss / len(dataloader),
            "auroc": auroc,
            "auprc": auprc,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
    ) -> dict[str, list[float]]:
        """
        Full training loop.

        Returns:
            history: Training history with losses and metrics
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "val_auprc": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_auroc"].append(val_metrics["auroc"])
                history["val_auprc"].append(val_metrics["auprc"])

                # Learning rate scheduling
                self.scheduler.step(val_metrics["loss"])

                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val AUROC: {val_metrics['auroc']:.4f}, "
                    f"Val AUPRC: {val_metrics['auprc']:.4f}"
                )

                # Early stopping
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

        return history


# Singleton instance
_transformer_predictor: Optional[TransformerPredictor] = None
_transformer_trainer: Optional[TransformerTrainer] = None


def get_transformer_predictor(
    config: Optional[TransformerConfig] = None,
) -> TransformerPredictor:
    """Get or create transformer predictor instance."""
    global _transformer_predictor
    if _transformer_predictor is None:
        _transformer_predictor = TransformerPredictor(config)
        logger.info("Created new Transformer predictor instance")
    return _transformer_predictor


def get_transformer_trainer(
    model: Optional[TransformerPredictor] = None,
    learning_rate: float = 1e-4,
    device: str = "cpu",
) -> TransformerTrainer:
    """Get or create transformer trainer instance."""
    global _transformer_trainer
    if _transformer_trainer is None:
        if model is None:
            model = get_transformer_predictor()
        _transformer_trainer = TransformerTrainer(
            model, learning_rate=learning_rate, device=device
        )
        logger.info("Created new Transformer trainer instance")
    return _transformer_trainer
