"""Meta-learning for few-shot drug repurposing.

Implements Model-Agnostic Meta-Learning (MAML) for quick adaptation
to new drug-disease associations with limited data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class MAMLConfig:
    """Configuration for MAML meta-learner."""

    input_dim: int = 2048 + 768  # Combined drug + disease features
    hidden_dims: list[int] = None  # [512, 256, 128]
    output_dim: int = 1
    inner_lr: float = 0.01  # Inner loop learning rate
    meta_lr: float = 0.001  # Meta (outer loop) learning rate
    num_inner_steps: int = 5  # Number of gradient steps in inner loop
    num_support: int = 5  # Support set size (k-shot)
    num_query: int = 15  # Query set size
    dropout: float = 0.1

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


class BaseModel(nn.Module):
    """Base model for MAML adaptation."""

    def __init__(self, config: MAMLConfig):
        super().__init__()
        self.config = config

        # Build MLP layers
        layers = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, config.output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.sigmoid(self.network(x))


class MetaLearner:
    """MAML meta-learner for few-shot drug repurposing.

    Learns to quickly adapt to new drug-disease associations with
    minimal training examples (few-shot learning).
    """

    def __init__(
        self,
        config: Optional[MAMLConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or MAMLConfig()
        self.device = device

        # Meta-model (will be cloned for each task)
        self.meta_model = BaseModel(self.config).to(device)

        # Meta-optimizer (updates meta-model parameters)
        self.meta_optimizer = torch.optim.Adam(
            self.meta_model.parameters(),
            lr=self.config.meta_lr,
        )

        self.criterion = nn.BCELoss()

        logger.info(
            f"MAML Meta-Learner initialized with "
            f"{self.config.num_support}-shot learning, "
            f"inner_lr={self.config.inner_lr}, "
            f"meta_lr={self.config.meta_lr}"
        )

    def inner_loop(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        """
        Perform inner loop adaptation on support set.

        Args:
            model: Model to adapt
            support_x: Support set features [num_support, input_dim]
            support_y: Support set labels [num_support]

        Returns:
            adapted_model: Model adapted to the task
        """
        # Clone model for adaptation
        adapted_model = deepcopy(model)
        adapted_model.train()

        # Create inner optimizer
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr,
        )

        # Perform gradient steps on support set
        for _ in range(self.config.num_inner_steps):
            predictions = adapted_model(support_x).squeeze()
            loss = self.criterion(predictions, support_y)

            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return adapted_model

    def meta_train_step(
        self,
        task_batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> dict[str, float]:
        """
        Perform one meta-training step across a batch of tasks.

        Args:
            task_batch: List of (support_x, support_y, query_x, query_y) tuples

        Returns:
            metrics: Training metrics
        """
        self.meta_model.train()

        meta_loss = 0.0
        meta_accuracy = 0.0

        for support_x, support_y, query_x, query_y in task_batch:
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # Inner loop: adapt to task using support set
            adapted_model = self.inner_loop(self.meta_model, support_x, support_y)

            # Compute loss on query set with adapted model
            adapted_model.eval()
            with torch.no_grad():
                query_pred = adapted_model(query_x).squeeze()
                task_loss = self.criterion(query_pred, query_y)
                task_acc = ((query_pred > 0.5) == query_y).float().mean()

            meta_loss += task_loss
            meta_accuracy += task_acc

        # Average across tasks
        meta_loss = meta_loss / len(task_batch)
        meta_accuracy = meta_accuracy / len(task_batch)

        # Meta-optimization: update meta-model to minimize query loss
        # Note: In true MAML, we'd compute gradients through the inner loop
        # For simplicity, we use first-order MAML (FOMAML)
        self.meta_optimizer.zero_grad()

        # Re-compute with gradients for meta-update
        batch_meta_loss = 0.0
        for support_x, support_y, query_x, query_y in task_batch:
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # Adapt model
            adapted_model = self.inner_loop(self.meta_model, support_x, support_y)

            # Query loss
            query_pred = adapted_model(query_x).squeeze()
            batch_meta_loss += self.criterion(query_pred, query_y)

        batch_meta_loss = batch_meta_loss / len(task_batch)
        batch_meta_loss.backward()
        self.meta_optimizer.step()

        return {
            "meta_loss": meta_loss.item(),
            "meta_accuracy": meta_accuracy.item(),
        }

    def adapt(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        num_steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Adapt meta-model to new task using support examples.

        Args:
            support_x: Support set features [num_support, input_dim]
            support_y: Support set labels [num_support]
            num_steps: Number of adaptation steps (default: config.num_inner_steps)

        Returns:
            adapted_model: Model adapted to the new task
        """
        if num_steps is None:
            num_steps = self.config.num_inner_steps

        support_x = torch.FloatTensor(support_x).to(self.device)
        support_y = torch.FloatTensor(support_y).to(self.device)

        # Clone meta-model
        adapted_model = deepcopy(self.meta_model)
        adapted_model.train()

        # Create optimizer for adaptation
        optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr,
        )

        # Adaptation loop
        for step in range(num_steps):
            predictions = adapted_model(support_x).squeeze()
            loss = self.criterion(predictions, support_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == 0 or (step + 1) % max(1, num_steps // 2) == 0:
                logger.debug(f"Adaptation step {step+1}/{num_steps}, loss: {loss.item():.4f}")

        adapted_model.eval()
        return adapted_model

    def predict(
        self,
        adapted_model: nn.Module,
        query_x: np.ndarray,
    ) -> np.ndarray:
        """
        Make predictions using adapted model.

        Args:
            adapted_model: Model adapted to task
            query_x: Query features [num_query, input_dim]

        Returns:
            predictions: Predicted probabilities [num_query]
        """
        adapted_model.eval()
        with torch.no_grad():
            query_x = torch.FloatTensor(query_x).to(self.device)
            predictions = adapted_model(query_x).squeeze()

        return predictions.cpu().numpy()

    def few_shot_predict(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        query_x: np.ndarray,
        num_adaptation_steps: int = 10,
    ) -> np.ndarray:
        """
        End-to-end few-shot prediction.

        Args:
            support_x: Support examples (few labeled examples)
            support_y: Support labels
            query_x: Query examples (to predict)
            num_adaptation_steps: Number of gradient steps for adaptation

        Returns:
            predictions: Predicted probabilities for query examples
        """
        # Adapt to task
        adapted_model = self.adapt(support_x, support_y, num_adaptation_steps)

        # Predict on query set
        predictions = self.predict(adapted_model, query_x)

        return predictions

    def save(self, path: str):
        """Save meta-model checkpoint."""
        torch.save(
            {
                "meta_model_state": self.meta_model.state_dict(),
                "meta_optimizer_state": self.meta_optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info(f"Meta-learner saved to {path}")

    def load(self, path: str):
        """Load meta-model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_model.load_state_dict(checkpoint["meta_model_state"])
        self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer_state"])
        logger.info(f"Meta-learner loaded from {path}")


class TaskSampler:
    """Sample tasks for meta-learning."""

    def __init__(
        self,
        drug_features: np.ndarray,
        disease_features: np.ndarray,
        labels: np.ndarray,
        num_support: int = 5,
        num_query: int = 15,
    ):
        """
        Args:
            drug_features: All drug features [N, drug_dim]
            disease_features: All disease features [N, disease_dim]
            labels: All labels [N]
            num_support: Support set size per task
            num_query: Query set size per task
        """
        self.features = np.concatenate([drug_features, disease_features], axis=1)
        self.labels = labels
        self.num_support = num_support
        self.num_query = num_query

        # Separate positive and negative examples
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]

        logger.info(
            f"Task sampler initialized with {len(self.pos_indices)} positive "
            f"and {len(self.neg_indices)} negative examples"
        )

    def sample_task(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample one task (support + query sets).

        Returns:
            support_x, support_y, query_x, query_y
        """
        # Sample balanced support set
        num_pos = self.num_support // 2
        num_neg = self.num_support - num_pos

        support_pos_idx = np.random.choice(self.pos_indices, num_pos, replace=False)
        support_neg_idx = np.random.choice(self.neg_indices, num_neg, replace=False)
        support_idx = np.concatenate([support_pos_idx, support_neg_idx])

        # Sample balanced query set
        num_pos_q = self.num_query // 2
        num_neg_q = self.num_query - num_pos_q

        # Exclude support examples from query
        available_pos = np.setdiff1d(self.pos_indices, support_pos_idx)
        available_neg = np.setdiff1d(self.neg_indices, support_neg_idx)

        query_pos_idx = np.random.choice(available_pos, num_pos_q, replace=False)
        query_neg_idx = np.random.choice(available_neg, num_neg_q, replace=False)
        query_idx = np.concatenate([query_pos_idx, query_neg_idx])

        # Get features and labels
        support_x = self.features[support_idx]
        support_y = self.labels[support_idx]
        query_x = self.features[query_idx]
        query_y = self.labels[query_idx]

        return support_x, support_y, query_x, query_y

    def sample_batch(self, batch_size: int) -> list:
        """Sample a batch of tasks."""
        return [self.sample_task() for _ in range(batch_size)]


# Singleton instance
_meta_learner: Optional[MetaLearner] = None


def get_meta_learner(
    config: Optional[MAMLConfig] = None,
    device: str = "cpu",
) -> MetaLearner:
    """Get or create meta-learner instance."""
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = MetaLearner(config, device)
        logger.info("Created new Meta-Learner instance")
    return _meta_learner
