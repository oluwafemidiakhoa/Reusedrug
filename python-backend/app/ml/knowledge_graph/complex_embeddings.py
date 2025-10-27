"""ComplEx knowledge graph embeddings for link prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ComplExConfig(BaseModel):
    """Configuration for ComplEx embeddings."""

    embedding_dim: int = Field(default=100, description="Embedding dimensionality")
    learning_rate: float = Field(default=0.01, description="Learning rate")
    num_epochs: int = Field(default=100, description="Training epochs")
    batch_size: int = Field(default=128, description="Batch size")
    regularization: float = Field(default=0.01, description="L2 regularization")
    negative_samples: int = Field(default=10, description="Negative samples per positive")


class ComplExModel(nn.Module):
    """ComplEx model for knowledge graph embeddings.

    ComplEx uses complex-valued embeddings to model asymmetric relations.
    Score function: Re(<e_s, w_r, conj(e_o)>)
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
    ):
        """Initialize ComplEx model.

        Args:
            num_entities: Number of entities
            num_relations: Number of relation types
            embedding_dim: Embedding dimensionality (for both real and imaginary)
        """
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # Entity embeddings (real and imaginary parts)
        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)

        # Relation embeddings (real and imaginary parts)
        self.relation_re = nn.Embedding(num_relations, embedding_dim)
        self.relation_im = nn.Embedding(num_relations, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_re.weight)
        nn.init.xavier_uniform_(self.entity_im.weight)
        nn.init.xavier_uniform_(self.relation_re.weight)
        nn.init.xavier_uniform_(self.relation_im.weight)

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ComplEx scores for triples.

        Args:
            heads: Head entity indices [batch_size]
            relations: Relation indices [batch_size]
            tails: Tail entity indices [batch_size]

        Returns:
            Scores [batch_size]
        """
        # Get embeddings
        h_re = self.entity_re(heads)  # [batch_size, dim]
        h_im = self.entity_im(heads)
        r_re = self.relation_re(relations)
        r_im = self.relation_im(relations)
        t_re = self.entity_re(tails)
        t_im = self.entity_im(tails)

        # ComplEx score: Re(<h, r, conj(t)>)
        # = Re(h_re + i*h_im, r_re + i*r_im, t_re - i*t_im)
        # = h_re * r_re * t_re + h_re * r_im * t_im + h_im * r_re * t_im - h_im * r_im * t_re

        score = torch.sum(
            h_re * r_re * t_re
            + h_re * r_im * t_im
            + h_im * r_re * t_im
            - h_im * r_im * t_re,
            dim=-1,
        )

        return score

    def get_entity_embedding(self, entity_idx: int) -> torch.Tensor:
        """Get full complex embedding for an entity.

        Args:
            entity_idx: Entity index

        Returns:
            Concatenated [real, imaginary] embedding
        """
        entity_tensor = torch.tensor([entity_idx], dtype=torch.long)
        re = self.entity_re(entity_tensor).squeeze(0)
        im = self.entity_im(entity_tensor).squeeze(0)
        return torch.cat([re, im])  # [2 * embedding_dim]


class ComplExEmbeddings:
    """ComplEx knowledge graph embeddings for link prediction."""

    def __init__(self, config: Optional[ComplExConfig] = None):
        """Initialize ComplEx embeddings.

        Args:
            config: ComplEx configuration
        """
        self.config = config or ComplExConfig()
        self.model: Optional[ComplExModel] = None
        self.entity2idx: Dict[str, int] = {}
        self.idx2entity: Dict[int, str] = {}
        self.relation2idx: Dict[str, int] = {}
        self.idx2relation: Dict[int, str] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_vocabulary(
        self,
        triples: List[Tuple[str, str, str]],
    ) -> None:
        """Build entity and relation vocabularies from triples.

        Args:
            triples: List of (head, relation, tail) string tuples
        """
        entities = set()
        relations = set()

        for head, relation, tail in triples:
            entities.add(head)
            entities.add(tail)
            relations.add(relation)

        # Create mappings
        self.entity2idx = {e: i for i, e in enumerate(sorted(entities))}
        self.idx2entity = {i: e for e, i in self.entity2idx.items()}

        self.relation2idx = {r: i for i, r in enumerate(sorted(relations))}
        self.idx2relation = {i: r for r, i in self.relation2idx.items()}

        logger.info(
            f"Built vocabulary: {len(self.entity2idx)} entities, "
            f"{len(self.relation2idx)} relations"
        )

    def _triples_to_tensors(
        self,
        triples: List[Tuple[str, str, str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert string triples to tensor indices.

        Args:
            triples: List of (head, relation, tail) tuples

        Returns:
            Tuple of (heads, relations, tails) tensors
        """
        heads = []
        relations = []
        tails = []

        for h, r, t in triples:
            if h in self.entity2idx and t in self.entity2idx and r in self.relation2idx:
                heads.append(self.entity2idx[h])
                relations.append(self.relation2idx[r])
                tails.append(self.entity2idx[t])

        return (
            torch.tensor(heads, dtype=torch.long),
            torch.tensor(relations, dtype=torch.long),
            torch.tensor(tails, dtype=torch.long),
        )

    def train(
        self,
        train_triples: List[Tuple[str, str, str]],
        valid_triples: Optional[List[Tuple[str, str, str]]] = None,
    ) -> Dict[str, List[float]]:
        """Train ComplEx model on triples.

        Args:
            train_triples: Training triples
            valid_triples: Optional validation triples

        Returns:
            Training history dict
        """
        # Build vocabulary
        self.build_vocabulary(train_triples)

        # Initialize model
        self.model = ComplExModel(
            num_entities=len(self.entity2idx),
            num_relations=len(self.relation2idx),
            embedding_dim=self.config.embedding_dim,
        ).to(self.device)

        # Convert triples to tensors
        heads, relations, tails = self._triples_to_tensors(train_triples)
        num_samples = len(heads)

        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        # Training loop
        history = {"train_loss": [], "valid_mrr": []}

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            # Mini-batch training
            indices = torch.randperm(num_samples)

            for batch_start in range(0, num_samples, self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, num_samples)
                batch_indices = indices[batch_start:batch_end]

                # Positive samples
                batch_heads = heads[batch_indices].to(self.device)
                batch_relations = relations[batch_indices].to(self.device)
                batch_tails = tails[batch_indices].to(self.device)

                # Positive scores
                pos_scores = self.model(batch_heads, batch_relations, batch_tails)

                # Negative sampling (corrupt tail)
                neg_tails = torch.randint(
                    0,
                    len(self.entity2idx),
                    (len(batch_indices), self.config.negative_samples),
                    device=self.device,
                )

                # Expand for negative samples
                batch_heads_expanded = batch_heads.unsqueeze(1).expand(-1, self.config.negative_samples)
                batch_relations_expanded = batch_relations.unsqueeze(1).expand(-1, self.config.negative_samples)

                neg_scores = self.model(
                    batch_heads_expanded.reshape(-1),
                    batch_relations_expanded.reshape(-1),
                    neg_tails.reshape(-1),
                ).reshape(len(batch_indices), self.config.negative_samples)

                # Margin loss
                pos_scores_expanded = pos_scores.unsqueeze(1)
                loss = torch.mean(
                    torch.clamp(1.0 - pos_scores_expanded + neg_scores, min=0.0)
                )

                # L2 regularization
                l2_reg = (
                    self.model.entity_re.weight.norm(2) ** 2
                    + self.model.entity_im.weight.norm(2) ** 2
                    + self.model.relation_re.weight.norm(2) ** 2
                    + self.model.relation_im.weight.norm(2) ** 2
                )
                loss = loss + self.config.regularization * l2_reg

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / (num_samples // self.config.batch_size)
            history["train_loss"].append(avg_loss)

            # Validation
            if valid_triples and epoch % 10 == 0:
                mrr = self.evaluate(valid_triples)
                history["valid_mrr"].append(mrr)
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, MRR={mrr:.4f}")
            elif epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")

        logger.info("Training complete")
        return history

    def evaluate(
        self,
        test_triples: List[Tuple[str, str, str]],
        metric: str = "mrr",
    ) -> float:
        """Evaluate model on test triples.

        Args:
            test_triples: Test triples
            metric: Evaluation metric ('mrr', 'hits@1', 'hits@10')

        Returns:
            Metric score
        """
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()

        heads, relations, tails = self._triples_to_tensors(test_triples)
        ranks = []

        with torch.no_grad():
            for i in range(len(heads)):
                h = heads[i:i+1].to(self.device)
                r = relations[i:i+1].to(self.device)
                t = tails[i:i+1].to(self.device)

                # Score against all entities
                all_entities = torch.arange(len(self.entity2idx), device=self.device)
                h_expanded = h.expand(len(all_entities))
                r_expanded = r.expand(len(all_entities))

                scores = self.model(h_expanded, r_expanded, all_entities)

                # Get rank of true tail
                true_score = scores[t.item()]
                rank = (scores > true_score).sum().item() + 1
                ranks.append(rank)

        ranks = np.array(ranks)

        if metric == "mrr":
            return float(np.mean(1.0 / ranks))
        elif metric == "hits@1":
            return float(np.mean(ranks <= 1))
        elif metric == "hits@10":
            return float(np.mean(ranks <= 10))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def predict_tail(
        self,
        head: str,
        relation: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Predict most likely tail entities for (head, relation, ?).

        Args:
            head: Head entity
            relation: Relation type
            top_k: Number of predictions

        Returns:
            List of (tail_entity, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained")

        if head not in self.entity2idx or relation not in self.relation2idx:
            return []

        self.model.eval()

        h_idx = torch.tensor([self.entity2idx[head]], device=self.device)
        r_idx = torch.tensor([self.relation2idx[relation]], device=self.device)

        with torch.no_grad():
            # Score all possible tails
            all_entities = torch.arange(len(self.entity2idx), device=self.device)
            h_expanded = h_idx.expand(len(all_entities))
            r_expanded = r_idx.expand(len(all_entities))

            scores = self.model(h_expanded, r_expanded, all_entities)

            # Get top-k
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))

        # Convert to entity names
        results = []
        for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()):
            entity = self.idx2entity[idx]
            results.append((entity, float(score)))

        return results

    def get_embedding(self, entity: str) -> Optional[np.ndarray]:
        """Get ComplEx embedding for an entity.

        Args:
            entity: Entity ID

        Returns:
            Embedding vector or None
        """
        if self.model is None or entity not in self.entity2idx:
            return None

        idx = self.entity2idx[entity]
        embedding = self.model.get_entity_embedding(idx)
        return embedding.detach().cpu().numpy()

    def save_model(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Save path
        """
        if self.model is None:
            raise ValueError("Model not trained")

        torch.save({
            "model_state": self.model.state_dict(),
            "entity2idx": self.entity2idx,
            "relation2idx": self.relation2idx,
            "config": self.config.dict(),
        }, path)

        logger.info(f"Saved ComplEx model to {path}")

    def load_model(self, path: Path) -> None:
        """Load model from disk.

        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.entity2idx = checkpoint["entity2idx"]
        self.idx2entity = {i: e for e, i in self.entity2idx.items()}
        self.relation2idx = checkpoint["relation2idx"]
        self.idx2relation = {i: r for r, i in self.relation2idx.items()}

        self.model = ComplExModel(
            num_entities=len(self.entity2idx),
            num_relations=len(self.relation2idx),
            embedding_dim=checkpoint["config"]["embedding_dim"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        logger.info(f"Loaded ComplEx model from {path}")


# Singleton instance
_complex_embeddings: Optional[ComplExEmbeddings] = None


def get_complex_embeddings(
    config: Optional[ComplExConfig] = None,
) -> ComplExEmbeddings:
    """Get or create singleton ComplExEmbeddings instance."""
    global _complex_embeddings
    if _complex_embeddings is None:
        _complex_embeddings = ComplExEmbeddings(config)
    return _complex_embeddings
