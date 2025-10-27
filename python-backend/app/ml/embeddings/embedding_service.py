"""Service for generating and retrieving entity embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.ml.config import MLConfig

logger = logging.getLogger(__name__)

try:
    import torch
    from sentence_transformers import SentenceTransformer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Text embeddings disabled.")


class EmbeddingService:
    """Service for managing entity embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
    ):
        """Initialize embedding service.

        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or MLConfig.EMBEDDINGS_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[SentenceTransformer] = None
        self.embedding_cache: Dict[str, np.ndarray] = {}

        if TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence transformer model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                self.model = None
        else:
            logger.warning("Sentence transformers not available")

    def encode_text(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: List of text strings
            normalize: Whether to normalize embeddings

        Returns:
            Array of embeddings [num_texts, embedding_dim]
        """
        if not self.model:
            # Return random embeddings as fallback
            logger.warning("Model not available, returning random embeddings")
            return np.random.randn(len(texts), 384).astype(np.float32)

        embeddings = self.model.encode(
            texts, normalize_embeddings=normalize, show_progress_bar=False
        )
        return embeddings

    def get_drug_embedding(
        self, drug_id: str, drug_name: str, description: Optional[str] = None
    ) -> np.ndarray:
        """Get embedding for a drug.

        Args:
            drug_id: Drug identifier
            drug_name: Drug name
            description: Optional drug description

        Returns:
            Embedding vector
        """
        cache_key = f"drug:{drug_id}"

        # Check cache
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Create text representation
        text = f"Drug: {drug_name}"
        if description:
            text += f". {description}"

        # Encode
        embedding = self.encode_text([text])[0]

        # Cache
        self.embedding_cache[cache_key] = embedding

        return embedding

    def get_disease_embedding(
        self, disease_id: str, disease_name: str, description: Optional[str] = None
    ) -> np.ndarray:
        """Get embedding for a disease.

        Args:
            disease_id: Disease identifier
            disease_name: Disease name
            description: Optional disease description

        Returns:
            Embedding vector
        """
        cache_key = f"disease:{disease_id}"

        # Check cache
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Create text representation
        text = f"Disease: {disease_name}"
        if description:
            text += f". {description}"

        # Encode
        embedding = self.encode_text([text])[0]

        # Cache
        self.embedding_cache[cache_key] = embedding

        return embedding

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        # Normalize
        emb1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        emb2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

        # Cosine similarity
        similarity = np.dot(emb1, emb2)

        return float(similarity)

    def find_similar_drugs(
        self,
        query_drug_id: str,
        query_drug_name: str,
        candidate_drugs: List[Tuple[str, str]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find similar drugs based on embeddings.

        Args:
            query_drug_id: Query drug ID
            query_drug_name: Query drug name
            candidate_drugs: List of (drug_id, drug_name) tuples
            top_k: Number of similar drugs to return

        Returns:
            List of (drug_id, similarity_score) tuples
        """
        # Get query embedding
        query_emb = self.get_drug_embedding(query_drug_id, query_drug_name)

        # Compute similarities
        similarities = []
        for drug_id, drug_name in candidate_drugs:
            if drug_id == query_drug_id:
                continue

            drug_emb = self.get_drug_embedding(drug_id, drug_name)
            similarity = self.compute_similarity(query_emb, drug_emb)
            similarities.append((drug_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def find_similar_diseases(
        self,
        query_disease_id: str,
        query_disease_name: str,
        candidate_diseases: List[Tuple[str, str]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find similar diseases based on embeddings.

        Args:
            query_disease_id: Query disease ID
            query_disease_name: Query disease name
            candidate_diseases: List of (disease_id, disease_name) tuples
            top_k: Number of similar diseases to return

        Returns:
            List of (disease_id, similarity_score) tuples
        """
        # Get query embedding
        query_emb = self.get_disease_embedding(query_disease_id, query_disease_name)

        # Compute similarities
        similarities = []
        for disease_id, disease_name in candidate_diseases:
            if disease_id == query_disease_id:
                continue

            disease_emb = self.get_disease_embedding(disease_id, disease_name)
            similarity = self.compute_similarity(query_emb, disease_emb)
            similarities.append((disease_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def save_embeddings(self, filepath: Path) -> None:
        """Save cached embeddings to disk.

        Args:
            filepath: Path to save embeddings
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(filepath, **self.embedding_cache)
        logger.info(f"Saved {len(self.embedding_cache)} embeddings to {filepath}")

    def load_embeddings(self, filepath: Path) -> None:
        """Load cached embeddings from disk.

        Args:
            filepath: Path to load embeddings from
        """
        if not filepath.exists():
            logger.warning(f"Embedding file not found: {filepath}")
            return

        data = np.load(filepath)
        self.embedding_cache = {key: data[key] for key in data.files}
        logger.info(f"Loaded {len(self.embedding_cache)} embeddings from {filepath}")


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
