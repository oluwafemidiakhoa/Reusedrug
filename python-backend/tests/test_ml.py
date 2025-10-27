"""Tests for ML module."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from app.ml.config import MLConfig
from app.ml.embeddings.embedding_service import EmbeddingService
from app.ml.models.base import PredictionResult


class TestEmbeddingService:
    """Tests for embedding service."""

    def test_embedding_service_init(self):
        """Test embedding service initialization."""
        service = EmbeddingService()
        assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert service.cache_dir == MLConfig.EMBEDDINGS_DIR

    def test_encode_text(self):
        """Test text encoding."""
        service = EmbeddingService()
        texts = ["Drug: Aspirin", "Disease: Diabetes"]
        embeddings = service.encode_text(texts)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Embedding dimension
        assert embeddings.dtype == np.float32

    def test_get_drug_embedding(self):
        """Test drug embedding generation."""
        service = EmbeddingService()
        embedding = service.get_drug_embedding(
            drug_id="CHEMBL123",
            drug_name="Aspirin",
            description="Pain reliever"
        )

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0

    def test_get_disease_embedding(self):
        """Test disease embedding generation."""
        service = EmbeddingService()
        embedding = service.get_disease_embedding(
            disease_id="MONDO:0005148",
            disease_name="Type 2 Diabetes",
            description="Metabolic disorder"
        )

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0

    def test_compute_similarity(self):
        """Test similarity computation."""
        service = EmbeddingService()

        # Similar drugs
        emb1 = service.get_drug_embedding("CHEMBL123", "Aspirin")
        emb2 = service.get_drug_embedding("CHEMBL456", "Ibuprofen")
        similarity = service.compute_similarity(emb1, emb2)

        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be somewhat similar (both NSAIDs)

    def test_find_similar_drugs(self):
        """Test finding similar drugs."""
        service = EmbeddingService()

        candidates = [
            ("CHEMBL456", "Ibuprofen"),
            ("CHEMBL789", "Acetaminophen"),
            ("CHEMBL012", "Metformin"),  # Not similar (diabetes drug)
        ]

        similar = service.find_similar_drugs(
            query_drug_id="CHEMBL123",
            query_drug_name="Aspirin",
            candidate_drugs=candidates,
            top_k=2
        )

        assert len(similar) == 2
        # First result should be Ibuprofen (most similar to Aspirin)
        assert similar[0][0] == "CHEMBL456"
        assert similar[0][1] > 0.5

    def test_embedding_cache(self):
        """Test embedding caching."""
        service = EmbeddingService()

        # First call
        emb1 = service.get_drug_embedding("CHEMBL123", "Aspirin")

        # Second call (should use cache)
        emb2 = service.get_drug_embedding("CHEMBL123", "Aspirin")

        # Should be exact same object (from cache)
        assert np.array_equal(emb1, emb2)
        assert "drug:CHEMBL123" in service.embedding_cache

    def test_save_load_embeddings(self, tmp_path):
        """Test saving and loading embeddings."""
        service = EmbeddingService()

        # Generate some embeddings
        service.get_drug_embedding("CHEMBL123", "Aspirin")
        service.get_disease_embedding("MONDO:0005148", "Diabetes")

        # Save
        save_path = tmp_path / "embeddings.npz"
        service.save_embeddings(save_path)
        assert save_path.exists()

        # Load into new service
        new_service = EmbeddingService()
        new_service.load_embeddings(save_path)

        assert len(new_service.embedding_cache) == 2
        assert "drug:CHEMBL123" in new_service.embedding_cache
        assert "disease:MONDO:0005148" in new_service.embedding_cache


class TestMLConfig:
    """Tests for ML configuration."""

    def test_ml_config_defaults(self):
        """Test default ML configuration values."""
        assert MLConfig.GNN_EMBEDDING_DIM == 128
        assert MLConfig.GNN_HIDDEN_DIM == 256
        assert MLConfig.GNN_NUM_LAYERS == 3
        assert MLConfig.GNN_DROPOUT == 0.2
        assert MLConfig.BATCH_SIZE == 32
        assert MLConfig.LEARNING_RATE == 0.001

    def test_ml_config_directories_exist(self):
        """Test that ML directories are created."""
        assert MLConfig.MODEL_DIR.exists()
        assert MLConfig.EMBEDDINGS_DIR.exists()


class TestPredictionResult:
    """Tests for PredictionResult model."""

    def test_prediction_result_creation(self):
        """Test creating a prediction result."""
        result = PredictionResult(
            drug_id="CHEMBL123",
            disease_id="MONDO:0005148",
            score=0.85,
            confidence_low=0.75,
            confidence_high=0.95,
            model_name="gnn",
            features_used=["graph_structure", "embeddings"],
            metadata={"device": "cpu"}
        )

        assert result.drug_id == "CHEMBL123"
        assert result.disease_id == "MONDO:0005148"
        assert result.score == 0.85
        assert result.model_name == "gnn"
        assert len(result.features_used) == 2

    def test_prediction_result_defaults(self):
        """Test prediction result with defaults."""
        result = PredictionResult(
            drug_id="CHEMBL123",
            disease_id="MONDO:0005148",
            score=0.5,
            model_name="baseline"
        )

        assert result.confidence_low is None
        assert result.confidence_high is None
        assert result.features_used == []
        assert result.metadata == {}


# Note: GNN predictor tests require PyTorch and are more complex
# They would test model initialization, forward pass, save/load
# Skipped here to avoid torch dependency in test environment
