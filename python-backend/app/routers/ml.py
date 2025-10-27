"""ML prediction and embedding endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.embeddings.embedding_service import get_embedding_service
from app.ml.models.base import PredictionResult
from app.ml.models.gnn_predictor import GNNPredictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/ml", tags=["machine-learning"])

# Global trained model
_trained_gnn: Optional[GNNPredictor] = None


# Request/Response models
class SimilarityRequest(BaseModel):
    """Request for finding similar entities."""

    entity_type: str = Field(..., description="Entity type: 'drug' or 'disease'")
    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(..., description="Entity name")
    candidates: List[tuple[str, str]] = Field(
        ..., description="List of (id, name) candidate tuples"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")


class SimilarityResult(BaseModel):
    """Similarity search result."""

    entity_id: str
    entity_name: Optional[str] = None
    similarity_score: float
    rank: int


class SimilarityResponse(BaseModel):
    """Response for similarity search."""

    query_entity_id: str
    query_entity_name: str
    entity_type: str
    results: List[SimilarityResult]


class PredictionRequest(BaseModel):
    """Request for ML prediction."""

    drug_id: str = Field(..., description="Drug identifier")
    disease_id: str = Field(..., description="Disease identifier")
    drug_name: Optional[str] = None
    disease_name: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Request for batch ML predictions."""

    pairs: List[tuple[str, str]] = Field(
        ..., description="List of (drug_id, disease_id) pairs"
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=1000)


class MLMetadata(BaseModel):
    """ML system metadata."""

    models_available: List[str]
    embedding_model: str
    features: dict


# Endpoints


@router.post("/similar/drugs", response_model=SimilarityResponse)
async def find_similar_drugs(request: SimilarityRequest) -> SimilarityResponse:
    """Find similar drugs using embeddings.

    This endpoint uses sentence transformer embeddings to find drugs
    with similar names/descriptions.
    """
    if request.entity_type != "drug":
        raise HTTPException(
            status_code=400, detail="entity_type must be 'drug' for this endpoint"
        )

    embedding_service = get_embedding_service()

    try:
        similar = embedding_service.find_similar_drugs(
            query_drug_id=request.entity_id,
            query_drug_name=request.entity_name,
            candidate_drugs=request.candidates,
            top_k=request.top_k,
        )

        results = [
            SimilarityResult(
                entity_id=drug_id,
                similarity_score=score,
                rank=i + 1,
            )
            for i, (drug_id, score) in enumerate(similar)
        ]

        return SimilarityResponse(
            query_entity_id=request.entity_id,
            query_entity_name=request.entity_name,
            entity_type="drug",
            results=results,
        )

    except Exception as e:
        logger.error(f"Error finding similar drugs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {e}")


@router.post("/similar/diseases", response_model=SimilarityResponse)
async def find_similar_diseases(request: SimilarityRequest) -> SimilarityResponse:
    """Find similar diseases using embeddings.

    This endpoint uses sentence transformer embeddings to find diseases
    with similar names/descriptions.
    """
    if request.entity_type != "disease":
        raise HTTPException(
            status_code=400, detail="entity_type must be 'disease' for this endpoint"
        )

    embedding_service = get_embedding_service()

    try:
        similar = embedding_service.find_similar_diseases(
            query_disease_id=request.entity_id,
            query_disease_name=request.entity_name,
            candidate_diseases=request.candidates,
            top_k=request.top_k,
        )

        results = [
            SimilarityResult(
                entity_id=disease_id,
                similarity_score=score,
                rank=i + 1,
            )
            for i, (disease_id, score) in enumerate(similar)
        ]

        return SimilarityResponse(
            query_entity_id=request.entity_id,
            query_entity_name=request.entity_name,
            entity_type="disease",
            results=results,
        )

    except Exception as e:
        logger.error(f"Error finding similar diseases: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {e}")


async def load_trained_model() -> None:
    """Load trained GNN model on startup."""
    global _trained_gnn

    model_path = Path("data/models/gnn_best.pt")
    if model_path.exists():
        try:
            _trained_gnn = GNNPredictor()
            _trained_gnn.load(model_path)
            logger.info(f"Loaded trained GNN model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load GNN model: {e}")
            _trained_gnn = None
    else:
        logger.info(f"No trained GNN model found at {model_path}")


@router.post("/predict", response_model=PredictionResult)
async def predict_association(request: PredictionRequest) -> PredictionResult:
    """Predict drug-disease association using trained GNN model.

    Uses the trained Graph Neural Network to predict the likelihood
    of a drug being effective for a disease based on knowledge graph structure.
    """
    if _trained_gnn is None:
        # Fallback to baseline if model not loaded
        return PredictionResult(
            drug_id=request.drug_id,
            disease_id=request.disease_id,
            score=0.5,
            confidence_low=0.3,
            confidence_high=0.7,
            model_name="baseline",
            features_used=["none"],
            metadata={
                "status": "model_not_loaded",
                "message": "GNN model not loaded. Train model with: python scripts/train_gnn.py",
            },
        )

    # Use trained model
    try:
        result = _trained_gnn.predict(request.drug_id, request.disease_id)
        return result
    except Exception as e:
        logger.error(f"GNN prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch", response_model=List[PredictionResult])
async def predict_batch(
    request: BatchPredictionRequest,
) -> List[PredictionResult]:
    """Batch prediction for multiple drug-disease pairs using trained GNN."""
    if _trained_gnn is None:
        # Fallback to baseline
        results = []
        for drug_id, disease_id in request.pairs:
            results.append(PredictionResult(
                drug_id=drug_id,
                disease_id=disease_id,
                score=0.5,
                model_name="baseline",
                metadata={"status": "model_not_loaded"},
            ))
    else:
        # Use trained model
        try:
            results = _trained_gnn.predict_batch(request.pairs)
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Batch prediction failed: {str(e)}"
            )

    if request.top_k:
        results = sorted(results, key=lambda r: r.score, reverse=True)[: request.top_k]

    return results


@router.get("/metadata", response_model=MLMetadata)
async def get_ml_metadata() -> MLMetadata:
    """Get ML system metadata and available features."""
    try:
        from app.ml.models.gnn_predictor import TORCH_AVAILABLE
        from app.ml.embeddings.embedding_service import TRANSFORMERS_AVAILABLE
    except ImportError:
        TORCH_AVAILABLE = False
        TRANSFORMERS_AVAILABLE = False

    models_available = []
    if TORCH_AVAILABLE and _trained_gnn is not None:
        models_available.append("gnn_trained")
    elif TORCH_AVAILABLE:
        models_available.append("gnn_framework")
    if TRANSFORMERS_AVAILABLE:
        models_available.append("embeddings")

    return MLMetadata(
        models_available=models_available,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        features={
            "similarity_search": TRANSFORMERS_AVAILABLE,
            "gnn_prediction": _trained_gnn is not None,
            "batch_prediction": True,
            "gnn_trained": _trained_gnn is not None,
        },
    )
