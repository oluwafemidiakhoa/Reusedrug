"""API endpoints for enhanced ML models (Phase 4).

Provides endpoints for:
- Transformer-based predictions
- Meta-learning (few-shot)
- Ensemble predictions with uncertainty
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.logging_conf import get_logger
from app.ml.enhanced_models.transformer_predictor import (
    get_transformer_predictor,
    get_transformer_trainer,
    TransformerConfig,
    DrugDiseaseDataset,
)
from app.ml.enhanced_models.meta_learner import (
    get_meta_learner,
    MAMLConfig,
    TaskSampler,
)
from app.ml.enhanced_models.ensemble_predictor import (
    get_ensemble_predictor,
    EnsembleConfig,
    EnsembleMethod,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/enhanced-ml", tags=["enhanced-ml"])


# ============================================================================
# Request/Response Models
# ============================================================================


class TransformerPredictRequest(BaseModel):
    """Request for transformer prediction."""

    drug_features: List[List[float]] = Field(..., description="Drug features [batch, 2048]")
    disease_features: List[List[float]] = Field(..., description="Disease features [batch, 768]")


class TransformerPredictResponse(BaseModel):
    """Response for transformer prediction."""

    predictions: List[float] = Field(..., description="Predicted probabilities")
    num_predictions: int


class TransformerTrainRequest(BaseModel):
    """Request for transformer training."""

    drug_features: List[List[float]]
    disease_features: List[List[float]]
    labels: List[float]
    num_epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=1e-4, gt=0, lt=1)


class MetaLearnRequest(BaseModel):
    """Request for meta-learning (few-shot prediction)."""

    support_drug_features: List[List[float]] = Field(..., description="Support set drugs")
    support_disease_features: List[List[float]] = Field(..., description="Support set diseases")
    support_labels: List[float] = Field(..., description="Support set labels")
    query_drug_features: List[List[float]] = Field(..., description="Query set drugs")
    query_disease_features: List[List[float]] = Field(..., description="Query set diseases")
    num_adaptation_steps: int = Field(default=10, ge=1, le=50)


class MetaLearnResponse(BaseModel):
    """Response for meta-learning prediction."""

    predictions: List[float]
    num_support: int
    num_query: int
    adaptation_steps: int


class EnsemblePredictRequest(BaseModel):
    """Request for ensemble prediction."""

    drug_features: List[List[float]]
    disease_features: List[List[float]]
    method: str = Field(default="weighted", description="Ensemble method")
    return_uncertainty: bool = Field(default=True)
    return_individual: bool = Field(default=False)
    confidence_threshold: Optional[float] = Field(default=None, ge=0, le=1)


class EnsemblePredictResponse(BaseModel):
    """Response for ensemble prediction."""

    predictions: List[float]
    uncertainty: Optional[List[float]] = None
    confidence: Optional[List[float]] = None
    lower_ci: Optional[List[float]] = None
    upper_ci: Optional[List[float]] = None
    diversity_score: Optional[float] = None
    individual_predictions: Optional[dict] = None
    high_confidence_mask: Optional[List[bool]] = None
    num_high_confidence: Optional[int] = None


class ModelContributionsResponse(BaseModel):
    """Response for model contributions analysis."""

    contributions: dict = Field(..., description="Per-model contribution statistics")
    ensemble_method: str


# ============================================================================
# Transformer Endpoints
# ============================================================================


@router.post("/transformer/predict", response_model=TransformerPredictResponse)
async def predict_transformer(request: TransformerPredictRequest):
    """Make predictions using Transformer model."""
    try:
        predictor = get_transformer_predictor()

        drug_features = np.array(request.drug_features)
        disease_features = np.array(request.disease_features)

        if len(drug_features) != len(disease_features):
            raise HTTPException(
                status_code=400,
                detail="Drug and disease features must have same batch size",
            )

        predictions = predictor.predict_proba(drug_features, disease_features)

        return TransformerPredictResponse(
            predictions=predictions.tolist(),
            num_predictions=len(predictions),
        )

    except Exception as e:
        logger.error(f"Transformer prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transformer/train")
async def train_transformer(request: TransformerTrainRequest):
    """Train Transformer model on provided data."""
    try:
        from torch.utils.data import DataLoader

        # Get predictor and trainer
        predictor = get_transformer_predictor()
        trainer = get_transformer_trainer(model=predictor, learning_rate=request.learning_rate)

        # Prepare data
        drug_features = np.array(request.drug_features)
        disease_features = np.array(request.disease_features)
        labels = np.array(request.labels)

        # Create dataset and dataloader
        dataset = DrugDiseaseDataset(drug_features, disease_features, labels)
        dataloader = DataLoader(dataset, batch_size=request.batch_size, shuffle=True)

        # Train
        history = trainer.train(dataloader, num_epochs=request.num_epochs)

        return {
            "status": "Training complete",
            "num_epochs": request.num_epochs,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
            "history": {
                "train_loss": [float(x) for x in history["train_loss"]],
            },
        }

    except Exception as e:
        logger.error(f"Transformer training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transformer/metadata")
async def get_transformer_metadata():
    """Get Transformer model metadata."""
    try:
        predictor = get_transformer_predictor()
        config = predictor.config

        return {
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_layers": config.num_encoder_layers,
            "dim_feedforward": config.dim_feedforward,
            "dropout": config.dropout,
            "num_drug_features": config.num_drug_features,
            "num_disease_features": config.num_disease_features,
            "total_parameters": sum(p.numel() for p in predictor.parameters()),
        }

    except Exception as e:
        logger.error(f"Failed to get transformer metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Meta-Learning Endpoints
# ============================================================================


@router.post("/meta-learning/predict", response_model=MetaLearnResponse)
async def few_shot_predict(request: MetaLearnRequest):
    """Few-shot prediction using meta-learning."""
    try:
        meta_learner = get_meta_learner()

        # Prepare support set (few labeled examples)
        support_drug = np.array(request.support_drug_features)
        support_disease = np.array(request.support_disease_features)
        support_x = np.concatenate([support_drug, support_disease], axis=1)
        support_y = np.array(request.support_labels)

        # Prepare query set (to predict)
        query_drug = np.array(request.query_drug_features)
        query_disease = np.array(request.query_disease_features)
        query_x = np.concatenate([query_drug, query_disease], axis=1)

        # Few-shot prediction
        predictions = meta_learner.few_shot_predict(
            support_x,
            support_y,
            query_x,
            num_adaptation_steps=request.num_adaptation_steps,
        )

        return MetaLearnResponse(
            predictions=predictions.tolist(),
            num_support=len(support_y),
            num_query=len(query_x),
            adaptation_steps=request.num_adaptation_steps,
        )

    except Exception as e:
        logger.error(f"Meta-learning prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/meta-learning/metadata")
async def get_meta_learner_metadata():
    """Get meta-learner metadata."""
    try:
        meta_learner = get_meta_learner()
        config = meta_learner.config

        return {
            "input_dim": config.input_dim,
            "hidden_dims": config.hidden_dims,
            "inner_lr": config.inner_lr,
            "meta_lr": config.meta_lr,
            "num_inner_steps": config.num_inner_steps,
            "num_support": config.num_support,
            "num_query": config.num_query,
            "total_parameters": sum(
                p.numel() for p in meta_learner.meta_model.parameters()
            ),
        }

    except Exception as e:
        logger.error(f"Failed to get meta-learner metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Ensemble Endpoints
# ============================================================================


@router.post("/ensemble/predict", response_model=EnsemblePredictResponse)
async def predict_ensemble(request: EnsemblePredictRequest):
    """Make ensemble predictions with uncertainty quantification."""
    try:
        # Get ensemble predictor
        config = EnsembleConfig(method=EnsembleMethod(request.method))
        ensemble = get_ensemble_predictor(config)

        drug_features = np.array(request.drug_features)
        disease_features = np.array(request.disease_features)

        # Register mock models for demonstration
        # In production, these would be actual trained models
        def mock_gnn_predict(drug_feat, disease_feat):
            # Simulate GNN predictions
            return np.random.rand(len(drug_feat)) * 0.7 + 0.2

        def mock_transformer_predict(drug_feat, disease_feat):
            # Simulate transformer predictions
            return np.random.rand(len(drug_feat)) * 0.6 + 0.25

        def mock_multimodal_predict(drug_feat, disease_feat):
            # Simulate multimodal predictions
            return np.random.rand(len(drug_feat)) * 0.5 + 0.3

        ensemble.register_model("gnn", mock_gnn_predict, weight=0.4)
        ensemble.register_model("transformer", mock_transformer_predict, weight=0.35)
        ensemble.register_model("multimodal", mock_multimodal_predict, weight=0.25)

        # Make predictions
        if request.confidence_threshold is not None:
            results = ensemble.predict_with_confidence(
                drug_features,
                disease_features,
                confidence_threshold=request.confidence_threshold,
            )
        else:
            results = ensemble.predict(
                drug_features,
                disease_features,
                return_uncertainty=request.return_uncertainty,
                return_individual=request.return_individual,
            )

        # Build response
        response = EnsemblePredictResponse(
            predictions=results["predictions"].tolist(),
        )

        if "uncertainty" in results:
            response.uncertainty = results["uncertainty"].tolist()

        if "confidence" in results:
            response.confidence = results["confidence"].tolist()

        if "lower_ci" in results:
            response.lower_ci = results["lower_ci"].tolist()
            response.upper_ci = results["upper_ci"].tolist()

        if "diversity_score" in results:
            response.diversity_score = float(results["diversity_score"])

        if "individual_predictions" in results:
            response.individual_predictions = {
                k: v.tolist() for k, v in results["individual_predictions"].items()
            }

        if "high_confidence_mask" in results:
            response.high_confidence_mask = results["high_confidence_mask"].tolist()
            response.num_high_confidence = int(results["num_high_confidence"])

        return response

    except Exception as e:
        logger.error(f"Ensemble prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ensemble/contributions")
async def get_model_contributions(request: EnsemblePredictRequest):
    """Analyze individual model contributions to ensemble."""
    try:
        ensemble = get_ensemble_predictor()

        drug_features = np.array(request.drug_features)
        disease_features = np.array(request.disease_features)

        # Register models (same as above)
        def mock_gnn_predict(drug_feat, disease_feat):
            return np.random.rand(len(drug_feat)) * 0.7 + 0.2

        def mock_transformer_predict(drug_feat, disease_feat):
            return np.random.rand(len(drug_feat)) * 0.6 + 0.25

        ensemble.register_model("gnn", mock_gnn_predict)
        ensemble.register_model("transformer", mock_transformer_predict)

        contributions = ensemble.get_model_contributions(drug_features, disease_features)

        return ModelContributionsResponse(
            contributions=contributions,
            ensemble_method=ensemble.config.method.value,
        )

    except Exception as e:
        logger.error(f"Failed to get model contributions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ensemble/metadata")
async def get_ensemble_metadata():
    """Get ensemble model metadata."""
    try:
        ensemble = get_ensemble_predictor()
        config = ensemble.config

        return {
            "method": config.method.value,
            "model_weights": config.model_weights,
            "num_registered_models": len(ensemble.models),
            "registered_models": list(ensemble.models.keys()),
            "calibrate_uncertainty": config.calibrate_uncertainty,
            "diversity_bonus": config.diversity_bonus,
            "min_models": config.min_models,
        }

    except Exception as e:
        logger.error(f"Failed to get ensemble metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# General Metadata
# ============================================================================


@router.get("/metadata")
async def get_enhanced_ml_metadata():
    """Get metadata for all enhanced ML features."""
    try:
        return {
            "transformer": await get_transformer_metadata(),
            "meta_learning": await get_meta_learner_metadata(),
            "ensemble": await get_ensemble_metadata(),
            "capabilities": [
                "transformer_predictions",
                "few_shot_learning",
                "ensemble_methods",
                "uncertainty_quantification",
                "model_stacking",
                "confidence_calibration",
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get enhanced ML metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))
