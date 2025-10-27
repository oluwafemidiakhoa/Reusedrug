"""API endpoints for explainability and interpretability (Phase 5).

Provides endpoints for:
- SHAP feature importance
- Counterfactual explanations
- Attention visualization
- Molecular substructure highlighting
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.logging_conf import get_logger
from app.ml.explainability.shap_explainer import (
    get_shap_explainer,
    SHAPConfig,
)
from app.ml.explainability.counterfactual_generator import (
    get_counterfactual_generator,
    CounterfactualConfig,
)
from app.ml.explainability.attention_visualizer import (
    get_attention_visualizer,
    AttentionVisualizationConfig,
)
from app.ml.explainability.molecular_highlighter import (
    get_molecular_highlighter,
    MolecularHighlighterConfig,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/explainability", tags=["explainability"])


# ============================================================================
# Request/Response Models
# ============================================================================


class SHAPRequest(BaseModel):
    """Request for SHAP explanation."""

    drug_features: List[float] = Field(..., description="Drug features")
    disease_features: List[float] = Field(..., description="Disease features")
    feature_names: Optional[List[str]] = None


class SHAPResponse(BaseModel):
    """Response for SHAP explanation."""

    prediction: float
    base_value: float
    shap_values: List[float]
    top_positive_features: List[dict]
    top_negative_features: List[dict]
    num_drug_features: int
    num_disease_features: int


class CounterfactualRequest(BaseModel):
    """Request for counterfactual explanation."""

    drug_features: List[float]
    disease_features: List[float]
    target_class: Optional[int] = Field(default=None, description="0 or 1, None for flip")
    feature_names: Optional[List[str]] = None


class CounterfactualResponse(BaseModel):
    """Response for counterfactual explanation."""

    success: bool
    original_prediction: float
    target_prediction: float
    counterfactual_prediction: Optional[float] = None
    distance: Optional[float] = None
    num_changes: Optional[int] = None
    changes: Optional[dict] = None
    message: Optional[str] = None


class AttentionRequest(BaseModel):
    """Request for attention visualization."""

    drug_features: List[List[float]] = Field(..., description="[batch, d_drug]")
    disease_features: List[List[float]] = Field(..., description="[batch, d_disease]")


class AttentionResponse(BaseModel):
    """Response for attention visualization."""

    top_drug_features: List[dict]
    top_disease_features: List[dict]
    drug_focus_ratio: float
    attention_entropy: float
    num_drug_features: int
    num_disease_features: int


class MolecularHighlightRequest(BaseModel):
    """Request for molecular substructure highlighting."""

    smiles: str = Field(..., description="SMILES string of molecule")
    feature_importance: List[float] = Field(..., description="Feature importance scores")


class MolecularHighlightResponse(BaseModel):
    """Response for molecular highlighting."""

    smiles: str
    num_atoms: int
    top_atoms: List[dict]
    functional_groups: List[dict]
    molecular_weight: float
    has_aromatic_rings: bool


class MolecularComparisonRequest(BaseModel):
    """Request for molecular comparison."""

    smiles1: str
    smiles2: str
    feature_importance1: List[float]
    feature_importance2: List[float]


class MolecularComparisonResponse(BaseModel):
    """Response for molecular comparison."""

    molecule1: dict
    molecule2: dict
    common_groups: List[dict]
    different_groups: List[dict]
    structural_similarity: float


# ============================================================================
# SHAP Endpoints
# ============================================================================


@router.post("/shap/explain", response_model=SHAPResponse)
async def explain_with_shap(request: SHAPRequest):
    """Generate SHAP explanation for a prediction."""
    try:
        # Create mock predict function for demo
        def mock_predict(drug_feat, disease_feat):
            # Simple mock: return random predictions
            return np.random.rand(len(drug_feat)) * 0.5 + 0.25

        explainer = get_shap_explainer(predict_fn=mock_predict)

        drug_features = np.array(request.drug_features)
        disease_features = np.array(request.disease_features)

        explanation = explainer.explain(
            drug_features,
            disease_features,
            feature_names=request.feature_names,
        )

        return SHAPResponse(
            prediction=explanation["prediction"],
            base_value=explanation["base_value"],
            shap_values=explanation["shap_values"],
            top_positive_features=explanation["top_positive_features"],
            top_negative_features=explanation["top_negative_features"],
            num_drug_features=explanation["num_drug_features"],
            num_disease_features=explanation["num_disease_features"],
        )

    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/shap/global")
async def global_shap_importance(
    drug_features: List[List[float]],
    disease_features: List[List[float]],
    feature_names: Optional[List[str]] = None,
):
    """Compute global feature importance across multiple samples."""
    try:
        def mock_predict(drug_feat, disease_feat):
            return np.random.rand(len(drug_feat)) * 0.5 + 0.25

        explainer = get_shap_explainer(predict_fn=mock_predict)

        drug_feat = np.array(drug_features)
        disease_feat = np.array(disease_features)

        global_importance = explainer.global_importance(
            drug_feat,
            disease_feat,
            feature_names=feature_names,
        )

        return global_importance

    except Exception as e:
        logger.error(f"Global SHAP failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Counterfactual Endpoints
# ============================================================================


@router.post("/counterfactual/generate", response_model=CounterfactualResponse)
async def generate_counterfactual(request: CounterfactualRequest):
    """Generate counterfactual explanation."""
    try:
        def mock_predict(drug_feat, disease_feat):
            # Mock prediction based on feature sums
            combined = np.concatenate([drug_feat, disease_feat], axis=1)
            scores = np.sum(combined, axis=1) / combined.shape[1]
            return 1 / (1 + np.exp(-scores))  # Sigmoid

        config = CounterfactualConfig(target_class=request.target_class)
        generator = get_counterfactual_generator(predict_fn=mock_predict, config=config)

        drug_features = np.array(request.drug_features)
        disease_features = np.array(request.disease_features)

        result = generator.generate(
            drug_features,
            disease_features,
            feature_names=request.feature_names,
        )

        return CounterfactualResponse(**result)

    except Exception as e:
        logger.error(f"Counterfactual generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Attention Visualization Endpoints
# ============================================================================


@router.post("/attention/visualize", response_model=AttentionResponse)
async def visualize_attention(request: AttentionRequest):
    """Visualize attention weights from Transformer model."""
    try:
        visualizer = get_attention_visualizer()

        drug_features = np.array(request.drug_features)
        disease_features = np.array(request.disease_features)

        attention_data = visualizer.extract_attention(drug_features, disease_features)

        return AttentionResponse(
            top_drug_features=attention_data["top_drug_features"],
            top_disease_features=attention_data["top_disease_features"],
            drug_focus_ratio=attention_data["drug_focus_ratio"],
            attention_entropy=attention_data["attention_entropy"],
            num_drug_features=attention_data["num_drug_features"],
            num_disease_features=attention_data["num_disease_features"],
        )

    except Exception as e:
        logger.error(f"Attention visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/attention/cross-attention")
async def cross_attention_matrix(request: AttentionRequest):
    """Get cross-attention matrix between drug and disease features."""
    try:
        visualizer = get_attention_visualizer()

        drug_features = np.array(request.drug_features)
        disease_features = np.array(request.disease_features)

        cross_attention = visualizer.visualize_cross_attention(
            drug_features,
            disease_features,
        )

        return cross_attention

    except Exception as e:
        logger.error(f"Cross-attention visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Molecular Highlighting Endpoints
# ============================================================================


@router.post("/molecular/highlight", response_model=MolecularHighlightResponse)
async def highlight_molecule(request: MolecularHighlightRequest):
    """Highlight important molecular substructures."""
    try:
        highlighter = get_molecular_highlighter()

        feature_importance = np.array(request.feature_importance)

        highlights = highlighter.highlight_molecule(
            request.smiles,
            feature_importance,
        )

        return MolecularHighlightResponse(
            smiles=highlights["smiles"],
            num_atoms=highlights["num_atoms"],
            top_atoms=highlights["top_atoms"],
            functional_groups=highlights["functional_groups"],
            molecular_weight=highlights["molecular_weight"],
            has_aromatic_rings=highlights["has_aromatic_rings"],
        )

    except Exception as e:
        logger.error(f"Molecular highlighting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/molecular/compare", response_model=MolecularComparisonResponse)
async def compare_molecules(request: MolecularComparisonRequest):
    """Compare molecular substructures between two molecules."""
    try:
        highlighter = get_molecular_highlighter()

        importance1 = np.array(request.feature_importance1)
        importance2 = np.array(request.feature_importance2)

        comparison = highlighter.compare_molecules(
            request.smiles1,
            request.smiles2,
            importance1,
            importance2,
        )

        return MolecularComparisonResponse(**comparison)

    except Exception as e:
        logger.error(f"Molecular comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# General Metadata
# ============================================================================


@router.get("/metadata")
async def get_explainability_metadata():
    """Get metadata for all explainability features."""
    return {
        "shap": {
            "method": "kernel",
            "max_evals": 1000,
            "n_samples": 100,
        },
        "counterfactual": {
            "max_iterations": 100,
            "distance_metrics": ["l1", "l2", "cosine"],
            "num_counterfactuals": 3,
        },
        "attention": {
            "aggregation_methods": ["mean", "max", "sum"],
            "top_k_features": 20,
        },
        "molecular": {
            "fragment_size": 3,
            "top_k_fragments": 10,
            "rdkit_available": False,  # Updated dynamically
        },
        "capabilities": [
            "shap_explanations",
            "global_feature_importance",
            "counterfactual_generation",
            "attention_visualization",
            "cross_attention_analysis",
            "molecular_highlighting",
            "functional_group_identification",
            "molecule_comparison",
        ],
    }
