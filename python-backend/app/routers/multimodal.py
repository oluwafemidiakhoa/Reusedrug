"""Multi-modal feature extraction and fusion API endpoints."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.multimodal.chemical_analyzer import (
    ChemicalFeatures,
    get_chemical_analyzer,
)
from app.ml.multimodal.clinical_trials import (
    TrialEvidence,
    get_clinical_trial_pipeline,
)
from app.ml.multimodal.fusion import (
    MultiModalFeatures,
    get_multimodal_fusion,
)
from app.ml.multimodal.gene_expression import (
    GeneExpressionProfile,
    GeneExpressionSimilarity,
    get_gene_expression_integrator,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/multimodal", tags=["multimodal"])


# Request/Response Models
class ChemicalAnalysisRequest(BaseModel):
    """Request for chemical structure analysis."""

    drug_id: str
    smiles: str = Field(..., description="SMILES notation of molecule")


class GeneExpressionRequest(BaseModel):
    """Request for gene expression analysis."""

    drug_id: str
    disease_id: str


class ClinicalTrialRequest(BaseModel):
    """Request for clinical trial evidence."""

    drug_id: str
    disease_id: str


class MultiModalPredictionRequest(BaseModel):
    """Request for multi-modal prediction."""

    drug_id: str
    disease_id: str
    drug_smiles: Optional[str] = None
    drug_name: Optional[str] = None
    disease_name: Optional[str] = None


class BatchMultiModalRequest(BaseModel):
    """Request for batch multi-modal predictions."""

    pairs: List[tuple[str, str]] = Field(..., description="List of (drug_id, disease_id)")
    smiles_map: Optional[Dict[str, str]] = Field(
        None, description="Optional mapping of drug_id to SMILES"
    )
    top_k: Optional[int] = Field(None, ge=1, le=100)


class MultiModalResponse(BaseModel):
    """Response with multi-modal prediction and explanation."""

    prediction: MultiModalFeatures
    explanation: Dict[str, str]


# Endpoints


@router.post("/chemical/analyze", response_model=ChemicalFeatures)
async def analyze_chemical_structure(
    request: ChemicalAnalysisRequest,
) -> ChemicalFeatures:
    """Analyze chemical structure and extract molecular features.

    This endpoint uses RDKit to:
    - Validate SMILES notation
    - Compute molecular descriptors (MW, LogP, H-donors/acceptors, etc.)
    - Generate Morgan fingerprints for similarity search
    - Check Lipinski's Rule of Five
    """
    analyzer = get_chemical_analyzer()

    try:
        features = analyzer.analyze_smiles(request.smiles, request.drug_id)
        return features
    except Exception as e:
        logger.error(f"Chemical analysis failed for {request.drug_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/chemical/similarity")
async def compute_chemical_similarity(
    drug1_id: str,
    drug1_smiles: str,
    drug2_id: str,
    drug2_smiles: str,
    method: str = "tanimoto",
) -> Dict[str, float]:
    """Compute structural similarity between two molecules.

    Similarity methods:
    - tanimoto: Jaccard/Tanimoto coefficient (most common)
    - dice: Dice coefficient
    - cosine: Cosine similarity
    """
    analyzer = get_chemical_analyzer()

    try:
        similarity = analyzer.compute_similarity(
            drug1_smiles, drug2_smiles, method=method
        )

        return {
            "drug1_id": drug1_id,
            "drug2_id": drug2_id,
            "similarity": similarity,
            "method": method,
        }
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chemical/druglikeness")
async def check_drug_likeness(
    drug_id: str,
    smiles: str,
):
    """Check Lipinski's Rule of Five for drug-likeness.

    Rule of Five criteria:
    - Molecular weight < 500 Da
    - LogP < 5
    - H-bond donors <= 5
    - H-bond acceptors <= 10

    Passing requires no more than 1 violation.
    """
    analyzer = get_chemical_analyzer()

    try:
        result = analyzer.check_lipinski_rule_of_five(smiles)
        return result
    except Exception as e:
        logger.error(f"Drug-likeness check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gene-expression/analyze", response_model=GeneExpressionSimilarity)
async def analyze_gene_expression(
    request: GeneExpressionRequest,
) -> GeneExpressionSimilarity:
    """Analyze gene expression similarity between drug and disease.

    Uses the "connectivity map" approach:
    - Negative correlation suggests therapeutic potential
    - Drug reverses disease expression signature
    """
    integrator = get_gene_expression_integrator()

    try:
        drug_profile = integrator.load_expression_profile(request.drug_id, "drug")
        disease_profile = integrator.load_expression_profile(request.disease_id, "disease")

        if drug_profile is None or disease_profile is None:
            raise HTTPException(
                status_code=404,
                detail="Expression profiles not available",
            )

        similarity = integrator.compute_expression_similarity(drug_profile, disease_profile)
        return similarity

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gene expression analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gene-expression/profile/{entity_type}/{entity_id}", response_model=GeneExpressionProfile)
async def get_expression_profile(
    entity_type: str,
    entity_id: str,
) -> GeneExpressionProfile:
    """Get gene expression profile for a drug or disease.

    entity_type: 'drug' or 'disease'
    """
    if entity_type not in ["drug", "disease"]:
        raise HTTPException(
            status_code=400,
            detail="entity_type must be 'drug' or 'disease'",
        )

    integrator = get_gene_expression_integrator()

    try:
        profile = integrator.load_expression_profile(entity_id, entity_type)

        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found for {entity_type} {entity_id}",
            )

        return profile

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clinical-trials/evidence", response_model=TrialEvidence)
async def get_clinical_evidence(
    request: ClinicalTrialRequest,
) -> TrialEvidence:
    """Get clinical trial evidence for drug-disease association.

    Returns:
    - Number of trials (all phases)
    - Highest phase reached
    - Positive/negative results
    - Overall evidence score
    """
    pipeline = get_clinical_trial_pipeline()

    try:
        evidence = pipeline.get_trial_evidence(request.drug_id, request.disease_id)
        return evidence

    except Exception as e:
        logger.error(f"Clinical trial lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=MultiModalResponse)
async def predict_multimodal(
    request: MultiModalPredictionRequest,
) -> MultiModalResponse:
    """Multi-modal drug repurposing prediction.

    Integrates:
    1. Chemical structure analysis (molecular descriptors, fingerprints)
    2. Gene expression profiles (connectivity map approach)
    3. Clinical trial evidence (phase, outcomes, success rate)

    Returns integrated prediction with explanations.
    """
    fusion = get_multimodal_fusion()

    try:
        # Fuse multi-modal features
        features = fusion.fuse_features(
            request.drug_id,
            request.disease_id,
            request.drug_smiles,
        )

        # Generate explanation
        explanation = fusion.explain_prediction(features)

        return MultiModalResponse(
            prediction=features,
            explanation=explanation,
        )

    except Exception as e:
        logger.error(f"Multi-modal prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch")
async def predict_multimodal_batch(
    request: BatchMultiModalRequest,
) -> List[MultiModalResponse]:
    """Batch multi-modal predictions for multiple drug-disease pairs.

    Optionally ranks by integrated score and returns top-k.
    """
    fusion = get_multimodal_fusion()

    try:
        # Batch fuse
        features_list = fusion.batch_fuse(request.pairs, request.smiles_map)

        # Create responses with explanations
        responses = []
        for features in features_list:
            explanation = fusion.explain_prediction(features)
            responses.append(
                MultiModalResponse(
                    prediction=features,
                    explanation=explanation,
                )
            )

        # Sort by score descending
        responses.sort(
            key=lambda r: r.prediction.integrated_score,
            reverse=True,
        )

        # Apply top_k if specified
        if request.top_k:
            responses = responses[: request.top_k]

        return responses

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata")
async def get_multimodal_metadata():
    """Get metadata about multi-modal capabilities."""
    analyzer = get_chemical_analyzer()

    return {
        "modalities": [
            "chemical_structure",
            "gene_expression",
            "clinical_trials",
        ],
        "chemical_features": {
            "rdkit_available": analyzer.enabled,
            "fingerprint_type": "Morgan (circular)",
            "fingerprint_radius": analyzer.fingerprint_radius,
            "fingerprint_bits": analyzer.fingerprint_bits,
            "descriptors": [
                "molecular_weight",
                "logp",
                "h_donors",
                "h_acceptors",
                "rotatable_bonds",
                "tpsa",
                "aromatic_rings",
                "ring_count",
                "fraction_csp3",
                "stereocenters",
            ],
        },
        "gene_expression": {
            "data_sources": ["GEO", "LINCS", "CREEDS"],
            "correlation_methods": ["pearson", "spearman", "cosine"],
        },
        "clinical_trials": {
            "data_sources": ["ClinicalTrials.gov", "EU-CTR", "WHO-ICTRP"],
            "phases_tracked": ["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
        },
        "fusion": {
            "strategies": ["weighted", "early", "late", "hybrid"],
            "default_strategy": "weighted",
            "default_weights": {
                "chemical": 0.25,
                "gene_expression": 0.35,
                "clinical_trials": 0.40,
            },
        },
    }
