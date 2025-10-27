"""Multi-modal feature fusion for drug repurposing predictions."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from app.ml.multimodal.chemical_analyzer import (
    ChemicalFeatures,
    get_chemical_analyzer,
)
from app.ml.multimodal.clinical_trials import (
    TrialEvidence,
    get_clinical_trial_pipeline,
)
from app.ml.multimodal.gene_expression import (
    GeneExpressionProfile,
    get_gene_expression_integrator,
)

logger = logging.getLogger(__name__)


class MultiModalFeatures(BaseModel):
    """Combined multi-modal features for a drug-disease pair."""

    drug_id: str
    disease_id: str

    # Chemical features
    chemical_features: Optional[ChemicalFeatures] = None
    chemical_similarity_score: float = 0.0

    # Gene expression features
    gene_expression_correlation: float = 0.0
    gene_overlap_score: float = 0.0
    pathway_enrichment_score: float = 0.0

    # Clinical trial features
    clinical_evidence_score: float = 0.0
    trial_phase_score: float = 0.0
    trial_success_rate: float = 0.0

    # Fusion scores
    integrated_score: float = Field(
        ..., ge=0.0, le=1.0, description="Fused multi-modal prediction score"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction confidence"
    )
    feature_contributions: Dict[str, float] = Field(
        default_factory=dict, description="Contribution of each modality"
    )


class FusionStrategy(str):
    """Fusion strategy for combining multi-modal features."""

    EARLY = "early"  # Concatenate features, single model
    LATE = "late"  # Separate models, combine predictions
    HYBRID = "hybrid"  # Attention-based fusion
    WEIGHTED = "weighted"  # Weighted average with learned weights


class MultiModalFusion:
    """Fuses multi-modal data for drug-disease prediction."""

    def __init__(
        self,
        fusion_strategy: str = FusionStrategy.WEIGHTED,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize multi-modal fusion.

        Args:
            fusion_strategy: Strategy for combining modalities
            weights: Optional modality weights (for weighted fusion)
        """
        self.fusion_strategy = fusion_strategy
        self.chemical_analyzer = get_chemical_analyzer()
        self.gene_integrator = get_gene_expression_integrator()
        self.clinical_pipeline = get_clinical_trial_pipeline()

        # Default weights for weighted fusion
        self.weights = weights or {
            "chemical": 0.25,
            "gene_expression": 0.35,
            "clinical_trials": 0.40,
        }

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def extract_chemical_features(
        self,
        drug_id: str,
        smiles: Optional[str] = None,
    ) -> Tuple[Optional[ChemicalFeatures], float]:
        """Extract chemical structure features.

        Args:
            drug_id: Drug identifier
            smiles: Optional SMILES string

        Returns:
            Tuple of (ChemicalFeatures, score)
        """
        if smiles is None:
            # TODO: Fetch SMILES from database (PubChem, ChEMBL)
            logger.warning(f"No SMILES provided for {drug_id}")
            return None, 0.0

        features = self.chemical_analyzer.analyze_smiles(smiles, drug_id)

        if not features.is_valid:
            return features, 0.0

        # Compute a basic drug-likeness score
        if features.molecular_descriptors:
            desc = features.molecular_descriptors

            # Check Lipinski's Rule of Five
            violations = 0
            if desc.molecular_weight > 500:
                violations += 1
            if desc.logp > 5:
                violations += 1
            if desc.num_h_donors > 5:
                violations += 1
            if desc.num_h_acceptors > 10:
                violations += 1

            # Score: 1.0 if no violations, decreases with violations
            score = max(0.0, 1.0 - (violations * 0.25))
        else:
            score = 0.5  # Neutral score

        return features, score

    def extract_gene_expression_features(
        self,
        drug_id: str,
        disease_id: str,
    ) -> Dict[str, float]:
        """Extract gene expression-based features.

        Args:
            drug_id: Drug identifier
            disease_id: Disease identifier

        Returns:
            Dictionary with gene expression scores
        """
        # Get expression profiles
        drug_profile = self.gene_integrator.load_expression_profile(drug_id, "drug")
        disease_profile = self.gene_integrator.load_expression_profile(
            disease_id, "disease"
        )

        if drug_profile is None or disease_profile is None:
            return {
                "correlation": 0.0,
                "overlap_score": 0.0,
                "pathway_score": 0.0,
            }

        # Compute expression similarity (negative = therapeutic)
        similarity = self.gene_integrator.compute_expression_similarity(
            drug_profile, disease_profile
        )

        # Reversal score (drug reverses disease signature)
        reversal_correlation = -similarity.correlation
        normalized_correlation = (reversal_correlation + 1) / 2  # Map [-1,1] to [0,1]

        # Gene overlap score
        drug_genes = set(drug_profile.gene_symbols)
        disease_genes = set(disease_profile.gene_symbols)
        common_genes = drug_genes.intersection(disease_genes)
        overlap_score = len(common_genes) / max(len(drug_genes), len(disease_genes))

        # Pathway enrichment (mock for now)
        pathway_score = 0.6  # TODO: Compute real pathway enrichment

        return {
            "correlation": normalized_correlation,
            "overlap_score": overlap_score,
            "pathway_score": pathway_score,
        }

    def extract_clinical_trial_features(
        self,
        drug_id: str,
        disease_id: str,
    ) -> Dict[str, float]:
        """Extract clinical trial evidence features.

        Args:
            drug_id: Drug identifier
            disease_id: Disease identifier

        Returns:
            Dictionary with clinical trial scores
        """
        evidence = self.clinical_pipeline.get_trial_evidence(drug_id, disease_id)

        # Phase score
        phase_scores = {
            "Early Phase 1": 0.2,
            "Phase 1": 0.4,
            "Phase 2": 0.6,
            "Phase 3": 0.8,
            "Phase 4": 1.0,
            "N/A": 0.0,
        }
        phase_score = phase_scores.get(evidence.highest_phase.value, 0.0)

        # Success rate
        total_completed = evidence.positive_results + evidence.negative_results
        if total_completed > 0:
            success_rate = evidence.positive_results / total_completed
        else:
            success_rate = 0.5  # Neutral

        return {
            "evidence_score": evidence.evidence_score,
            "phase_score": phase_score,
            "success_rate": success_rate,
        }

    def fuse_features(
        self,
        drug_id: str,
        disease_id: str,
        drug_smiles: Optional[str] = None,
    ) -> MultiModalFeatures:
        """Fuse multi-modal features for drug-disease prediction.

        Args:
            drug_id: Drug identifier
            disease_id: Disease identifier
            drug_smiles: Optional SMILES string for drug

        Returns:
            MultiModalFeatures with integrated prediction
        """
        # Extract features from each modality
        chemical_features, chemical_score = self.extract_chemical_features(
            drug_id, drug_smiles
        )

        gene_features = self.extract_gene_expression_features(drug_id, disease_id)

        clinical_features = self.extract_clinical_trial_features(drug_id, disease_id)

        # Compute modality-specific scores
        chemical_modality_score = chemical_score
        gene_modality_score = np.mean(list(gene_features.values()))
        clinical_modality_score = clinical_features["evidence_score"]

        # Feature contributions
        contributions = {
            "chemical": chemical_modality_score * self.weights["chemical"],
            "gene_expression": gene_modality_score * self.weights["gene_expression"],
            "clinical_trials": clinical_modality_score * self.weights["clinical_trials"],
        }

        # Fuse based on strategy
        if self.fusion_strategy == FusionStrategy.WEIGHTED:
            integrated_score = sum(contributions.values())
        elif self.fusion_strategy == FusionStrategy.LATE:
            # Simple average
            scores = [chemical_modality_score, gene_modality_score, clinical_modality_score]
            integrated_score = np.mean([s for s in scores if s > 0])
        else:
            # Default to weighted
            integrated_score = sum(contributions.values())

        # Compute confidence based on data availability
        available_modalities = sum(
            [
                chemical_features is not None and chemical_features.is_valid,
                gene_modality_score > 0,
                clinical_modality_score > 0,
            ]
        )
        confidence = available_modalities / 3.0

        return MultiModalFeatures(
            drug_id=drug_id,
            disease_id=disease_id,
            chemical_features=chemical_features,
            chemical_similarity_score=chemical_score,
            gene_expression_correlation=gene_features["correlation"],
            gene_overlap_score=gene_features["overlap_score"],
            pathway_enrichment_score=gene_features["pathway_score"],
            clinical_evidence_score=clinical_features["evidence_score"],
            trial_phase_score=clinical_features["phase_score"],
            trial_success_rate=clinical_features["success_rate"],
            integrated_score=float(integrated_score),
            confidence=float(confidence),
            feature_contributions=contributions,
        )

    def batch_fuse(
        self,
        pairs: List[Tuple[str, str]],
        smiles_map: Optional[Dict[str, str]] = None,
    ) -> List[MultiModalFeatures]:
        """Fuse features for multiple drug-disease pairs in batch.

        Args:
            pairs: List of (drug_id, disease_id) tuples
            smiles_map: Optional mapping of drug_id to SMILES

        Returns:
            List of MultiModalFeatures
        """
        smiles_map = smiles_map or {}
        results = []

        for drug_id, disease_id in pairs:
            smiles = smiles_map.get(drug_id)
            features = self.fuse_features(drug_id, disease_id, smiles)
            results.append(features)

        logger.info(f"Batch fused features for {len(results)} drug-disease pairs")
        return results

    def get_feature_vector(
        self,
        features: MultiModalFeatures,
    ) -> np.ndarray:
        """Convert MultiModalFeatures to numeric feature vector.

        Useful for feeding into ML models.

        Args:
            features: MultiModalFeatures object

        Returns:
            Numpy array with concatenated features
        """
        vector_parts = []

        # Chemical features (11 descriptors + fingerprint)
        if features.chemical_features and features.chemical_features.molecular_descriptors:
            desc = features.chemical_features.molecular_descriptors
            chem_feats = [
                desc.molecular_weight / 500.0,  # Normalize
                desc.logp / 5.0,
                desc.num_h_donors / 5.0,
                desc.num_h_acceptors / 10.0,
                desc.num_rotatable_bonds / 10.0,
                desc.tpsa / 150.0,
                desc.num_aromatic_rings / 5.0,
                desc.num_rings / 10.0,
                desc.fraction_csp3,
                desc.num_stereocenters / 10.0,
                features.chemical_similarity_score,
            ]
            vector_parts.extend(chem_feats)

            # Add fingerprint if available
            if features.chemical_features.morgan_fingerprint:
                vector_parts.extend(features.chemical_features.morgan_fingerprint)
        else:
            # Pad with zeros if chemical features unavailable
            vector_parts.extend([0.0] * (11 + 2048))  # 11 descriptors + 2048 fingerprint

        # Gene expression features (3)
        vector_parts.extend(
            [
                features.gene_expression_correlation,
                features.gene_overlap_score,
                features.pathway_enrichment_score,
            ]
        )

        # Clinical trial features (3)
        vector_parts.extend(
            [
                features.clinical_evidence_score,
                features.trial_phase_score,
                features.trial_success_rate,
            ]
        )

        return np.array(vector_parts, dtype=np.float32)

    def explain_prediction(
        self,
        features: MultiModalFeatures,
    ) -> Dict[str, str]:
        """Generate human-readable explanation of prediction.

        Args:
            features: MultiModalFeatures object

        Returns:
            Dictionary with explanation strings
        """
        explanations = []

        # Chemical explanation
        if features.chemical_features and features.chemical_features.is_valid:
            if features.chemical_similarity_score > 0.7:
                explanations.append(
                    "Strong chemical drug-likeness properties (Lipinski's Rule of Five)"
                )
            elif features.chemical_similarity_score > 0.5:
                explanations.append("Moderate chemical drug-likeness")
            else:
                explanations.append("Limited chemical drug-likeness")

        # Gene expression explanation
        if features.gene_expression_correlation > 0.7:
            explanations.append(
                "Drug expression profile strongly reverses disease signature"
            )
        elif features.gene_expression_correlation > 0.5:
            explanations.append("Moderate gene expression reversal pattern")

        # Clinical trial explanation
        if features.clinical_evidence_score > 0.7:
            explanations.append(
                f"Strong clinical evidence (Phase {features.trial_phase_score})"
            )
        elif features.clinical_evidence_score > 0.4:
            explanations.append("Moderate clinical trial evidence")
        elif features.clinical_evidence_score > 0:
            explanations.append("Limited clinical trial data available")

        # Overall verdict
        if features.integrated_score > 0.7:
            verdict = "Highly promising drug repurposing candidate"
        elif features.integrated_score > 0.5:
            verdict = "Moderate repurposing potential"
        else:
            verdict = "Limited repurposing evidence"

        return {
            "verdict": verdict,
            "confidence": f"{features.confidence * 100:.1f}%",
            "reasons": "; ".join(explanations) if explanations else "Insufficient data",
        }


# Singleton instance
_multimodal_fusion: Optional[MultiModalFusion] = None


def get_multimodal_fusion() -> MultiModalFusion:
    """Get or create singleton MultiModalFusion instance."""
    global _multimodal_fusion
    if _multimodal_fusion is None:
        _multimodal_fusion = MultiModalFusion()
    return _multimodal_fusion
