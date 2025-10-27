"""Clinical trial data integration from ClinicalTrials.gov and other sources."""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrialPhase(str, Enum):
    """Clinical trial phase."""

    EARLY_PHASE_1 = "Early Phase 1"
    PHASE_1 = "Phase 1"
    PHASE_2 = "Phase 2"
    PHASE_3 = "Phase 3"
    PHASE_4 = "Phase 4"
    NOT_APPLICABLE = "N/A"


class TrialStatus(str, Enum):
    """Clinical trial recruitment status."""

    NOT_YET_RECRUITING = "Not yet recruiting"
    RECRUITING = "Recruiting"
    ENROLLING_BY_INVITATION = "Enrolling by invitation"
    ACTIVE_NOT_RECRUITING = "Active, not recruiting"
    SUSPENDED = "Suspended"
    TERMINATED = "Terminated"
    COMPLETED = "Completed"
    WITHDRAWN = "Withdrawn"
    UNKNOWN = "Unknown"


class TrialResult(str, Enum):
    """Clinical trial outcome."""

    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    INCONCLUSIVE = "Inconclusive"
    ONGOING = "Ongoing"
    UNKNOWN = "Unknown"


class ClinicalTrial(BaseModel):
    """Clinical trial information."""

    nct_id: str = Field(..., description="ClinicalTrials.gov NCT identifier")
    title: str
    phase: TrialPhase
    status: TrialStatus
    drug_id: Optional[str] = None
    drug_name: Optional[str] = None
    disease_id: Optional[str] = None
    disease_name: Optional[str] = None
    condition: str
    intervention: str
    start_date: Optional[str] = None
    completion_date: Optional[str] = None
    enrollment: Optional[int] = None
    sponsor: Optional[str] = None
    primary_outcome: Optional[str] = None
    result: Optional[TrialResult] = None
    result_summary: Optional[str] = None
    efficacy_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Efficacy score from trial results"
    )
    safety_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Safety score from trial results"
    )
    publication_links: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


class TrialEvidence(BaseModel):
    """Evidence from clinical trials for drug-disease association."""

    drug_id: str
    disease_id: str
    num_trials: int
    phases: List[TrialPhase]
    highest_phase: TrialPhase
    completed_trials: int
    ongoing_trials: int
    positive_results: int
    negative_results: int
    evidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall evidence strength"
    )
    confidence_level: str = Field(..., description="High, Medium, or Low")
    trials: List[ClinicalTrial] = Field(default_factory=list)


class ClinicalTrialDataPipeline:
    """Pipeline for fetching and analyzing clinical trial data."""

    def __init__(self):
        """Initialize clinical trial data pipeline."""
        self.trial_cache: Dict[str, ClinicalTrial] = {}
        self.association_cache: Dict[Tuple[str, str], TrialEvidence] = {}

    def fetch_trials_for_drug(
        self,
        drug_id: str,
        drug_name: Optional[str] = None,
    ) -> List[ClinicalTrial]:
        """Fetch clinical trials for a specific drug.

        In production, this would query:
        - ClinicalTrials.gov API
        - EU Clinical Trials Register
        - WHO ICTRP (International Clinical Trials Registry Platform)

        Args:
            drug_id: Drug identifier
            drug_name: Optional drug name for search

        Returns:
            List of ClinicalTrial objects
        """
        logger.info(f"Fetching clinical trials for drug {drug_id}")

        # TODO: Implement real API calls to ClinicalTrials.gov
        # Example: https://clinicaltrials.gov/api/query/study_fields?expr={drug_name}

        # For now, return mock data
        return self._generate_mock_trials(drug_id, drug_name, entity_type="drug")

    def fetch_trials_for_disease(
        self,
        disease_id: str,
        disease_name: Optional[str] = None,
    ) -> List[ClinicalTrial]:
        """Fetch clinical trials for a specific disease.

        Args:
            disease_id: Disease identifier
            disease_name: Optional disease name for search

        Returns:
            List of ClinicalTrial objects
        """
        logger.info(f"Fetching clinical trials for disease {disease_id}")

        # TODO: Implement real API calls
        return self._generate_mock_trials(disease_id, disease_name, entity_type="disease")

    def _generate_mock_trials(
        self,
        entity_id: str,
        entity_name: Optional[str],
        entity_type: str,
    ) -> List[ClinicalTrial]:
        """Generate mock clinical trial data for testing.

        Args:
            entity_id: Drug or disease ID
            entity_name: Drug or disease name
            entity_type: 'drug' or 'disease'

        Returns:
            List of mock ClinicalTrial objects
        """
        import hashlib
        import random

        # Use entity_id for reproducible randomness
        seed = int(hashlib.md5(entity_id.encode()).hexdigest()[:8], 16)
        random.seed(seed)

        num_trials = random.randint(2, 8)
        trials = []

        for i in range(num_trials):
            nct_id = f"NCT{random.randint(1000000, 9999999):08d}"

            trial = ClinicalTrial(
                nct_id=nct_id,
                title=f"Study of {entity_name or entity_id} - Trial {i+1}",
                phase=random.choice(list(TrialPhase)),
                status=random.choice(list(TrialStatus)),
                drug_id=entity_id if entity_type == "drug" else None,
                drug_name=entity_name if entity_type == "drug" else None,
                disease_id=entity_id if entity_type == "disease" else None,
                disease_name=entity_name if entity_type == "disease" else None,
                condition=entity_name or "Unknown Condition",
                intervention=f"Drug: {entity_name or entity_id}",
                start_date="2020-01-15",
                completion_date="2023-12-31" if random.random() > 0.3 else None,
                enrollment=random.randint(50, 500),
                sponsor="Academic Medical Center",
                primary_outcome="Overall Response Rate",
                result=random.choice(list(TrialResult)),
                efficacy_score=random.uniform(0.4, 0.9),
                safety_score=random.uniform(0.6, 0.95),
            )
            trials.append(trial)

        return trials

    def get_trial_evidence(
        self,
        drug_id: str,
        disease_id: str,
    ) -> TrialEvidence:
        """Get evidence from clinical trials for drug-disease association.

        Args:
            drug_id: Drug identifier
            disease_id: Disease identifier

        Returns:
            TrialEvidence summarizing all relevant trials
        """
        cache_key = (drug_id, disease_id)

        if cache_key in self.association_cache:
            return self.association_cache[cache_key]

        # In production, search for trials matching both drug AND disease
        # For now, generate mock evidence
        trials = self._generate_mock_association_trials(drug_id, disease_id)

        # Analyze trials
        phases = [trial.phase for trial in trials]
        highest_phase = self._get_highest_phase(phases)

        completed = sum(1 for t in trials if t.status == TrialStatus.COMPLETED)
        ongoing = sum(
            1
            for t in trials
            if t.status
            in {
                TrialStatus.RECRUITING,
                TrialStatus.ACTIVE_NOT_RECRUITING,
                TrialStatus.NOT_YET_RECRUITING,
            }
        )

        positive = sum(1 for t in trials if t.result == TrialResult.POSITIVE)
        negative = sum(1 for t in trials if t.result == TrialResult.NEGATIVE)

        # Compute evidence score
        evidence_score = self._compute_evidence_score(
            trials, highest_phase, positive, negative
        )

        # Determine confidence level
        if evidence_score > 0.7:
            confidence = "High"
        elif evidence_score > 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

        evidence = TrialEvidence(
            drug_id=drug_id,
            disease_id=disease_id,
            num_trials=len(trials),
            phases=phases,
            highest_phase=highest_phase,
            completed_trials=completed,
            ongoing_trials=ongoing,
            positive_results=positive,
            negative_results=negative,
            evidence_score=evidence_score,
            confidence_level=confidence,
            trials=trials,
        )

        self.association_cache[cache_key] = evidence
        return evidence

    def _generate_mock_association_trials(
        self,
        drug_id: str,
        disease_id: str,
    ) -> List[ClinicalTrial]:
        """Generate mock trials for specific drug-disease pair."""
        import hashlib
        import random

        # Reproducible randomness
        combined_id = f"{drug_id}:{disease_id}"
        seed = int(hashlib.md5(combined_id.encode()).hexdigest()[:8], 16)
        random.seed(seed)

        # 30% chance of having trials
        if random.random() > 0.3:
            return []

        num_trials = random.randint(1, 5)
        trials = []

        for i in range(num_trials):
            nct_id = f"NCT{random.randint(1000000, 9999999):08d}"

            trials.append(
                ClinicalTrial(
                    nct_id=nct_id,
                    title=f"Drug {drug_id} for Disease {disease_id} - Phase {i+1}",
                    phase=random.choice(list(TrialPhase)),
                    status=random.choice(list(TrialStatus)),
                    drug_id=drug_id,
                    disease_id=disease_id,
                    condition=f"Disease {disease_id}",
                    intervention=f"Drug: {drug_id}",
                    enrollment=random.randint(50, 300),
                    result=random.choice(list(TrialResult)),
                    efficacy_score=random.uniform(0.3, 0.9),
                    safety_score=random.uniform(0.5, 0.95),
                )
            )

        return trials

    def _get_highest_phase(self, phases: List[TrialPhase]) -> TrialPhase:
        """Determine highest phase from list of trial phases."""
        if not phases:
            return TrialPhase.NOT_APPLICABLE

        phase_order = {
            TrialPhase.EARLY_PHASE_1: 0,
            TrialPhase.PHASE_1: 1,
            TrialPhase.PHASE_2: 2,
            TrialPhase.PHASE_3: 3,
            TrialPhase.PHASE_4: 4,
            TrialPhase.NOT_APPLICABLE: -1,
        }

        return max(phases, key=lambda p: phase_order.get(p, -1))

    def _compute_evidence_score(
        self,
        trials: List[ClinicalTrial],
        highest_phase: TrialPhase,
        positive_results: int,
        negative_results: int,
    ) -> float:
        """Compute overall evidence score from trial data.

        Args:
            trials: List of trials
            highest_phase: Highest phase reached
            positive_results: Number of positive outcomes
            negative_results: Number of negative outcomes

        Returns:
            Evidence score between 0 and 1
        """
        if not trials:
            return 0.0

        # Phase score (0-0.4)
        phase_scores = {
            TrialPhase.EARLY_PHASE_1: 0.1,
            TrialPhase.PHASE_1: 0.15,
            TrialPhase.PHASE_2: 0.25,
            TrialPhase.PHASE_3: 0.35,
            TrialPhase.PHASE_4: 0.4,
            TrialPhase.NOT_APPLICABLE: 0.05,
        }
        phase_score = phase_scores.get(highest_phase, 0.0)

        # Results score (0-0.4)
        total_completed = positive_results + negative_results
        if total_completed > 0:
            success_rate = positive_results / total_completed
            results_score = success_rate * 0.4
        else:
            results_score = 0.2  # Neutral if no results

        # Volume score (0-0.2)
        volume_score = min(len(trials) / 10.0, 1.0) * 0.2

        total_score = phase_score + results_score + volume_score
        return min(total_score, 1.0)

    def search_similar_trials(
        self,
        reference_trial: ClinicalTrial,
        top_k: int = 10,
    ) -> List[Tuple[ClinicalTrial, float]]:
        """Find similar clinical trials based on intervention and condition.

        Args:
            reference_trial: Reference trial to find similar ones
            top_k: Number of results to return

        Returns:
            List of (trial, similarity_score) tuples
        """
        # TODO: Implement using text embeddings and similarity search
        logger.info(f"Searching for trials similar to {reference_trial.nct_id}")

        # Mock implementation
        return []


# Singleton instance
_clinical_trial_pipeline: Optional[ClinicalTrialDataPipeline] = None


def get_clinical_trial_pipeline() -> ClinicalTrialDataPipeline:
    """Get or create singleton ClinicalTrialDataPipeline instance."""
    global _clinical_trial_pipeline
    if _clinical_trial_pipeline is None:
        _clinical_trial_pipeline = ClinicalTrialDataPipeline()
    return _clinical_trial_pipeline
