from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ScoringWeights(BaseModel):
    mechanism: Optional[float] = Field(default=None, ge=0)
    network: Optional[float] = Field(default=None, ge=0)
    signature: Optional[float] = Field(default=None, ge=0)
    clinical: Optional[float] = Field(default=None, ge=0)
    safety: Optional[float] = Field(default=None, ge=0)

    def as_dict(self) -> Dict[str, float]:
        return {
            key: value
            for key, value in self.model_dump(exclude_none=True).items()
        }


class RankRequest(BaseModel):
    disease: str = Field(..., min_length=3, max_length=200)
    persona: Optional[str] = Field(default=None, min_length=1, max_length=64)
    weights: Optional[ScoringWeights] = None
    exclude_contraindicated: bool = False


class EvidenceLink(BaseModel):
    source: str
    url: str
    summary: Optional[str] = None


class NarrativeCitation(BaseModel):
    source: str
    label: str
    url: Optional[str] = None
    detail: Optional[str] = None


class MechanisticNarrative(BaseModel):
    summary: str
    reasoning_steps: List[str] = Field(default_factory=list)
    citations: List[NarrativeCitation] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    mechanism_fit: float
    network_proximity: float
    signature_reversal: float
    clinical_signal: float
    safety_penalty: float
    final_score: float
    confidence_low: float | None = None
    confidence_high: float | None = None
    mechanism_raw: float | None = None
    network_raw: float | None = None
    signature_raw: float | None = None
    clinical_raw: float | None = None
    safety_raw: float | None = None


class GraphNode(BaseModel):
    id: str
    label: str | None = None
    category: str | None = None


class GraphEdge(BaseModel):
    id: str | None = None
    source: str
    target: str
    predicate: str | None = None


class GraphSnapshot(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


class GraphInsight(BaseModel):
    node_count: int
    edge_count: int
    density: float
    average_shortest_path: float | None = None
    top_central_nodes: List[str] = Field(default_factory=list)
    summary: str | None = None
    graph: GraphSnapshot | None = None


class ConceptMapping(BaseModel):
    cui: str
    name: str
    preferred_name: str
    semantic_types: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)


class DrugCandidate(BaseModel):
    drug_id: str
    name: str
    score: ScoreBreakdown
    evidence: List[EvidenceLink] = Field(default_factory=list)
    graph_insights: GraphInsight | None = None
    narrative: Optional["MechanisticNarrative"] = None
    confidence: Optional["CandidateConfidence"] = None
    indications: List[str] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)
    annotation_sources: List["DrugAnnotationSource"] = Field(default_factory=list)
    pathways: List["PathwayInsight"] = Field(default_factory=list)


class RankWarning(BaseModel):
    source: str
    detail: str


class RankResponse(BaseModel):
    query: str
    normalized_disease: Optional[str]
    candidates: List[DrugCandidate]
    warnings: List[RankWarning] = Field(default_factory=list)
    related_concepts: List[ConceptMapping] = Field(default_factory=list)
    graph_overview: GraphInsight | None = None
    cached: bool = False
    scoring: Optional["AppliedWeights"] = None
    counterfactuals: List["CounterfactualScenario"] = Field(default_factory=list)
    pathway_summary: List[PathwaySummaryItem] = Field(default_factory=list)


class SavedQuery(BaseModel):
    id: int
    disease: str
    response: RankResponse
    created_at: float
    note: Optional[str] = None


class SaveQueryRequest(BaseModel):
    disease: str
    response: RankResponse
    note: Optional[str] = None


class UpdateQueryRequest(BaseModel):
    note: Optional[str] = None


class ServiceResult(BaseModel):
    candidates: List[DrugCandidate] = Field(default_factory=list)
    warnings: List[RankWarning] = Field(default_factory=list)


class RawEvidence(BaseModel):
    disease_id: Optional[str] = None
    targets: List[dict[str, Any]] = Field(default_factory=list)
    activities: List[dict[str, Any]] = Field(default_factory=list)
    trials: List[dict[str, Any]] = Field(default_factory=list)
    safety_events: List[dict[str, Any]] = Field(default_factory=list)


class AppliedWeights(BaseModel):
    persona: str
    weights: Dict[str, float]
    delta_vs_default: Dict[str, float]
    overrides: Dict[str, float] = Field(default_factory=dict)


class ScoringPersona(BaseModel):
    name: str
    label: str
    description: Optional[str] = None
    weights: Dict[str, float]


class ScoringMetadata(BaseModel):
    default_persona: str
    default_weights: Dict[str, float]
    personas: List[ScoringPersona] = Field(default_factory=list)


class CounterfactualScenario(BaseModel):
    label: str
    weights: Dict[str, float]
    candidates: List[DrugCandidate]
    description: Optional[str] = None


class CandidateConfidence(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    tier: Literal["exploratory", "hypothesis-ready", "decision-grade"]
    signals: List[str] = Field(default_factory=list)


class DrugAnnotationSource(BaseModel):
    label: str
    url: Optional[str] = None


class PathwayInsight(BaseModel):
    name: str
    source: Optional[str] = None
    url: Optional[str] = None
    genes: List[str] = Field(default_factory=list)


class PathwaySummaryItem(BaseModel):
    name: str
    source: Optional[str] = None
    url: Optional[str] = None
    genes: List[str] = Field(default_factory=list)
    count: int = 0


DrugCandidate.model_rebuild()
RankResponse.model_rebuild()


