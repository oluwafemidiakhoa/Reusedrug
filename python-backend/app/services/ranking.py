from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional

from app.db import get_cached_rank, store_rank
from app.models import (
    AppliedWeights,
    CandidateConfidence,
    DrugAnnotationSource,
    GraphInsight,
    GraphSnapshot,
    GraphNode,
    GraphEdge,
    DrugCandidate,
    EvidenceLink,
    PathwayInsight,
    PathwaySummaryItem,
    RankResponse,
    RankWarning,
    CounterfactualScenario,
    ScoreBreakdown,
)
from app.services import data_sources, pathways
from app.services.clients import drugcentral, lincs, sider, translator
from app.services.chembl import fetch_strongest_activity
from app.services.clinicaltrials import fetch_trials
from app.services.faers import fetch_top_reactions
from app.services.normalize import normalize_disease
from app.services.graph import analyze_translator_paths, GraphInsights
from app.services.opentargets import fetch_known_drugs
from app.services.pubmed import fetch_pubmed_summaries
from app.services.scoring import resolve_scoring_context, score_candidate
from app.services.narratives import generate_mechanistic_narrative
from app.services.umls import lookup_concepts
from app.utils.identifiers import normalize_drug_id

logger = logging.getLogger(__name__)

_WEIGHT_KEYS = ("mechanism", "network", "signature", "clinical", "safety")
_CONFIDENCE_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("decision-grade", 0.75),
    ("hypothesis-ready", 0.45),
)

def _cache_ttl_seconds() -> float:
    try:
        return float(os.getenv("RESULT_CACHE_TTL_SECONDS", "900"))
    except ValueError:
        return 900.0


def _contraindication_penalty() -> float:
    try:
        return max(0.0, float(os.getenv("CONTRAINDICATION_PENALTY", "0.2")))
    except ValueError:
        return 0.2


def _weights_match_default(
    weights: Mapping[str, float],
    default_weights: Mapping[str, float],
) -> bool:
    for key, default_value in default_weights.items():
        if abs(weights.get(key, 0.0) - default_value) > 1e-6:
            return False
    return True


def _renormalize(weights: Mapping[str, float]) -> Dict[str, float]:
    total = sum(weights.get(key, 0.0) for key in _WEIGHT_KEYS)
    if total <= 0:
        return {key: max(weights.get(key, 0.0), 0.0) for key in _WEIGHT_KEYS}
    return {key: max(weights.get(key, 0.0), 0.0) / total for key in _WEIGHT_KEYS}


def _scale_weight(weights: Mapping[str, float], *, key: str, factor: float) -> Dict[str, float]:
    adjusted = {k: max(weights.get(k, 0.0), 0.0) for k in _WEIGHT_KEYS}
    adjusted[key] = max(adjusted.get(key, 0.0) * factor, 0.0)
    return _renormalize(adjusted)


def _apply_weights_to_score(
    score: ScoreBreakdown,
    weights: Mapping[str, float],
    has_graph: bool,
) -> ScoreBreakdown:
    mech_raw = score.mechanism_raw or 0.0
    network_raw = score.network_raw or 0.0
    signature_raw = score.signature_raw or 0.0
    clinical_raw = score.clinical_raw or 0.0
    safety_raw = score.safety_raw or 0.0

    mech = mech_raw * weights["mechanism"]
    network = network_raw * weights["network"]
    signature = signature_raw * weights["signature"]
    clinical = clinical_raw * weights["clinical"]
    penalty = safety_raw * weights["safety"]

    raw_score = mech + network + signature + clinical + penalty
    final = max(min(raw_score, 1.0), 0.0)

    evidence_signals = [
        mech_raw > 0.1,
        network_raw > 0.1,
        signature_raw > 0.2,
        clinical_raw > 0.2,
        safety_raw < 0.0,
        has_graph,
    ]
    signal_strength = sum(1 for flag in evidence_signals if flag)
    margin = max(0.05, 0.25 - 0.03 * signal_strength)
    confidence_low = max(0.0, final - margin)
    confidence_high = min(1.0, final + margin)

    return ScoreBreakdown(
        mechanism_fit=round(mech, 4),
        network_proximity=round(network, 4),
        signature_reversal=round(signature, 4),
        clinical_signal=round(clinical, 4),
        safety_penalty=round(penalty, 4),
        final_score=round(final, 4),
        confidence_low=round(confidence_low, 4),
        confidence_high=round(confidence_high, 4),
        mechanism_raw=round(mech_raw, 4),
        network_raw=round(network_raw, 4),
        signature_raw=round(signature_raw, 4),
        clinical_raw=round(clinical_raw, 4),
        safety_raw=round(safety_raw, 4),
    )


def _assess_candidate_confidence(
    score: ScoreBreakdown,
    *,
    trials: list[dict[str, Any]],
    translator_paths: Any,
    signature_records: Optional[List[dict[str, Any]]],
    adverse_events: Optional[list[dict[str, Any]]],
    graph_metrics: GraphInsights | None,
    warnings: List[RankWarning],
    evidence: List[EvidenceLink],
    pathways: List[dict],
    contraindicated: bool = False,
) -> CandidateConfidence:
    signals: list[str] = []
    evidence_score = 0.05 if evidence else 0.0

    if (score.mechanism_raw or 0.0) > 0.25:
        evidence_score += 0.18
        signals.append("Mechanism assays")
    if trials:
        evidence_score += 0.22
        signals.append("Clinical studies")
    if signature_records:
        evidence_score += 0.07
        signals.append("Transcriptomic reversal")
    if graph_metrics:
        evidence_score += 0.12
        signals.append("Knowledge-graph connectivity")

    translator_present = False
    if translator_paths:
        if isinstance(translator_paths, list):
            translator_present = any(bool(item) for item in translator_paths)
        else:
            translator_present = True
    if translator_present:
        evidence_score += 0.08
        signals.append("Translator graph paths")

    if adverse_events:
        evidence_score += 0.04
        signals.append("Pharmacovigilance coverage")

    if contraindicated:
        signals.append("Documented contraindication")
        evidence_score = max(evidence_score - 0.1, 0.0)

    if pathways:
        evidence_score += 0.06
        signals.append(f"Pathway support ({len(pathways)})")

    warning_penalty = min(len(warnings) * 0.05, 0.2)
    adjusted_evidence = max(evidence_score - warning_penalty, 0.0)

    composite = max(
        0.0,
        min(
            1.0,
            0.55 * (score.final_score or 0.0) + 0.45 * adjusted_evidence,
        ),
    )

    tier = "exploratory"
    for name, threshold in _CONFIDENCE_THRESHOLDS:
        if composite >= threshold:
            tier = name
            break

    if not signals:
        signals.append("Limited direct evidence detected")

    return CandidateConfidence(
        score=round(composite, 4),
        tier=tier,
        signals=signals,
    )


def _build_counterfactuals(
    candidates: List[DrugCandidate],
    base_weights: Mapping[str, float],
) -> List[CounterfactualScenario]:
    scenarios: List[CounterfactualScenario] = []

    safety_down = _scale_weight(base_weights, key="safety", factor=0.8)
    scenarios.append(
        (
            "Safety -20%",
            safety_down,
            "Safety contribution reduced by 20% with renormalized weights.",
        )
    )

    results: List[CounterfactualScenario] = []
    for label, counter_weights, description in scenarios:
        counter_candidates: List[DrugCandidate] = []
        for candidate in candidates:
            updated_score = _apply_weights_to_score(
                candidate.score,
                counter_weights,
                bool(candidate.graph_insights),
            )
            counter_candidates.append(
                candidate.model_copy(update={"score": updated_score})
            )
        counter_candidates.sort(key=lambda c: c.score.final_score, reverse=True)
        results.append(
            CounterfactualScenario(
                label=label,
                description=description,
                weights={key: round(counter_weights[key], 4) for key in counter_weights},
                candidates=counter_candidates[:5],
            )
        )
    return results


def _make_cache_key(
    query: str,
    persona: str,
    weights: Mapping[str, float],
    default_weights: Mapping[str, float],
    exclude_contraindicated: bool,
) -> str:
    base = query.strip().lower()
    if (
        _weights_match_default(weights, default_weights)
        and persona == "balanced"
        and not exclude_contraindicated
    ):
        return base
    payload = {
        "persona": persona,
        "weights": {key: round(weights.get(key, 0.0), 6) for key in sorted(weights)},
        "exclude_contraindicated": exclude_contraindicated,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{base}:{digest}"


def _dedupe_warnings(warnings: List[RankWarning]) -> List[RankWarning]:
    deduped: List[RankWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        key = (warning.source, warning.detail)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(warning)
    return deduped


async def compute_rank(
    query: str,
    *,
    persona: str | None = None,
    weight_overrides: Optional[Mapping[str, float]] = None,
    exclude_contraindicated: bool = False,
    force_refresh: bool = False,
    background: bool = False,
) -> RankResponse:
    (
        active_persona,
        applied_weights,
        default_weights,
        applied_overrides,
    ) = resolve_scoring_context(persona, weight_overrides)

    cache_key = _make_cache_key(
        query,
        active_persona,
        applied_weights,
        default_weights,
        exclude_contraindicated,
    )

    if not force_refresh:
        cached = await get_cached_rank(cache_key, _cache_ttl_seconds())
        if cached:
            response = RankResponse(**cached)
            response.cached = True
            return response

    disease_id, normalized_label, warnings = await normalize_disease(query)

    ot_query_id: Optional[str] = None
    if disease_id and disease_id.upper().startswith(("EFO_", "ORPHANET_")):
        ot_query_id = disease_id

    if ot_query_id:
        opentargets_rows, ot_warnings = await fetch_known_drugs(ot_query_id)
    else:
        opentargets_rows = []
        ot_warnings = [
            RankWarning(
                source="opentargets",
                detail="Skipping Open Targets query because disease could not be normalized to an EFO identifier",
            )
        ]

    if disease_id:
        translator_rows, translator_warnings = await translator.fetch_disease_treatments([disease_id])
    else:
        translator_rows = []
        translator_warnings = [
            RankWarning(
                source="translator",
                detail="Skipping Translator query because no disease identifier was resolved",
            )
        ]

    umls_concepts, umls_warnings = await lookup_concepts(normalized_label or query)

    all_warnings = warnings + ot_warnings + translator_warnings + umls_warnings

    candidate_map: Dict[str, dict] = {}

    def ensure_candidate(drug_id: str, name: str) -> dict:
        entry = candidate_map.setdefault(
            drug_id,
            {
                "drug_id": drug_id,
                "name": name or "Unknown",
                "targets": set(),
                "evidence": [],
                "translator_paths": [],
            },
        )
        if name and entry["name"] == "Unknown":
            entry["name"] = name
        return entry

    all_translator_paths: List[Dict[str, Any]] = []

    for row in translator_rows:
        drug_id = row.get("drug_id") or "unknown"
        name = row.get("drug_name") or "Unknown"
        entry = ensure_candidate(drug_id, name)
        path = row.get("evidence") or {}
        if isinstance(path, dict):
            all_translator_paths.append(path)
        elif isinstance(path, list):
            all_translator_paths.extend([item for item in path if isinstance(item, dict)])
        entry["translator_paths"].append(path)
        edge_count = 0
        if isinstance(path, dict):
            edge_count = sum(len(bindings) for bindings in path.values())
        summary = (
            f"Knowledge graph path with {edge_count} edges" if edge_count else "Knowledge graph support"
        )
        entry["evidence"].append(
            EvidenceLink(
                source="NCATS Translator",
                url="https://api.bte.ncats.io/v1/query",
                summary=summary,
            )
        )

    for row in opentargets_rows:
        drug_id = row.get("drug_id") or "unknown"
        entry = ensure_candidate(drug_id, row.get("drug_name") or "Unknown")
        target_id = row.get("target_id")
        if target_id:
            entry["targets"].add(target_id)
        target_symbol = row.get("target_symbol")
        if target_symbol:
            entry["targets"].add(target_symbol)
        entry["evidence"].append(
            EvidenceLink(
                source="Open Targets",
                url=f"https://platform.opentargets.org/drug/{drug_id}",
                summary=f"Target {row.get('target_symbol')} phase {row.get('phase')} status {row.get('status')}",
            )
        )

    if not candidate_map:
        final_warnings = _dedupe_warnings(
            all_warnings
            + [
                RankWarning(
                    source="pipeline",
                    detail="No candidates returned from upstream data sources",
                )
            ]
        )
        response = RankResponse(
            query=query,
            normalized_disease=normalized_label,
            candidates=[],
            warnings=final_warnings,
            cached=False,
            scoring=AppliedWeights(
                persona=active_persona,
                weights={key: round(applied_weights[key], 4) for key in applied_weights},
                delta_vs_default={
                    key: round(applied_weights[key] - default_weights.get(key, 0.0), 4)
                    for key in applied_weights
                },
                overrides={key: round(applied_overrides[key], 4) for key in applied_overrides},
            ),
        )
        await store_rank(cache_key, normalized_label, response.model_dump(exclude={"cached"}))
        return response

    async def enrich_candidate(raw: dict) -> tuple[Optional[DrugCandidate], List[RankWarning], bool]:
        drug_id = raw["drug_id"]
        drug_name = raw["name"]
        target_ids = [tid for tid in raw["targets"] if tid]
        translator_paths = raw.get("translator_paths")

        normalized_drug_id = normalize_drug_id(drug_id) or drug_id
        chembl_lookup = (
            normalized_drug_id.replace("CHEMBL_", "CHEMBL")
            if normalized_drug_id.upper().startswith("CHEMBL_")
            else normalized_drug_id
        )

        (
            (activity, activity_warnings),
            (trials, trial_warnings),
            (safety, safety_warnings),
            (moa_annotations, moa_warnings),
            (signature_records, signature_warnings),
            (sider_records, sider_warnings),
            (pubmed_records, pubmed_warnings),
        ) = await asyncio.gather(
            fetch_strongest_activity(
                molecule_id=chembl_lookup, target_id=target_ids[0] if target_ids else None
            ),
            fetch_trials(condition=normalized_label or query, intervention=drug_name),
            fetch_top_reactions(drug_name),
            drugcentral.fetch_drug_moa(drug_name),
            lincs.fetch_signature_scores(normalized_label or query),
            sider.fetch_safety(normalized_drug_id),
            fetch_pubmed_summaries(normalized_label or query, drug_name),
        )

        graph_insights = analyze_translator_paths(translator_paths)
        graph_model = _to_graph_model(graph_insights)

        activity_record = activity[0] if activity else None
        evidence = list(raw["evidence"])

        annotations = data_sources.drug_annotations(normalized_drug_id)
        annotation_sources: list[DrugAnnotationSource] = []
        indications: list[str] = []
        contraindications: list[str] = []
        contraindication_warning: RankWarning | None = None
        if annotations:
            indications = [str(item) for item in annotations.get("indications", [])]
            contraindications = [str(item) for item in annotations.get("contraindications", [])]
            for source in annotations.get("sources", []):
                annotation_sources.append(
                    DrugAnnotationSource(
                        label=str(source.get("label", "Annotation")),
                        url=source.get("url"),
                    )
                )
            if indications:
                evidence.append(
                    EvidenceLink(
                        source="DrugBank/CTD",
                        url=annotation_sources[0].url if annotation_sources else None,
                        summary=f"Documented indications: {', '.join(indications[:3])}",
                    )
                )
            if contraindications:
                disease_key = (normalized_label or query).lower()
                for entry in contraindications:
                    entry_lower = entry.lower()
                    if entry_lower in disease_key or disease_key in entry_lower:
                        contraindication_warning = RankWarning(
                            source="drug_annotations",
                            detail=f"Contraindicated for {normalized_label or query}",
                        )
                        break

        pathway_records = pathways.related_pathways(target_ids)
        pathway_insights: list[PathwayInsight] = []
        if pathway_records:
            for record in pathway_records:
                pathway_insights.append(
                    PathwayInsight(
                        name=record["name"],
                        source=record.get("source"),
                        url=record.get("url"),
                        genes=record.get("genes", []),
                    )
                )
            for record in pathway_records[:2]:
                evidence.append(
                    EvidenceLink(
                        source=record.get("source") or "Pathway",
                        url=record.get("url"),
                        summary=f"Pathway: {record.get('name')} (genes: {', '.join(record.get('genes', []))})",
                    )
                )
        else:
            pathway_insights = []

        if activity_record:
            evidence.append(
                EvidenceLink(
                    source="ChEMBL",
                    url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_lookup}/",
                    summary=f"Potency: {activity_record.get('potency')}",
                )
            )
        if trials:
            for trial in trials:
                evidence.append(
                    EvidenceLink(
                        source="ClinicalTrials.gov",
                        url=f"https://clinicaltrials.gov/study/{trial.get('nct_id')}",
                        summary=f"{trial.get('phase')} - {trial.get('status')}",
                    )
                )
        if safety:
            evidence.append(
                EvidenceLink(
                    source="openFDA FAERS",
                    url="https://open.fda.gov/apis/drug/event",
                    summary="Safety signals available",
                )
            )
        if moa_annotations:
            for annotation in moa_annotations[:3]:
                mechanism = annotation.get("mechanism_of_action") or annotation.get("action_type")
                if mechanism:
                    evidence.append(
                        EvidenceLink(
                            source="DrugCentral",
                            url="https://drugcentral.org",
                            summary=f"Mechanism: {mechanism}",
                        )
                    )
        if signature_records:
            top_signature = signature_records[0]
            tau = top_signature.get("tau") or top_signature.get("score")
            evidence.append(
                EvidenceLink(
                    source="CLUE/LINCS",
                    url="https://clue.io",
                    summary=f"Signature reversal tau={tau}",
                )
            )
        if sider_records:
            serious = sum(
                1 for entry in sider_records if str(entry.get("seriousness", "")).lower() == "serious"
            )
            evidence.append(
                EvidenceLink(
                    source="SIDER",
                    url="http://sideeffects.embl.de/",
                    summary=f"{len(sider_records)} adverse events ({serious} serious)",
                )
            )
        if pubmed_records:
            for record in pubmed_records:
                title = record.get("title") or "PubMed article"
                journal = record.get("journal")
                pub_date = record.get("pub_date")
                summary_bits = [title]
                if journal:
                    summary_bits.append(journal)
                if pub_date:
                    summary_bits.append(pub_date)
                summary = " — ".join(summary_bits)
                evidence.append(
                    EvidenceLink(
                        source="PubMed",
                        url=record.get("url") or "https://pubmed.ncbi.nlm.nih.gov",
                        summary=summary,
                    )
                )

        if graph_insights:
            evidence.append(
                EvidenceLink(
                    source="Graph Analytics",
                    url="https://translator.broadinstitute.org",
                    summary=graph_insights.summary(),
                )
            )

        score = score_candidate(
            drug_name,
            activity=activity_record,
            target_count=len(target_ids),
            trials=trials,
            adverse_events=safety,
            translator_paths=translator_paths,
            moa_annotations=moa_annotations,
            signature_records=signature_records,
            sider_records=sider_records,
            graph_metrics=graph_insights,
            weights=applied_weights,
        )

        contraindicated = bool(contraindication_warning)

        if contraindicated and not exclude_contraindicated:
            penalty = _contraindication_penalty()
            if penalty > 0:
                adjusted = score.model_dump()
                adjusted_final = max(0.0, score.final_score - penalty)
                if score.confidence_low is not None:
                    adjusted["confidence_low"] = round(max(0.0, score.confidence_low - penalty), 4)
                if score.confidence_high is not None:
                    adjusted["confidence_high"] = round(max(0.0, score.confidence_high - penalty), 4)
                adjusted["final_score"] = round(adjusted_final, 4)
                score = ScoreBreakdown(**adjusted)

        warnings = (
            activity_warnings
            + trial_warnings
            + safety_warnings
            + moa_warnings
            + signature_warnings
            + sider_warnings
            + pubmed_warnings
        )
        if contraindication_warning:
            warnings.append(contraindication_warning)

        excluded = contraindicated and exclude_contraindicated
        if excluded:
            return (None, warnings, True)

        narrative = generate_mechanistic_narrative(
            drug_name=drug_name,
            score=score,
            evidence=evidence,
            graph=graph_insights,
            pathways=pathway_insights,
        )

        confidence = _assess_candidate_confidence(
            score,
            trials=trials,
            translator_paths=translator_paths,
            signature_records=signature_records,
            adverse_events=safety,
            graph_metrics=graph_insights,
            warnings=warnings,
            evidence=evidence,
            pathways=pathway_records,
            contraindicated=contraindicated,
        )

        return (
            DrugCandidate(
                drug_id=drug_id,
                name=drug_name,
                score=score,
                evidence=evidence,
                graph_insights=graph_model,
                narrative=narrative,
                confidence=confidence,
                indications=indications,
                contraindications=contraindications,
                annotation_sources=annotation_sources,
                pathways=pathway_insights,
            ),
            warnings,
            False,
        )

    enrichment_results = await asyncio.gather(
        *(enrich_candidate(raw) for raw in candidate_map.values()),
        return_exceptions=True,
    )

    candidates: List[DrugCandidate] = []
    for result in enrichment_results:
        if isinstance(result, Exception):
            all_warnings.append(
                RankWarning(
                    source="pipeline",
                    detail=f"Candidate enrichment failed: {result}",
                )
            )
            continue
        candidate, candidate_warnings, was_excluded = result
        all_warnings.extend(candidate_warnings)
        if was_excluded or candidate is None:
            continue
        candidates.append(candidate)

    overview_graph = _to_graph_model(analyze_translator_paths(all_translator_paths))
    sorted_candidates = sorted(candidates, key=lambda c: c.score.final_score, reverse=True)
    final_warnings = _dedupe_warnings(all_warnings)

    scoring_summary = AppliedWeights(
        persona=active_persona,
        weights={key: round(applied_weights[key], 4) for key in applied_weights},
        delta_vs_default={
            key: round(applied_weights[key] - default_weights.get(key, 0.0), 4) for key in applied_weights
        },
        overrides={key: round(applied_overrides[key], 4) for key in applied_overrides},
    )

    counterfactuals = _build_counterfactuals(sorted_candidates, applied_weights)

    pathway_summary_map: dict[tuple[str, Optional[str], Optional[str]], dict[str, Any]] = {}
    for candidate in sorted_candidates:
        for pathway in candidate.pathways:
            if not pathway.name:
                continue
            key = (pathway.name, pathway.source, pathway.url)
            entry = pathway_summary_map.setdefault(
                key,
                {
                    "name": pathway.name,
                    "source": pathway.source,
                    "url": pathway.url,
                    "genes": set(),
                    "count": 0,
                },
            )
            entry["count"] += 1
            for gene in pathway.genes:
                entry["genes"].add(gene)

    pathway_summary = [
        PathwaySummaryItem(
            name=value["name"],
            source=value["source"],
            url=value["url"],
            genes=sorted(value["genes"]),
            count=value["count"],
        )
        for value in pathway_summary_map.values()
    ]
    pathway_summary.sort(key=lambda item: (-item.count, item.name.lower()))

    if not background:
        override_keys = ",".join(sorted(applied_overrides)) or "none"
        logger.info(
            "compute_rank persona=%s overrides=%s",
            active_persona,
            override_keys,
        )

    response = RankResponse(
        query=query,
        normalized_disease=normalized_label,
        candidates=sorted_candidates,
        warnings=final_warnings,
        related_concepts=umls_concepts,
        graph_overview=overview_graph,
        cached=False,
        scoring=scoring_summary,
        counterfactuals=counterfactuals,
        pathway_summary=pathway_summary,
    )

    await store_rank(cache_key, normalized_label, response.model_dump(exclude={"cached"}))

    return response

def _to_graph_model(insights) -> GraphInsight | None:
    if not insights:
        return None
    snapshot = GraphSnapshot(
        nodes=[
            GraphNode(
                id=node.get("id", ""),
                label=node.get("label"),
                category=node.get("category"),
            )
            for node in insights.nodes
            if node.get("id")
        ],
        edges=[
            GraphEdge(
                id=edge.get("id"),
                source=edge.get("source", ""),
                target=edge.get("target", ""),
                predicate=edge.get("predicate"),
            )
            for edge in insights.edges
            if edge.get("source") and edge.get("target")
        ],
    )
    return GraphInsight(
        node_count=insights.node_count,
        edge_count=insights.edge_count,
        density=insights.density,
        average_shortest_path=insights.average_shortest_path,
        top_central_nodes=insights.top_central_nodes,
        summary=insights.summary(),
        graph=snapshot,
    )











