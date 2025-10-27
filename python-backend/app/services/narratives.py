from __future__ import annotations

from typing import List, Sequence

from app.models import EvidenceLink, MechanisticNarrative, NarrativeCitation, PathwayInsight, ScoreBreakdown
from app.services.graph import GraphInsights

_MECHANISM_KEYS = ("ChEMBL", "DrugCentral")
_CLINICAL_KEYS = ("ClinicalTrials.gov",)
_SAFETY_KEYS = ("SIDER", "openFDA FAERS")
_TRANSCRIPTOMIC_KEYS = ("CLUE/LINCS",)


def _top_score_components(score: ScoreBreakdown) -> List[str]:
    pairs = [
        ("mechanism fit", score.mechanism_fit),
        ("network proximity", score.network_proximity),
        ("transcriptomic reversal", score.signature_reversal),
        ("clinical signal", score.clinical_signal),
        ("favorable safety profile", max(0.0, -score.safety_penalty)),
    ]
    ranked = sorted(pairs, key=lambda item: item[1], reverse=True)
    return [name for name, value in ranked if value >= 0.05][:3]


def _group_sources(evidence: Sequence[EvidenceLink]) -> dict[str, list[EvidenceLink]]:
    grouped: dict[str, list[EvidenceLink]] = {}
    for item in evidence:
        grouped.setdefault(item.source, []).append(item)
    return grouped


def _select_citations(evidence: Sequence[EvidenceLink]) -> List[NarrativeCitation]:
    citations: list[NarrativeCitation] = []
    for link in evidence[:5]:
        citations.append(
            NarrativeCitation(
                source=link.source,
                label=link.summary or link.source,
                url=link.url,
                detail=link.summary,
            )
        )
    return citations


def _pathway_statement(graph_insights: GraphInsights | None) -> str | None:
    if not graph_insights:
        return None
    if graph_insights.node_count == 0 or graph_insights.edge_count == 0:
        return None
    summary = graph_insights.summary()
    if summary:
        return summary
    dense = graph_insights.density if graph_insights.density is not None else 0.0
    return (
        f"Translator graph exploration stitched {graph_insights.node_count} nodes and "
        f"{graph_insights.edge_count} edges with density {dense:.2f}."
    )


def generate_mechanistic_narrative(
    *,
    drug_name: str,
    score: ScoreBreakdown,
    evidence: Sequence[EvidenceLink],
    graph: GraphInsights | None,
    pathways: Sequence[PathwayInsight] | None = None,
) -> MechanisticNarrative:
    grouped = _group_sources(evidence)
    components = _top_score_components(score)
    unique_pathways: List[str] = []
    pathway_genes: set[str] = set()
    if pathways:
        for pathway in pathways:
            if pathway.name and pathway.name not in unique_pathways:
                unique_pathways.append(pathway.name)
            for gene in pathway.genes:
                pathway_genes.add(gene)

    summary_parts: List[str] = []
    if components:
        summary_parts.append(
            f"{drug_name} stands out for {', '.join(components[:-1])} and {components[-1]}"
            if len(components) > 1
            else f"{drug_name} stands out for {components[0]}"
        )
    else:
        summary_parts.append(f"{drug_name} shows a balanced but modest evidence blend")

    if unique_pathways:
        highlight = ", ".join(unique_pathways[:3])
        if len(unique_pathways) > 3:
            highlight += ", ..."
        summary_parts.append(f"Pathway overlays highlight {highlight}.")

    if grouped:
        if any(src in grouped for src in _MECHANISM_KEYS):
            summary_parts.append("Mechanistic assays support pathway alignment.")
        if any(src in grouped for src in _CLINICAL_KEYS):
            summary_parts.append("Clinical development history reinforces translational viability.")
        if any(src in grouped for src in _SAFETY_KEYS):
            summary_parts.append("Safety surveillance data moderates overall confidence.")
        if any(src in grouped for src in _TRANSCRIPTOMIC_KEYS):
            summary_parts.append("Perturbational signatures indicate transcriptomic reversal.")

    pathway_line = _pathway_statement(graph)
    if pathway_line:
        summary_parts.append(pathway_line)

    reasoning: List[str] = []
    mechanism_evidence = sum(len(grouped.get(source, [])) for source in _MECHANISM_KEYS)
    if mechanism_evidence:
        reasoning.append(
            f"Mechanism fit leverages {mechanism_evidence} direct assay{'s' if mechanism_evidence != 1 else ''} from ChEMBL/DrugCentral."
        )
    if grouped.get("ClinicalTrials.gov"):
        trials = grouped["ClinicalTrials.gov"]
        summaries = [entry.summary for entry in trials if entry.summary]
        trial_summary = ", ".join(summaries[:2])
        reasoning.append(
            "ClinicalTrials.gov records show late-stage activity" + (f" ({trial_summary})" if trial_summary else ".")
        )
    if grouped.get("CLUE/LINCS"):
        reasoning.append("CLUE/LINCS signatures report strong reversal scores supporting anti-disease transcriptional shifts.")
    if grouped.get("SIDER") or grouped.get("openFDA FAERS"):
        reasoning.append("Pharmacovigilance datasets inform safety adjustments incorporated into the composite score.")
    if pathway_genes:
        gene_list = ", ".join(sorted(pathway_genes)[:5])
        if len(pathway_genes) > 5:
            gene_list += ", ..."
        reasoning.append(f"Pathway insights implicate genes {gene_list}.")
    elif unique_pathways:
        reasoning.append("Pathway overlays reinforce disease-relevant signaling cascades.")
    if not reasoning:
        reasoning.append("Evidence sources contribute evenly without a dominant channel.")

    citations = _select_citations(list(evidence))

    return MechanisticNarrative(
        summary=" ".join(summary_parts),
        reasoning_steps=reasoning,
        citations=citations,
    )





