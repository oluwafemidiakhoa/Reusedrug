"use client";

import clsx from "clsx";
import { ReactNode, useEffect, useMemo, useState, useTransition } from "react";
import { useSession, signIn } from "next-auth/react";
import Card from "@/components/ui/Card";
import Metric from "@/components/ui/Metric";
import Chip from "@/components/ui/Chip";
import SectionHeader from "@/components/ui/SectionHeader";
import EvidenceGraphPreview, { GraphInsight as GraphInsightModel } from "@/components/EvidenceGraphPreview";
import GraphExplorer from "@/components/GraphExplorer";

export type EvidenceLink = {
  source: string;
  url: string;
  summary?: string | null;
};

export type DrugAnnotationSource = {
  label: string;
  url?: string | null;
};

export type PathwayInsight = {
  name: string;
  source?: string | null;
  url?: string | null;
  genes: string[];
};

export type PathwaySummaryItem = {
  name: string;
  source?: string | null;
  url?: string | null;
  genes: string[];
  count: number;
};

export type ScoreBreakdown = {
  mechanism_fit: number;
  network_proximity: number;
  signature_reversal: number;
  clinical_signal: number;
  safety_penalty: number;
  final_score: number;
  confidence_low?: number | null;
  confidence_high?: number | null;
};

export type CandidateConfidence = {
  score: number;
  tier: "exploratory" | "hypothesis-ready" | "decision-grade";
  signals: string[];
};

export type Candidate = {
  drug_id: string;
  name: string;
  score: ScoreBreakdown;
  evidence: EvidenceLink[];
  graph_insights?: GraphInsightModel | null;
  narrative?: {
    summary: string;
    reasoning_steps: string[];
    citations: Array<{ source: string; label: string; url?: string | null; detail?: string | null }>;
  } | null;
  confidence?: CandidateConfidence | null;
  indications?: string[] | null;
  contraindications?: string[] | null;
  annotation_sources?: DrugAnnotationSource[] | null;
  pathways?: PathwayInsight[] | null;
};

export type CounterfactualScenario = {
  label: string;
  description?: string | null;
  weights: Record<string, number>;
  candidates: Candidate[];
};

export type ConceptMapping = {
  cui: string;
  name: string;
  preferred_name: string;
  synonyms: string[];
  semantic_types: string[];
};

type Props = {
  query: string;
  loading: boolean;
  candidates: Candidate[];
  warnings: { source: string; detail: string }[];
  normalizedDisease?: string | null;
  backendCached?: boolean;
  bffCached?: boolean;
  relatedConcepts?: ConceptMapping[];
  graphOverview?: GraphInsightModel | null;
  scoringSummary?: {
    persona: string;
    weights: Record<string, number>;
    delta_vs_default: Record<string, number>;
    overrides: Record<string, number>;
  } | null;
  personaLabel?: string;
  scoringControls?: ReactNode;
  counterfactuals?: CounterfactualScenario[];
  pathwaySummary?: PathwaySummaryItem[];
  excludeContraindicated?: boolean;
  onToggleContraindicated?: (value: boolean) => void;
};

const FILTER_DEFS: Record<string, { label: string; sources: string[]; description: string }> = {
  mechanism: {
    label: "Mechanism signalling",
    sources: ["ChEMBL", "DrugCentral"],
    description: "Mechanism-of-action, potency, pathway support",
  },
  network: {
    label: "Graph proximity",
    sources: ["Open Targets", "NCATS Translator", "Graph Analytics"],
    description: "Target coverage and knowledge-graph connectivity",
  },
  clinical: {
    label: "Clinical signal",
    sources: ["ClinicalTrials.gov"],
    description: "Recruiting, active, and completed trials",
  },
  signature: {
    label: "Transcriptomic",
    sources: ["CLUE/LINCS"],
    description: "Signature-reversal compatibility",
  },
  safety: {
    label: "Safety",
    sources: ["openFDA FAERS", "SIDER"],
    description: "Adverse events and pharmacovigilance evidence",
  },
  literature: {
    label: "Literature",
    sources: ["PubMed"],
    description: "Peer-reviewed publication support",
  },
};

const MAX_EVIDENCE_PREVIEW = 5;

export default function ResultsList({
  query,
  loading,
  candidates,
  warnings,
  normalizedDisease,
  backendCached,
  bffCached,
  relatedConcepts = [],
  graphOverview,
  scoringSummary = null,
  personaLabel,
  scoringControls,
  counterfactuals = [],
  pathwaySummary = [],
  excludeContraindicated = false,
  onToggleContraindicated,
}: Props) {
  const { data: session, status } = useSession();
  const [expanded, setExpanded] = useState<string | null>(null);
  const [filters, setFilters] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(Object.keys(FILTER_DEFS).map((key) => [key, false]))
  );
  const [message, setMessage] = useState<string | null>(null);
  const [saveNote, setSaveNote] = useState<string>("");
  const [isPending, startTransition] = useTransition();
  const pathwaySummaryItems = pathwaySummary ?? [];
  const topPathways = useMemo(() => pathwaySummaryItems.slice(0, 4), [pathwaySummaryItems]);

  useEffect(() => {
    setMessage(null);
  }, [query]);

  const activeFilters = useMemo(
    () => Object.entries(filters).filter(([, value]) => value).map(([key]) => key),
    [filters]
  );

  const filteredCandidates = useMemo(() => {
    if (activeFilters.length === 0) {
      return candidates;
    }
    return candidates.filter((candidate) =>
      activeFilters.every((filterKey) =>
        candidate.evidence.some((item) => FILTER_DEFS[filterKey].sources.includes(item.source))
      )
    );
  }, [activeFilters, candidates]);

  const conceptSummary = useMemo(() => {
    if (!relatedConcepts.length) {
      return null;
    }
    return relatedConcepts
      .slice(0, 3)
      .map((concept) => `${concept.preferred_name || concept.name} (${concept.cui})`)
      .join(", ");
  }, [relatedConcepts]);

  const conceptSynonyms = useMemo(() => {
    if (!relatedConcepts.length) {
      return [];
    }
    return (relatedConcepts[0].synonyms || []).slice(0, 6);
  }, [relatedConcepts]);

  const classificationChips = useMemo(() => {
    if (!filteredCandidates.length) {
      return [];
    }
    const set = new Set<string>();
    filteredCandidates.forEach((candidate) => {
      Object.values(FILTER_DEFS).forEach((def) => {
        if (candidate.evidence.some((item) => def.sources.includes(item.source))) {
          set.add(def.label);
        }
      });
    });
    return Array.from(set);
  }, [filteredCandidates]);

  const topCandidate = filteredCandidates[0];
  const baseRankMap = useMemo(() => {
    const map = new Map<string, { rank: number; score: number }>();
    candidates.forEach((candidate, index) => {
      map.set(candidate.drug_id, { rank: index + 1, score: candidate.score.final_score });
    });
    return map;
  }, [candidates]);
  const highConfidence = topCandidate?.score.confidence_high ?? null;
  const lowConfidence = topCandidate?.score.confidence_low ?? null;
  const totalWarnings = warnings.length;

  if (!query) {
    return (
      <div className="glass-panel flex flex-col items-center gap-3 p-12 text-slate-300">
        <h3 className="text-lg font-semibold text-slate-100">Enter a disease to activate the workspace</h3>
        <p className="max-w-lg text-center text-sm text-slate-400">
          Start from a disease, phenotype, or pathway to orchestrate multi-source evidence and surface ranked repurposing opportunities.
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="glass-panel flex flex-col items-center gap-4 p-12">
        <span className="h-10 w-10 animate-spin rounded-full border-4 border-primary-500/60 border-t-transparent" />
        <p className="text-sm text-slate-300">Synthesising graph signals, clinical evidence, and literature...</p>
      </div>
    );
  }

  const cacheMessage =
    backendCached || bffCached
      ? `Served from ${backendCached ? "backend cache" : "BFF cache"}`
      : null;

  const authenticated = status === "authenticated" && Boolean(session?.apiKey);

  const handleFilterToggle = (filterKey: string) => {
    setFilters((prev) => ({
      ...prev,
      [filterKey]: !prev[filterKey],
    }));
  };

  const handleSave = () => {
    if (!authenticated) {
      signIn(undefined, { callbackUrl: "/workspace" });
      return;
    }
    if (!candidates.length) {
      setMessage("No results to save yet.");
      return;
    }
    setMessage(null);
    startTransition(async () => {
      try {
        const response = await fetch("/api/workspace/queries", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            disease: query,
            response: {
              query,
              normalized_disease: normalizedDisease,
              candidates,
              warnings,
              cached: Boolean(backendCached),
              related_concepts: relatedConcepts,
            },
            note: saveNote.trim() || undefined,
          }),
        });
        if (!response.ok) {
          throw new Error("Failed to save");
        }
        setSaveNote("");
        setMessage("Saved to workspace");
      } catch (error) {
        setMessage((error as Error).message ?? "Failed to save");
      }
    });
  };

  const candidatePanel = filteredCandidates.length === 0 ? (
    <Card
      title={
        candidates.length === 0
          ? "No candidates were returned."
          : "No candidates match the selected evidence filters."
      }
      description="Adjust evidence filters or broaden the disease query to surface additional hypotheses."
    />
  ) : (
    <div className="flex flex-col gap-6">
      {filteredCandidates.map((candidate) => {
        const isOpen = expanded === candidate.drug_id;
        const categories = Object.entries(FILTER_DEFS)
          .filter(({ 1: def }) =>
            candidate.evidence.some((item) => def.sources.includes(item.source))
          )
          .map(([key, def]) => ({ key, label: def.label }));

        return (
          <Card
            key={candidate.drug_id}
            title={candidate.name}
            description={
              <span className="text-xs uppercase tracking-wide text-slate-500">
                {candidate.drug_id}
              </span>
            }
            action={
              <button
                onClick={() => setExpanded(isOpen ? null : candidate.drug_id)}
                className="text-sm font-semibold text-primary-200 underline-offset-4 transition hover:underline"
              >
                {isOpen ? "Collapse evidence" : "Inspect evidence"}
              </button>
            }
          >
            <div className="flex flex-wrap items-center gap-2">
              {categories.map(({ key, label }) => (
                <Chip key={`${candidate.drug_id}-${key}`} tone="info">
                  {label}
                </Chip>
              ))}
              {candidate.confidence ? (
                <Chip tone={confidenceTone(candidate.confidence.tier)}>
                  {formatConfidenceTier(candidate.confidence.tier)} - {candidate.confidence.score.toFixed(2)}
                </Chip>
              ) : (
                <Chip tone="info">Score {candidate.score.final_score.toFixed(2)}</Chip>
              )}
              {candidate.indications && candidate.indications.length > 0 && (
                <Chip tone="success">
                  Indications: {candidate.indications.slice(0, 2).join(", ")}
                  {candidate.indications.length > 2 ? " +" : ""}
                </Chip>
              )}
              {candidate.contraindications && candidate.contraindications.length > 0 && (
                <Chip tone="warning">
                  Contraindications: {candidate.contraindications.slice(0, 2).join(", ")}
                  {candidate.contraindications.length > 2 ? " +" : ""}
                </Chip>
              )}
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <Metric label="Composite score" value={candidate.score.final_score.toFixed(2)} trend="up" hint="Calibrated" />
              <Metric
                label="Confidence window"
                value={
                  candidate.score.confidence_low !== undefined &&
                  candidate.score.confidence_high !== undefined
                    ? `${candidate.score.confidence_low.toFixed(2)} - ${candidate.score.confidence_high.toFixed(2)}`
                    : "-"
                }
                hint="95% credible"
              />
              <Metric
                label="Confidence tier"
                value={candidate.confidence ? formatConfidenceTier(candidate.confidence.tier) : "Exploratory"}
                hint={candidate.confidence ? `Score ${candidate.confidence.score.toFixed(2)}` : undefined}
              />
              <Metric label="Evidence sources" value={candidate.evidence.length} hint="unique channels" />
            </div>

                {isOpen && (
                  <div className="mt-4 grid gap-4 lg:grid-cols-2">
                    {candidate.narrative && (
                      <div className="surface rounded-xl p-4">
                        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                          Mechanistic narrative
                        </p>
                        <p className="mt-3 text-sm text-slate-200">{candidate.narrative.summary}</p>
                        {candidate.narrative.reasoning_steps.length > 0 && (
                          <ul className="mt-3 space-y-2 text-xs text-slate-400">
                            {candidate.narrative.reasoning_steps.map((step, index) => (
                              <li key={index} className="leading-relaxed">
                                {step}
                              </li>
                            ))}
                          </ul>
                        )}
                        {candidate.narrative.citations.length > 0 && (
                          <div className="mt-4 space-y-2">
                            <p className="text-[11px] uppercase tracking-wide text-slate-500">Cited evidence</p>
                            <ul className="space-y-2 text-xs text-slate-400">
                              {candidate.narrative.citations.slice(0, 4).map((citation, index) => (
                                <li key={`${citation.source}-${index}`} className="flex flex-col">
                                  <span className="font-semibold text-slate-200">{citation.source}</span>
                                  {citation.label && <span>{citation.label}</span>}
                                  {citation.url && (
                                    <a
                                      href={citation.url}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="text-primary-300 underline-offset-2 hover:underline"
                                    >
                                      View source
                                    </a>
                                  )}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                    <div className="surface rounded-xl p-4">
                      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                        Evidence spotlight
                      </p>
                  <ul className="mt-3 space-y-3 text-sm text-slate-300">
                    {candidate.evidence.slice(0, MAX_EVIDENCE_PREVIEW).map((item, index) => (
                      <li key={`${item.url}-${index}`} className="flex flex-col gap-1">
                        <span className="font-semibold text-slate-100">{item.source}</span>
                        {item.summary && <span className="text-xs text-slate-400">{item.summary}</span>}
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noreferrer"
                          className="text-xs text-primary-300 underline-offset-2 hover:underline"
                        >
                          View source
                        </a>
                      </li>
                    ))}
                    {candidate.evidence.length > MAX_EVIDENCE_PREVIEW && (
                      <li className="text-xs text-slate-500">
                        + {candidate.evidence.length - MAX_EVIDENCE_PREVIEW} additional evidence sources
                      </li>
                    )}
                  </ul>
                </div>

                {candidate.pathways && candidate.pathways.length > 0 && (
                  <div className="surface rounded-xl p-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Pathway overlays
                    </p>
                    <ul className="mt-3 space-y-2 text-xs text-slate-300">
                      {candidate.pathways.map((pathway, index) => (
                        <li key={`${candidate.drug_id}-pathway-${index}`} className="rounded-lg border border-slate-800/60 bg-slate-900/60 p-3">
                          <div className="flex items-center justify-between text-sm text-slate-100">
                            <span>{pathway.name}</span>
                            {pathway.source && <span className="text-xs text-slate-500">{pathway.source}</span>}
                          </div>
                          {pathway.genes.length > 0 && (
                            <p className="mt-1 text-[11px] text-slate-400">Genes: {pathway.genes.join(", ")}</p>
                          )}
                          {pathway.url && (
                            <a
                              href={pathway.url}
                              target="_blank"
                              rel="noreferrer"
                              className="mt-2 inline-flex text-[11px] text-primary-300 underline-offset-2 hover:underline"
                            >
                              View pathway
                            </a>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {candidate.annotation_sources && candidate.annotation_sources.length > 0 && (
                  <div className="surface rounded-xl p-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Annotation sources
                    </p>
                    <ul className="mt-3 space-y-2 text-xs text-slate-300">
                      {candidate.annotation_sources.map((source, index) => (
                        <li key={`${candidate.drug_id}-annotation-${index}`} className="flex items-center justify-between gap-3 rounded-lg border border-slate-800/60 bg-slate-900/60 p-3">
                          <span className="font-medium text-slate-100">{source.label}</span>
                          {source.url && (
                            <a
                              href={source.url}
                              target="_blank"
                              rel="noreferrer"
                              className="text-[11px] text-primary-300 underline-offset-2 hover:underline"
                            >
                              View source
                            </a>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="surface rounded-xl p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Confidence intelligence
                  </p>
                  {candidate.confidence ? (
                    <>
                      <div className="mt-3 flex flex-wrap gap-2">
                        <Chip tone={confidenceTone(candidate.confidence.tier)}>
                          {formatConfidenceTier(candidate.confidence.tier)} tier
                        </Chip>
                        <Chip tone="info">Evidence score {candidate.confidence.score.toFixed(2)}</Chip>
                      </div>
                      <p className="mt-3 text-sm text-slate-300">
                        Signals aggregated from {candidate.confidence.signals.length} evidence channel
                        {candidate.confidence.signals.length === 1 ? "" : "s"} with calibrated penalties for warnings and safety.
                      </p>
                      <ul className="mt-3 space-y-2 text-xs text-slate-300">
                        {candidate.confidence.signals.map((signal, index) => (
                          <li key={`${candidate.drug_id}-confidence-${index}`} className="flex items-start gap-2">
                            <span className="mt-1 inline-flex h-2 w-2 flex-shrink-0 rounded-full bg-primary-400" />
                            <span className="flex-1 leading-relaxed">{signal}</span>
                          </li>
                        ))}
                      </ul>
                    </>
                  ) : (
                    <p className="mt-3 text-sm text-slate-300">
                      Final score blends mechanism fit, graph analytics, transcriptomics, clinical evidence, and safety adjustments with configurable weights.
                    </p>
                  )}
                  <div className="mt-4 space-y-3 text-xs text-slate-400">
                    {buildContributionEntries(candidate.score).map((entry) => (
                      <div key={`${candidate.drug_id}-${entry.key}`} className="space-y-1">
                        <div className="flex items-center justify-between">
                          <span>{entry.label}</span>
                          <span>{entry.display}</span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-slate-800/60">
                          <div
                            className={clsx(
                              "h-2 rounded-full transition-all",
                              entry.positive ? "bg-primary-400" : "bg-rose-400"
                            )}
                            style={{ width: `${entry.width}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {candidate.graph_insights && candidate.graph_insights.graph && candidate.graph_insights.graph.nodes.length > 0 && (
                  <div className="lg:col-span-2">
                    <EvidenceGraphPreview insight={candidate.graph_insights} variant="inline" />
                  </div>
                )}
              </div>
            )}
          </Card>
        );
      })}
    </div>
  );

  return (
    <div className="flex flex-col gap-8">
      <SectionHeader
        eyebrow="Evidence synthesis"
        title={`Results for "${query}"`}
        description="Ranked by composite score with calibrated confidence intervals across all evidence channels."
        actions={
          <button
            onClick={handleSave}
            disabled={isPending || !candidates.length}
            className="inline-flex items-center gap-2 rounded-full border border-primary-500/50 px-4 py-2 text-sm font-semibold text-primary-200 transition hover:border-primary-300 hover:text-primary-100 disabled:cursor-not-allowed disabled:border-slate-700 disabled:text-slate-500"
          >
            {authenticated ? (isPending ? "Saving..." : "Save to workspace") : "Sign in to save"}
          </button>
        }
      />

      <div className="rounded-2xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-inner">
        {onToggleContraindicated && (
          <label className="mb-3 flex items-center gap-2 text-xs text-slate-300">
            <input
              type="checkbox"
              className="h-4 w-4 rounded border border-slate-600 bg-slate-900 text-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-400/40"
              checked={excludeContraindicated}
              onChange={(event) => onToggleContraindicated(event.target.checked)}
            />
            <span>Exclude contraindicated candidates automatically</span>
          </label>
        )}
        <label htmlFor="workspace-note" className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Workspace note
        </label>
        <textarea
          id="workspace-note"
          value={saveNote}
          onChange={(event) => setSaveNote(event.target.value)}
          placeholder={authenticated ? "Add context or next actions for this run..." : "Sign in to annotate and save notes"}
          maxLength={500}
          rows={3}
          disabled={!authenticated || isPending}
          className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950/70 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-500 focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500/40 disabled:cursor-not-allowed disabled:opacity-60"
        />
        <p className="mt-2 text-[11px] text-slate-500">
          Notes are stored with the saved query so your team can review decisions later.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Metric
          label="Top confidence"
          value={highConfidence ? highConfidence.toFixed(2) : "-"}
          trend="up"
          hint={lowConfidence ? `floor ${lowConfidence.toFixed(2)}` : undefined}
        />
        <Metric
          label="Candidates surfaced"
          value={filteredCandidates.length}
          hint={`${candidates.length - filteredCandidates.length} filtered`}
        />
        <Metric
          label="Warnings"
          value={totalWarnings}
          trend={totalWarnings ? "warning" : "steady"}
          hint={cacheMessage ?? "fresh"}
        />
        <Metric
          label="Evidence density"
          value={topCandidate ? topCandidate.evidence.length : 0}
          hint="sources per lead"
        />
      </div>

      {topPathways.length > 0 && (
        <Card
          title="Pathway focus"
          description="Most frequent pathway signals across returned candidates."
        >
          <ul className="space-y-3 text-sm text-slate-300">
            {topPathways.map((item) => {
              const genePreview = item.genes.slice(0, 5).join(", ");
              const geneSuffix = item.genes.length > 5 ? " ..." : "";
              return (
                <li
                  key={`pathway-summary-${item.name}-${item.source ?? "unknown"}`}
                  className="rounded-xl border border-slate-800/60 bg-slate-900/60 p-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-slate-100">{item.name}</span>
                    <span className="text-xs text-slate-500">
                      {item.count} candidate{item.count === 1 ? "" : "s"}
                    </span>
                  </div>
                  {item.source && (
                    <p className="text-[11px] text-slate-500">Source: {item.source}</p>
                  )}
                  {item.genes.length > 0 && (
                    <p className="text-[11px] text-slate-400">
                      Genes: {genePreview}
                      {geneSuffix}
                    </p>
                  )}
                  {item.url && (
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-2 inline-flex text-[11px] text-primary-300 underline-offset-2 hover:underline"
                    >
                      View pathway
                    </a>
                  )}
                </li>
              );
            })}
          </ul>
        </Card>
      )}

      {scoringSummary && (
        <Card
          title="Active scoring blend"
          description="Current weighting profile relative to the balanced default."
        >
          <div className="flex flex-wrap items-center gap-2">
            <Chip tone="info" className="persona-chip persona-chip--active">
              {personaLabel ?? scoringSummary.persona}
            </Chip>
            {Object.entries(scoringSummary.delta_vs_default)
              .filter(([, delta]) => Math.abs(delta) >= 0.005)
              .map(([key, delta]) => (
                <Chip
                  key={key}
                  tone={delta >= 0 ? "success" : "warning"}
                  className="persona-chip"
                >
                  {formatAxisLabel(key)} {formatDelta(delta)}
                </Chip>
              ))}
            {Object.values(scoringSummary.delta_vs_default).every((delta) => Math.abs(delta) < 0.005) && (
              <Chip className="persona-chip persona-chip--quiet">Aligned with default</Chip>
            )}
          </div>
        </Card>
      )}

      <Card
        title={normalizedDisease ? `${normalizedDisease} (normalized)` : "Disease normalization"}
        description="Concept harmonisation via UMLS and Translator ontologies ensures downstream evidence is aligned."
      >
        <div className="flex flex-wrap gap-2">
          {conceptSummary ? <Chip tone="info">{conceptSummary}</Chip> : <Chip>No normalized concept available</Chip>}
          {conceptSynonyms.map((synonym) => (
            <Chip key={synonym}>{synonym}</Chip>
          ))}
        </div>
      </Card>

      <Card
        title="Focus evidence channels"
        description="Toggle evidence filters to hone in on candidates supported by specific modalities."
      >
        <div className="grid gap-3 md:grid-cols-3 lg:grid-cols-6">
          {Object.entries(FILTER_DEFS).map(([key, def]) => {
            const active = filters[key];
            return (
              <button
                key={key}
                type="button"
                onClick={() => handleFilterToggle(key)}
                className={clsx(
                  "h-full rounded-2xl border px-3 py-3 text-left text-sm transition",
                  active
                    ? "border-primary-500/60 bg-primary-500/10 text-primary-100"
                    : "border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700"
                )}
              >
                <span className="font-semibold">{def.label}</span>
                <span className="mt-1 block text-xs text-slate-400">{def.description}</span>
              </button>
            );
          })}
        </div>
      </Card>

      {warnings.length > 0 && (
        <Card
          title="Pipeline warnings"
          description="Upstream providers may be unavailable or returned limited evidence."
        >
          <ul className="grid gap-3 md:grid-cols-2">
            {warnings.map((warning, index) => (
              <li
                key={`${warning.source}-${index}`}
                className="surface rounded-xl border border-slate-800/60 p-4"
              >
                <p className="text-sm font-semibold text-slate-200">{warning.source}</p>
                <p className="text-xs text-slate-400">{warning.detail}</p>
              </li>
            ))}
          </ul>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <div className="flex flex-col gap-6">{candidatePanel}</div>

        <div className="flex flex-col gap-6">
          {scoringControls}
          {graphOverview && graphOverview.graph && graphOverview.graph.nodes.length > 0 && (
            <GraphExplorer insight={graphOverview} />
          )}
          {counterfactuals.length > 0 && (
            <Card
              title="Counterfactual scenarios"
              description="How rankings shift when safety weight is reduced."
            >
              <div className="space-y-4">
                {counterfactuals.map((scenario) => (
                  <div key={scenario.label} className="rounded-2xl border border-slate-800/60 bg-slate-900/60 p-4">
                    <div className="flex flex-col gap-1 md:flex-row md:items-center md:justify-between">
                      <h4 className="text-sm font-semibold text-slate-100">{scenario.label}</h4>
                      <p className="text-xs text-slate-500">
                        {scenario.description ?? "Weights renormalized across all evidence channels."}
                      </p>
                    </div>
                    <ul className="mt-3 space-y-2 text-xs text-slate-300">
                      {scenario.candidates.map((candidate, index) => {
                        const base = baseRankMap.get(candidate.drug_id);
                        const deltaRank = base ? base.rank - (index + 1) : 0;
                        const deltaScore = base ? candidate.score.final_score - base.score : 0;
                        return (
                          <li
                            key={`${scenario.label}-${candidate.drug_id}`}
                            className="flex items-center justify-between gap-2 rounded-xl border border-slate-800/60 bg-slate-900/70 px-3 py-2"
                          >
                            <div className="flex flex-col">
                              <span className="text-sm font-semibold text-slate-100">
                                #{index + 1} - {candidate.name}
                              </span>
                              <span className="text-[11px] text-slate-500">
                                Base rank {base ? `#${base.rank}` : "-"} | Delta score {deltaScore >= 0 ? "+" : ""}
                                {deltaScore.toFixed(3)} | Delta rank {deltaRank >= 0 ? "+" : ""}
                                {deltaRank}
                              </span>
                            </div>
                            <span className="text-sm font-semibold text-primary-200">
                              {candidate.score.final_score.toFixed(3)}
                            </span>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                ))}
              </div>
            </Card>
          )}
          <Card
            title="Cache status"
            description="Understand data freshness and caching layer used for this response."
          >
            <div className="flex flex-col gap-3 text-sm text-slate-300">
              <div className="surface rounded-xl border border-slate-800/60 p-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Serving tier</p>
                <p className="text-sm text-slate-200">{cacheMessage ?? "Generated fresh from upstream providers"}</p>
              </div>
              <div className="surface rounded-xl border border-slate-800/60 p-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Filters active</p>
                <p className="text-sm text-slate-200">
                  {activeFilters.length
                    ? activeFilters.map((filter) => FILTER_DEFS[filter].label).join(", ")
                    : "All evidence channels"}
                </p>
              </div>
              {classificationChips.length > 0 && (
                <div className="surface rounded-xl border border-slate-800/60 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Evidence coverage</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {classificationChips.map((chip) => (
                      <Chip key={chip}>{chip}</Chip>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>

          <Card
            title="🎉 New: Knowledge Graph Explorer"
            description="Visualize drug-disease relationships in an interactive Neo4j-powered graph."
          >
            <div className="space-y-4">
              <div className="rounded-lg bg-gradient-to-r from-indigo-900/50 to-purple-900/50 p-4 border border-indigo-700/50">
                <div className="flex items-start gap-3">
                  <div className="text-3xl">🌐</div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-white mb-1">Interactive Graph Now Live!</h4>
                    <p className="text-sm text-slate-300 mb-3">
                      Explore {candidates.length} drugs and their relationships with dynamic layouts,
                      real-time statistics, and exportable visualizations.
                    </p>
                    <a
                      href="/graph"
                      className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-medium transition-all text-sm"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                      Open Graph Explorer
                    </a>
                  </div>
                </div>
              </div>

              <div className="text-xs text-slate-400">
                <p className="font-semibold text-slate-300 mb-2">Coming Next:</p>
                <ul className="space-y-1.5 text-slate-400">
                  <li>• Cohort-level signal detection and real-world evidence</li>
                  <li>• Workspace comparisons & annotation trails</li>
                  <li>• Advanced graph filters and path analysis</li>
                </ul>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {message && (
        <div className="glass-panel border-primary-400/40 bg-primary-500/10 p-4 text-sm text-primary-100">{message}</div>
      )}
    </div>
  );
}

function formatConfidenceTier(tier: CandidateConfidence["tier"]): string {
  switch (tier) {
    case "decision-grade":
      return "Decision-grade";
    case "hypothesis-ready":
      return "Hypothesis-ready";
    default:
      return "Exploratory";
  }
}

function confidenceTone(tier: CandidateConfidence["tier"]): "success" | "info" | "warning" {
  if (tier === "decision-grade") {
    return "success";
  }
  if (tier === "hypothesis-ready") {
    return "info";
  }
  return "warning";
}

type ContributionEntry = {
  key: string;
  label: string;
  value: number;
  display: string;
  positive: boolean;
  width: number;
};

function buildContributionEntries(score: ScoreBreakdown): ContributionEntry[] {
  const entries: ContributionEntry[] = [
    {
      key: "mechanism_fit",
      label: "Mechanism contribution",
      value: score.mechanism_fit,
      display: score.mechanism_fit.toFixed(2),
      positive: score.mechanism_fit >= 0,
      width: Math.min(100, Math.round(Math.abs(score.mechanism_fit) * 100))
    },
    {
      key: "network_proximity",
      label: "Graph contribution",
      value: score.network_proximity,
      display: score.network_proximity.toFixed(2),
      positive: score.network_proximity >= 0,
      width: Math.min(100, Math.round(Math.abs(score.network_proximity) * 100))
    },
    {
      key: "signature_reversal",
      label: "Transcriptomic contribution",
      value: score.signature_reversal,
      display: score.signature_reversal.toFixed(2),
      positive: score.signature_reversal >= 0,
      width: Math.min(100, Math.round(Math.abs(score.signature_reversal) * 100))
    },
    {
      key: "clinical_signal",
      label: "Clinical contribution",
      value: score.clinical_signal,
      display: score.clinical_signal.toFixed(2),
      positive: score.clinical_signal >= 0,
      width: Math.min(100, Math.round(Math.abs(score.clinical_signal) * 100))
    },
    {
      key: "safety_penalty",
      label: "Safety adjustment",
      value: score.safety_penalty,
      display: score.safety_penalty.toFixed(2),
      positive: score.safety_penalty >= 0,
      width: Math.min(100, Math.round(Math.abs(score.safety_penalty) * 100))
    }
  ];

  return entries;
}

function formatDelta(delta: number): string {
  const percent = Math.round(delta * 100);
  if (percent === 0) {
    return "0%";
  }
  return `${percent > 0 ? "+" : ""}${percent}%`;
}

function formatAxisLabel(key: string): string {
  switch (key) {
    case "mechanism":
      return "Mechanism";
    case "network":
      return "Network";
    case "signature":
      return "Signature";
    case "clinical":
      return "Clinical";
    case "safety":
      return "Safety";
    default:
      return key;
  }
}






