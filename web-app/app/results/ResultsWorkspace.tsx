type PathwaySummaryItem = {
  name: string;
  source?: string | null;
  url?: string | null;
  genes: string[];
  count: number;
};

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ResultsList, {
  Candidate,
  ConceptMapping
} from "@/components/ResultsList";
import ScoringPersonasPanel from "@/components/ScoringPersonasPanel";
import { trackPersonaUsage } from "@/lib/analytics";
import type { CounterfactualScenario } from "@/components/ResultsList";

type PersonaDefinition = {
  name: string;
  label: string;
  description?: string | null;
  weights: Record<string, number>;
};

type ScoringMetadata = {
  default_persona: string;
  default_weights: Record<string, number>;
  personas: PersonaDefinition[];
};

type ScoringSummary = {
  persona: string;
  weights: Record<string, number>;
  delta_vs_default: Record<string, number>;
  overrides: Record<string, number>;
};

type RankResponse = {
  candidates: Candidate[];
  pathway_summary?: PathwaySummaryItem[];
  warnings: { source: string; detail: string }[];
  normalized_disease: string | null;
  cached: boolean;
  related_concepts: ConceptMapping[];
  graph_overview?: unknown;
  scoring?: ScoringSummary | null;
  counterfactuals?: CounterfactualScenario[];
};

const WEIGHT_KEYS = ["mechanism", "network", "signature", "clinical", "safety"] as const;
const BALANCED_DEFAULT: Record<string, number> = {
  mechanism: 0.3,
  network: 0.25,
  signature: 0.2,
  clinical: 0.15,
  safety: 0.1
};

type Props = {
  query: string;
};

export default function ResultsWorkspace({ query }: Props) {
  const [metadata, setMetadata] = useState<{
    defaultPersona: string;
    defaultWeights: Record<string, number>;
    personas: PersonaDefinition[];
  } | null>(null);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [selectedPersona, setSelectedPersona] = useState<string>("balanced");
  const [weights, setWeights] = useState<Record<string, number> | null>(null);
  const [results, setResults] = useState<RankResponse | null>(null);
  const [backendCached, setBackendCached] = useState<boolean>(false);
  const [bffCached, setBffCached] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [excludeContraindicated, setExcludeContraindicated] = useState<boolean>(false);
  const lastSignatureRef = useRef<string | null>(null);
  const fetchTimeoutRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    const loadMetadata = async () => {
      try {
        const response = await fetch("/api/repurpose/metadata", { cache: "no-store" });
        if (!response.ok) {
          throw new Error("Failed to load scoring metadata");
        }
        const payload: ScoringMetadata = await response.json();
        if (cancelled) {
          return;
        }
        const personas = normalisePersonas(payload.personas, payload.default_weights);
        setMetadata({
          defaultPersona: payload.default_persona ?? "balanced",
          defaultWeights: normalizeWeights(payload.default_weights),
          personas
        });
        setSelectedPersona(payload.default_persona ?? "balanced");
        setWeights((prev) => prev ?? normalizeWeights(payload.default_weights));
        setMetadataError(null);
      } catch (err) {
        if (!cancelled) {
          setMetadataError((err as Error).message ?? "Metadata unavailable");
          const fallbackWeights = normalizeWeights(BALANCED_DEFAULT);
          setMetadata({
            defaultPersona: "balanced",
            defaultWeights: fallbackWeights,
            personas: [
              {
                name: "balanced",
                label: "Balanced",
                description: "Default weighting across evidence channels.",
                weights: fallbackWeights
              }
            ]
          });
          setWeights((prev) => prev ?? fallbackWeights);
          setSelectedPersona("balanced");
        }
      }
    };
    loadMetadata();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!query) {
      setResults(null);
      setBackendCached(false);
      setBffCached(false);
      setError(null);
      return;
    }
    lastSignatureRef.current = null;
  }, [query]);

  const personas = useMemo(() => metadata?.personas ?? [], [metadata]);
  const personaLookup = useMemo(
    () => Object.fromEntries(personas.map((persona) => [persona.name, persona])),
    [personas]
  );

  const personaForRequest = useMemo(() => {
    if (!metadata) {
      return "balanced";
    }
    if (selectedPersona === "custom") {
      return "custom";
    }
    if (personaLookup[selectedPersona]) {
      return selectedPersona;
    }
    return metadata.defaultPersona;
  }, [metadata, personaLookup, selectedPersona]);

  const currentWeights = useMemo(() => weights ?? metadata?.defaultWeights ?? null, [metadata, weights]);

  const requestSignature = useMemo(() => {
    if (!query || !currentWeights) {
      return null;
    }
    const persona = personaForRequest;
    const weightsPayload = persona === "custom" ? currentWeights : undefined;
    return JSON.stringify({
      query,
      persona,
      weights: weightsPayload,
      exclude_contraindicated: excludeContraindicated
    });
  }, [currentWeights, personaForRequest, query, excludeContraindicated]);

  const scheduleFetch = useCallback(
    (signal: AbortController) => {
      if (!query || !currentWeights) {
        return;
      }
      const persona = personaForRequest;
      const requestPayload = {
        disease: query,
        ...(persona ? { persona } : {}),
        ...(persona === "custom" ? { weights: currentWeights } : {}),
        exclude_contraindicated: excludeContraindicated
      };
      setLoading(true);
      setError(null);
      void (async () => {
        try {
          const response = await fetch("/api/repurpose", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestPayload),
            cache: "no-store",
            signal: signal.signal
          });
          if (!response.ok) {
            throw new Error("Ranking request failed");
          }
          const body = await response.json();
          const responseData = body.data ?? {};
          const rawCandidates = Array.isArray(responseData.candidates) ? responseData.candidates : [];
          const normalizedCandidates = rawCandidates.map((candidate: any) => ({
            ...candidate,
            graph_insights: candidate?.graph_insights ?? null,
            narrative: candidate?.narrative ?? null
          })) as Candidate[];
          const normalizedWarnings = Array.isArray(responseData.warnings) ? responseData.warnings : [];
          const relatedConcepts = Array.isArray(responseData.related_concepts)
            ? (responseData.related_concepts as ConceptMapping[])
            : [];
          const counterfactuals = Array.isArray(responseData.counterfactuals)
            ? (responseData.counterfactuals as any[]).map((scenario) => ({
                label: scenario.label,
                description: scenario.description ?? null,
                weights: scenario.weights ?? {},
                candidates: (scenario.candidates ?? []).map((candidate: any) => ({
                  ...candidate,
                  graph_insights: candidate?.graph_insights ?? null,
                  narrative: candidate?.narrative ?? null
                })) as Candidate[]
              }))
            : [];

          const structured: RankResponse = {
            candidates: normalizedCandidates,
            pathway_summary: responseData.pathway_summary ?? [],
            warnings: normalizedWarnings,
            normalized_disease: responseData.normalized_disease ?? null,
            cached: Boolean(responseData.cached),
            related_concepts: relatedConcepts,
            graph_overview: responseData.graph_overview ?? null,
            scoring: responseData.scoring ?? null,
            counterfactuals
          };
          setResults(structured);
          setBackendCached(Boolean(structured.cached));
          setBffCached(Boolean(body.cached));
          setError(null);
          console.info("persona_fetch", {
            persona: requestPayload.persona ?? "balanced",
            overrides: requestPayload.weights ? Object.keys(requestPayload.weights) : [],
            exclude_contraindicated: requestPayload.exclude_contraindicated ?? false,
            cached: Boolean(body.cached)
          });
          void trackPersonaUsage({
            persona: requestPayload.persona ?? "balanced",
            weights:
              requestPayload.persona === "custom"
                ? requestPayload.weights ?? currentWeights ?? {}
                : {},
            diseaseQuery: query,
            normalizedDisease: structured.normalized_disease,
            backendCached: structured.cached,
            bffCached: Boolean(body.cached),
            excludeContraindicated: excludeContraindicated,
          }).catch(() => undefined);
        } catch (err) {
          if (signal.signal.aborted) {
            return;
          }
          setResults(null);
          setBackendCached(false);
          setBffCached(false);
          setError((err as Error).message ?? "Unexpected error");
        } finally {
          if (!signal.signal.aborted) {
            setLoading(false);
          }
        }
      })();
    },
    [currentWeights, personaForRequest, query, excludeContraindicated]
  );

  useEffect(() => {
    if (!metadata || !currentWeights || !query) {
      return;
    }
    if (requestSignature && requestSignature === lastSignatureRef.current) {
      return;
    }
    lastSignatureRef.current = requestSignature;
    if (fetchTimeoutRef.current) {
      window.clearTimeout(fetchTimeoutRef.current);
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    fetchTimeoutRef.current = window.setTimeout(() => {
      scheduleFetch(controller);
    }, 250);
    return () => {
      if (fetchTimeoutRef.current) {
        window.clearTimeout(fetchTimeoutRef.current);
      }
      controller.abort();
    };
  }, [metadata, currentWeights, query, requestSignature, scheduleFetch]);

  const handlePersonaSelect = useCallback(
    (nextPersona: string) => {
      if (!metadata) {
        return;
      }
      const persona = personaLookup[nextPersona];
      if (!persona) {
        setSelectedPersona("custom");
        return;
      }
      setSelectedPersona(persona.name);
      setWeights(normalizeWeights(persona.weights));
    },
    [metadata, personaLookup]
  );

  const handleAdjustWeight = useCallback(
    (key: string, percent: number) => {
      setWeights((prev) => {
        const base = prev ?? metadata?.defaultWeights ?? {};
        const rawValue = Math.max(percent / 100, 0);
        const updated = { ...base, [key]: rawValue };
        const normalized = normalizeWeights(updated);
        return normalized;
      });
      setSelectedPersona("custom");
    },
    [metadata?.defaultWeights]
  );

  const scoringSummary = results?.scoring ?? null;
  const personaDisplayName =
    (scoringSummary && personaLookup[scoringSummary.persona]?.label) ??
    (scoringSummary && scoringSummary.persona === "custom"
      ? "Custom blend"
      : personaLookup[selectedPersona]?.label ?? "Balanced");

  const scoringPanel = metadata && currentWeights ? (
    <ScoringPersonasPanel
      personas={personas}
      defaultPersona={metadata.defaultPersona}
      selectedPersona={selectedPersona}
      weights={currentWeights}
      defaultWeights={metadata.defaultWeights}
      onSelectPersona={handlePersonaSelect}
      onAdjustWeight={handleAdjustWeight}
      scoringSummary={scoringSummary}
      busy={loading}
    />
  ) : null;

  const handleToggleContraindicated = useCallback((value: boolean) => {
    setExcludeContraindicated(value);
  }, []);

  return (
    <ResultsList
      query={query}
      loading={loading}
      candidates={results?.candidates ?? []}
      warnings={augmentWarnings(results?.warnings ?? [], error ?? metadataError)}
      normalizedDisease={results?.normalized_disease ?? null}
      backendCached={backendCached}
      bffCached={bffCached}
      relatedConcepts={results?.related_concepts ?? []}
      graphOverview={results?.graph_overview as any}
      scoringSummary={scoringSummary}
      personaLabel={personaDisplayName}
      scoringControls={scoringPanel}
      counterfactuals={results?.counterfactuals ?? []}
      pathwaySummary={results?.pathway_summary ?? []}
      excludeContraindicated={excludeContraindicated}
      onToggleContraindicated={handleToggleContraindicated}
    />
  );
}

function normalizeWeights(weights: Record<string, number>): Record<string, number> {
  const sanitized = WEIGHT_KEYS.map((key) => [key, Math.max(weights[key] ?? 0, 0)] as const);
  const total = sanitized.reduce((sum, [, value]) => sum + value, 0);
  if (total <= 0) {
    const share = 1 / WEIGHT_KEYS.length;
    return Object.fromEntries(WEIGHT_KEYS.map((key) => [key, share]));
  }
  return Object.fromEntries(
    sanitized.map(([key, value]) => [key, Number((value / total).toFixed(6))])
  );
}

function normalisePersonas(
  personas: PersonaDefinition[],
  defaults: Record<string, number>
): PersonaDefinition[] {
  const map = new Map<string, PersonaDefinition>();
  personas.forEach((persona) => {
    map.set(persona.name, {
      ...persona,
      weights: normalizeWeights(persona.weights ?? defaults)
    });
  });
  if (!map.has("balanced")) {
    map.set("balanced", {
      name: "balanced",
      label: "Balanced",
      description: "Default weighting across evidence channels.",
      weights: normalizeWeights(defaults)
    });
  }
  return Array.from(map.values());
}

function augmentWarnings(
  warnings: { source: string; detail: string }[],
  message: string | null
) {
  if (!message) {
    return warnings;
  }
  return [
    ...warnings,
    {
      source: "ui",
      detail: message
    }
  ];
}













