"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";
import Card from "@/components/ui/Card";
import Chip from "@/components/ui/Chip";

type PersonaDefinition = {
  name: string;
  label: string;
  description?: string | null;
  weights: Record<string, number>;
};

type ScoringSummary = {
  persona: string;
  weights: Record<string, number>;
  delta_vs_default: Record<string, number>;
  overrides: Record<string, number>;
};

type Props = {
  personas: PersonaDefinition[];
  defaultPersona: string;
  selectedPersona: string;
  weights: Record<string, number>;
  defaultWeights: Record<string, number>;
  onSelectPersona: (name: string) => void;
  onAdjustWeight: (key: string, percent: number) => void;
  busy?: boolean;
  scoringSummary?: ScoringSummary | null;
};

const WEIGHT_LABELS: Record<string, string> = {
  mechanism: "Mechanism",
  network: "Network",
  signature: "Signature",
  clinical: "Clinical",
  safety: "Safety"
};

export default function ScoringPersonasPanel({
  personas,
  defaultPersona,
  selectedPersona,
  weights,
  defaultWeights,
  onSelectPersona,
  onAdjustWeight,
  busy = false,
  scoringSummary
}: Props) {
  const [showAdvanced, setShowAdvanced] = useState(true);

  const activePersona = useMemo(() => {
    if (selectedPersona === "custom") {
      return personas.find((persona) => persona.name === scoringSummary?.persona) ?? null;
    }
    return personas.find((persona) => persona.name === selectedPersona) ?? null;
  }, [personas, scoringSummary?.persona, selectedPersona]);

  const activeLabel = selectedPersona === "custom"
    ? "Custom blend"
    : activePersona?.label ?? "Balanced";

  const deltas = useMemo(() => {
    const summaryDelta = scoringSummary?.delta_vs_default;
    if (summaryDelta) {
      return summaryDelta;
    }
    const entries = Object.entries(weights).map(([key, value]) => {
      const baseline = defaultWeights[key] ?? 0;
      return [key, Number((value - baseline).toFixed(4))];
    });
    return Object.fromEntries(entries);
  }, [defaultWeights, scoringSummary?.delta_vs_default, weights]);

  const overrides = useMemo(() => {
    if (scoringSummary?.overrides && Object.keys(scoringSummary.overrides).length > 0) {
      return Object.keys(scoringSummary.overrides);
    }
    const diffKeys = Object.entries(deltas)
      .filter(([, delta]) => Math.abs(delta) >= 0.005)
      .map(([key]) => key);
    if (selectedPersona === "custom") {
      return diffKeys;
    }
    return [];
  }, [deltas, scoringSummary?.overrides, selectedPersona]);

  const weightEntries = useMemo(
    () => Object.keys(weights).map((key) => [key, weights[key] ?? 0]),
    [weights]
  );

  return (
    <Card
      title="Scoring personas"
      description="Blend evidence weights via quick presets or fine-grained tuning. Scores update instantly."
      action={
        <button
          type="button"
          className="text-xs font-semibold text-primary-200 underline-offset-4 transition hover:underline"
          onClick={() => setShowAdvanced((prev) => !prev)}
        >
          {showAdvanced ? "Hide advanced sliders" : "Advanced sliders"}
        </button>
      }
    >
      <div className="flex flex-col gap-4">
        <div className="surface rounded-xl border border-slate-700/60 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Active persona
          </p>
          <p className="mt-1 text-sm font-semibold text-slate-100">{activeLabel}</p>
          <p className="text-xs text-slate-400">
            {selectedPersona === "custom"
              ? "Weights adjusted from default to craft a custom scoring blend."
              : activePersona?.description ?? "Default weighting across evidence channels."}
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            {overrides.length > 0 ? (
              overrides.map((key) => (
                <Chip
                  key={key}
                  tone="info"
                  className="persona-chip persona-chip--active"
                >
                  {WEIGHT_LABELS[key] ?? key}: Δ{formatDelta(deltas[key] ?? 0)}
                </Chip>
              ))
            ) : (
              <Chip tone="success" className="persona-chip persona-chip--quiet">
                Aligned with default weights
              </Chip>
            )}
          </div>
        </div>

        <div className="grid gap-2 sm:grid-cols-2">
          {personas.map((persona) => {
            const isActive = selectedPersona === persona.name;
            return (
              <button
                key={persona.name}
                type="button"
                disabled={busy}
                onClick={() => onSelectPersona(persona.name)}
                className={clsx(
                  "rounded-xl border px-4 py-3 text-left text-sm transition shadow-sm",
                  isActive
                    ? "border-primary-500/70 bg-primary-500/15 text-primary-100 shadow-[0_14px_40px_-30px_rgba(37,99,235,0.8)]"
                    : "border-slate-800/70 bg-slate-900/50 text-slate-300 hover:border-slate-700/80 hover:bg-slate-900/70",
                  busy && "opacity-60"
                )}
              >
                <span className="font-semibold">{persona.label}</span>
                {persona.description && (
                  <span className="mt-1 block text-xs text-slate-400">{persona.description}</span>
                )}
              </button>
            );
          })}
        </div>

        <div className="flex items-center justify-end">
          <button
            type="button"
            onClick={() => onSelectPersona(defaultPersona)}
            disabled={busy}
            className="text-xs font-semibold text-primary-200 underline-offset-4 transition hover:underline disabled:text-slate-500"
          >
            Reset to default
          </button>
        </div>

        {showAdvanced && (
          <div className="space-y-4 rounded-xl border border-slate-800/40 bg-slate-900/70 p-4 shadow-inner">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Weight distribution
            </p>
            {weightEntries.map(([key, value]) => (
              <div key={key} className="space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>{WEIGHT_LABELS[key] ?? key}</span>
                  <span className="text-slate-200">{Math.round(value * 100)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={Math.round(value * 100)}
                  onChange={(event) => onAdjustWeight(key, Number(event.target.value))}
                  className="persona-slider"
                />
                <div className="flex justify-between text-[11px] text-slate-500">
                  <span>Δ default: {formatDelta(deltas[key] ?? 0)}</span>
                  {defaultWeights[key] !== undefined && (
                    <span>Baseline {Math.round((defaultWeights[key] ?? 0) * 100)}%</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Card>
  );
}

function formatDelta(delta: number): string {
  const percent = Math.round(delta * 100);
  if (percent === 0) {
    return "0%";
  }
  return `${percent > 0 ? "+" : ""}${percent}%`;
}
