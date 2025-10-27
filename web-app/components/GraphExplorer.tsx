"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import Card from "@/components/ui/Card";
import Chip from "@/components/ui/Chip";
import type { GraphInsight } from "@/components/EvidenceGraphPreview";

const CytoscapeComponent = dynamic(() => import("react-cytoscapejs"), { ssr: false });

type Props = {
  insight: GraphInsight;
};

const layout = {
  name: "cose",
  padding: 20,
};

const baseStylesheet = [
  {
    selector: "node",
    style: {
      label: "data(label)",
      "background-color": "#3b82f6",
      color: "#e2e8f0",
      "font-size": "9px",
      "text-wrap": "wrap",
      "text-max-width": 80,
      "border-width": 1,
      "border-color": "#1d4ed8",
    },
  },
  {
    selector: "edge",
    style: {
      width: 1.5,
      "line-color": "#64748b",
      "target-arrow-color": "#64748b",
      "target-arrow-shape": "triangle",
      label: "data(predicate)",
      "font-size": "8px",
      "curve-style": "bezier",
      "text-background-color": "rgba(15,23,42,0.85)",
      "text-background-opacity": 1,
      "text-background-padding": 2,
      color: "#cbd5f5",
    },
  },
  {
    selector: "node.highlight",
    style: {
      "background-color": "#10b981",
      "border-color": "#34d399",
      "border-width": 2,
    },
  },
];

export default function GraphExplorer({ insight }: Props) {
  const [selectedPredicate, setSelectedPredicate] = useState<string>("__all");
  const [highlightCentral, setHighlightCentral] = useState<boolean>(false);

  const predicates = useMemo(() => {
    const set = new Set<string>();
    (insight.graph?.edges ?? []).forEach((edge) => {
      if (edge.predicate) {
        set.add(edge.predicate);
      }
    });
    return Array.from(set).sort();
  }, [insight]);

  const elements = useMemo(() => {
    const allowedPredicates = new Set<string>(
      selectedPredicate === "__all" ? predicates : [selectedPredicate]
    );

    const highlighted = new Set<string>(highlightCentral ? insight.top_central_nodes : []);

    const visibleEdges = (insight.graph?.edges ?? []).filter((edge) => {
      if (!edge.source || !edge.target) return false;
      if (selectedPredicate === "__all") return true;
      return edge.predicate ? allowedPredicates.has(edge.predicate) : false;
    });

    const allowedNodes = new Set<string>();
    visibleEdges.forEach((edge) => {
      allowedNodes.add(edge.source);
      allowedNodes.add(edge.target);
    });

    const nodes = (insight.graph?.nodes ?? [])
      .filter((node) => allowedNodes.size === 0 || allowedNodes.has(node.id))
      .map((node) => ({
        data: {
          id: node.id,
          label: node.label || node.id,
        },
        classes: highlighted.has(node.id) ? "highlight" : undefined,
      }));

    const edges = visibleEdges.map((edge) => ({
      data: {
        id: edge.id ?? [edge.source, edge.target].join("-"),
        source: edge.source,
        target: edge.target,
        predicate: edge.predicate,
      },
    }));

    return [...nodes, ...edges];
  }, [insight, predicates, selectedPredicate, highlightCentral]);

  return (
    <Card
      title="Knowledge graph explorer"
      description="Inspect the Translator evidence graph, filter by relationship, and spotlight central nodes."
      className="p-0"
    >
      <div className="flex flex-col gap-4 p-4">
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-xs uppercase tracking-wide text-slate-400">
            Predicate filter
            <select
              value={selectedPredicate}
              onChange={(event) => setSelectedPredicate(event.target.value)}
              className="ml-2 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 focus:border-primary-400 focus:outline-none"
            >
              <option value="__all">All predicates</option>
              {predicates.map((predicate) => (
                <option key={predicate} value={predicate}>
                  {predicate}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={highlightCentral}
              onChange={(event) => setHighlightCentral(event.target.checked)}
              className="h-3 w-3 rounded border-slate-600 bg-slate-900"
            />
            Highlight central nodes
          </label>
          <Chip tone="default">{insight.node_count} nodes</Chip>
          <Chip tone="default">{insight.edge_count} edges</Chip>
        </div>
        <div className="h-80 w-full overflow-hidden rounded-2xl border border-slate-800">
          <CytoscapeComponent
            elements={elements as any}
            layout={layout as any}
            stylesheet={baseStylesheet as any}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
      </div>
    </Card>
  );
}
