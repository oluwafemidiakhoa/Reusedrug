"use client";

import { memo } from "react";
import Chip from "@/components/ui/Chip";
import Card from "@/components/ui/Card";

export type GraphNode = {
  id: string;
  label?: string | null;
  category?: string | null;
};

export type GraphEdge = {
  id?: string | null;
  source: string;
  target: string;
  predicate?: string | null;
};

export type GraphInsight = {
  node_count: number;
  edge_count: number;
  density: number;
  average_shortest_path?: number | null;
  top_central_nodes: string[];
  summary?: string | null;
  graph?: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  } | null;
};

type Props = {
  insight: GraphInsight;
  variant?: "card" | "inline";
};

function GraphContent({ insight }: { insight: GraphInsight }) {
  return (
    <div className="surface rounded-2xl p-4">
      <div className="flex flex-wrap gap-3">
        <Chip tone="info">{insight.node_count} nodes</Chip>
        <Chip tone="info">{insight.edge_count} edges</Chip>
        <Chip tone="warning">density {insight.density.toFixed(2)}</Chip>
        {typeof insight.average_shortest_path === "number" && (
          <Chip tone="default">avg path {insight.average_shortest_path.toFixed(2)}</Chip>
        )}
      </div>
      <p className="mt-4 text-sm text-slate-300">
        {insight.summary ?? "Graph metrics available"}
      </p>
      {insight.top_central_nodes.length > 0 && (
        <div className="mt-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Central entities</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {insight.top_central_nodes.slice(0, 5).map((node) => (
              <Chip key={node} tone="default">
                {node}
              </Chip>
            ))}
          </div>
        </div>
      )}
      <div className="mt-6 rounded-xl border border-dashed border-slate-700 bg-slate-900/40 p-6 text-center text-xs text-slate-500">
        Interactive knowledge-graph visualisation coming soon — prioritise the entities above when exploring Translator paths.
      </div>
    </div>
  );
}

function EvidenceGraphPreview({ insight, variant = "card" }: Props) {
  if (variant === "inline") {
    return <GraphContent insight={insight} />;
  }

  return (
    <Card
      title="Graph insight"
      description="Knowledge-graph analysis highlights connectivity strength and central intermediaries."
      className="p-0"
    >
      <GraphContent insight={insight} />
    </Card>
  );
}

export default memo(EvidenceGraphPreview);
