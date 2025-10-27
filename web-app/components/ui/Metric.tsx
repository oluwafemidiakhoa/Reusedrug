"use client";

import clsx from "clsx";

type Props = {
  label: string;
  value: string | number;
  trend?: "up" | "down" | "steady";
  hint?: string;
  className?: string;
};

const TREND_COPY: Record<Required<Props>["trend"], string> = {
  up: "▲",
  down: "▼",
  steady: "■",
};

export default function Metric({ label, value, trend = "steady", hint, className }: Props) {
  return (
    <div
      className={clsx(
        "surface flex flex-col gap-2 rounded-2xl border border-slate-800/60 bg-slate-900/60 p-4",
        className
      )}
    >
      <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">{label}</p>
      <div className="flex items-baseline gap-2">
        <span className="font-display text-3xl font-bold text-slate-50">{value}</span>
        {hint && (
          <span
            className={clsx("text-xs font-semibold", {
              "text-success-500": trend === "up",
              "text-danger-500": trend === "down",
              "text-slate-400": trend === "steady",
            })}
          >
            {TREND_COPY[trend]} {hint}
          </span>
        )}
      </div>
    </div>
  );
}
