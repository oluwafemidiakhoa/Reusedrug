"use client";

import clsx from "clsx";
import { ReactNode } from "react";

type Props = {
  children: ReactNode;
  tone?: "default" | "info" | "success" | "warning";
  icon?: ReactNode;
  className?: string;
};

const TONE_STYLES: Record<Required<Props>["tone"], string> = {
  default: "bg-slate-800/80 border-slate-700 text-slate-200",
  info: "bg-primary-500/10 border-primary-500/30 text-primary-200",
  success: "bg-success-500/10 border-success-500/30 text-success-400",
  warning: "bg-warning-500/10 border-warning-500/30 text-warning-300",
};

export default function Chip({ children, tone = "default", icon, className }: Props) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs font-medium",
        TONE_STYLES[tone],
        className
      )}
    >
      {icon}
      {children}
    </span>
  );
}
