"use client";

import clsx from "clsx";
import { ReactNode } from "react";

type Props = {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
};

export default function SectionHeader({ eyebrow, title, description, actions, className }: Props) {
  return (
    <div className={clsx("flex flex-col gap-2 md:flex-row md:items-center md:justify-between", className)}>
      <div className="flex flex-col gap-1">
        {eyebrow && <p className="text-xs uppercase tracking-[0.35em] text-primary-200">{eyebrow}</p>}
        <h2 className="font-display text-2xl font-semibold text-slate-100 md:text-3xl">{title}</h2>
        {description && <p className="max-w-2xl text-sm text-slate-400">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-3">{actions}</div>}
    </div>
  );
}
