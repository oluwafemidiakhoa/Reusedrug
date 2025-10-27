"use client";

import clsx from "clsx";
import { ReactNode } from "react";

type Props = {
  title?: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  className?: string;
  action?: ReactNode;
};

export default function Card({ title, description, children, className, action }: Props) {
  return (
    <section className={clsx("glass-panel p-6 transition hover:shadow-card hover:scale-102", className)}>
      {(title || description || action) && (
        <header className="mb-5 flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
          <div>
            {title && <h3 className="text-lg font-semibold text-slate-100">{title}</h3>}
            {description && <p className="text-sm text-slate-400">{description}</p>}
          </div>
          {action}
        </header>
      )}
      <div className="space-y-4 md:space-y-6">{children}</div>
    </section>
  );
}
