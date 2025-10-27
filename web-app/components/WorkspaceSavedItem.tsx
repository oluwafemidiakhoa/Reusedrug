"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";
import clsx from "clsx";

type Props = {
  record: {
    id: number;
    disease: string;
    created_at: number;
    topScore: number | null;
    normalizedDisease: string | null;
    note?: string | null;
  };
};

export default function WorkspaceSavedItem({ record }: Props) {
  const router = useRouter();
  const [note, setNote] = useState(record.note ?? "");
  const [isEditing, setIsEditing] = useState(false);
  const [isPending, startTransition] = useTransition();

  const handleSave = () => {
    startTransition(async () => {
      await fetch(`/api/workspace/queries/${record.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ note: note.trim() || null }),
        cache: "no-store"
      });
      setIsEditing(false);
      router.refresh();
    });
  };

  const handleDelete = () => {
    if (!confirm("Delete this saved query?")) {
      return;
    }
    startTransition(async () => {
      await fetch(`/api/workspace/queries/${record.id}`, {
        method: "DELETE",
        cache: "no-store"
      });
      router.refresh();
    });
  };

  return (
    <li className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div className="flex-1 space-y-1">
          <h3 className="text-lg font-semibold text-slate-900">{record.disease}</h3>
          <p className="text-xs text-slate-500">
            Saved {new Date(record.created_at * 1000).toLocaleString(undefined, { hour12: false })}
          </p>
          {record.normalizedDisease && (
            <p className="text-xs text-slate-500">Normalized to {record.normalizedDisease}</p>
          )}
        </div>
        <div className="flex items-center gap-4">
          <p className="text-sm text-slate-600">Top score</p>
          <span className="rounded-md bg-blue-50 px-3 py-1 text-lg font-semibold text-blue-700">
            {record.topScore !== null ? record.topScore.toFixed(2) : "-"}
          </span>
        </div>
      </div>

      <div className="mt-3 space-y-2">
        <textarea
          value={note}
          onChange={(event) => setNote(event.target.value)}
          disabled={!isEditing || isPending}
          rows={3}
          maxLength={500}
          className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500/40 disabled:cursor-not-allowed disabled:opacity-60"
          placeholder="Add context or follow-up notes for this result"
        />
        <div className="flex items-center justify-end gap-2">
          <button
            onClick={() => setIsEditing((prev) => !prev)}
            className="rounded-md border border-slate-300 px-3 py-1 text-xs font-semibold text-slate-600 transition hover:border-slate-400 hover:text-slate-800"
            disabled={isPending}
          >
            {isEditing ? "Cancel" : "Edit note"}
          </button>
          <button
            onClick={handleSave}
            disabled={!isEditing || isPending}
            className={clsx(
              "rounded-md px-3 py-1 text-xs font-semibold text-white transition",
              isPending
                ? "cursor-not-allowed bg-primary-300"
                : "bg-primary-600 hover:bg-primary-700"
            )}
          >
            {isPending ? "Saving..." : "Save"}
          </button>
          <button
            onClick={handleDelete}
            disabled={isPending}
            className="rounded-md border border-rose-200 px-3 py-1 text-xs font-semibold text-rose-600 transition hover:border-rose-400 hover:text-rose-700"
          >
            Delete
          </button>
        </div>
      </div>
    </li>
  );
}
