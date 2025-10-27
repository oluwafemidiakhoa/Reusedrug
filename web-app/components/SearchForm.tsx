"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

export default function SearchForm() {
  const [query, setQuery] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const router = useRouter();

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }
    setIsSubmitting(true);
    const target = `/results?q=${encodeURIComponent(query.trim())}`;
    router.push(target);
    setTimeout(() => {
      setIsSubmitting(false);
    }, 300);
  };

  return (
    <form
      onSubmit={onSubmit}
      className="flex w-full flex-col gap-3 rounded-lg border border-slate-200 bg-white p-4 shadow-sm"
    >
      <label className="text-sm font-medium text-slate-600" htmlFor="disease-input">
        Disease or phenotype
      </label>
      <input
        id="disease-input"
        name="disease"
        type="text"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="e.g. idiopathic pulmonary fibrosis"
        className="w-full rounded-md border border-slate-300 px-3 py-2 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
        minLength={3}
        maxLength={200}
        required
      />
      <button
        type="submit"
        disabled={isSubmitting}
        className="inline-flex items-center justify-center rounded-md bg-blue-700 px-4 py-2 text-white transition hover:bg-blue-800 disabled:cursor-not-allowed disabled:bg-blue-400"
      >
        {isSubmitting ? "Submitting..." : "Rank repurposing candidates"}
      </button>
    </form>
  );
}

