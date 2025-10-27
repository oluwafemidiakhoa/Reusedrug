import { getServerSession } from "next-auth";
import Link from "next/link";
import { redirect } from "next/navigation";
import WorkspaceSavedItem from "@/components/WorkspaceSavedItem";
import { authOptions } from "../api/auth/[...nextauth]/route";

function backendBase(): string {
  return (
    process.env.API_BASE ??
    process.env.NEXT_PUBLIC_API_BASE ??
    "http://localhost:8080"
  );
}

async function fetchSavedQueries(apiKey: string) {
  const response = await fetch(`${backendBase()}/v1/workspaces/queries`, {
    headers: {
      "x-api-key": apiKey
    },
    cache: "no-store"
  });

  if (!response.ok) {
    return [];
  }

  const payload = await response.json();
  return (payload ?? []) as Array<{
    id: number;
    disease: string;
    created_at: number;
    response: {
      normalized_disease: string | null;
      candidates: Array<{ name: string; score: { final_score: number } }>;
    };
    note?: string | null;
  }>;
}

export default async function WorkspacePage() {
  const session = await getServerSession(authOptions);
  if (!session || !session.apiKey) {
    redirect("/api/auth/signin");
  }

  const savedQueries = await fetchSavedQueries(session.apiKey);

  return (
    <section className="flex flex-col gap-4">
      <div className="rounded-lg bg-white p-6 shadow-sm">
        <h2 className="text-2xl font-semibold text-slate-900">Workspace</h2>
        <p className="mt-2 text-sm text-slate-600">
          Saved query runs are stored here. Capture promising hypotheses and revisit their evidence over time.
        </p>
      </div>
      {savedQueries.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500">
          No saved queries yet. Run a search and use the "Save" button to keep it here.
        </div>
      ) : (
        <ul className="space-y-3">
          {savedQueries.map((record) => (
            <WorkspaceSavedItem
              key={record.id}
              record={{
                id: record.id,
                disease: record.disease,
                created_at: record.created_at,
                normalizedDisease: record.response.normalized_disease,
                topScore: record.response.candidates?.[0]?.score.final_score ?? null,
                note: record.note ?? null
              }}
            />
          ))}
        </ul>
      )}
      <Link
        href="/results"
        className="inline-flex w-fit items-center justify-center rounded-md bg-blue-700 px-4 py-2 text-sm font-medium text-white hover:bg-blue-800"
      >
        Run new query
      </Link>
    </section>
  );
}
