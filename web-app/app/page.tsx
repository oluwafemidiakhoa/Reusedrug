import SearchForm from "@/components/SearchForm";
import Link from "next/link";

export default function HomePage() {
  return (
    <section className="flex flex-col gap-6">
      {/* Knowledge Graph Banner */}
      <div className="rounded-lg bg-gradient-to-r from-indigo-600 to-purple-600 p-6 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <h3 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              New: Interactive Knowledge Graph Explorer
            </h3>
            <p className="text-indigo-100 text-sm">
              Visualize drug-disease relationships in an interactive Neo4j-powered graph with pathway overlays
            </p>
          </div>
          <Link
            href="/graph"
            className="ml-6 px-6 py-3 bg-white text-indigo-600 rounded-lg font-semibold hover:bg-indigo-50 transition-all shadow-md hover:shadow-lg"
          >
            Explore Graph →
          </Link>
        </div>
      </div>

      <div className="rounded-lg bg-white p-6 shadow-sm">
        <h2 className="text-2xl font-semibold text-slate-900">
          Generate ranked drug repurposing hypotheses
        </h2>
        <p className="mt-2 text-sm text-slate-600">
          This MVP orchestrates knowledge graph, clinical, and safety evidence to surface
          promising candidates for further evaluation. Enter a disease to begin.
        </p>
      </div>
      <SearchForm />
    </section>
  );
}

