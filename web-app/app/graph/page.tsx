'use client';

import { Suspense } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import the graph explorer to avoid SSR issues with Cytoscape
const Neo4jGraphExplorer = dynamic(
  () => import('@/components/Neo4jGraphExplorer'),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-indigo-500 mx-auto mb-4"></div>
          <p className="text-white text-lg font-semibold">Initializing Graph Explorer...</p>
        </div>
      </div>
    ),
  }
);

export default function GraphPage() {
  return (
    <div className="h-screen flex flex-col bg-gray-900">
      {/* Page Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-1">
              Drug Repurposing Knowledge Graph
            </h1>
            <p className="text-gray-400">
              Interactive visualization of drug-disease relationships from Neo4j
            </p>
          </div>
          <a
            href="/"
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-all font-medium"
          >
            Back to Search
          </a>
        </div>
      </header>

      {/* Graph Explorer */}
      <main className="flex-1 overflow-hidden">
        <Suspense
          fallback={
            <div className="flex items-center justify-center h-full bg-gray-900">
              <div className="text-center">
                <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-indigo-500 mx-auto mb-4"></div>
                <p className="text-white text-lg font-semibold">Loading Graph...</p>
              </div>
            </div>
          }
        >
          <Neo4jGraphExplorer
            diseaseId="EFO_0001360"
            diseaseName="type 2 diabetes mellitus"
            autoLoad={true}
          />
        </Suspense>
      </main>

      {/* Footer Info */}
      <footer className="bg-gray-800 border-t border-gray-700 px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between text-sm text-gray-400">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Real-time Neo4j Connection</span>
            </div>
            <div>
              <span className="text-gray-500">Controls:</span> Click nodes to inspect | Drag to pan | Scroll to zoom
            </div>
          </div>
          <div>
            Powered by Neo4j Aura + Cytoscape.js
          </div>
        </div>
      </footer>
    </div>
  );
}
