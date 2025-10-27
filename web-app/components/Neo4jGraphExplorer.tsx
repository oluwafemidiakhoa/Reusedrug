'use client';

import React, { useEffect, useRef, useState } from 'react';
import cytoscape, { Core, ElementDefinition, NodeSingular } from 'cytoscape';
import cola from 'cytoscape-cola';

// Register the cola layout
if (typeof window !== 'undefined') {
  cytoscape.use(cola);
}

interface GraphNode {
  id: string;
  label: string;
  type: 'drug' | 'disease' | 'target' | 'pathway';
  properties?: Record<string, any>;
}

interface GraphEdge {
  source: string;
  target: string;
  type: string;
  score?: number;
  confidence?: number;
  evidence?: string[];
}

interface GraphStats {
  neo4j_stats: {
    node_counts: Record<string, number>;
    total_nodes: number;
    relationship_counts: Record<string, number>;
    total_relationships: number;
  };
  connected: boolean;
}

interface Neo4jGraphExplorerProps {
  diseaseId?: string;
  diseaseName?: string;
  autoLoad?: boolean;
}

export default function Neo4jGraphExplorer({
  diseaseId,
  diseaseName,
  autoLoad = true
}: Neo4jGraphExplorerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [layoutType, setLayoutType] = useState<'cola' | 'circle' | 'grid' | 'concentric'>('cola');

  // Fetch graph stats
  const fetchStats = async () => {
    try {
      const response = await fetch('/api/neo4j?endpoint=stats');
      if (!response.ok) throw new Error('Failed to fetch graph stats');
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Stats fetch error:', err);
    }
  };

  // Fetch graph data from Neo4j
  const fetchGraphData = async (): Promise<{ nodes: GraphNode[]; edges: GraphEdge[] }> => {
    setLoading(true);
    setError(null);

    try {
      // First get stats to understand what's in the graph
      const statsResponse = await fetch('/api/neo4j?endpoint=stats');
      if (!statsResponse.ok) throw new Error('Failed to fetch graph stats');
      const statsData = await statsResponse.json();
      setStats(statsData);

      // Fetch all drug connections (simplified approach for MVP)
      // In production, you'd query Neo4j directly or use a dedicated endpoint
      const nodes: GraphNode[] = [];
      const edges: GraphEdge[] = [];
      const seenNodes = new Set<string>();

      // Add disease node if provided
      if (diseaseId && diseaseName) {
        nodes.push({
          id: diseaseId,
          label: diseaseName,
          type: 'disease',
        });
        seenNodes.add(diseaseId);
      }

      // For MVP, we'll create a mock graph structure based on stats
      // In production, you'd fetch actual Neo4j data
      const drugCount = statsData.neo4j_stats?.node_counts?.Drug || 0;
      const diseaseCount = statsData.neo4j_stats?.node_counts?.Disease || 0;

      // Mock some sample drugs and relationships
      const sampleDrugs = [
        { id: 'CHEMBL1073', name: 'TELMISARTAN', score: 0.87 },
        { id: 'CHEMBL1280', name: 'VALSARTAN', score: 0.84 },
        { id: 'CHEMBL1431', name: 'RAMIPRIL', score: 0.81 },
        { id: 'CHEMBL1201497', name: 'INSULIN GLARGINE', score: 0.95 },
        { id: 'CHEMBL1014', name: 'METFORMIN', score: 0.96 },
      ];

      sampleDrugs.forEach((drug) => {
        nodes.push({
          id: drug.id,
          label: drug.name,
          type: 'drug',
          properties: { score: drug.score },
        });
        seenNodes.add(drug.id);

        // Add edge to disease
        if (diseaseId) {
          edges.push({
            source: drug.id,
            target: diseaseId,
            type: 'TREATS',
            score: drug.score,
            confidence: drug.score * 0.9,
          });
        }
      });

      return { nodes, edges };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load graph data';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Initialize Cytoscape
  const initializeCytoscape = async () => {
    if (!containerRef.current) return;

    try {
      const { nodes, edges } = await fetchGraphData();

      // Convert to Cytoscape format
      const elements: ElementDefinition[] = [
        ...nodes.map((node) => ({
          data: {
            id: node.id,
            label: node.label,
            type: node.type,
            ...node.properties,
          },
        })),
        ...edges.map((edge, idx) => ({
          data: {
            id: `edge-${idx}`,
            source: edge.source,
            target: edge.target,
            type: edge.type,
            score: edge.score,
            confidence: edge.confidence,
          },
        })),
      ];

      // Initialize Cytoscape
      const cy = cytoscape({
        container: containerRef.current,
        elements,
        style: [
          {
            selector: 'node',
            style: {
              'background-color': '#4F46E5',
              'label': 'data(label)',
              'color': '#fff',
              'text-valign': 'center',
              'text-halign': 'center',
              'font-size': '12px',
              'font-weight': 'bold',
              'width': '60px',
              'height': '60px',
              'border-width': 3,
              'border-color': '#312E81',
              'text-outline-color': '#000',
              'text-outline-width': 2,
            },
          },
          {
            selector: 'node[type="drug"]',
            style: {
              'background-color': '#10B981',
              'border-color': '#065F46',
              'shape': 'roundrectangle',
            },
          },
          {
            selector: 'node[type="disease"]',
            style: {
              'background-color': '#EF4444',
              'border-color': '#991B1B',
              'shape': 'ellipse',
              'width': '80px',
              'height': '80px',
              'font-size': '14px',
            },
          },
          {
            selector: 'node[type="target"]',
            style: {
              'background-color': '#F59E0B',
              'border-color': '#92400E',
              'shape': 'diamond',
            },
          },
          {
            selector: 'node[type="pathway"]',
            style: {
              'background-color': '#8B5CF6',
              'border-color': '#5B21B6',
              'shape': 'hexagon',
            },
          },
          {
            selector: 'node:selected',
            style: {
              'border-width': 5,
              'border-color': '#FBBF24',
              'background-color': '#FBBF24',
            },
          },
          {
            selector: 'edge',
            style: {
              'width': 3,
              'line-color': '#94A3B8',
              'target-arrow-color': '#94A3B8',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'arrow-scale': 1.5,
            },
          },
          {
            selector: 'edge[type="TREATS"]',
            style: {
              'line-color': (ele) => {
                const score = ele.data('score') || 0;
                if (score > 0.9) return '#10B981';
                if (score > 0.8) return '#3B82F6';
                return '#94A3B8';
              },
              'width': (ele) => {
                const score = ele.data('score') || 0;
                return 2 + score * 6;
              },
            },
          },
          {
            selector: 'edge:selected',
            style: {
              'line-color': '#FBBF24',
              'target-arrow-color': '#FBBF24',
              'width': 5,
            },
          },
        ],
        layout: {
          name: 'cola',
          animate: true,
          randomize: false,
          maxSimulationTime: 2000,
          fit: true,
          padding: 50,
          nodeSpacing: 100,
        },
        wheelSensitivity: 0.2,
        minZoom: 0.3,
        maxZoom: 3,
      });

      // Add event listeners
      cy.on('tap', 'node', (evt) => {
        const node: NodeSingular = evt.target;
        setSelectedNode({
          id: node.id(),
          label: node.data('label'),
          type: node.data('type'),
          score: node.data('score'),
          ...node.data(),
        });
      });

      cy.on('tap', (evt) => {
        if (evt.target === cy) {
          setSelectedNode(null);
        }
      });

      cyRef.current = cy;
    } catch (err) {
      console.error('Cytoscape initialization error:', err);
    }
  };

  // Apply layout
  const applyLayout = (layout: typeof layoutType) => {
    if (!cyRef.current) return;

    setLayoutType(layout);

    const layoutOptions: any = {
      name: layout,
      animate: true,
      animationDuration: 500,
      fit: true,
      padding: 50,
    };

    if (layout === 'cola') {
      layoutOptions.randomize = false;
      layoutOptions.maxSimulationTime = 2000;
      layoutOptions.nodeSpacing = 100;
    }

    cyRef.current.layout(layoutOptions).run();
  };

  // Export graph as image
  const exportImage = () => {
    if (!cyRef.current) return;

    const png = cyRef.current.png({
      output: 'blob',
      bg: '#1F2937',
      full: true,
      scale: 2,
    });

    const url = URL.createObjectURL(png as Blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `graph-${Date.now()}.png`;
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    if (autoLoad) {
      fetchStats();
      initializeCytoscape();
    }

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
      }
    };
  }, [diseaseId]);

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden shadow-2xl">
      {/* Header Controls */}
      <div className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <svg className="w-8 h-8 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Knowledge Graph Explorer
          </h2>

          {stats && (
            <div className="flex gap-4 text-sm">
              <div className="bg-indigo-900 px-3 py-1 rounded-full">
                <span className="text-indigo-200">Nodes: </span>
                <span className="text-white font-bold">{stats.neo4j_stats.total_nodes}</span>
              </div>
              <div className="bg-green-900 px-3 py-1 rounded-full">
                <span className="text-green-200">Edges: </span>
                <span className="text-white font-bold">{stats.neo4j_stats.total_relationships}</span>
              </div>
              <div className={`px-3 py-1 rounded-full ${stats.connected ? 'bg-green-900' : 'bg-red-900'}`}>
                <span className={stats.connected ? 'text-green-200' : 'text-red-200'}>
                  {stats.connected ? '● Connected' : '● Disconnected'}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Layout Controls */}
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex gap-2">
            <button
              onClick={() => applyLayout('cola')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                layoutType === 'cola'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Force Layout
            </button>
            <button
              onClick={() => applyLayout('circle')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                layoutType === 'circle'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Circle
            </button>
            <button
              onClick={() => applyLayout('grid')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                layoutType === 'grid'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Grid
            </button>
            <button
              onClick={() => applyLayout('concentric')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                layoutType === 'concentric'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Concentric
            </button>
          </div>

          <div className="flex gap-2 ml-auto">
            <button
              onClick={() => cyRef.current?.fit(undefined, 50)}
              className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-all font-medium"
            >
              Fit View
            </button>
            <button
              onClick={exportImage}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-all font-medium"
            >
              Export PNG
            </button>
            <button
              onClick={initializeCytoscape}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all font-medium disabled:opacity-50"
            >
              {loading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </div>
      </div>

      {/* Graph Container and Side Panel */}
      <div className="flex-1 flex overflow-hidden">
        {/* Main Graph */}
        <div className="flex-1 relative">
          {error && (
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 bg-red-900 text-white px-6 py-3 rounded-lg shadow-lg">
              <p className="font-semibold">Error loading graph</p>
              <p className="text-sm text-red-200">{error}</p>
            </div>
          )}

          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75 z-10">
              <div className="text-center">
                <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-indigo-500 mx-auto mb-4"></div>
                <p className="text-white text-lg font-semibold">Loading Knowledge Graph...</p>
              </div>
            </div>
          )}

          <div ref={containerRef} className="w-full h-full bg-gray-900" />

          {/* Legend */}
          <div className="absolute bottom-4 left-4 bg-gray-800 bg-opacity-95 p-4 rounded-lg shadow-xl border border-gray-700">
            <h3 className="text-white font-semibold mb-2">Legend</h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-500 rounded-full border-2 border-red-900"></div>
                <span className="text-gray-300">Disease</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500 rounded border-2 border-green-900"></div>
                <span className="text-gray-300">Drug</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-orange-500 rounded border-2 border-orange-900 transform rotate-45"></div>
                <span className="text-gray-300">Target</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-purple-500 border-2 border-purple-900" style={{ clipPath: 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)' }}></div>
                <span className="text-gray-300">Pathway</span>
              </div>
            </div>
          </div>
        </div>

        {/* Side Panel - Node Details */}
        {selectedNode && (
          <div className="w-80 bg-gray-800 border-l border-gray-700 p-6 overflow-y-auto">
            <div className="mb-4">
              <h3 className="text-xl font-bold text-white mb-2">Node Details</h3>
              <div className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
                selectedNode.type === 'drug' ? 'bg-green-900 text-green-200' :
                selectedNode.type === 'disease' ? 'bg-red-900 text-red-200' :
                selectedNode.type === 'target' ? 'bg-orange-900 text-orange-200' :
                'bg-purple-900 text-purple-200'
              }`}>
                {selectedNode.type.toUpperCase()}
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <p className="text-gray-400 text-sm">Name</p>
                <p className="text-white font-semibold text-lg">{selectedNode.label}</p>
              </div>

              <div>
                <p className="text-gray-400 text-sm">ID</p>
                <p className="text-gray-300 font-mono text-sm">{selectedNode.id}</p>
              </div>

              {selectedNode.score !== undefined && (
                <div>
                  <p className="text-gray-400 text-sm mb-1">Score</p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all"
                        style={{ width: `${selectedNode.score * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-white font-bold">{(selectedNode.score * 100).toFixed(1)}%</span>
                  </div>
                </div>
              )}

              {selectedNode.confidence !== undefined && (
                <div>
                  <p className="text-gray-400 text-sm">Confidence</p>
                  <p className="text-white font-semibold">{(selectedNode.confidence * 100).toFixed(1)}%</p>
                </div>
              )}

              {/* Additional properties */}
              <div>
                <p className="text-gray-400 text-sm mb-2">Properties</p>
                <div className="bg-gray-900 rounded-lg p-3 space-y-1">
                  {Object.entries(selectedNode)
                    .filter(([key]) => !['id', 'label', 'type', 'score', 'confidence'].includes(key))
                    .map(([key, value]) => (
                      <div key={key} className="flex justify-between text-xs">
                        <span className="text-gray-400">{key}:</span>
                        <span className="text-gray-200">{String(value)}</span>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
