# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A drug repurposing monorepo with a **FastAPI science engine** ([python-backend/](python-backend/)) and **Next.js web UI** ([web-app/](web-app/)). The backend orchestrates async evidence gathering from multiple public APIs (Translator, PubMed, DrugCentral, LINCS, SIDER, UMLS) and scores drug candidates using configurable weights and personas. The frontend provides a BFF layer for rate limiting, validation, and caching.

## Development Commands

### Backend (Python 3.11+)
```bash
cd python-backend
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Unix/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Run backend (port 8080)
uvicorn app.main:app --reload --port 8080

# Run tests
pytest

# Run specific test
pytest tests/test_scoring.py
pytest tests/test_scoring.py::test_function_name
```

### Frontend (Node.js 20+)
```bash
cd web-app
npm install

# Run dev server (port 3000)
npm run dev

# Build production
npm run build

# Run tests
npm test

# Watch mode
npm run test:watch
```

### Docker Compose
```bash
docker-compose up --build
# Frontend: http://localhost:3000
# Backend: http://localhost:8080
```

## Architecture

### Backend Flow ([python-backend/app/](python-backend/app/))

1. **Entry point**: [main.py](python-backend/app/main.py:18) - FastAPI app with CORS, telemetry, and startup hooks
2. **Routers**:
   - [routers/repurpose.py](python-backend/app/routers/repurpose.py) - `/v1/rank` endpoint and `/v1/metadata/scoring`
   - [routers/workspace.py](python-backend/app/routers/workspace.py) - `/workspace/*` endpoints for saved queries
3. **Core ranking logic**: [services/ranking.py](python-backend/app/services/ranking.py) - Orchestrates disease normalization, evidence gathering, scoring, and caching
4. **Scoring system**: [services/scoring.py](python-backend/app/services/scoring.py) - Defines personas (balanced, mechanism-first, clinical-first) and weight resolution
5. **Evidence services**: [services/](python-backend/app/services/)
   - [services/clients/](python-backend/app/services/clients/) - External API clients (translator, drugcentral, lincs, sider)
   - Other services: chembl, clinicaltrials, faers, opentargets, pubmed, umls
6. **Graph analysis**: [services/graph.py](python-backend/app/services/graph.py) - NetworkX-based path analysis for Translator knowledge graphs
7. **Narratives**: [services/narratives.py](python-backend/app/services/narratives.py) - Auto-generated mechanistic narratives with citations
8. **Database**: [db.py](python-backend/app/db.py) - MongoDB operations for rank caching (`rank_cache`) and saved queries (`saved_queries`)
9. **Background worker**: [background.py](python-backend/app/background.py) - Async prefetcher to warm cache
10. **Observability**: [telemetry.py](python-backend/app/telemetry.py), [logging_conf.py](python-backend/app/logging_conf.py) - OpenTelemetry tracing and structured logging

### Frontend Flow ([web-app/](web-app/))

1. **Pages**: [app/page.tsx](web-app/app/page.tsx) - Search form, [app/results/](web-app/app/results/) - Results display
2. **API BFF**: [app/api/repurpose/route.ts](web-app/app/api/repurpose/route.ts) - Rate limiting, validation (Zod), LRU caching, backend proxy
3. **Components**: [components/](web-app/components/)
   - [SearchForm.tsx](web-app/components/SearchForm.tsx) - Disease query input
   - [ResultsList.tsx](web-app/components/ResultsList.tsx) - Drug candidate cards with expandable evidence
   - [ScoringPersonasPanel.tsx](web-app/components/ScoringPersonasPanel.tsx) - Persona selector and weight overrides
   - [GraphExplorer.tsx](web-app/components/GraphExplorer.tsx) - Cytoscape visualization for Translator graphs
4. **Auth**: [app/api/auth/](web-app/app/api/auth/) - NextAuth with credentials provider
5. **Workspace**: [app/workspace/](web-app/app/workspace/) - Saved queries UI

### Data Models ([python-backend/app/models.py](python-backend/app/models.py))

- **RankRequest**: `{disease, persona?, weights?, exclude_contraindicated?}`
- **RankResponse**: Contains `candidates[]`, `scoring`, `normalized_disease`, `warnings`, `graph_insight`, `counterfactuals`, `cached`
- **DrugCandidate**: Includes `score_breakdown`, `mechanistic_narrative`, `evidence_links`, `confidence`, `pathway_insights`, `annotations`
- **ScoringWeights**: Five dimensions: `mechanism`, `network`, `signature`, `clinical`, `safety`

### Environment Variables

**Backend** ([python-backend/.env.example](python-backend/.env.example)):
- Toggle providers: `TRANSLATOR_ENABLED`, `DRUGCENTRAL_ENABLED`, `LINCS_ENABLED`, `SIDER_ENABLED`, `PUBMED_ENABLED`, `UMLS_ENABLED`
- Scoring weights: `SCORE_WEIGHT_MECHANISM`, `SCORE_WEIGHT_NETWORK`, `SCORE_WEIGHT_SIGNATURE`, `SCORE_WEIGHT_CLINICAL`, `SCORE_WEIGHT_SAFETY`
- Cache/DB: `MONGODB_URI`, `MONGODB_DB`, `RESULT_CACHE_TTL_SECONDS`
- Observability: `ENABLE_OTEL`, `OTEL_EXPORTER_OTLP_ENDPOINT`
- Auth: `WORKSPACE_API_KEY`, `WORKSPACE_USER_ID`

**Frontend** ([web-app/.env.example](web-app/.env.example)):
- `NEXT_PUBLIC_API_BASE` - Backend URL
- Rate limiting: `RATE_LIMIT_TOKENS`, `RATE_LIMIT_WINDOW_MS`
- Auth: `NEXTAUTH_SECRET`, `AUTH_USERNAME`, `AUTH_PASSWORD`, `WORKSPACE_API_KEY`

## Key Patterns

### Adding a New Evidence Source

1. Create client in [python-backend/app/services/clients/](python-backend/app/services/clients/) or standalone service in [python-backend/app/services/](python-backend/app/services/)
2. Add feature flag to [python-backend/.env.example](python-backend/.env.example)
3. Register in [services/data_sources.py](python-backend/app/services/data_sources.py)
4. Call from [services/ranking.py](python-backend/app/services/ranking.py) `compute_rank()` or related functions
5. Update scoring logic in [services/scoring.py](python-backend/app/services/scoring.py) if introducing new dimension
6. Add tests to [python-backend/tests/](python-backend/tests/)

### Modifying Scoring Weights

- Edit defaults in [services/scoring.py](python-backend/app/services/scoring.py) `_BASE_DEFAULT_WEIGHTS`
- Add/modify personas in `_PERSONA_TEMPLATES` and `_PERSONA_METADATA`
- Frontend reads from `/v1/metadata/scoring` endpoint to populate UI

### Caching Strategy

- **Frontend BFF**: LRU cache (5 min TTL) in [app/api/repurpose/route.ts](web-app/app/api/repurpose/route.ts)
- **Backend MongoDB**: Persistent rank cache in [db.py](python-backend/app/db.py), controlled by `RESULT_CACHE_TTL_SECONDS`
- **Background prefetch**: [background.py](python-backend/app/background.py) enqueues heavy evidence gathering for balanced persona

### Graph Analysis

- Translator responses parsed in [services/clients/translator.py](python-backend/app/services/clients/translator.py)
- NetworkX graph constructed and analyzed in [services/graph.py](python-backend/app/services/graph.py)
- Metrics: node/edge count, density, betweenness centrality, average path length

## Machine Learning Features (Phase 1 - NEW)

### ML Module Structure ([python-backend/app/ml/](python-backend/app/ml/))

1. **Embeddings** ([ml/embeddings/embedding_service.py](python-backend/app/ml/embeddings/embedding_service.py))
   - Sentence transformer embeddings for drugs and diseases
   - Semantic similarity search
   - Embedding cache for performance

2. **Models** ([ml/models/](python-backend/app/ml/models/))
   - [base.py](python-backend/app/ml/models/base.py) - Base predictor interface
   - [gnn_predictor.py](python-backend/app/ml/models/gnn_predictor.py) - Graph Neural Network for link prediction
   - GNN architecture: Graph Convolution layers + Link Prediction head

3. **ML API** ([routers/ml.py](python-backend/app/routers/ml.py))
   - `POST /v1/ml/similar/drugs` - Find similar drugs via embeddings
   - `POST /v1/ml/similar/diseases` - Find similar diseases via embeddings
   - `POST /v1/ml/predict` - ML prediction (placeholder, requires training)
   - `POST /v1/ml/predict/batch` - Batch predictions
   - `GET /v1/ml/metadata` - Available ML features

4. **Configuration** ([ml/config.py](python-backend/app/ml/config.py))
   - GNN hyperparameters (embedding dim, hidden dim, layers, dropout)
   - Training settings (batch size, learning rate, epochs)
   - Model/embedding storage paths

### Using ML Features

**Similarity Search** (Ready to Use):
```python
# Find similar drugs
POST /v1/ml/similar/drugs
{
  "entity_id": "CHEMBL123",
  "entity_name": "Aspirin",
  "candidates": [["CHEMBL456", "Ibuprofen"], ...],
  "top_k": 10
}
```

**GNN Predictions** (Requires Training):
- Models need training on knowledge graph data
- See [ML_GUIDE.md](python-backend/ML_GUIDE.md) for training instructions
- After training, load model and use `/v1/ml/predict` endpoint

### ML Dependencies

Added to [requirements.txt](python-backend/requirements.txt):
- `torch` - PyTorch for deep learning
- `torch-geometric` - Graph Neural Networks
- `scikit-learn` - Traditional ML algorithms
- `transformers` - Pre-trained language models
- `sentence-transformers` - Semantic embeddings
- `mlflow` - Experiment tracking

### Environment Variables

ML-specific variables in [.env.example](python-backend/.env.example):
- `ML_MODEL_DIR`, `ML_EMBEDDINGS_DIR` - Storage paths
- `MLFLOW_ENABLED`, `MLFLOW_TRACKING_URI` - Experiment tracking
- `GNN_EMBEDDING_DIM`, `GNN_HIDDEN_DIM`, `GNN_NUM_LAYERS` - Model architecture
- `ML_USE_GNN`, `ML_USE_EMBEDDINGS` - Feature toggles

## Testing Notes

- Backend uses pytest with `respx` for HTTP mocking and `mongomock` for database tests
- Frontend uses Vitest with React Testing Library
- Test files mirror source structure: [python-backend/tests/](python-backend/tests/), [web-app/tests/](web-app/tests/)
- ML tests: Test embeddings, similarity search, model save/load
