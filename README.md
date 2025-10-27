# Drug Repurposing MVP Monorepo

This monorepo packages a FastAPI science engine and a Next.js/Tailwind web experience to explore drug repurposing hypotheses from public data.

## Why FastAPI for the science engine?
- **Async-first orchestration**: Normalizing diseases and fanning out to multiple evidence providers benefits from HTTP concurrency and backpressure primitives that FastAPI/anyio expose cleanly.
- **Pydantic v2 typing**: Rich typing keeps I/O and scoring logic honest while giving fast JSON serialization for downstream consumers.
- **Observability and resilience**: Structured logging plus tenacity-powered retries provide a stable backbone when querying brittle public APIs.
- **Python ecosystem fit**: Translational science teams often prototype scoring formulas and statistics in Python notebooks; the engine mirrors that workflow with minimal impedance.

## When a Node/TypeScript-only stack could be better
- If your signal processing is already authored in TypeScript or you rely solely on JavaScript-compatible SDKs.
- When your deployment team standardizes on a single runtime (Node) for serverless footprints and wants tight integration with V8-based tooling.
- For lightweight prototypes that only need deterministic transformations or simple aggregations without heavy numerical dependencies.

## Repository layout
- `python-backend/`: FastAPI app with service clients, scoring logic, and pytest coverage.
- `web-app/`: Next.js App Router UI with a BFF route that validates, caches, and rate-limits calls to the Python engine.
- `docker-compose.yml`: One command to run both services locally.

## Getting started

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker (optional, for container workflow)

### Local setup
1. **Backend**
   ```bash
   cd python-backend
   python -m venv .venv && source .venv/bin/activate  # use `.\.venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   uvicorn app.main:app --reload --port 8080
   ```
2. **Frontend**
   ```bash
   cd web-app
   npm install
   npm run dev
   ```
3. Visit `http://localhost:3000` and submit a disease query.

### Tests
- Backend: `cd python-backend && pytest`
- Frontend: `cd web-app && npm test`

### Observability
- The backend emits structured logs with `trace_id`/`span_id` correlation by default.  
- Enable OpenTelemetry export by setting `ENABLE_OTEL=true` and pointing `OTEL_EXPORTER_OTLP_ENDPOINT` (or `..._TRACES_ENDPOINT`) at your collector. Traces cover FastAPI routes and outbound httpx calls automatically.
- Persona usage analytics can be fanned out to an external sink by setting `PERSONA_ANALYTICS_ENDPOINT` (and optional `PERSONA_ANALYTICS_API_KEY`) in `python-backend`; the frontend will also mirror events when `PERSONA_ANALYTICS_ENDPOINT` or `NEXT_PUBLIC_PERSONA_ANALYTICS_ENDPOINT` is present. Events include persona name, override keys, cache status, and disease context so downstream dashboards can compare how presets perform.

### Data Providers & Enrichment (opt-in)
- Toggle integrations via environment variables in `python-backend/.env.example`: `TRANSLATOR_ENABLED`, `DRUGCENTRAL_ENABLED`, `LINCS_ENABLED`, `CLUE_ENABLED`, `SIDER_ENABLED`, `PUBMED_ENABLED`, `UMLS_ENABLED`, and related endpoints/paths.  
- SIDER/OFFSIDES datasets must be downloaded separately; place TSVs under `data/sider/` (see `SIDER_DATA_PATH`) before enabling.
- PubMed enrichment is enabled by default. Provide `PUBMED_API_KEY` (optional) plus `PUBMED_RESULT_LIMIT`/`PUBMED_CACHE_SECONDS` to tune rate limits and caching.
- UMLS concept mapping (disabled by default) surfaces CUIs, semantic types, and synonyms for each query. Configure `UMLS_SEARCH_ENDPOINT`, `UMLS_API_KEY`, and cache/page settings to taste.
- Evidence scoring weights are adjustable via `SCORE_WEIGHT_*` env vars, and responses include confidence bounds plus Translator graph analytics (nodes, edges, density, centrality) for deeper interpretation.

### Result caching & prefetch
- Ranked responses now persist in MongoDB (`rank_cache` collection) and respect `RESULT_CACHE_TTL_SECONDS`. Provide `MONGODB_URI` (for example, your Atlas connection string) and optional `MONGODB_DB` to point the backend at your cluster.
- A lightweight background worker re-runs heavy evidence gathering asynchronously to refresh the cache; adjust queue behavior by toggling feature flags for external providers.
- Each candidate now includes an auto-generated mechanistic narrative that stitches together assay, clinical, safety, and transcriptomic evidence with source-linked citations. Use it to brief stakeholders quickly or as grounding context when exporting results downstream.
- Counterfactual analysis lets you down-weight safety (or other factors) on the fly to see which drugs climb the leaderboard - rank deltas and score shifts are summarized next to the primary results.
- Confidence tiers (exploratory / hypothesis-ready / decision-grade) summarize the breadth, provenance, and recency of supporting evidence, with signal callouts surfaced directly in the UI.
- DrugBank/CTD annotations and pathway overlays enrich each candidate with documented indications, contraindications, and Reactome/KEGG-signature callouts.

### Authenticated workspace
- Enable WORKSPACE_API_KEY on the backend plus matching AUTH_USERNAME/AUTH_PASSWORD/WORKSPACE_API_KEY and NEXTAUTH_SECRET on the frontend to turn on login-protected workspaces.
- Signed-in users can save ranked results via the UI; the backend stores them in MongoDB (`saved_queries` collection) and exposes them at /workspace.
- Analysts can attach notes to each saved run, captured alongside the serialized response for collaborative review.

### CI/CD
- GitHub Actions `CI` workflow runs lint-free backend/ frontend test suites and builds container images on every push or pull request.  
- `Build and Deploy Staging` workflow (manual or push to `main`) publishes production-ready images to GHCR; extend the final placeholder step with your staging deploy commands (kubectl/helm/compose).

### Docker Compose
```bash
docker-compose up --build
```
Frontend: `http://localhost:3000`, Backend: `http://localhost:8080`.

## Licensing & disclaimer
- All third-party data queried via public APIs retains its original licensing terms. Review each provider’s ToS before production use.
- This repository is provided “as is” for research prototyping. **It is not medical advice.** Validate findings clinically before action.






