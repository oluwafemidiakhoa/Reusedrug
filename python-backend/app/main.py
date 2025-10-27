from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.background import prefetcher
from app.db import init_db
from app.logging_conf import configure_logging, get_logger
from app.models import HealthResponse
from app.routers import repurpose, workspace, ml, multimodal, knowledge_graph, enhanced_ml, explainability, neo4j_graph
from app.telemetry import setup_telemetry

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Drug Repurposing Science Engine",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

setup_telemetry(app)


def _allowed_origins() -> list[str]:
    default_origin = "http://localhost:3000"
    origin_env = os.getenv("WEB_APP_ORIGIN")
    if not origin_env:
        return [default_origin]
    return [origin.strip() for origin in origin_env.split(",")]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(repurpose.router)
app.include_router(workspace.router)
app.include_router(ml.router)
app.include_router(multimodal.router)
app.include_router(knowledge_graph.router)
app.include_router(enhanced_ml.router)
app.include_router(explainability.router)
app.include_router(neo4j_graph.router)


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()
    await prefetcher.start()
    # Load trained ML model if available
    from app.routers.ml import load_trained_model
    await load_trained_model()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await prefetcher.stop()


@app.get("/healthz", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()

