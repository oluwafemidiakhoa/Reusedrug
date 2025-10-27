"""Knowledge graph API endpoints for embeddings, reasoning, and temporal analysis."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.knowledge_graph.node2vec_embeddings import (
    Node2VecConfig,
    NodeEmbedding,
    get_node2vec_embeddings,
)
from app.ml.knowledge_graph.complex_embeddings import (
    ComplExConfig,
    get_complex_embeddings,
)
from app.ml.knowledge_graph.temporal_kg import (
    TemporalFact,
    TemporalQuery,
    get_temporal_kg,
)
from app.ml.knowledge_graph.multihop_reasoner import (
    ReasoningPath,
    PathPattern,
    get_multihop_reasoner,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/knowledge-graph", tags=["knowledge-graph"])


# Request/Response Models

class BuildGraphRequest(BaseModel):
    """Request to build knowledge graph from edges."""

    edges: List[Tuple[str, str, str]] = Field(
        ..., description="List of (source, relation, target) tuples"
    )
    node_types: Optional[Dict[str, str]] = Field(
        None, description="Optional node type mappings"
    )


class TrainNode2VecRequest(BaseModel):
    """Request to train Node2Vec embeddings."""

    config: Optional[Node2VecConfig] = None


class TrainComplExRequest(BaseModel):
    """Request to train ComplEx embeddings."""

    triples: List[Tuple[str, str, str]]
    config: Optional[ComplExConfig] = None


class SimilarityRequest(BaseModel):
    """Request for similarity search."""

    query_node: str
    top_k: int = Field(default=10, ge=1, le=100)
    node_type_filter: Optional[str] = None
    method: str = Field(default="cosine", description="Similarity method")


class PathRequest(BaseModel):
    """Request for path finding."""

    source: str
    target: str
    max_length: int = Field(default=4, ge=1, le=6)
    max_paths: int = Field(default=5, ge=1, le=20)


class PredictTailRequest(BaseModel):
    """Request for ComplEx tail prediction."""

    head: str
    relation: str
    top_k: int = Field(default=10, ge=1, le=50)


class TemporalFactRequest(BaseModel):
    """Request to add temporal fact."""

    head: str
    relation: str
    tail: str
    timestamp: datetime
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="api")


class TemporalQueryRequest(BaseModel):
    """Request for temporal query."""

    head: Optional[str] = None
    relation: Optional[str] = None
    tail: Optional[str] = None
    time_point: Optional[datetime] = None


# Node2Vec Endpoints

@router.post("/node2vec/build")
async def build_graph_for_node2vec(request: BuildGraphRequest):
    """Build knowledge graph from edges for Node2Vec."""
    try:
        embedder = get_node2vec_embeddings()
        graph = embedder.build_graph_from_edges(request.edges, request.node_types)

        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "status": "Graph built successfully",
        }
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/node2vec/train")
async def train_node2vec(request: TrainNode2VecRequest):
    """Train Node2Vec embeddings on the graph."""
    try:
        embedder = get_node2vec_embeddings(request.config)

        if embedder.graph is None:
            raise HTTPException(
                status_code=400,
                detail="No graph built. Call /node2vec/build first",
            )

        embeddings = embedder.train_embeddings()

        return {
            "num_embeddings": len(embeddings),
            "embedding_dim": embedder.config.dimensions,
            "status": "Training complete",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to train Node2Vec: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/node2vec/embedding/{node_id}", response_model=NodeEmbedding)
async def get_node2vec_embedding(node_id: str):
    """Get Node2Vec embedding for a node."""
    embedder = get_node2vec_embeddings()
    embedding_obj = embedder.get_node_embedding_obj(node_id)

    if embedding_obj is None:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{node_id}' not found or no embedding available",
        )

    return embedding_obj


@router.post("/node2vec/similar")
async def find_similar_nodes(request: SimilarityRequest):
    """Find similar nodes using Node2Vec embeddings."""
    embedder = get_node2vec_embeddings()

    try:
        similar = embedder.find_similar_nodes(
            request.query_node,
            top_k=request.top_k,
            node_type_filter=request.node_type_filter,
        )

        return {
            "query_node": request.query_node,
            "results": [
                {"node_id": node_id, "similarity": float(score)}
                for node_id, score in similar
            ],
        }
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ComplEx Endpoints

@router.post("/complex/train")
async def train_complex(request: TrainComplExRequest):
    """Train ComplEx embeddings for link prediction."""
    try:
        complex_model = get_complex_embeddings(request.config)
        history = complex_model.train(request.triples)

        return {
            "num_entities": len(complex_model.entity2idx),
            "num_relations": len(complex_model.relation2idx),
            "training_epochs": len(history["train_loss"]),
            "final_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "status": "Training complete",
        }
    except Exception as e:
        logger.error(f"ComplEx training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complex/predict")
async def predict_tail_complex(request: PredictTailRequest):
    """Predict tail entities using ComplEx."""
    complex_model = get_complex_embeddings()

    if complex_model.model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Call /complex/train first",
        )

    try:
        predictions = complex_model.predict_tail(
            request.head,
            request.relation,
            top_k=request.top_k,
        )

        return {
            "head": request.head,
            "relation": request.relation,
            "predictions": [
                {"tail": tail, "score": float(score)}
                for tail, score in predictions
            ],
        }
    except Exception as e:
        logger.error(f"ComplEx prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Temporal KG Endpoints

@router.post("/temporal/fact")
async def add_temporal_fact(request: TemporalFactRequest):
    """Add a temporal fact to the knowledge graph."""
    try:
        temporal_kg = get_temporal_kg()

        fact = TemporalFact(
            head=request.head,
            relation=request.relation,
            tail=request.tail,
            timestamp=request.timestamp,
            valid_from=request.valid_from,
            valid_until=request.valid_until,
            confidence=request.confidence,
            source=request.source,
        )

        temporal_kg.add_fact(fact)

        return {
            "status": "Fact added",
            "fact": fact.dict(),
        }
    except Exception as e:
        logger.error(f"Failed to add temporal fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/temporal/query")
async def query_temporal_kg(request: TemporalQueryRequest):
    """Query the temporal knowledge graph."""
    try:
        temporal_kg = get_temporal_kg()

        query = TemporalQuery(
            head=request.head,
            relation=request.relation,
            tail=request.tail,
            time_point=request.time_point,
        )

        results = temporal_kg.query(query)

        return {
            "num_results": len(results),
            "facts": [fact.dict() for fact in results],
        }
    except Exception as e:
        logger.error(f"Temporal query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/temporal/entity/{entity_id}/history")
async def get_entity_history(
    entity_id: str,
    relation_filter: Optional[str] = None,
):
    """Get chronological history of an entity."""
    temporal_kg = get_temporal_kg()

    try:
        history = temporal_kg.get_entity_history(entity_id, relation_filter)

        return {
            "entity_id": entity_id,
            "num_events": len(history),
            "timeline": temporal_kg.visualize_timeline(entity_id),
        }
    except Exception as e:
        logger.error(f"Failed to get entity history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/temporal/stats")
async def get_temporal_stats():
    """Get temporal knowledge graph statistics."""
    temporal_kg = get_temporal_kg()
    return temporal_kg.get_temporal_statistics()


# Multi-Hop Reasoning Endpoints

@router.post("/reasoning/paths", response_model=List[ReasoningPath])
async def find_reasoning_paths(request: PathRequest):
    """Find reasoning paths between two entities."""
    reasoner = get_multihop_reasoner()

    if reasoner.graph.number_of_nodes() == 0:
        raise HTTPException(
            status_code=400,
            detail="Graph not initialized. Build graph first.",
        )

    try:
        paths = reasoner.find_all_paths(
            request.source,
            request.target,
            max_length=request.max_length,
            cutoff=request.max_paths,
        )

        return paths
    except Exception as e:
        logger.error(f"Path finding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ExplainRequest(BaseModel):
    """Request for explanation generation."""

    drug_id: str
    disease_id: str
    max_paths: int = Field(default=3, ge=1, le=10)


@router.post("/reasoning/explain")
async def explain_drug_disease_prediction(request: ExplainRequest):
    """Explain a drug-disease prediction using reasoning paths."""
    reasoner = get_multihop_reasoner()

    try:
        explanation = reasoner.explain_prediction(
            request.drug_id, request.disease_id, request.max_paths
        )
        return explanation
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reasoning/patterns", response_model=List[PathPattern])
async def discover_path_patterns(
    min_frequency: int = 2,
    max_length: int = 3,
):
    """Discover common path patterns in the knowledge graph."""
    reasoner = get_multihop_reasoner()

    try:
        patterns = reasoner.discover_path_patterns(min_frequency, max_length)
        return patterns
    except Exception as e:
        logger.error(f"Pattern discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reasoning/centrality/{node_id}")
async def get_node_centrality(
    node_id: str,
    method: str = "betweenness",
):
    """Get centrality score for a node."""
    reasoner = get_multihop_reasoner()

    try:
        all_centralities = reasoner.compute_node_centrality(method)

        if node_id not in all_centralities:
            raise HTTPException(
                status_code=404,
                detail=f"Node '{node_id}' not found in graph",
            )

        return {
            "node_id": node_id,
            "centrality_method": method,
            "score": all_centralities[node_id],
            "rank": sorted(all_centralities.values(), reverse=True).index(all_centralities[node_id]) + 1,
            "total_nodes": len(all_centralities),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Centrality computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata")
async def get_kg_metadata():
    """Get knowledge graph system metadata."""
    embedder = get_node2vec_embeddings()
    complex_model = get_complex_embeddings()
    temporal_kg = get_temporal_kg()
    reasoner = get_multihop_reasoner()

    return {
        "node2vec": {
            "num_embeddings": len(embedder.embeddings),
            "embedding_dim": embedder.config.dimensions,
            "graph_nodes": embedder.graph.number_of_nodes() if embedder.graph else 0,
            "graph_edges": embedder.graph.number_of_edges() if embedder.graph else 0,
        },
        "complex": {
            "num_entities": len(complex_model.entity2idx),
            "num_relations": len(complex_model.relation2idx),
            "embedding_dim": complex_model.config.embedding_dim,
            "model_trained": complex_model.model is not None,
        },
        "temporal": temporal_kg.get_temporal_statistics(),
        "reasoning": {
            "graph_nodes": reasoner.graph.number_of_nodes(),
            "graph_edges": reasoner.graph.number_of_edges(),
            "cached_paths": len(reasoner.path_cache),
        },
        "capabilities": [
            "node2vec_embeddings",
            "complex_link_prediction",
            "temporal_reasoning",
            "multi_hop_paths",
            "pattern_discovery",
            "centrality_analysis",
        ],
    }
