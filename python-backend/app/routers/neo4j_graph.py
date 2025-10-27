"""API endpoints for Neo4j graph visualization and exploration."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.logging_conf import get_logger
from app.services.neo4j_service import get_neo4j_service

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/neo4j", tags=["neo4j-graph"])


# ============================================================================
# Request/Response Models
# ============================================================================


class DrugNode(BaseModel):
    """Drug node properties."""

    drug_id: str = Field(..., description="Drug identifier (CHEMBL ID)")
    name: Optional[str] = None
    properties: dict = Field(default_factory=dict)


class DiseaseNode(BaseModel):
    """Disease node properties."""

    disease_id: str = Field(..., description="Disease identifier")
    name: Optional[str] = None
    properties: dict = Field(default_factory=dict)


class TreatsRelationship(BaseModel):
    """TREATS relationship."""

    drug_id: str
    disease_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    evidence: dict = Field(default_factory=dict)


class PathQuery(BaseModel):
    """Path finding query."""

    start_id: str = Field(..., description="Start node ID")
    end_id: str = Field(..., description="End node ID")
    max_hops: int = Field(default=4, ge=1, le=10)


class PopulateRequest(BaseModel):
    """Request to populate graph from predictions."""

    predictions: List[dict] = Field(..., description="List of prediction results")


# ============================================================================
# Graph Creation Endpoints
# ============================================================================


@router.post("/drug/create")
async def create_drug_node(drug: DrugNode):
    """Create a drug node in Neo4j."""
    try:
        neo4j = get_neo4j_service()

        properties = drug.properties.copy()
        if drug.name:
            properties['name'] = drug.name

        success = neo4j.create_drug_node(drug.drug_id, properties)

        if success:
            return {
                "success": True,
                "drug_id": drug.drug_id,
                "message": "Drug node created/updated"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create drug node")

    except Exception as e:
        logger.error(f"Failed to create drug node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disease/create")
async def create_disease_node(disease: DiseaseNode):
    """Create a disease node in Neo4j."""
    try:
        neo4j = get_neo4j_service()

        properties = disease.properties.copy()
        if disease.name:
            properties['name'] = disease.name

        success = neo4j.create_disease_node(disease.disease_id, properties)

        if success:
            return {
                "success": True,
                "disease_id": disease.disease_id,
                "message": "Disease node created/updated"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create disease node")

    except Exception as e:
        logger.error(f"Failed to create disease node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relationship/treats")
async def create_treats_relationship(rel: TreatsRelationship):
    """Create a TREATS relationship between drug and disease."""
    try:
        neo4j = get_neo4j_service()

        success = neo4j.create_treats_relationship(
            rel.drug_id,
            rel.disease_id,
            rel.score,
            rel.evidence
        )

        if success:
            return {
                "success": True,
                "drug_id": rel.drug_id,
                "disease_id": rel.disease_id,
                "score": rel.score,
                "message": "TREATS relationship created"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create relationship")

    except Exception as e:
        logger.error(f"Failed to create relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Graph Query Endpoints
# ============================================================================


@router.post("/paths/find")
async def find_paths(query: PathQuery):
    """Find paths between two nodes in the graph."""
    try:
        neo4j = get_neo4j_service()

        paths = neo4j.find_paths(query.start_id, query.end_id, query.max_hops)

        return {
            "start_id": query.start_id,
            "end_id": query.end_id,
            "max_hops": query.max_hops,
            "num_paths": len(paths),
            "paths": paths
        }

    except Exception as e:
        logger.error(f"Failed to find paths: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drug/{drug_id}/connections")
async def get_drug_connections(drug_id: str, limit: int = 20):
    """Get all connections for a drug."""
    try:
        neo4j = get_neo4j_service()

        connections = neo4j.get_drug_connections(drug_id, limit)

        return {
            "drug_id": drug_id,
            "connections": connections,
            "total": sum(len(v) for v in connections.values())
        }

    except Exception as e:
        logger.error(f"Failed to get drug connections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_graph_stats():
    """Get Neo4j graph statistics."""
    try:
        neo4j = get_neo4j_service()
        stats = neo4j.get_graph_stats()

        return {
            "neo4j_stats": stats,
            "connected": neo4j.driver is not None
        }

    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Bulk Operations
# ============================================================================


@router.post("/populate")
async def populate_from_predictions(request: PopulateRequest):
    """Populate Neo4j graph from prediction results."""
    try:
        neo4j = get_neo4j_service()

        result = neo4j.populate_from_predictions(request.predictions)

        return {
            "success": True,
            "populated": result,
            "message": f"Created {result['drugs']} drugs, {result['diseases']} diseases, "
                      f"{result['relationships']} relationships"
        }

    except Exception as e:
        logger.error(f"Failed to populate graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_all_data(confirm: str = ""):
    """Clear all data from Neo4j (dangerous operation!)."""
    if confirm != "yes_delete_everything":
        raise HTTPException(
            status_code=400,
            detail="Must provide confirm='yes_delete_everything' to clear data"
        )

    try:
        neo4j = get_neo4j_service()
        success = neo4j.clear_all_data()

        if success:
            return {
                "success": True,
                "message": "All data cleared from Neo4j"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear data")

    except Exception as e:
        logger.error(f"Failed to clear data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================


@router.get("/health")
async def neo4j_health():
    """Check Neo4j connection health."""
    try:
        neo4j = get_neo4j_service()

        if neo4j.driver is None:
            return {
                "status": "disconnected",
                "connected": False,
                "message": "Neo4j driver not initialized"
            }

        # Try to verify connectivity
        neo4j.driver.verify_connectivity()

        return {
            "status": "connected",
            "connected": True,
            "uri": neo4j.uri,
            "database": neo4j.database,
            "message": "Neo4j connection healthy"
        }

    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        return {
            "status": "error",
            "connected": False,
            "error": str(e)
        }


@router.get("/metadata")
async def get_neo4j_metadata():
    """Get Neo4j configuration and capabilities."""
    neo4j = get_neo4j_service()

    return {
        "uri": neo4j.uri if neo4j.uri else "Not configured",
        "database": neo4j.database,
        "connected": neo4j.driver is not None,
        "capabilities": [
            "drug_node_creation",
            "disease_node_creation",
            "treats_relationships",
            "path_finding",
            "connection_exploration",
            "graph_statistics",
            "bulk_population"
        ],
        "endpoints": {
            "health": "/v1/neo4j/health",
            "stats": "/v1/neo4j/stats",
            "create_drug": "/v1/neo4j/drug/create",
            "create_disease": "/v1/neo4j/disease/create",
            "create_relationship": "/v1/neo4j/relationship/treats",
            "find_paths": "/v1/neo4j/paths/find",
            "drug_connections": "/v1/neo4j/drug/{drug_id}/connections",
            "populate": "/v1/neo4j/populate",
        }
    }
