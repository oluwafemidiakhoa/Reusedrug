"""
Biomedical Data Integration API Router

Provides endpoints to enrich drug repurposing results with data from:
- UniProt, STRING, DrugBank, PubChem, KEGG, DisGeNET
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
import os

from app.services.biomedical_integrations import get_biomedical_hub

router = APIRouter(prefix="/v1/biomedical", tags=["biomedical"])

# Get API keys from environment
DRUGBANK_API_KEY = os.getenv("DRUGBANK_API_KEY")
DISGENET_API_KEY = os.getenv("DISGENET_API_KEY")


class DrugEnrichmentResponse(BaseModel):
    drug_name: str
    chembl_id: Optional[str] = None
    chemical_properties: dict = {}
    drugbank: dict = {}
    kegg: dict = {}
    bioactivity: List[dict] = []


class DiseaseEnrichmentResponse(BaseModel):
    disease_name: str
    disease_id: Optional[str] = None
    associated_genes: List[dict] = []
    protein_network: List[dict] = []
    pathways: List[dict] = []


class ProteinInteraction(BaseModel):
    partner_protein: str
    score: float
    interaction_type: str


class NetworkResponse(BaseModel):
    nodes: List[dict]
    edges: List[dict]


@router.get("/enrich/drug/{drug_name}", response_model=DrugEnrichmentResponse)
async def enrich_drug(
    drug_name: str,
    chembl_id: Optional[str] = Query(None, description="ChEMBL ID if available")
):
    """
    Enrich drug information from multiple biomedical databases

    Fetches data from:
    - PubChem (chemical properties, bioactivity)
    - DrugBank (mechanism, targets, indications)
    - KEGG (pathways)

    Example: `/v1/biomedical/enrich/drug/metformin`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        data = await hub.enrich_drug_data(drug_name, chembl_id)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enrich drug data: {str(e)}")


@router.get("/enrich/disease/{disease_name}", response_model=DiseaseEnrichmentResponse)
async def enrich_disease(
    disease_name: str,
    disease_id: Optional[str] = Query(None, description="Disease ID (UMLS, MONDO, etc.)")
):
    """
    Enrich disease information from multiple biomedical databases

    Fetches data from:
    - DisGeNET (associated genes)
    - KEGG (pathways)
    - STRING (protein interactions)

    Example: `/v1/biomedical/enrich/disease/diabetes?disease_id=MONDO:0005015`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        data = await hub.enrich_disease_data(disease_name, disease_id)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enrich disease data: {str(e)}")


@router.get("/protein/info/{protein_id}")
async def get_protein_info(protein_id: str):
    """
    Get detailed protein information from UniProt

    Args:
        protein_id: UniProt accession (e.g., P04637 for TP53)

    Example: `/v1/biomedical/protein/info/P04637`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        data = await hub.uniprot.get_protein_info(protein_id)

        if not data:
            raise HTTPException(status_code=404, detail=f"Protein {protein_id} not found")

        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch protein info: {str(e)}")


@router.get("/protein/search/gene/{gene_name}")
async def search_protein_by_gene(
    gene_name: str,
    limit: int = Query(10, ge=1, le=50)
):
    """
    Search proteins by gene name

    Example: `/v1/biomedical/protein/search/gene/TP53`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        proteins = await hub.uniprot.search_proteins_by_gene(gene_name, limit)
        return {"gene_name": gene_name, "proteins": proteins, "count": len(proteins)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search proteins: {str(e)}")


@router.get("/protein/interactions/{protein_name}", response_model=List[ProteinInteraction])
async def get_protein_interactions(
    protein_name: str,
    species: int = Query(9606, description="NCBI taxonomy ID (9606=human)"),
    limit: int = Query(10, ge=1, le=50),
    min_score: int = Query(400, ge=0, le=1000, description="Minimum confidence score (0-1000)")
):
    """
    Get protein-protein interactions from STRING database

    Args:
        protein_name: Gene/protein name (e.g., TP53, EGFR)
        species: NCBI taxonomy ID (default: 9606 for human)
        limit: Maximum number of interactions
        min_score: Minimum confidence score (0-1000)

    Example: `/v1/biomedical/protein/interactions/EGFR?limit=20&min_score=700`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        interactions = await hub.string_db.get_protein_interactions(
            protein_name, species, limit, min_score
        )
        return interactions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch interactions: {str(e)}")


@router.get("/protein/network/image")
async def get_protein_network_image(
    proteins: List[str] = Query(..., description="List of protein/gene names"),
    species: int = Query(9606, description="NCBI taxonomy ID")
):
    """
    Get URL for protein-protein interaction network visualization

    Example: `/v1/biomedical/protein/network/image?proteins=TP53&proteins=MDM2&proteins=CDKN1A`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        image_url = await hub.string_db.get_network_image(proteins, species)
        return {"proteins": proteins, "image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate network image: {str(e)}")


@router.get("/compound/pubchem/{compound_name}")
async def get_compound_info(compound_name: str):
    """
    Get chemical compound information from PubChem

    Returns molecular formula, weight, SMILES, InChI, etc.

    Example: `/v1/biomedical/compound/pubchem/aspirin`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        data = await hub.pubchem.get_compound_by_name(compound_name)

        if not data:
            raise HTTPException(status_code=404, detail=f"Compound {compound_name} not found")

        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch compound info: {str(e)}")


@router.get("/compound/bioactivity/{cid}")
async def get_compound_bioactivity(cid: str):
    """
    Get bioactivity assay results for a PubChem compound

    Args:
        cid: PubChem Compound ID

    Example: `/v1/biomedical/compound/bioactivity/2244`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        activities = await hub.pubchem.get_bioactivity(cid)
        return {"cid": cid, "bioactivity_count": len(activities), "assays": activities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch bioactivity: {str(e)}")


@router.get("/pathway/disease/{disease_id}")
async def get_disease_pathways(disease_id: str):
    """
    Get KEGG pathways associated with a disease

    Args:
        disease_id: KEGG disease ID (e.g., H00409 for Type 2 diabetes)

    Example: `/v1/biomedical/pathway/disease/H00409`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        pathways = await hub.kegg.get_disease_pathways(disease_id)
        return {"disease_id": disease_id, "pathway_count": len(pathways), "pathways": pathways}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch pathways: {str(e)}")


@router.get("/pathway/info/{pathway_id}")
async def get_pathway_info(pathway_id: str):
    """
    Get detailed information about a KEGG pathway

    Args:
        pathway_id: KEGG pathway ID (e.g., hsa04930)

    Example: `/v1/biomedical/pathway/info/hsa04930`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        info = await hub.kegg.get_pathway_info(pathway_id)

        if not info:
            raise HTTPException(status_code=404, detail=f"Pathway {pathway_id} not found")

        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch pathway info: {str(e)}")


@router.get("/drug/kegg/{drug_name}")
async def search_kegg_drug(drug_name: str):
    """
    Search for drug in KEGG database

    Example: `/v1/biomedical/drug/kegg/metformin`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        data = await hub.kegg.search_drug(drug_name)

        if not data:
            raise HTTPException(status_code=404, detail=f"Drug {drug_name} not found in KEGG")

        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search KEGG: {str(e)}")


@router.post("/network/build", response_model=NetworkResponse)
async def build_drug_target_network(
    targets: List[str],
    include_interactions: bool = Query(True, description="Include protein-protein interactions")
):
    """
    Build a comprehensive drug target network

    Creates a network graph with drug targets and their protein interactions

    Request body:
    ```json
    {
        "targets": ["EGFR", "KRAS", "TP53"]
    }
    ```

    Example: `POST /v1/biomedical/network/build`
    """
    try:
        hub = get_biomedical_hub(DRUGBANK_API_KEY, DISGENET_API_KEY)
        network = await hub.get_drug_target_network(targets, include_interactions)
        return network
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build network: {str(e)}")


@router.get("/metadata")
async def get_metadata():
    """
    Get information about available biomedical data sources
    """
    return {
        "sources": {
            "uniprot": {
                "name": "UniProt",
                "description": "Protein sequences and functional information",
                "url": "https://www.uniprot.org/",
                "requires_api_key": False,
                "status": "active"
            },
            "string": {
                "name": "STRING",
                "description": "Protein-protein interaction networks",
                "url": "https://string-db.org/",
                "requires_api_key": False,
                "status": "active"
            },
            "pubchem": {
                "name": "PubChem",
                "description": "Chemical compounds and bioactivity data",
                "url": "https://pubchem.ncbi.nlm.nih.gov/",
                "requires_api_key": False,
                "status": "active"
            },
            "kegg": {
                "name": "KEGG",
                "description": "Biological pathways and diseases",
                "url": "https://www.kegg.jp/",
                "requires_api_key": False,
                "status": "active"
            },
            "drugbank": {
                "name": "DrugBank",
                "description": "Comprehensive drug database",
                "url": "https://go.drugbank.com/",
                "requires_api_key": True,
                "status": "available" if DRUGBANK_API_KEY else "disabled",
                "note": "Free academic API keys available"
            },
            "disgenet": {
                "name": "DisGeNET",
                "description": "Gene-disease associations",
                "url": "https://www.disgenet.org/",
                "requires_api_key": True,
                "status": "available" if DISGENET_API_KEY else "disabled",
                "note": "Registration required for API access"
            }
        },
        "endpoints": {
            "drug_enrichment": "/v1/biomedical/enrich/drug/{drug_name}",
            "disease_enrichment": "/v1/biomedical/enrich/disease/{disease_name}",
            "protein_info": "/v1/biomedical/protein/info/{protein_id}",
            "protein_interactions": "/v1/biomedical/protein/interactions/{protein_name}",
            "compound_info": "/v1/biomedical/compound/pubchem/{compound_name}",
            "pathways": "/v1/biomedical/pathway/disease/{disease_id}",
            "network_builder": "/v1/biomedical/network/build"
        }
    }
