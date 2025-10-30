"""
Enhanced Biomedical Data Integrations for Drug Repurposing

This module provides connectors to major biomedical databases:
- UniProt (Protein sequences and functions)
- STRING (Protein-protein interactions)
- DrugBank (Comprehensive drug database)
- ChEBI (Chemical entities of biological interest)
- PubChem (Chemical information)
- ClinVar (Genetic variants)
- KEGG (Pathways and diseases)
- DisGeNET (Gene-disease associations)
"""

import httpx
from typing import Dict, List, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)


class UniProtConnector:
    """Connect to UniProt protein database"""

    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_protein_info(self, protein_id: str) -> Dict[str, Any]:
        """Get protein information from UniProt"""
        url = f"{self.BASE_URL}/{protein_id}.json"

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()

            return {
                "protein_id": protein_id,
                "name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                "gene": data.get("genes", [{}])[0].get("geneName", {}).get("value", ""),
                "organism": data.get("organism", {}).get("scientificName", ""),
                "function": self._extract_function(data),
                "subcellular_location": self._extract_location(data),
                "interactions": len(data.get("interactions", [])),
            }
        except Exception as e:
            logger.error(f"UniProt API error for {protein_id}: {e}")
            return {}

    def _extract_function(self, data: Dict) -> str:
        """Extract protein function description"""
        comments = data.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "FUNCTION":
                return comment.get("texts", [{}])[0].get("value", "")
        return ""

    def _extract_location(self, data: Dict) -> List[str]:
        """Extract subcellular locations"""
        comments = data.get("comments", [])
        locations = []
        for comment in comments:
            if comment.get("commentType") == "SUBCELLULAR_LOCATION":
                for loc in comment.get("subcellularLocations", []):
                    locations.append(loc.get("location", {}).get("value", ""))
        return locations

    async def search_proteins_by_gene(self, gene_name: str, limit: int = 10) -> List[Dict]:
        """Search proteins by gene name"""
        url = f"{self.BASE_URL}/search"
        params = {
            "query": f"gene:{gene_name}",
            "format": "json",
            "size": limit
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for entry in data.get("results", []):
                results.append({
                    "protein_id": entry.get("primaryAccession", ""),
                    "name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                    "organism": entry.get("organism", {}).get("scientificName", ""),
                })

            return results
        except Exception as e:
            logger.error(f"UniProt search error for {gene_name}: {e}")
            return []


class STRINGConnector:
    """Connect to STRING protein-protein interaction database"""

    BASE_URL = "https://string-db.org/api/json"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_protein_interactions(
        self,
        protein_name: str,
        species: int = 9606,  # Human
        limit: int = 10,
        required_score: int = 400  # Confidence threshold (0-1000)
    ) -> List[Dict[str, Any]]:
        """Get protein-protein interactions from STRING"""
        url = f"{self.BASE_URL}/interaction_partners"
        params = {
            "identifiers": protein_name,
            "species": species,
            "limit": limit,
            "required_score": required_score
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            interactions = []
            for item in data:
                interactions.append({
                    "partner_protein": item.get("preferredName_B", ""),
                    "score": item.get("score", 0) / 1000.0,  # Convert to 0-1 scale
                    "interaction_type": self._categorize_score(item.get("score", 0)),
                })

            return interactions
        except Exception as e:
            logger.error(f"STRING API error for {protein_name}: {e}")
            return []

    def _categorize_score(self, score: int) -> str:
        """Categorize interaction confidence"""
        if score >= 900:
            return "highest_confidence"
        elif score >= 700:
            return "high_confidence"
        elif score >= 400:
            return "medium_confidence"
        else:
            return "low_confidence"

    async def get_network_image(self, proteins: List[str], species: int = 9606) -> str:
        """Get URL for network visualization"""
        protein_list = "%0d".join(proteins)  # URL encoded newline
        url = f"https://string-db.org/api/image/network?identifiers={protein_list}&species={species}"
        return url


class DrugBankConnector:
    """
    Connect to DrugBank (requires API key from https://go.drugbank.com/releases/latest)
    Note: Free tier available for academic use
    """

    BASE_URL = "https://api.drugbank.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {}
        )

    async def search_drug(self, drug_name: str) -> Dict[str, Any]:
        """Search for drug information"""
        if not self.api_key:
            logger.warning("DrugBank API key not provided")
            return {}

        url = f"{self.BASE_URL}/drugs"
        params = {"q": drug_name}

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data:
                drug = data[0]
                return {
                    "drugbank_id": drug.get("drugbank_id", ""),
                    "name": drug.get("name", ""),
                    "description": drug.get("description", ""),
                    "indication": drug.get("indication", ""),
                    "pharmacodynamics": drug.get("pharmacodynamics", ""),
                    "mechanism_of_action": drug.get("mechanism_of_action", ""),
                    "targets": [t.get("name") for t in drug.get("targets", [])],
                    "categories": [c.get("category") for c in drug.get("categories", [])],
                }
            return {}
        except Exception as e:
            logger.error(f"DrugBank API error for {drug_name}: {e}")
            return {}


class PubChemConnector:
    """Connect to PubChem for chemical information"""

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_compound_by_name(self, compound_name: str) -> Dict[str, Any]:
        """Get compound information by name"""
        url = f"{self.BASE_URL}/compound/name/{compound_name}/JSON"

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()

            if "PC_Compounds" in data and data["PC_Compounds"]:
                compound = data["PC_Compounds"][0]
                return {
                    "cid": compound.get("id", {}).get("id", {}).get("cid", ""),
                    "molecular_formula": self._get_property(compound, "Molecular Formula"),
                    "molecular_weight": self._get_property(compound, "Molecular Weight"),
                    "smiles": self._get_property(compound, "SMILES", "Canonical"),
                    "inchi": self._get_property(compound, "InChI"),
                    "iupac_name": self._get_property(compound, "IUPAC Name"),
                }
            return {}
        except Exception as e:
            logger.error(f"PubChem API error for {compound_name}: {e}")
            return {}

    def _get_property(self, compound: Dict, label: str, name: str = None) -> str:
        """Extract property from compound data"""
        props = compound.get("props", [])
        for prop in props:
            urn = prop.get("urn", {})
            if urn.get("label") == label:
                if name and urn.get("name") != name:
                    continue
                value = prop.get("value", {})
                return str(value.get("sval") or value.get("fval") or value.get("ival", ""))
        return ""

    async def get_bioactivity(self, cid: str) -> List[Dict]:
        """Get bioactivity data for compound"""
        url = f"{self.BASE_URL}/compound/cid/{cid}/assaysummary/JSON"

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()

            activities = []
            for item in data.get("Table", {}).get("Row", [])[:10]:  # Limit to 10
                activities.append({
                    "assay_id": item.get("aid", ""),
                    "activity": item.get("activity", ""),
                    "target": item.get("target", ""),
                })

            return activities
        except Exception as e:
            logger.error(f"PubChem bioactivity error for CID {cid}: {e}")
            return []


class KEGGConnector:
    """Connect to KEGG pathway database"""

    BASE_URL = "https://rest.kegg.jp"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_disease_pathways(self, disease_id: str) -> List[Dict]:
        """Get pathways associated with disease"""
        url = f"{self.BASE_URL}/link/pathway/{disease_id}"

        try:
            response = await self.client.get(url)
            response.raise_for_status()

            pathways = []
            for line in response.text.strip().split("\n"):
                if "\t" in line:
                    _, pathway_id = line.split("\t")
                    pathway_info = await self.get_pathway_info(pathway_id)
                    if pathway_info:
                        pathways.append(pathway_info)

            return pathways
        except Exception as e:
            logger.error(f"KEGG pathways error for {disease_id}: {e}")
            return []

    async def get_pathway_info(self, pathway_id: str) -> Dict:
        """Get detailed pathway information"""
        url = f"{self.BASE_URL}/get/{pathway_id}"

        try:
            response = await self.client.get(url)
            response.raise_for_status()

            lines = response.text.split("\n")
            info = {"pathway_id": pathway_id}

            for line in lines:
                if line.startswith("NAME"):
                    info["name"] = line.split("NAME")[1].strip()
                elif line.startswith("DESCRIPTION"):
                    info["description"] = line.split("DESCRIPTION")[1].strip()

            return info
        except Exception as e:
            logger.error(f"KEGG pathway info error for {pathway_id}: {e}")
            return {}

    async def search_drug(self, drug_name: str) -> Dict:
        """Search for drug in KEGG"""
        url = f"{self.BASE_URL}/find/drug/{drug_name}"

        try:
            response = await self.client.get(url)
            response.raise_for_status()

            if response.text.strip():
                first_line = response.text.strip().split("\n")[0]
                kegg_id, description = first_line.split("\t")

                return {
                    "kegg_id": kegg_id.replace("drug:", ""),
                    "description": description,
                }
            return {}
        except Exception as e:
            logger.error(f"KEGG drug search error for {drug_name}: {e}")
            return {}


class DisGeNETConnector:
    """
    Connect to DisGeNET gene-disease associations
    Note: Requires registration for API key at https://www.disgenet.org/
    """

    BASE_URL = "https://www.disgenet.org/api"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {}
        )

    async def get_disease_genes(self, disease_id: str, limit: int = 20) -> List[Dict]:
        """Get genes associated with disease"""
        if not self.api_key:
            logger.warning("DisGeNET API key not provided")
            return []

        url = f"{self.BASE_URL}/gda/disease/{disease_id}"
        params = {"limit": limit}

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            genes = []
            for item in data:
                genes.append({
                    "gene_symbol": item.get("gene_symbol", ""),
                    "gene_id": item.get("gene_id", ""),
                    "score": item.get("score", 0),
                    "evidence": item.get("source", ""),
                })

            return genes
        except Exception as e:
            logger.error(f"DisGeNET API error for {disease_id}: {e}")
            return []


# Unified interface for all biomedical data sources
class BiomedicalDataHub:
    """Central hub for all biomedical data sources"""

    def __init__(
        self,
        drugbank_api_key: Optional[str] = None,
        disgenet_api_key: Optional[str] = None
    ):
        self.uniprot = UniProtConnector()
        self.string_db = STRINGConnector()
        self.drugbank = DrugBankConnector(drugbank_api_key)
        self.pubchem = PubChemConnector()
        self.kegg = KEGGConnector()
        self.disgenet = DisGeNETConnector(disgenet_api_key)

    async def enrich_drug_data(self, drug_name: str, chembl_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enrich drug information from multiple sources

        Returns comprehensive drug profile with:
        - Chemical properties (PubChem)
        - Drug database info (DrugBank)
        - Pathway information (KEGG)
        - Bioactivity data
        """
        enriched_data = {
            "drug_name": drug_name,
            "chembl_id": chembl_id,
            "chemical_properties": {},
            "drugbank": {},
            "kegg": {},
            "bioactivity": [],
        }

        # Get PubChem data
        pubchem_data = await self.pubchem.get_compound_by_name(drug_name)
        if pubchem_data:
            enriched_data["chemical_properties"] = pubchem_data

        # Get DrugBank data (if API key available)
        drugbank_data = await self.drugbank.search_drug(drug_name)
        if drugbank_data:
            enriched_data["drugbank"] = drugbank_data

        # Get KEGG data
        kegg_data = await self.kegg.search_drug(drug_name)
        if kegg_data:
            enriched_data["kegg"] = kegg_data

        # Get bioactivity if we have CID
        if pubchem_data and pubchem_data.get("cid"):
            bioactivity = await self.pubchem.get_bioactivity(pubchem_data["cid"])
            if bioactivity:
                enriched_data["bioactivity"] = bioactivity

        return enriched_data

    async def enrich_disease_data(self, disease_name: str, disease_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enrich disease information from multiple sources

        Returns comprehensive disease profile with:
        - Associated genes (DisGeNET)
        - Pathways (KEGG)
        - Protein interactions
        """
        enriched_data = {
            "disease_name": disease_name,
            "disease_id": disease_id,
            "associated_genes": [],
            "protein_network": [],
            "pathways": [],
        }

        # Get associated genes
        if disease_id and self.disgenet.api_key:
            genes = await self.disgenet.get_disease_genes(disease_id)
            if genes:
                enriched_data["associated_genes"] = genes

                # Get protein interactions for top genes
                if genes:
                    top_gene = genes[0]["gene_symbol"]
                    interactions = await self.string_db.get_protein_interactions(top_gene)
                    if interactions:
                        enriched_data["protein_network"] = interactions

        # Get pathways
        if disease_id:
            pathways = await self.kegg.get_disease_pathways(disease_id)
            if pathways:
                enriched_data["pathways"] = pathways

        return enriched_data

    async def get_drug_target_network(
        self,
        drug_targets: List[str],
        include_interactions: bool = True
    ) -> Dict[str, Any]:
        """
        Build a network of drug targets and their interactions

        Args:
            drug_targets: List of target gene/protein names
            include_interactions: Whether to fetch protein-protein interactions

        Returns:
            Network data with nodes and edges
        """
        network = {
            "nodes": [],
            "edges": [],
        }

        for target in drug_targets:
            # Get protein info
            protein_info = await self.uniprot.search_proteins_by_gene(target, limit=1)
            if protein_info:
                network["nodes"].append({
                    "id": target,
                    "type": "protein",
                    "data": protein_info[0]
                })

                # Get interactions
                if include_interactions:
                    interactions = await self.string_db.get_protein_interactions(target, limit=5)
                    for interaction in interactions:
                        network["edges"].append({
                            "source": target,
                            "target": interaction["partner_protein"],
                            "score": interaction["score"],
                            "type": interaction["interaction_type"]
                        })

                        # Add partner as node if not exists
                        if not any(n["id"] == interaction["partner_protein"] for n in network["nodes"]):
                            network["nodes"].append({
                                "id": interaction["partner_protein"],
                                "type": "protein",
                                "data": {}
                            })

        return network

    async def close(self):
        """Close all HTTP clients"""
        await self.uniprot.client.aclose()
        await self.string_db.client.aclose()
        await self.drugbank.client.aclose()
        await self.pubchem.client.aclose()
        await self.kegg.client.aclose()
        await self.disgenet.client.aclose()


# Singleton instance
_biomedical_hub: Optional[BiomedicalDataHub] = None


def get_biomedical_hub(
    drugbank_api_key: Optional[str] = None,
    disgenet_api_key: Optional[str] = None
) -> BiomedicalDataHub:
    """Get or create biomedical data hub instance"""
    global _biomedical_hub
    if _biomedical_hub is None:
        _biomedical_hub = BiomedicalDataHub(
            drugbank_api_key=drugbank_api_key,
            disgenet_api_key=disgenet_api_key
        )
    return _biomedical_hub
