"""Gene expression data integration for drug-disease associations."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GeneExpressionProfile(BaseModel):
    """Gene expression profile for a drug or disease."""

    entity_id: str = Field(..., description="Drug or disease identifier")
    entity_type: str = Field(..., description="'drug' or 'disease'")
    gene_symbols: List[str] = Field(default_factory=list, description="Gene symbols")
    expression_values: List[float] = Field(
        default_factory=list, description="Expression fold changes"
    )
    pvalues: Optional[List[float]] = Field(
        None, description="Statistical significance values"
    )
    data_source: str = Field(default="unknown", description="Data source name")
    sample_size: Optional[int] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class GeneSetEnrichment(BaseModel):
    """Gene set enrichment analysis result."""

    pathway_id: str
    pathway_name: str
    enrichment_score: float
    pvalue: float
    adjusted_pvalue: float
    genes_in_pathway: List[str]
    num_genes: int
    database: str = "KEGG"  # KEGG, GO, Reactome, etc.


class GeneExpressionSimilarity(BaseModel):
    """Similarity between two gene expression profiles."""

    entity1_id: str
    entity2_id: str
    correlation: float
    method: str = "pearson"
    num_common_genes: int
    pvalue: Optional[float] = None


class GeneExpressionIntegrator:
    """Integrates gene expression data for drug repurposing."""

    def __init__(self):
        """Initialize gene expression integrator."""
        self.expression_cache: Dict[str, GeneExpressionProfile] = {}
        self.pathway_cache: Dict[str, List[str]] = {}

    def load_expression_profile(
        self,
        entity_id: str,
        entity_type: str,
        data_source: str = "GEO",
    ) -> Optional[GeneExpressionProfile]:
        """Load gene expression profile for a drug or disease.

        In production, this would fetch from databases like:
        - GEO (Gene Expression Omnibus)
        - LINCS L1000 for drugs
        - DisGeNET for disease-gene associations
        - CREEDS for curated expression signatures

        Args:
            entity_id: Drug or disease identifier
            entity_type: 'drug' or 'disease'
            data_source: Data source name

        Returns:
            GeneExpressionProfile or None if not found
        """
        cache_key = f"{entity_type}:{entity_id}:{data_source}"

        if cache_key in self.expression_cache:
            return self.expression_cache[cache_key]

        # TODO: Implement actual data fetching from APIs
        # For now, return mock data
        logger.info(
            f"Loading expression profile for {entity_type} {entity_id} from {data_source}"
        )

        # Mock profile for demonstration
        profile = self._generate_mock_profile(entity_id, entity_type, data_source)
        self.expression_cache[cache_key] = profile

        return profile

    def _generate_mock_profile(
        self,
        entity_id: str,
        entity_type: str,
        data_source: str,
    ) -> GeneExpressionProfile:
        """Generate mock gene expression profile for testing.

        Args:
            entity_id: Entity identifier
            entity_type: 'drug' or 'disease'
            data_source: Data source name

        Returns:
            Mock GeneExpressionProfile
        """
        # Common cancer-related genes for demo
        genes = [
            "TP53", "EGFR", "KRAS", "PIK3CA", "PTEN",
            "BRAF", "MYC", "AKT1", "MTOR", "RB1",
            "VEGFA", "CDKN2A", "ERBB2", "BCL2", "MDM2",
            "NRAS", "FGFR1", "ALK", "MET", "RET",
        ]

        # Generate random expression values
        np.random.seed(hash(entity_id) % (2**32))
        expression = np.random.randn(len(genes)) * 2.0  # Fold changes
        pvalues = np.random.uniform(0.001, 0.05, len(genes))

        return GeneExpressionProfile(
            entity_id=entity_id,
            entity_type=entity_type,
            gene_symbols=genes,
            expression_values=expression.tolist(),
            pvalues=pvalues.tolist(),
            data_source=data_source,
            sample_size=100,
            metadata={"mock": "true"},
        )

    def compute_expression_similarity(
        self,
        profile1: GeneExpressionProfile,
        profile2: GeneExpressionProfile,
        method: str = "pearson",
    ) -> GeneExpressionSimilarity:
        """Compute similarity between two gene expression profiles.

        Args:
            profile1: First expression profile
            profile2: Second expression profile
            method: Correlation method ('pearson', 'spearman', 'cosine')

        Returns:
            GeneExpressionSimilarity result
        """
        # Find common genes
        genes1 = set(profile1.gene_symbols)
        genes2 = set(profile2.gene_symbols)
        common_genes = genes1.intersection(genes2)

        if not common_genes:
            return GeneExpressionSimilarity(
                entity1_id=profile1.entity_id,
                entity2_id=profile2.entity_id,
                correlation=0.0,
                method=method,
                num_common_genes=0,
            )

        # Extract expression values for common genes
        expr1 = []
        expr2 = []

        for gene in common_genes:
            idx1 = profile1.gene_symbols.index(gene)
            idx2 = profile2.gene_symbols.index(gene)
            expr1.append(profile1.expression_values[idx1])
            expr2.append(profile2.expression_values[idx2])

        expr1_arr = np.array(expr1)
        expr2_arr = np.array(expr2)

        if method == "pearson":
            correlation = np.corrcoef(expr1_arr, expr2_arr)[0, 1]
        elif method == "spearman":
            # Spearman = Pearson on ranks
            rank1 = np.argsort(np.argsort(expr1_arr))
            rank2 = np.argsort(np.argsort(expr2_arr))
            correlation = np.corrcoef(rank1, rank2)[0, 1]
        elif method == "cosine":
            # Cosine similarity
            dot_product = np.dot(expr1_arr, expr2_arr)
            norm1 = np.linalg.norm(expr1_arr)
            norm2 = np.linalg.norm(expr2_arr)
            correlation = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        # Handle NaN
        if np.isnan(correlation):
            correlation = 0.0

        return GeneExpressionSimilarity(
            entity1_id=profile1.entity_id,
            entity2_id=profile2.entity_id,
            correlation=float(correlation),
            method=method,
            num_common_genes=len(common_genes),
        )

    def find_differentially_expressed_genes(
        self,
        profile: GeneExpressionProfile,
        fold_change_threshold: float = 1.5,
        pvalue_threshold: float = 0.05,
    ) -> Tuple[List[str], List[str]]:
        """Find up-regulated and down-regulated genes.

        Args:
            profile: Gene expression profile
            fold_change_threshold: Minimum absolute fold change
            pvalue_threshold: Maximum p-value

        Returns:
            Tuple of (upregulated_genes, downregulated_genes)
        """
        upregulated = []
        downregulated = []

        for i, gene in enumerate(profile.gene_symbols):
            expr = profile.expression_values[i]
            pval = profile.pvalues[i] if profile.pvalues else 0.01

            if pval > pvalue_threshold:
                continue

            if expr > fold_change_threshold:
                upregulated.append(gene)
            elif expr < -fold_change_threshold:
                downregulated.append(gene)

        logger.info(
            f"Found {len(upregulated)} upregulated and {len(downregulated)} "
            f"downregulated genes in {profile.entity_id}"
        )

        return upregulated, downregulated

    def compute_gene_set_enrichment(
        self,
        gene_list: List[str],
        database: str = "KEGG",
    ) -> List[GeneSetEnrichment]:
        """Perform gene set enrichment analysis.

        In production, this would use:
        - GSEA (Gene Set Enrichment Analysis)
        - Enrichr API
        - DAVID functional annotation
        - g:Profiler

        Args:
            gene_list: List of gene symbols
            database: Pathway database (KEGG, GO, Reactome)

        Returns:
            List of enriched pathways
        """
        # TODO: Implement real GSEA via API calls
        logger.info(f"Computing enrichment for {len(gene_list)} genes in {database}")

        # Mock enrichment results
        mock_pathways = [
            ("hsa04151", "PI3K-Akt signaling pathway", 0.85, 0.001),
            ("hsa04010", "MAPK signaling pathway", 0.78, 0.005),
            ("hsa05200", "Pathways in cancer", 0.92, 0.0001),
            ("hsa04510", "Focal adhesion", 0.72, 0.01),
            ("hsa04110", "Cell cycle", 0.68, 0.02),
        ]

        results = []
        for pathway_id, pathway_name, score, pval in mock_pathways:
            # Mock genes in pathway (subset of input)
            genes_in_pathway = gene_list[:min(5, len(gene_list))]

            results.append(
                GeneSetEnrichment(
                    pathway_id=pathway_id,
                    pathway_name=pathway_name,
                    enrichment_score=score,
                    pvalue=pval,
                    adjusted_pvalue=pval * len(mock_pathways),  # Bonferroni
                    genes_in_pathway=genes_in_pathway,
                    num_genes=len(genes_in_pathway),
                    database=database,
                )
            )

        return results

    def predict_drug_disease_association_from_expression(
        self,
        drug_id: str,
        disease_id: str,
    ) -> Dict[str, float]:
        """Predict drug-disease association from gene expression.

        Uses the "connectivity map" approach:
        - If drug reverses disease expression signature -> potential treatment
        - Negative correlation = therapeutic potential

        Args:
            drug_id: Drug identifier
            disease_id: Disease identifier

        Returns:
            Dictionary with association score and confidence
        """
        drug_profile = self.load_expression_profile(drug_id, "drug")
        disease_profile = self.load_expression_profile(disease_id, "disease")

        if drug_profile is None or disease_profile is None:
            return {
                "association_score": 0.5,
                "confidence": 0.0,
                "error": "Profiles not available",
            }

        # Compute correlation
        similarity = self.compute_expression_similarity(drug_profile, disease_profile)

        # Negative correlation suggests therapeutic potential
        # (drug reverses disease signature)
        reversal_score = -similarity.correlation  # Flip sign

        # Normalize to [0, 1]
        association_score = (reversal_score + 1) / 2

        return {
            "association_score": association_score,
            "correlation": similarity.correlation,
            "num_common_genes": similarity.num_common_genes,
            "confidence": min(similarity.num_common_genes / 100.0, 1.0),
        }

    def get_drug_targets(
        self,
        drug_id: str,
        expression_threshold: float = 2.0,
    ) -> List[str]:
        """Get predicted molecular targets from drug expression profile.

        Args:
            drug_id: Drug identifier
            expression_threshold: Minimum expression change

        Returns:
            List of gene symbols (predicted targets)
        """
        profile = self.load_expression_profile(drug_id, "drug")

        if profile is None:
            return []

        targets = []
        for i, gene in enumerate(profile.gene_symbols):
            if abs(profile.expression_values[i]) > expression_threshold:
                targets.append(gene)

        return targets


# Singleton instance
_gene_expression_integrator: Optional[GeneExpressionIntegrator] = None


def get_gene_expression_integrator() -> GeneExpressionIntegrator:
    """Get or create singleton GeneExpressionIntegrator instance."""
    global _gene_expression_integrator
    if _gene_expression_integrator is None:
        _gene_expression_integrator = GeneExpressionIntegrator()
    return _gene_expression_integrator
