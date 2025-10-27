"""Chemical structure analysis using RDKit and molecular fingerprints."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Optional RDKit import with graceful degradation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Chemical structure analysis will be limited.")
    RDKIT_AVAILABLE = False


class MolecularDescriptors(BaseModel):
    """Molecular descriptors computed from chemical structure."""

    molecular_weight: float
    logp: float  # Lipophilicity
    num_h_donors: int
    num_h_acceptors: int
    num_rotatable_bonds: int
    tpsa: float  # Topological polar surface area
    num_aromatic_rings: int
    num_rings: int
    fraction_csp3: float
    num_stereocenters: int


class ChemicalFeatures(BaseModel):
    """Chemical features extracted from molecular structure."""

    smiles: str
    molecular_descriptors: Optional[MolecularDescriptors] = None
    morgan_fingerprint: Optional[List[int]] = None
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    is_valid: bool = True
    error_message: Optional[str] = None


class ChemicalStructureAnalyzer:
    """Analyzes chemical structures and extracts molecular features."""

    def __init__(
        self,
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
        use_chirality: bool = True,
    ):
        """Initialize chemical structure analyzer.

        Args:
            fingerprint_radius: Radius for Morgan fingerprints (default: 2)
            fingerprint_bits: Number of bits in fingerprint (default: 2048)
            use_chirality: Include chirality in fingerprints (default: True)
        """
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        self.use_chirality = use_chirality
        self.enabled = RDKIT_AVAILABLE

        if not self.enabled:
            logger.warning("ChemicalStructureAnalyzer initialized but RDKit unavailable")

    def smiles_to_mol(self, smiles: str) -> Optional[object]:
        """Convert SMILES string to RDKit molecule object.

        Args:
            smiles: SMILES representation of molecule

        Returns:
            RDKit molecule object or None if invalid
        """
        if not self.enabled:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception as e:
            logger.error(f"Failed to parse SMILES '{smiles}': {e}")
            return None

    def compute_descriptors(self, mol: object) -> Optional[MolecularDescriptors]:
        """Compute molecular descriptors from RDKit molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            MolecularDescriptors or None if computation fails
        """
        if not self.enabled or mol is None:
            return None

        try:
            return MolecularDescriptors(
                molecular_weight=Descriptors.MolWt(mol),
                logp=Descriptors.MolLogP(mol),
                num_h_donors=Lipinski.NumHDonors(mol),
                num_h_acceptors=Lipinski.NumHAcceptors(mol),
                num_rotatable_bonds=Lipinski.NumRotatableBonds(mol),
                tpsa=Descriptors.TPSA(mol),
                num_aromatic_rings=Lipinski.NumAromaticRings(mol),
                num_rings=Lipinski.RingCount(mol),
                fraction_csp3=Lipinski.FractionCSP3(mol),
                num_stereocenters=len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            )
        except Exception as e:
            logger.error(f"Failed to compute descriptors: {e}")
            return None

    def compute_fingerprint(self, mol: object) -> Optional[np.ndarray]:
        """Compute Morgan (circular) fingerprint.

        Args:
            mol: RDKit molecule object

        Returns:
            Binary fingerprint as numpy array or None
        """
        if not self.enabled or mol is None:
            return None

        try:
            fp = GetMorganFingerprintAsBitVect(
                mol,
                radius=self.fingerprint_radius,
                nBits=self.fingerprint_bits,
                useChirality=self.use_chirality,
            )
            # Convert to numpy array
            arr = np.zeros((self.fingerprint_bits,), dtype=np.int32)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception as e:
            logger.error(f"Failed to compute fingerprint: {e}")
            return None

    def analyze_smiles(self, smiles: str, drug_id: str = "") -> ChemicalFeatures:
        """Analyze a chemical structure from SMILES.

        Args:
            smiles: SMILES string representation
            drug_id: Optional drug identifier for logging

        Returns:
            ChemicalFeatures object with extracted features
        """
        if not self.enabled:
            return ChemicalFeatures(
                smiles=smiles,
                is_valid=False,
                error_message="RDKit not available",
            )

        mol = self.smiles_to_mol(smiles)

        if mol is None:
            return ChemicalFeatures(
                smiles=smiles,
                is_valid=False,
                error_message="Invalid SMILES string",
            )

        descriptors = self.compute_descriptors(mol)
        fingerprint = self.compute_fingerprint(mol)

        return ChemicalFeatures(
            smiles=smiles,
            molecular_descriptors=descriptors,
            morgan_fingerprint=fingerprint.tolist() if fingerprint is not None else None,
            fingerprint_radius=self.fingerprint_radius,
            fingerprint_bits=self.fingerprint_bits,
            is_valid=True,
        )

    def compute_similarity(
        self,
        smiles1: str,
        smiles2: str,
        method: str = "tanimoto",
    ) -> float:
        """Compute structural similarity between two molecules.

        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            method: Similarity method ('tanimoto', 'dice', 'cosine')

        Returns:
            Similarity score between 0 and 1
        """
        if not self.enabled:
            return 0.5  # Default similarity

        mol1 = self.smiles_to_mol(smiles1)
        mol2 = self.smiles_to_mol(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        fp1 = self.compute_fingerprint(mol1)
        fp2 = self.compute_fingerprint(mol2)

        if fp1 is None or fp2 is None:
            return 0.0

        if method == "tanimoto":
            # Tanimoto similarity for binary fingerprints
            intersection = np.sum(np.logical_and(fp1, fp2))
            union = np.sum(np.logical_or(fp1, fp2))
            return float(intersection / union) if union > 0 else 0.0

        elif method == "dice":
            # Dice coefficient
            intersection = np.sum(np.logical_and(fp1, fp2))
            return float(2 * intersection / (np.sum(fp1) + np.sum(fp2)))

        elif method == "cosine":
            # Cosine similarity
            dot_product = np.dot(fp1, fp2)
            norm1 = np.linalg.norm(fp1)
            norm2 = np.linalg.norm(fp2)
            return float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def check_lipinski_rule_of_five(self, smiles: str) -> Dict[str, bool]:
        """Check Lipinski's Rule of Five for drug-likeness.

        Rule of Five criteria:
        - Molecular weight < 500 Da
        - LogP < 5
        - H-bond donors <= 5
        - H-bond acceptors <= 10

        Args:
            smiles: SMILES string

        Returns:
            Dictionary with rule checks
        """
        if not self.enabled:
            return {
                "passes_rule_of_five": False,
                "error": "RDKit not available",
            }

        features = self.analyze_smiles(smiles)

        if not features.is_valid or features.molecular_descriptors is None:
            return {
                "passes_rule_of_five": False,
                "error": "Invalid molecule",
            }

        desc = features.molecular_descriptors

        checks = {
            "mw_check": desc.molecular_weight < 500,
            "logp_check": desc.logp < 5,
            "hbd_check": desc.num_h_donors <= 5,
            "hba_check": desc.num_h_acceptors <= 10,
        }

        # Pass if no more than one violation
        violations = sum(not v for v in checks.values())
        checks["passes_rule_of_five"] = violations <= 1
        checks["num_violations"] = violations

        return checks

    def batch_analyze(
        self,
        smiles_list: List[Tuple[str, str]],
    ) -> List[ChemicalFeatures]:
        """Analyze multiple chemical structures in batch.

        Args:
            smiles_list: List of (drug_id, smiles) tuples

        Returns:
            List of ChemicalFeatures objects
        """
        results = []
        for drug_id, smiles in smiles_list:
            features = self.analyze_smiles(smiles, drug_id)
            results.append(features)

        logger.info(f"Batch analyzed {len(results)} chemical structures")
        return results


# Singleton instance
_chemical_analyzer: Optional[ChemicalStructureAnalyzer] = None


def get_chemical_analyzer() -> ChemicalStructureAnalyzer:
    """Get or create singleton ChemicalStructureAnalyzer instance."""
    global _chemical_analyzer
    if _chemical_analyzer is None:
        _chemical_analyzer = ChemicalStructureAnalyzer()
    return _chemical_analyzer
