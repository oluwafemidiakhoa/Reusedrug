"""Molecular substructure highlighting for drug explanations.

Identifies and highlights important molecular fragments that contribute
to drug-disease predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class MolecularHighlighterConfig:
    """Configuration for molecular highlighter."""

    fragment_size: int = 3  # Size of fragments to analyze (num atoms)
    top_k_fragments: int = 10  # Top K important fragments
    importance_threshold: float = 0.1  # Minimum importance to highlight


class MolecularHighlighter:
    """Highlight important molecular substructures.

    Uses feature importance to identify which parts of a molecule
    contribute most to the prediction.
    """

    def __init__(self, config: Optional[MolecularHighlighterConfig] = None):
        """
        Args:
            config: Molecular highlighter configuration
        """
        self.config = config or MolecularHighlighterConfig()

        logger.info(
            f"Molecular Highlighter initialized with "
            f"fragment_size={self.config.fragment_size}"
        )

    def highlight_molecule(
        self,
        smiles: str,
        feature_importance: np.ndarray,
        feature_to_atom_mapping: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Highlight important substructures in a molecule.

        Args:
            smiles: SMILES string of the molecule
            feature_importance: Importance scores for each feature [num_features]
            feature_to_atom_mapping: Mapping from feature indices to atom indices

        Returns:
            highlighted: Dictionary with atom/bond importance scores
        """
        try:
            # Try to use RDKit if available
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return self._mock_highlights(smiles, feature_importance)

            return self._highlight_with_rdkit(mol, feature_importance, feature_to_atom_mapping)

        except ImportError:
            logger.warning("RDKit not available, using mock highlights")
            return self._mock_highlights(smiles, feature_importance)

    def _highlight_with_rdkit(
        self,
        mol,
        feature_importance: np.ndarray,
        feature_to_atom_mapping: Optional[Dict[int, List[int]]],
    ):
        """Highlight molecule using RDKit."""
        from rdkit import Chem

        num_atoms = mol.GetNumAtoms()
        atom_importance = np.zeros(num_atoms)

        # Map feature importance to atoms
        if feature_to_atom_mapping:
            for feat_idx, importance in enumerate(feature_importance):
                if feat_idx in feature_to_atom_mapping:
                    for atom_idx in feature_to_atom_mapping[feat_idx]:
                        if atom_idx < num_atoms:
                            atom_importance[atom_idx] += importance
        else:
            # Use Morgan fingerprint bits to map features to atoms
            info = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048, bitInfo=info
            )

            # Map bits to atoms
            for bit_idx, atom_list in info.items():
                if bit_idx < len(feature_importance):
                    importance = feature_importance[bit_idx]
                    for atom_idx, _ in atom_list:
                        if atom_idx < num_atoms:
                            atom_importance[atom_idx] += importance

        # Normalize
        if np.max(atom_importance) > 0:
            atom_importance = atom_importance / np.max(atom_importance)

        # Identify functional groups
        functional_groups = self._identify_functional_groups(mol, atom_importance)

        # Get atom details
        atom_details = []
        for idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(idx)
            atom_details.append({
                "atom_index": idx,
                "element": atom.GetSymbol(),
                "importance": float(atom_importance[idx]),
                "is_aromatic": atom.GetIsAromatic(),
                "formal_charge": atom.GetFormalCharge(),
            })

        # Sort by importance
        atom_details.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "smiles": Chem.MolToSmiles(mol),
            "num_atoms": num_atoms,
            "num_bonds": mol.GetNumBonds(),
            "atom_importance": atom_importance.tolist(),
            "top_atoms": atom_details[:self.config.top_k_fragments],
            "functional_groups": functional_groups,
            "molecular_weight": self._get_molecular_weight(mol),
            "has_aromatic_rings": any(atom.GetIsAromatic() for atom in mol.GetAtoms()),
        }

    def _identify_functional_groups(self, mol, atom_importance: np.ndarray) -> List[Dict]:
        """Identify important functional groups."""
        from rdkit import Chem

        functional_groups = []

        # Common functional group SMARTS patterns
        patterns = {
            "carboxyl": "C(=O)O",
            "hydroxyl": "[OH]",
            "amine": "N",
            "amide": "C(=O)N",
            "ester": "C(=O)O[C]",
            "ether": "[C]O[C]",
            "ketone": "[C]C(=O)[C]",
            "aldehyde": "C(=O)[H]",
            "phenyl": "c1ccccc1",
        }

        for group_name, smarts in patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    # Compute average importance for this group
                    group_importance = np.mean([atom_importance[idx] for idx in match])

                    if group_importance > self.config.importance_threshold:
                        functional_groups.append({
                            "name": group_name,
                            "atom_indices": list(match),
                            "importance": float(group_importance),
                        })

        # Sort by importance
        functional_groups.sort(key=lambda x: x["importance"], reverse=True)

        return functional_groups[:self.config.top_k_fragments]

    def _get_molecular_weight(self, mol) -> float:
        """Calculate molecular weight."""
        try:
            from rdkit.Chem import Descriptors
            return Descriptors.MolWt(mol)
        except Exception:
            return 0.0

    def _mock_highlights(
        self,
        smiles: str,
        feature_importance: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate mock highlights when RDKit is not available."""
        # Estimate number of atoms from SMILES length
        num_atoms = len(smiles) // 2  # Rough estimate

        # Generate random importance scores
        atom_importance = np.random.rand(num_atoms)
        atom_importance = atom_importance / np.max(atom_importance)

        top_atoms = [
            {
                "atom_index": i,
                "element": "C",  # Mock
                "importance": float(atom_importance[i]),
                "is_aromatic": False,
                "formal_charge": 0,
            }
            for i in np.argsort(atom_importance)[::-1][:self.config.top_k_fragments]
        ]

        return {
            "smiles": smiles,
            "num_atoms": num_atoms,
            "num_bonds": num_atoms - 1,
            "atom_importance": atom_importance.tolist(),
            "top_atoms": top_atoms,
            "functional_groups": [],
            "molecular_weight": 0.0,
            "has_aromatic_rings": False,
            "note": "Mock highlights - RDKit not available",
        }

    def compare_molecules(
        self,
        smiles1: str,
        smiles2: str,
        feature_importance1: np.ndarray,
        feature_importance2: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compare highlighted substructures between two molecules.

        Args:
            smiles1: First molecule SMILES
            smiles2: Second molecule SMILES
            feature_importance1: Feature importance for molecule 1
            feature_importance2: Feature importance for molecule 2

        Returns:
            comparison: Structural differences and importance changes
        """
        highlights1 = self.highlight_molecule(smiles1, feature_importance1)
        highlights2 = self.highlight_molecule(smiles2, feature_importance2)

        # Find common and different substructures
        common_groups = []
        different_groups = []

        groups1 = {g["name"]: g for g in highlights1.get("functional_groups", [])}
        groups2 = {g["name"]: g for g in highlights2.get("functional_groups", [])}

        for name in set(groups1.keys()) | set(groups2.keys()):
            if name in groups1 and name in groups2:
                importance_change = groups2[name]["importance"] - groups1[name]["importance"]
                common_groups.append({
                    "name": name,
                    "importance_change": float(importance_change),
                    "molecule1_importance": groups1[name]["importance"],
                    "molecule2_importance": groups2[name]["importance"],
                })
            else:
                present_in = "molecule1" if name in groups1 else "molecule2"
                different_groups.append({
                    "name": name,
                    "present_in": present_in,
                    "importance": groups1.get(name, groups2.get(name))["importance"],
                })

        return {
            "molecule1": highlights1,
            "molecule2": highlights2,
            "common_groups": common_groups,
            "different_groups": different_groups,
            "structural_similarity": self._compute_similarity(smiles1, smiles2),
        }

    def _compute_similarity(self, smiles1: str, smiles2: str) -> float:
        """Compute structural similarity between molecules."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit import DataStructs

            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return 0.0

            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)

            return DataStructs.TanimotoSimilarity(fp1, fp2)

        except Exception as e:
            logger.warning(f"Failed to compute similarity: {e}")
            return 0.0


# Singleton instance
_molecular_highlighter: Optional[MolecularHighlighter] = None


def get_molecular_highlighter(
    config: Optional[MolecularHighlighterConfig] = None,
) -> MolecularHighlighter:
    """Get or create molecular highlighter instance."""
    global _molecular_highlighter

    if _molecular_highlighter is None:
        _molecular_highlighter = MolecularHighlighter(config)
        logger.info("Created new Molecular Highlighter instance")

    return _molecular_highlighter
