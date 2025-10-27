"""Prepare knowledge graph data for GNN training.

This script creates a sample knowledge graph from synthetic data
that mimics the structure of real biomedical knowledge.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch


def generate_sample_data() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Generate sample drugs, diseases, and associations.

    In production, this would fetch from:
    - Open Targets for known drug-disease pairs
    - Translator for knowledge graph
    - ChEMBL for drug properties
    - MONDO/DO for disease ontology
    """

    # Sample drugs (in reality: fetch from ChEMBL, DrugBank)
    drugs = [
        {"id": "CHEMBL25", "name": "Aspirin", "type": "NSAID"},
        {"id": "CHEMBL521", "name": "Ibuprofen", "type": "NSAID"},
        {"id": "CHEMBL1200958", "name": "Metformin", "type": "Antidiabetic"},
        {"id": "CHEMBL1201194", "name": "Glipizide", "type": "Antidiabetic"},
        {"id": "CHEMBL88", "name": "Warfarin", "type": "Anticoagulant"},
        {"id": "CHEMBL1656", "name": "Simvastatin", "type": "Statin"},
        {"id": "CHEMBL1433", "name": "Atorvastatin", "type": "Statin"},
        {"id": "CHEMBL941", "name": "Lisinopril", "type": "ACE_Inhibitor"},
        {"id": "CHEMBL1566", "name": "Losartan", "type": "ARB"},
        {"id": "CHEMBL835", "name": "Amlodipine", "type": "Calcium_Channel_Blocker"},
        {"id": "CHEMBL597", "name": "Insulin", "type": "Hormone"},
        {"id": "CHEMBL1200766", "name": "Sitagliptin", "type": "DPP4_Inhibitor"},
        {"id": "CHEMBL278020", "name": "Canagliflozin", "type": "SGLT2_Inhibitor"},
        {"id": "CHEMBL1096999", "name": "Levothyroxine", "type": "Thyroid_Hormone"},
        {"id": "CHEMBL220492", "name": "Adalimumab", "type": "TNF_Inhibitor"},
    ]

    # Sample diseases (in reality: fetch from MONDO, Disease Ontology)
    diseases = [
        {"id": "MONDO:0005015", "name": "Diabetes Mellitus"},
        {"id": "MONDO:0005148", "name": "Type 2 Diabetes"},
        {"id": "MONDO:0005044", "name": "Hypertension"},
        {"id": "MONDO:0004994", "name": "Cardiovascular Disease"},
        {"id": "MONDO:0005267", "name": "Heart Failure"},
        {"id": "MONDO:0011382", "name": "Obesity"},
        {"id": "MONDO:0005180", "name": "Parkinsons Disease"},
        {"id": "MONDO:0004975", "name": "Alzheimers Disease"},
        {"id": "MONDO:0008383", "name": "Rheumatoid Arthritis"},
        {"id": "MONDO:0005290", "name": "Inflammatory Bowel Disease"},
        {"id": "MONDO:0007254", "name": "Breast Cancer"},
        {"id": "MONDO:0008903", "name": "Lung Cancer"},
        {"id": "MONDO:0009061", "name": "Hypothyroidism"},
        {"id": "MONDO:0021166", "name": "Hyperlipidemia"},
        {"id": "MONDO:0002492", "name": "Thrombosis"},
    ]

    # Known associations (positive training examples)
    # In reality: fetch from clinical trials, DrugBank, literature
    known_associations = [
        # Diabetes drugs
        {"drug_id": "CHEMBL1200958", "disease_id": "MONDO:0005148", "confidence": 0.95},  # Metformin - T2D
        {"drug_id": "CHEMBL1201194", "disease_id": "MONDO:0005148", "confidence": 0.90},  # Glipizide - T2D
        {"drug_id": "CHEMBL1200766", "disease_id": "MONDO:0005148", "confidence": 0.88},  # Sitagliptin - T2D
        {"drug_id": "CHEMBL278020", "disease_id": "MONDO:0005148", "confidence": 0.87},  # Canagliflozin - T2D
        {"drug_id": "CHEMBL597", "disease_id": "MONDO:0005015", "confidence": 0.98},  # Insulin - Diabetes

        # Cardiovascular drugs
        {"drug_id": "CHEMBL1656", "disease_id": "MONDO:0021166", "confidence": 0.92},  # Simvastatin - Hyperlipidemia
        {"drug_id": "CHEMBL1433", "disease_id": "MONDO:0021166", "confidence": 0.93},  # Atorvastatin - Hyperlipidemia
        {"drug_id": "CHEMBL941", "disease_id": "MONDO:0005044", "confidence": 0.91},  # Lisinopril - Hypertension
        {"drug_id": "CHEMBL1566", "disease_id": "MONDO:0005044", "confidence": 0.90},  # Losartan - Hypertension
        {"drug_id": "CHEMBL835", "disease_id": "MONDO:0005044", "confidence": 0.89},  # Amlodipine - Hypertension
        {"drug_id": "CHEMBL88", "disease_id": "MONDO:0002492", "confidence": 0.94},  # Warfarin - Thrombosis

        # Pain/Inflammation
        {"drug_id": "CHEMBL25", "disease_id": "MONDO:0008383", "confidence": 0.75},  # Aspirin - RA
        {"drug_id": "CHEMBL521", "disease_id": "MONDO:0008383", "confidence": 0.78},  # Ibuprofen - RA
        {"drug_id": "CHEMBL220492", "disease_id": "MONDO:0008383", "confidence": 0.92},  # Adalimumab - RA
        {"drug_id": "CHEMBL220492", "disease_id": "MONDO:0005290", "confidence": 0.90},  # Adalimumab - IBD

        # Other
        {"drug_id": "CHEMBL1096999", "disease_id": "MONDO:0009061", "confidence": 0.96},  # Levothyroxine - Hypothyroidism

        # Cross-indications (repurposing examples)
        {"drug_id": "CHEMBL1200958", "disease_id": "MONDO:0011382", "confidence": 0.65},  # Metformin - Obesity
        {"drug_id": "CHEMBL1656", "disease_id": "MONDO:0004994", "confidence": 0.80},  # Simvastatin - CVD
        {"drug_id": "CHEMBL25", "disease_id": "MONDO:0004994", "confidence": 0.72},  # Aspirin - CVD prevention
    ]

    return drugs, diseases, known_associations


def build_knowledge_graph(
    drugs: List[Dict],
    diseases: List[Dict],
    associations: List[Dict],
) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str], List[Tuple[str, str, float]]]:
    """Build knowledge graph from entities and relationships.

    Returns:
        edge_index: Tensor of shape [2, num_edges]
        node_to_idx: Mapping from node ID to index
        idx_to_node: Mapping from index to node ID
        labeled_pairs: List of (drug_id, disease_id, label) for training
    """

    # Create node mappings
    nodes = []
    node_to_idx = {}

    # Add drugs
    for drug in drugs:
        drug_id = drug["id"]
        nodes.append(drug_id)
        node_to_idx[drug_id] = len(node_to_idx)

    # Add diseases
    for disease in diseases:
        disease_id = disease["id"]
        nodes.append(disease_id)
        node_to_idx[disease_id] = len(node_to_idx)

    idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}

    # Build edges from associations
    edges = []
    labeled_pairs = []

    for assoc in associations:
        drug_id = assoc["drug_id"]
        disease_id = assoc["disease_id"]
        confidence = assoc["confidence"]

        if drug_id in node_to_idx and disease_id in node_to_idx:
            drug_idx = node_to_idx[drug_id]
            disease_idx = node_to_idx[disease_id]

            # Add bidirectional edges
            edges.append([drug_idx, disease_idx])
            edges.append([disease_idx, drug_idx])

            # Store as positive training example
            labeled_pairs.append((drug_id, disease_id, 1.0))

    # Add some drug-drug similarity edges (same type)
    drug_types: Dict[str, List[str]] = {}
    for drug in drugs:
        drug_type = drug["type"]
        if drug_type not in drug_types:
            drug_types[drug_type] = []
        drug_types[drug_type].append(drug["id"])

    for drug_type, drug_ids in drug_types.items():
        # Connect drugs of same type
        for i, drug1 in enumerate(drug_ids):
            for drug2 in drug_ids[i+1:]:
                if drug1 in node_to_idx and drug2 in node_to_idx:
                    idx1 = node_to_idx[drug1]
                    idx2 = node_to_idx[drug2]
                    edges.append([idx1, idx2])
                    edges.append([idx2, idx1])

    # Generate negative examples (no known association)
    positive_pairs = {(d, dis) for d, dis, _ in labeled_pairs}
    all_drug_ids = [d["id"] for d in drugs]
    all_disease_ids = [d["id"] for d in diseases]

    # Sample negative pairs
    num_negatives = len(labeled_pairs) * 2  # 2x negatives
    negative_pairs = []
    attempts = 0
    max_attempts = num_negatives * 10

    while len(negative_pairs) < num_negatives and attempts < max_attempts:
        drug_id = random.choice(all_drug_ids)
        disease_id = random.choice(all_disease_ids)

        if (drug_id, disease_id) not in positive_pairs:
            negative_pairs.append((drug_id, disease_id, 0.0))
            positive_pairs.add((drug_id, disease_id))  # Don't sample again

        attempts += 1

    labeled_pairs.extend(negative_pairs)

    # Convert edges to tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    print(f"Built knowledge graph:")
    print(f"  Nodes: {len(nodes)} ({len(drugs)} drugs + {len(diseases)} diseases)")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Positive examples: {len([p for p in labeled_pairs if p[2] == 1.0])}")
    print(f"  Negative examples: {len([p for p in labeled_pairs if p[2] == 0.0])}")

    return edge_index, node_to_idx, idx_to_node, labeled_pairs


def save_data(
    edge_index: torch.Tensor,
    node_to_idx: Dict[str, int],
    idx_to_node: Dict[int, str],
    labeled_pairs: List[Tuple[str, str, float]],
    output_dir: Path,
) -> None:
    """Save prepared data for training."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save graph structure
    torch.save({
        "edge_index": edge_index,
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "num_nodes": len(node_to_idx),
    }, output_dir / "knowledge_graph.pt")

    # Save training pairs
    with open(output_dir / "training_pairs.json", "w") as f:
        json.dump([
            {"drug_id": d, "disease_id": dis, "label": float(label)}
            for d, dis, label in labeled_pairs
        ], f, indent=2)

    print(f"\nSaved data to {output_dir}:")
    print(f"  - knowledge_graph.pt")
    print(f"  - training_pairs.json")


def main():
    """Main function."""

    print("=" * 60)
    print("Preparing GNN Training Data")
    print("=" * 60)
    print()

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Generate sample data
    print("Generating sample biomedical data...")
    drugs, diseases, associations = generate_sample_data()

    # Build knowledge graph
    print("\nBuilding knowledge graph...")
    edge_index, node_to_idx, idx_to_node, labeled_pairs = build_knowledge_graph(
        drugs, diseases, associations
    )

    # Save data
    output_dir = Path("data/graph")
    save_data(edge_index, node_to_idx, idx_to_node, labeled_pairs, output_dir)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run training: python scripts/train_gnn.py")
    print("  2. Check results in data/models/")


if __name__ == "__main__":
    main()
