"""Test trained GNN model."""

import sys
sys.path.insert(0, '.')

from pathlib import Path
from app.ml.models.gnn_predictor import GNNPredictor

def main():
    print("="*60)
    print("Testing Trained GNN Model")
    print("="*60)

    # Load model
    model_path = Path("data/models/gnn_best.pt")
    print(f"\nLoading model from: {model_path}")

    predictor = GNNPredictor()
    predictor.load(model_path)

    print(f"Model loaded successfully!")
    print(f"Number of nodes in graph: {len(predictor.node_to_idx)}")

    # Test predictions
    print("\n" + "="*60)
    print("Sample Predictions:")
    print("="*60)

    test_pairs = [
        ("CHEMBL1200958", "MONDO:0005148", "Metformin", "Type 2 Diabetes"),  # Should be high
        ("CHEMBL941", "MONDO:0005044", "Lisinopril", "Hypertension"),  # Should be high
        ("CHEMBL25", "MONDO:0005180", "Aspirin", "Parkinsons"),  # Should be low
    ]

    for drug_id, disease_id, drug_name, disease_name in test_pairs:
        result = predictor.predict(drug_id, disease_id)

        print(f"\n{drug_name} -> {disease_name}")
        print(f"  IDs: {drug_id} + {disease_id}")
        print(f"  Score: {result.score:.4f}")
        if result.confidence_low and result.confidence_high:
            print(f"  Confidence: [{result.confidence_low:.4f}, {result.confidence_high:.4f}]")
        print(f"  Model: {result.model_name}")

    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
    print("\nThe model is ready to use!")
    print("To use in API, start the backend:")
    print("  uvicorn app.main:app --reload --port 8080")

if __name__ == "__main__":
    main()
