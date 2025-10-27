"""Train GNN model for drug-disease link prediction."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from app.ml.config import MLConfig
from app.ml.models.gnn_predictor import GNNPredictor


def load_training_data(data_dir: Path) -> Tuple[torch.Tensor, dict, dict, List[dict]]:
    """Load prepared training data."""

    # Load graph
    graph_data = torch.load(data_dir / "knowledge_graph.pt")
    edge_index = graph_data["edge_index"]
    node_to_idx = graph_data["node_to_idx"]
    idx_to_node = graph_data["idx_to_node"]

    # Load training pairs
    with open(data_dir / "training_pairs.json") as f:
        training_pairs = json.load(f)

    return edge_index, node_to_idx, idx_to_node, training_pairs


def prepare_splits(
    training_pairs: List[dict],
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split data into train/val/test sets."""

    # Separate positive and negative examples
    positives = [p for p in training_pairs if p["label"] == 1.0]
    negatives = [p for p in training_pairs if p["label"] == 0.0]

    # Split positives
    pos_train_val, pos_test = train_test_split(
        positives, test_size=test_size, random_state=42
    )
    pos_train, pos_val = train_test_split(
        pos_train_val, test_size=val_size/(1-test_size), random_state=42
    )

    # Split negatives
    neg_train_val, neg_test = train_test_split(
        negatives, test_size=test_size, random_state=42
    )
    neg_train, neg_val = train_test_split(
        neg_train_val, test_size=val_size/(1-test_size), random_state=42
    )

    # Combine
    train_data = pos_train + neg_train
    val_data = pos_val + neg_val
    test_data = pos_test + neg_test

    # Shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    print(f"\nData splits:")
    print(f"  Train: {len(train_data)} ({len(pos_train)} pos, {len(neg_train)} neg)")
    print(f"  Val:   {len(val_data)} ({len(pos_val)} pos, {len(neg_val)} neg)")
    print(f"  Test:  {len(test_data)} ({len(pos_test)} pos, {len(neg_test)} neg)")

    return train_data, val_data, test_data


def compute_loss(
    predictor: GNNPredictor,
    pairs: List[dict],
    node_to_idx: dict,
    device: str,
) -> Tuple[torch.Tensor, float, float]:
    """Compute loss for a batch of pairs."""

    predictor.encoder.train()
    predictor.link_predictor.train()

    # Get all node embeddings
    all_node_ids = torch.arange(len(node_to_idx), device=device)
    embeddings = predictor.encoder(all_node_ids, predictor.edge_index)

    # Prepare batch
    drug_indices = []
    disease_indices = []
    labels = []

    for pair in pairs:
        drug_idx = node_to_idx.get(pair["drug_id"])
        disease_idx = node_to_idx.get(pair["disease_id"])

        if drug_idx is not None and disease_idx is not None:
            drug_indices.append(drug_idx)
            disease_indices.append(disease_idx)
            labels.append(pair["label"])

    if not drug_indices:
        return torch.tensor(0.0, device=device), 0.0, 0.0

    drug_indices = torch.tensor(drug_indices, device=device)
    disease_indices = torch.tensor(disease_indices, device=device)
    labels = torch.tensor(labels, dtype=torch.float32, device=device)

    # Get embeddings
    drug_emb = embeddings[drug_indices]
    disease_emb = embeddings[disease_indices]

    # Predict
    predictions = predictor.link_predictor(drug_emb, disease_emb).squeeze()

    # Compute loss
    loss = F.binary_cross_entropy(predictions, labels)

    # Compute accuracy
    pred_labels = (predictions > 0.5).float()
    accuracy = (pred_labels == labels).float().mean().item()

    return loss, accuracy, predictions.detach().cpu()


def evaluate(
    predictor: GNNPredictor,
    pairs: List[dict],
    node_to_idx: dict,
    device: str,
) -> dict:
    """Evaluate model on a dataset."""

    predictor.encoder.eval()
    predictor.link_predictor.eval()

    with torch.no_grad():
        # Get all node embeddings
        all_node_ids = torch.arange(len(node_to_idx), device=device)
        embeddings = predictor.encoder(all_node_ids, predictor.edge_index)

        # Prepare batch
        drug_indices = []
        disease_indices = []
        labels = []

        for pair in pairs:
            drug_idx = node_to_idx.get(pair["drug_id"])
            disease_idx = node_to_idx.get(pair["disease_id"])

            if drug_idx is not None and disease_idx is not None:
                drug_indices.append(drug_idx)
                disease_indices.append(disease_idx)
                labels.append(pair["label"])

        drug_indices = torch.tensor(drug_indices, device=device)
        disease_indices = torch.tensor(disease_indices, device=device)
        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)

        # Get embeddings
        drug_emb = embeddings[drug_indices]
        disease_emb = embeddings[disease_indices]

        # Predict
        predictions = predictor.link_predictor(drug_emb, disease_emb).squeeze()

        # Compute metrics
        loss = F.binary_cross_entropy(predictions, labels_tensor).item()

        pred_labels = (predictions > 0.5).float()
        accuracy = (pred_labels == labels_tensor).float().mean().item()

        # Convert to numpy for sklearn metrics
        predictions_np = predictions.cpu().numpy()
        labels_np = torch.tensor(labels).numpy()

        try:
            auroc = roc_auc_score(labels_np, predictions_np)
            auprc = average_precision_score(labels_np, predictions_np)
        except:
            auroc = 0.0
            auprc = 0.0

    return {
        "loss": loss,
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
    }


def train_epoch(
    predictor: GNNPredictor,
    train_data: List[dict],
    node_to_idx: dict,
    optimizer: optim.Optimizer,
    batch_size: int,
    device: str,
) -> dict:
    """Train for one epoch."""

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    # Shuffle data
    random.shuffle(train_data)

    # Train in batches
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]

        optimizer.zero_grad()
        loss, accuracy, _ = compute_loss(predictor, batch, node_to_idx, device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_accuracy / num_batches,
    }


def main():
    """Main training function."""

    print("=" * 60)
    print("Training GNN Model for Drug Repurposing")
    print("=" * 60)
    print()

    # Set random seeds
    random.seed(42)
    torch.manual_seed(42)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\nLoading training data...")
    data_dir = Path("data/graph")
    edge_index, node_to_idx, idx_to_node, training_pairs = load_training_data(data_dir)

    # Prepare splits
    train_data, val_data, test_data = prepare_splits(training_pairs)

    # Initialize model
    print("\nInitializing GNN model...")
    predictor = GNNPredictor(device=device)
    predictor.initialize_model(
        num_nodes=len(node_to_idx),
        edge_index=edge_index,
        node_to_idx=node_to_idx,
    )

    print(f"  Embedding dim: {MLConfig.GNN_EMBEDDING_DIM}")
    print(f"  Hidden dim: {MLConfig.GNN_HIDDEN_DIM}")
    print(f"  Num layers: {MLConfig.GNN_NUM_LAYERS}")
    print(f"  Dropout: {MLConfig.GNN_DROPOUT}")

    # Optimizer
    optimizer = optim.Adam(
        list(predictor.encoder.parameters()) +
        list(predictor.link_predictor.parameters()),
        lr=MLConfig.LEARNING_RATE,
    )

    print(f"\nTraining parameters:")
    print(f"  Learning rate: {MLConfig.LEARNING_RATE}")
    print(f"  Batch size: {MLConfig.BATCH_SIZE}")
    print(f"  Max epochs: {MLConfig.MAX_EPOCHS}")
    print(f"  Early stopping patience: {MLConfig.EARLY_STOPPING_PATIENCE}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_auroc = 0.0
    patience_counter = 0
    best_model_path = Path("data/models/gnn_best.pt")

    for epoch in range(MLConfig.MAX_EPOCHS):
        # Train
        train_metrics = train_epoch(
            predictor, train_data, node_to_idx, optimizer, MLConfig.BATCH_SIZE, device
        )

        # Validate
        val_metrics = evaluate(predictor, val_data, node_to_idx, device)

        print(f"\nEpoch {epoch+1}/{MLConfig.MAX_EPOCHS}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUROC: {val_metrics['auroc']:.4f}, AUPRC: {val_metrics['auprc']:.4f}")

        # Save best model
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            predictor.save(best_model_path)
            print(f"  [BEST] New best model saved! (AUROC: {best_val_auroc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= MLConfig.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping after {epoch+1} epochs (patience={MLConfig.EARLY_STOPPING_PATIENCE})")
            break

    # Load best model and evaluate on test set
    print("\n" + "=" * 60)
    print("Training complete! Evaluating on test set...")
    print("=" * 60)

    predictor.load(best_model_path)
    test_metrics = evaluate(predictor, test_data, node_to_idx, device)

    print(f"\nTest Set Results:")
    print(f"  Loss:     {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  AUROC:    {test_metrics['auroc']:.4f}")
    print(f"  AUPRC:    {test_metrics['auprc']:.4f}")

    # Save final model
    final_model_path = Path("data/models/gnn_final.pt")
    predictor.save(final_model_path)

    print(f"\nModels saved:")
    print(f"  Best: {best_model_path}")
    print(f"  Final: {final_model_path}")

    # Test predictions on sample pairs
    print("\n" + "=" * 60)
    print("Sample Predictions:")
    print("=" * 60)

    sample_pairs = test_data[:5]
    for pair in sample_pairs:
        result = predictor.predict(pair["drug_id"], pair["disease_id"])
        true_label = "Positive" if pair["label"] == 1.0 else "Negative"
        print(f"\n  {pair['drug_id']} + {pair['disease_id']}")
        print(f"    True:      {true_label}")
        print(f"    Predicted: {result.score:.4f}")
        print(f"    Confidence: [{result.confidence_low:.4f}, {result.confidence_high:.4f}]")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Test predictions: python scripts/test_predictions.py")
    print("  2. Use in API: The model is now available via /v1/ml/predict")


if __name__ == "__main__":
    main()
