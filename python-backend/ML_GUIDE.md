# Machine Learning Features - Usage Guide

## Overview

The ML module adds AI-powered prediction capabilities to the drug repurposing platform:

- **Similarity Search**: Find similar drugs/diseases using semantic embeddings
- **GNN Predictions**: Graph Neural Network-based drug-disease association prediction (requires training)
- **Embeddings**: Sentence transformer embeddings for biomedical entities

## Installation

### Install ML Dependencies

```bash
cd python-backend
pip install torch torch-geometric scikit-learn transformers sentence-transformers mlflow
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## API Endpoints

### 1. Find Similar Drugs

**Endpoint**: `POST /v1/ml/similar/drugs`

Find drugs similar to a query drug based on semantic embeddings.

**Request**:
```json
{
  "entity_type": "drug",
  "entity_id": "CHEMBL123",
  "entity_name": "Aspirin",
  "candidates": [
    ["CHEMBL456", "Ibuprofen"],
    ["CHEMBL789", "Acetaminophen"],
    ["CHEMBL012", "Naproxen"]
  ],
  "top_k": 10
}
```

**Response**:
```json
{
  "query_entity_id": "CHEMBL123",
  "query_entity_name": "Aspirin",
  "entity_type": "drug",
  "results": [
    {
      "entity_id": "CHEMBL456",
      "similarity_score": 0.89,
      "rank": 1
    },
    {
      "entity_id": "CHEMBL012",
      "similarity_score": 0.82,
      "rank": 2
    }
  ]
}
```

**Example (Python)**:
```python
import httpx

async def find_similar_drugs():
    url = "http://localhost:8080/v1/ml/similar/drugs"
    payload = {
        "entity_type": "drug",
        "entity_id": "CHEMBL123",
        "entity_name": "Aspirin",
        "candidates": [
            ["CHEMBL456", "Ibuprofen"],
            ["CHEMBL789", "Acetaminophen"],
        ],
        "top_k": 5
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        return response.json()
```

**Example (cURL)**:
```bash
curl -X POST http://localhost:8080/v1/ml/similar/drugs \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "drug",
    "entity_id": "CHEMBL123",
    "entity_name": "Aspirin",
    "candidates": [["CHEMBL456", "Ibuprofen"]],
    "top_k": 5
  }'
```

---

### 2. Find Similar Diseases

**Endpoint**: `POST /v1/ml/similar/diseases`

Find diseases similar to a query disease based on semantic embeddings.

**Request**:
```json
{
  "entity_type": "disease",
  "entity_id": "MONDO:0005148",
  "entity_name": "Type 2 Diabetes",
  "candidates": [
    ["MONDO:0005147", "Type 1 Diabetes"],
    ["MONDO:0011382", "Obesity"],
    ["MONDO:0005044", "Hypertension"]
  ],
  "top_k": 5
}
```

**Response**: Same structure as similar drugs endpoint

---

### 3. ML Prediction (Placeholder)

**Endpoint**: `POST /v1/ml/predict`

Predict drug-disease association score using ML models.

**Note**: This endpoint currently returns baseline predictions. To enable GNN predictions, you need to train a model first (see Training section below).

**Request**:
```json
{
  "drug_id": "CHEMBL123",
  "disease_id": "MONDO:0005148",
  "drug_name": "Aspirin",
  "disease_name": "Type 2 Diabetes"
}
```

**Response**:
```json
{
  "drug_id": "CHEMBL123",
  "disease_id": "MONDO:0005148",
  "score": 0.5,
  "confidence_low": 0.3,
  "confidence_high": 0.7,
  "model_name": "baseline",
  "features_used": ["none"],
  "metadata": {
    "status": "not_implemented",
    "message": "ML prediction requires trained model."
  }
}
```

---

### 4. Batch Prediction

**Endpoint**: `POST /v1/ml/predict/batch`

Predict multiple drug-disease pairs in one request.

**Request**:
```json
{
  "pairs": [
    ["CHEMBL123", "MONDO:0005148"],
    ["CHEMBL456", "MONDO:0005148"],
    ["CHEMBL789", "MONDO:0011382"]
  ],
  "top_k": 10
}
```

**Response**: Array of prediction results

---

### 5. ML Metadata

**Endpoint**: `GET /v1/ml/metadata`

Get information about available ML models and features.

**Response**:
```json
{
  "models_available": ["embeddings"],
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "features": {
    "similarity_search": true,
    "gnn_prediction": false,
    "batch_prediction": true
  }
}
```

---

## Python SDK Usage

### Embedding Service

```python
from app.ml.embeddings.embedding_service import EmbeddingService

# Initialize service
embedding_service = EmbeddingService()

# Get drug embedding
drug_emb = embedding_service.get_drug_embedding(
    drug_id="CHEMBL123",
    drug_name="Aspirin",
    description="Non-steroidal anti-inflammatory drug"
)

# Get disease embedding
disease_emb = embedding_service.get_disease_embedding(
    disease_id="MONDO:0005148",
    disease_name="Type 2 Diabetes",
    description="Metabolic disorder characterized by high blood sugar"
)

# Compute similarity
similarity = embedding_service.compute_similarity(drug_emb, disease_emb)
print(f"Similarity: {similarity:.4f}")

# Find similar drugs
similar_drugs = embedding_service.find_similar_drugs(
    query_drug_id="CHEMBL123",
    query_drug_name="Aspirin",
    candidate_drugs=[
        ("CHEMBL456", "Ibuprofen"),
        ("CHEMBL789", "Acetaminophen"),
    ],
    top_k=5
)

for drug_id, score in similar_drugs:
    print(f"{drug_id}: {score:.4f}")
```

### GNN Predictor (After Training)

```python
from app.ml.models.gnn_predictor import GNNPredictor
import torch

# Initialize predictor
predictor = GNNPredictor()

# Load trained model
predictor.load(Path("data/models/gnn_trained.pt"))

# Make prediction
result = predictor.predict(
    drug_id="CHEMBL123",
    disease_id="MONDO:0005148"
)

print(f"Prediction Score: {result.score:.4f}")
print(f"Confidence Interval: [{result.confidence_low:.4f}, {result.confidence_high:.4f}]")
```

---

## Training GNN Models (Advanced)

To enable full GNN predictions, you need to train a model on your knowledge graph.

### Step 1: Prepare Training Data

Create a script to build your knowledge graph:

```python
# scripts/prepare_graph_data.py
import torch
from pathlib import Path

def prepare_knowledge_graph():
    """Build knowledge graph from your data sources."""

    # Collect nodes (drugs, diseases, proteins, etc.)
    nodes = []
    node_to_idx = {}

    # Example: Add drugs
    drugs = fetch_all_drugs()  # Your data source
    for i, drug in enumerate(drugs):
        node_id = drug["drug_id"]
        nodes.append(node_id)
        node_to_idx[node_id] = i

    # Example: Add diseases
    diseases = fetch_all_diseases()  # Your data source
    for i, disease in enumerate(diseases):
        node_id = disease["disease_id"]
        nodes.append(node_id)
        node_to_idx[node_id] = len(node_to_idx)

    # Collect edges
    edges = []

    # Example: Drug-disease associations from clinical trials
    associations = fetch_known_associations()
    for assoc in associations:
        drug_idx = node_to_idx[assoc["drug_id"]]
        disease_idx = node_to_idx[assoc["disease_id"]]
        edges.append([drug_idx, disease_idx])
        edges.append([disease_idx, drug_idx])  # Undirected

    # Create edge index tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # Save
    output_dir = Path("data/graph")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "edge_index": edge_index,
        "node_to_idx": node_to_idx,
        "nodes": nodes,
    }, output_dir / "knowledge_graph.pt")

    print(f"Saved graph with {len(nodes)} nodes and {edge_index.shape[1]} edges")

if __name__ == "__main__":
    prepare_knowledge_graph()
```

### Step 2: Train GNN Model

```python
# scripts/train_gnn.py
import torch
import torch.optim as optim
from pathlib import Path
from app.ml.models.gnn_predictor import GNNPredictor
from app.ml.config import MLConfig

def train_gnn_model():
    # Load graph data
    graph_data = torch.load("data/graph/knowledge_graph.pt")
    edge_index = graph_data["edge_index"]
    node_to_idx = graph_data["node_to_idx"]
    num_nodes = len(node_to_idx)

    # Initialize predictor
    predictor = GNNPredictor()
    predictor.initialize_model(num_nodes, edge_index, node_to_idx)

    # Prepare training data (positive and negative samples)
    train_pairs = prepare_training_pairs()  # Your implementation

    # Training loop
    optimizer = optim.Adam(
        list(predictor.encoder.parameters()) +
        list(predictor.link_predictor.parameters()),
        lr=MLConfig.LEARNING_RATE
    )

    for epoch in range(MLConfig.MAX_EPOCHS):
        predictor.encoder.train()
        predictor.link_predictor.train()

        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            loss = compute_loss(predictor, batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{MLConfig.MAX_EPOCHS}, Loss: {total_loss:.4f}")

    # Save trained model
    output_path = Path("data/models/gnn_trained.pt")
    predictor.save(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train_gnn_model()
```

---

## Configuration

All ML settings are in [.env.example](python-backend/.env.example):

```bash
# Machine Learning Settings
ML_MODEL_DIR=data/models
ML_EMBEDDINGS_DIR=data/embeddings
MLFLOW_ENABLED=false
GNN_EMBEDDING_DIM=128
GNN_HIDDEN_DIM=256
GNN_NUM_LAYERS=3
GNN_DROPOUT=0.2
ML_BATCH_SIZE=32
ML_LEARNING_RATE=0.001
ML_MAX_EPOCHS=100
```

---

## Testing

Run ML-specific tests:

```bash
cd python-backend
pytest tests/test_ml.py -v
```

---

## Performance Tips

1. **GPU Acceleration**: Install PyTorch with CUDA support for faster training
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Caching**: Embeddings are automatically cached. Save/load cache:
   ```python
   embedding_service.save_embeddings(Path("data/embeddings/cache.npz"))
   embedding_service.load_embeddings(Path("data/embeddings/cache.npz"))
   ```

3. **Batch Processing**: Use batch endpoints for multiple predictions

---

## Next Steps

1. ✅ **Similarity search is ready to use** - Try the `/v1/ml/similar/*` endpoints
2. **Train GNN model** - Follow training guide above to enable ML predictions
3. **Integrate with ranking** - Combine ML scores with existing evidence-based scoring
4. **Add more models** - Implement ensemble predictions (XGBoost, transformers)

## Support

For issues or questions, see the [ADVANCED_ROADMAP.md](../ADVANCED_ROADMAP.md) for the full ML development plan.
