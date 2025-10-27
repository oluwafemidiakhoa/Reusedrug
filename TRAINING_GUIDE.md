# GNN Model Training Guide

## Prerequisites

The GNN training scripts are ready, but require PyTorch installation. Due to the large size (~2GB), you should install when you have a stable internet connection.

## Installation

### Option 1: CPU Version (Faster Download)

```bash
cd python-backend
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install scikit-learn
```

### Option 2: GPU Version (Recommended for Training)

```bash
cd python-backend
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install scikit-learn
```

### Option 3: Install All ML Dependencies

```bash
cd python-backend
pip install -r requirements.txt
```

**Note**: This installs ~2GB of packages and may take 10-30 minutes depending on your connection.

---

## Training Workflow

Once dependencies are installed, follow these steps:

### Step 1: Prepare Training Data

```bash
cd python-backend
python scripts/prepare_training_data.py
```

**What this does**:
- Creates sample knowledge graph with drugs, diseases, and associations
- Generates positive examples (known drug-disease pairs)
- Generates negative examples (random non-associations)
- Splits data into train/val/test sets
- Saves to `data/graph/`

**Output**:
```
data/graph/
├── knowledge_graph.pt      # Graph structure (nodes, edges)
└── training_pairs.json     # Labeled training examples
```

---

### Step 2: Train the Model

```bash
python scripts/train_gnn.py
```

**What this does**:
- Loads knowledge graph and training pairs
- Initializes GNN model (Graph Convolutional Network)
- Trains using binary cross-entropy loss
- Validates after each epoch
- Saves best model based on AUROC
- Evaluates on held-out test set

**Training Progress**:
```
Epoch 1/100
  Train Loss: 0.6234, Acc: 0.6500
  Val   Loss: 0.5892, Acc: 0.7100, AUROC: 0.7234, AUPRC: 0.6892
  ✓ New best model saved! (AUROC: 0.7234)

Epoch 2/100
  Train Loss: 0.5123, Acc: 0.7300
  Val   Loss: 0.4821, Acc: 0.7600, AUROC: 0.7945, AUPRC: 0.7523
  ✓ New best model saved! (AUROC: 0.7945)

...

Training complete! Evaluating on test set...
Test Set Results:
  Loss:     0.4234
  Accuracy: 0.7850
  AUROC:    0.8234
  AUPRC:    0.7892
```

**Output**:
```
data/models/
├── gnn_best.pt       # Best model from training
└── gnn_final.pt      # Final model after all epochs
```

---

### Step 3: Test Predictions

```bash
python scripts/test_predictions.py
```

(This script will be created after training to test the model)

---

## Using the Trained Model

### In Python Code

```python
from pathlib import Path
from app.ml.models.gnn_predictor import GNNPredictor

# Load trained model
predictor = GNNPredictor()
predictor.load(Path("data/models/gnn_best.pt"))

# Make prediction
result = predictor.predict(
    drug_id="CHEMBL1200958",  # Metformin
    disease_id="MONDO:0005148"  # Type 2 Diabetes
)

print(f"Prediction Score: {result.score:.4f}")
print(f"Confidence: [{result.confidence_low:.4f}, {result.confidence_high:.4f}]")
```

### Via API

Once trained, update the ML router to load the model:

```python
# In app/routers/ml.py

# Load trained model on startup
from app.ml.models.gnn_predictor import GNNPredictor
from pathlib import Path

_trained_predictor = None

@router.on_event("startup")
async def load_trained_model():
    global _trained_predictor
    model_path = Path("data/models/gnn_best.pt")
    if model_path.exists():
        _trained_predictor = GNNPredictor()
        _trained_predictor.load(model_path)
        logger.info("Loaded trained GNN model")

@router.post("/v1/ml/predict")
async def predict_association(request: PredictionRequest):
    if _trained_predictor is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")

    return _trained_predictor.predict(request.drug_id, request.disease_id)
```

Then use the API:

```bash
curl -X POST http://localhost:8080/v1/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "drug_id": "CHEMBL1200958",
    "disease_id": "MONDO:0005148"
  }'
```

---

## Training with Real Data

The sample training script uses synthetic data. To train on real biomedical data:

### 1. Collect Drug-Disease Associations

```python
# scripts/collect_real_data.py

import asyncio
from app.services.opentargets import fetch_known_drugs
from app.services.clients.translator import fetch_disease_treatments

async def collect_associations():
    """Collect real drug-disease associations."""

    associations = []

    # Fetch from Open Targets
    for disease_id in disease_list:
        drugs, _ = await fetch_known_drugs(disease_id)
        for drug in drugs:
            associations.append({
                "drug_id": drug["drug_id"],
                "disease_id": disease_id,
                "source": "opentargets",
                "confidence": 1.0  # Positive example
            })

    # Fetch from Translator
    for disease_id in disease_list:
        drugs, _ = await fetch_disease_treatments([disease_id])
        for drug in drugs:
            associations.append({
                "drug_id": drug["drug_id"],
                "disease_id": disease_id,
                "source": "translator",
                "confidence": 1.0
            })

    return associations
```

### 2. Build Knowledge Graph

```python
# Add more entity types
# - Proteins/genes (from STRING-DB, Translator)
# - Pathways (from Reactome, KEGG)
# - Side effects (from SIDER)
# - Chemical structures (from PubChem)

# Add more edge types
# - Drug-target interactions
# - Protein-protein interactions
# - Disease-gene associations
# - Pathway memberships
```

### 3. Increase Model Capacity

```python
# In .env
GNN_EMBEDDING_DIM=256    # Increase from 128
GNN_HIDDEN_DIM=512       # Increase from 256
GNN_NUM_LAYERS=4         # Increase from 3
ML_MAX_EPOCHS=200        # More training
```

---

## Configuration

All training parameters in `.env`:

```bash
# Model Architecture
GNN_EMBEDDING_DIM=128        # Node embedding dimensions
GNN_HIDDEN_DIM=256           # Hidden layer size
GNN_NUM_LAYERS=3             # Number of GCN layers
GNN_DROPOUT=0.2              # Dropout rate

# Training
ML_BATCH_SIZE=32             # Batch size
ML_LEARNING_RATE=0.001       # Learning rate
ML_MAX_EPOCHS=100            # Maximum epochs
ML_EARLY_STOPPING_PATIENCE=10  # Stop if no improvement

# Data
ML_MODEL_DIR=data/models     # Where to save models
```

---

## Troubleshooting

### Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
ML_BATCH_SIZE=16

# Or use CPU
# (Slower but more memory)
```

### Poor Performance (Low AUROC)

**Solutions**:
1. **More data**: Collect more drug-disease associations
2. **Better features**: Add more node/edge types to graph
3. **Larger model**: Increase embedding dimensions
4. **More epochs**: Increase `ML_MAX_EPOCHS`
5. **Learning rate**: Try 0.0001 or 0.01

### Training Too Slow

**Solutions**:
1. **Use GPU**: Install CUDA version of PyTorch
2. **Reduce graph size**: Sample smaller subgraph
3. **Batch processing**: Increase `ML_BATCH_SIZE`

---

## Expected Performance

With the sample data (~15 drugs, ~15 diseases, ~40 associations):

| Metric | Expected Value |
|--------|----------------|
| Train Accuracy | 75-85% |
| Val Accuracy | 70-80% |
| Test Accuracy | 70-80% |
| AUROC | 0.75-0.85 |
| AUPRC | 0.70-0.80 |

With real data (1000+ drugs, 100+ diseases, 5000+ associations):

| Metric | Expected Value |
|--------|----------------|
| Train Accuracy | 85-95% |
| Val Accuracy | 80-90% |
| Test Accuracy | 80-90% |
| AUROC | 0.85-0.95 |
| AUPRC | 0.80-0.92 |

---

## Next Steps After Training

1. **Integrate with Ranking**: Combine GNN score with evidence-based scoring
2. **Uncertainty Calibration**: Calibrate confidence intervals
3. **Explainability**: Add attention mechanisms for interpretability
4. **Continuous Learning**: Retrain periodically with new data
5. **A/B Testing**: Compare GNN predictions vs baseline

---

## Current Status

✅ **Training scripts created**
- `scripts/prepare_training_data.py`
- `scripts/train_gnn.py`

⏳ **Pending**: Install PyTorch dependencies
```bash
pip install torch torch-geometric scikit-learn
```

⏳ **After installation**: Run training
```bash
python scripts/prepare_training_data.py
python scripts/train_gnn.py
```

---

## Support

- **Installation issues**: Check [PyTorch website](https://pytorch.org/get-started/locally/)
- **Training questions**: See [ML_GUIDE.md](python-backend/ML_GUIDE.md)
- **API integration**: See [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md)

