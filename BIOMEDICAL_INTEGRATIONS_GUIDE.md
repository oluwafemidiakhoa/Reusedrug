# 🧬 Biomedical Data Integrations Guide

## Overview

Your drug repurposing app now has **world-class biomedical data integration** capabilities! Connect to 6 major biomedical databases to enrich your drug repurposing insights.

---

## 🌐 Integrated Data Sources

### 1. **UniProt** ✅ (No API Key Required)
- **What**: Protein sequences and functional information
- **Use For**: Understanding drug targets, protein functions
- **URL**: https://www.uniprot.org/
- **Status**: Active

### 2. **STRING** ✅ (No API Key Required)
- **What**: Protein-protein interaction networks
- **Use For**: Building drug target networks, interaction maps
- **URL**: https://string-db.org/
- **Status**: Active

### 3. **PubChem** ✅ (No API Key Required)
- **What**: Chemical compounds and bioactivity data
- **Use For**: Chemical properties, bioassays, structural info
- **URL**: https://pubchem.ncbi.nlm.nih.gov/
- **Status**: Active

### 4. **KEGG** ✅ (No API Key Required)
- **What**: Biological pathways and diseases
- **Use For**: Pathway analysis, disease mechanisms
- **URL**: https://www.kegg.jp/
- **Status**: Active

### 5. **DrugBank** 🔑 (API Key Required - Free for Academic Use)
- **What**: Comprehensive drug database
- **Use For**: Detailed drug information, mechanisms, targets
- **URL**: https://go.drugbank.com/
- **API Access**: https://go.drugbank.com/releases/latest
- **Status**: Optional

### 6. **DisGeNET** 🔑 (API Key Required - Free Registration)
- **What**: Gene-disease associations
- **Use For**: Finding disease-related genes, associations
- **URL**: https://www.disgenet.org/
- **API Access**: Register at https://www.disgenet.org/signup/
- **Status**: Optional

---

## 🚀 Quick Start

### Without API Keys (4 sources work immediately!)
```bash
# No setup needed! These endpoints work right away:

# Get protein info
curl http://localhost:8080/v1/biomedical/protein/info/P04637

# Get protein interactions
curl http://localhost:8080/v1/biomedical/protein/interactions/EGFR

# Get compound info from PubChem
curl http://localhost:8080/v1/biomedical/compound/pubchem/aspirin

# Search KEGG pathways
curl http://localhost:8080/v1/biomedical/pathway/disease/H00409
```

### With API Keys (Full Power!)
```bash
# Add to your .env file:
DRUGBANK_API_KEY=your_drugbank_key
DISGENET_API_KEY=your_disgenet_key

# Then use enhanced endpoints:
curl http://localhost:8080/v1/biomedical/enrich/drug/metformin
curl http://localhost:8080/v1/biomedical/enrich/disease/diabetes
```

---

## 📍 API Endpoints

### Drug Enrichment
```http
GET /v1/biomedical/enrich/drug/{drug_name}
```

**Example**:
```bash
curl http://localhost:8080/v1/biomedical/enrich/drug/metformin?chembl_id=CHEMBL1201
```

**Response**:
```json
{
  "drug_name": "metformin",
  "chembl_id": "CHEMBL1201",
  "chemical_properties": {
    "cid": "4091",
    "molecular_formula": "C4H11N5",
    "molecular_weight": "129.16",
    "smiles": "CN(C)C(=N)NC(=N)N",
    "iupac_name": "3-(diaminomethylidene)-1,1-dimethylguanidine"
  },
  "drugbank": {
    "drugbank_id": "DB00331",
    "mechanism_of_action": "Decreases hepatic glucose production...",
    "targets": ["Mitochondrial complex I", "AMP-activated protein kinase"]
  },
  "bioactivity": [...]
}
```

---

### Disease Enrichment
```http
GET /v1/biomedical/enrich/disease/{disease_name}
```

**Example**:
```bash
curl "http://localhost:8080/v1/biomedical/enrich/disease/diabetes?disease_id=MONDO:0005015"
```

**Response**:
```json
{
  "disease_name": "diabetes",
  "disease_id": "MONDO:0005015",
  "associated_genes": [
    {
      "gene_symbol": "INS",
      "score": 0.85,
      "evidence": "curated"
    }
  ],
  "protein_network": [...],
  "pathways": [...]
}
```

---

### Protein Information
```http
GET /v1/biomedical/protein/info/{protein_id}
```

**Example**:
```bash
curl http://localhost:8080/v1/biomedical/protein/info/P04637
```

**Response**:
```json
{
  "protein_id": "P04637",
  "name": "Cellular tumor antigen p53",
  "gene": "TP53",
  "organism": "Homo sapiens",
  "function": "Acts as a tumor suppressor in many tumor types...",
  "subcellular_location": ["Nucleus", "Cytoplasm"],
  "interactions": 350
}
```

---

### Protein-Protein Interactions
```http
GET /v1/biomedical/protein/interactions/{protein_name}
```

**Example**:
```bash
curl "http://localhost:8080/v1/biomedical/protein/interactions/EGFR?limit=10&min_score=700"
```

**Response**:
```json
[
  {
    "partner_protein": "GRB2",
    "score": 0.999,
    "interaction_type": "highest_confidence"
  },
  {
    "partner_protein": "SHC1",
    "score": 0.998,
    "interaction_type": "highest_confidence"
  }
]
```

---

### Protein Network Visualization
```http
GET /v1/biomedical/protein/network/image
```

**Example**:
```bash
curl "http://localhost:8080/v1/biomedical/protein/network/image?proteins=TP53&proteins=MDM2&proteins=CDKN1A"
```

**Response**:
```json
{
  "proteins": ["TP53", "MDM2", "CDKN1A"],
  "image_url": "https://string-db.org/api/image/network?identifiers=TP53%0dMDM2%0dCDKN1A"
}
```

---

### Chemical Compound Information
```http
GET /v1/biomedical/compound/pubchem/{compound_name}
```

**Example**:
```bash
curl http://localhost:8080/v1/biomedical/compound/pubchem/aspirin
```

---

### Bioactivity Assays
```http
GET /v1/biomedical/compound/bioactivity/{cid}
```

**Example**:
```bash
curl http://localhost:8080/v1/biomedical/compound/bioactivity/2244
```

---

### KEGG Pathways
```http
GET /v1/biomedical/pathway/disease/{disease_id}
GET /v1/biomedical/pathway/info/{pathway_id}
```

**Example**:
```bash
# Get pathways for Type 2 Diabetes
curl http://localhost:8080/v1/biomedical/pathway/disease/H00409

# Get specific pathway info
curl http://localhost:8080/v1/biomedical/pathway/info/hsa04930
```

---

### Build Drug Target Network
```http
POST /v1/biomedical/network/build
```

**Example**:
```bash
curl -X POST http://localhost:8080/v1/biomedical/network/build \
  -H "Content-Type: application/json" \
  -d '{"targets": ["EGFR", "KRAS", "TP53"], "include_interactions": true}'
```

**Response**:
```json
{
  "nodes": [
    {
      "id": "EGFR",
      "type": "protein",
      "data": {...}
    }
  ],
  "edges": [
    {
      "source": "EGFR",
      "target": "GRB2",
      "score": 0.999,
      "type": "highest_confidence"
    }
  ]
}
```

---

## 🎯 Use Cases

### Use Case 1: Enrich Drug Candidates
```python
# For each drug candidate in your results, fetch enriched data
async def enrich_candidates(drug_list):
    for drug in drug_list:
        enriched = await fetch(f"/v1/biomedical/enrich/drug/{drug.name}")
        drug.chemical_properties = enriched["chemical_properties"]
        drug.targets = enriched["drugbank"]["targets"]
        drug.mechanism = enriched["drugbank"]["mechanism_of_action"]
```

### Use Case 2: Build Target Network
```python
# Build interaction network for drug targets
targets = ["EGFR", "VEGFA", "KDR"]
network = await fetch("/v1/biomedical/network/build", {
    "targets": targets,
    "include_interactions": true
})
# Visualize network in Neo4j or Cytoscape
```

### Use Case 3: Pathway Analysis
```python
# Find disease pathways and check drug coverage
disease_pathways = await fetch("/v1/biomedical/pathway/disease/H00409")
for pathway in disease_pathways:
    print(f"Pathway: {pathway['name']}")
    # Check which drugs target genes in this pathway
```

### Use Case 4: Protein Function Analysis
```python
# For each drug target, get detailed protein info
for target in drug_targets:
    protein_info = await fetch(f"/v1/biomedical/protein/info/{target}")
    print(f"Function: {protein_info['function']}")
    print(f"Location: {protein_info['subcellular_location']}")
```

---

## 🔧 Setup Instructions

### Step 1: No Setup Required!
4 out of 6 data sources work immediately without any API keys:
- ✅ UniProt
- ✅ STRING
- ✅ PubChem
- ✅ KEGG

Just start using them!

### Step 2: Get Optional API Keys (For Full Power)

#### DrugBank (Recommended for Academic Use)
1. Visit: https://go.drugbank.com/releases/latest
2. Click "Request Academic License"
3. Fill out form (free for academic/research use)
4. Receive API key by email
5. Add to `.env`: `DRUGBANK_API_KEY=your_key`

#### DisGeNET
1. Visit: https://www.disgenet.org/signup/
2. Register for free account
3. Request API access in your profile
4. Receive API key
5. Add to `.env`: `DISGENET_API_KEY=your_key`

### Step 3: Test Your Integration
```bash
# Test without API keys
curl http://localhost:8080/v1/biomedical/metadata

# Test with API keys (if configured)
curl http://localhost:8080/v1/biomedical/enrich/drug/metformin
```

---

## 📊 Integration Examples

### Example 1: Frontend Integration
```typescript
// Fetch enriched drug data in your React component
const enrichDrug = async (drugName: string) => {
  const response = await fetch(
    `http://localhost:8080/v1/biomedical/enrich/drug/${drugName}`
  );
  const data = await response.json();

  return {
    chemicalProperties: data.chemical_properties,
    targets: data.drugbank.targets,
    bioactivity: data.bioactivity
  };
};
```

### Example 2: Batch Enrichment
```python
# Enrich all drug candidates
import asyncio

async def enrich_all_drugs(drug_list):
    tasks = [
        fetch(f"/v1/biomedical/enrich/drug/{drug}")
        for drug in drug_list
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### Example 3: Neo4j Integration
```python
# Populate Neo4j with protein interaction network
network = await fetch("/v1/biomedical/network/build", {
    "targets": drug_targets
})

for node in network["nodes"]:
    neo4j.create_node("Protein", node)

for edge in network["edges"]:
    neo4j.create_relationship(
        edge["source"],
        edge["target"],
        "INTERACTS_WITH",
        {"score": edge["score"]}
    )
```

---

## 🌟 Advanced Features

### Feature 1: Multi-Source Data Fusion
The enrichment endpoints automatically combine data from multiple sources:
- PubChem: Chemical properties
- DrugBank: Clinical information
- KEGG: Pathway involvement
- UniProt: Target protein details

### Feature 2: Confidence Scoring
STRING interactions include confidence scores:
- `0.9-1.0`: Highest confidence
- `0.7-0.9`: High confidence
- `0.4-0.7`: Medium confidence

### Feature 3: Network Building
Automatically builds protein interaction networks with:
- Target proteins as nodes
- Interactions as edges
- Confidence scores as weights

---

## 📈 Performance Tips

1. **Cache Results**: Store enriched data to avoid repeated API calls
2. **Batch Requests**: Use `asyncio.gather()` for parallel requests
3. **Set Limits**: Use `limit` parameters to control response size
4. **Filter by Score**: Use `min_score` to get only high-confidence interactions

---

## 🐛 Troubleshooting

### Issue: 404 Not Found
**Solution**: Check that the ID/name is correct. Try searching first.

### Issue: 500 Server Error
**Solution**: Check logs. Might be rate limiting or network issues.

### Issue: Empty Results
**Solution**: API key might be missing (for DrugBank/DisGeNET) or ID format incorrect.

### Issue: Slow Responses
**Solution**: External APIs can be slow. Consider caching or reducing `limit` parameters.

---

## 📝 API Documentation

Full interactive API documentation available at:
**http://localhost:8080/docs**

Look for the **"biomedical"** tag in Swagger UI.

---

## 🎓 Learning Resources

- **UniProt**: https://www.uniprot.org/help/
- **STRING**: https://string-db.org/cgi/help
- **PubChem**: https://pubchemdocs.ncbi.nlm.nih.gov/
- **KEGG**: https://www.kegg.jp/kegg/docs/
- **DrugBank**: https://docs.drugbank.com/
- **DisGeNET**: https://www.disgenet.org/dbinfo

---

## 🚀 Next Steps

1. **Start with free endpoints** - No API keys needed!
2. **Test on your drug candidates** - Enrich your diabetes results
3. **Build visualizations** - Show protein networks in Neo4j
4. **Get API keys** - Unlock DrugBank and DisGeNET
5. **Integrate with frontend** - Add enrichment to UI

---

## 💡 Example Workflow

```python
# Complete drug repurposing enrichment workflow

# 1. Get drug repurposing results
results = await rank_drugs("diabetes")

# 2. Enrich top candidates
for drug in results[:10]:
    # Get comprehensive drug data
    enriched = await fetch(f"/v1/biomedical/enrich/drug/{drug.name}")

    # Get protein interactions for targets
    if enriched["drugbank"]["targets"]:
        network = await fetch("/v1/biomedical/network/build", {
            "targets": enriched["drugbank"]["targets"][:5]
        })

    # Get disease pathways
    disease_data = await fetch("/v1/biomedical/enrich/disease/diabetes")

    # Combine all data
    drug.enriched_profile = {
        **enriched,
        "target_network": network,
        "disease_pathways": disease_data["pathways"]
    }

# 3. Display in UI with interactive network visualization
```

---

**Your drug repurposing app now has enterprise-grade biomedical data integration!** 🎉

Start using it immediately - 4 out of 6 sources require zero setup! 🚀
