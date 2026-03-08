# Semantic Search Engine with Fuzzy Clustering & Intelligent Caching

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-orange)
![ML](https://img.shields.io/badge/Machine%20Learning-GMM-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A lightweight **semantic search system** that combines **vector embeddings, fuzzy clustering, and a cluster-aware semantic cache** to deliver fast and intelligent document retrieval.

The system is built using the **20 Newsgroups Dataset** and leverages transformer-based embeddings for semantic understanding.

---

# Project Highlights

* Transformer-based **semantic embeddings**
* **Vector similarity search** powered by **FAISS**
* **Fuzzy topic clustering** using Gaussian Mixture Models
* **Cluster-aware semantic caching** for accelerated query responses
* **REST API interface** built with **FastAPI**

---

# System Architecture

```
                User Query
                    │
                    ▼
           Sentence Embedding
                    │
                    ▼
            Semantic Cache Check
              │           │
           Hit │           │ Miss
              ▼           ▼
        Return Cached   FAISS Vector
           Result        Search
                              │
                              ▼
                   Cluster Assignment (GMM)
                              │
                              ▼
                     Store Result in Cache
                              │
                              ▼
                         API Response
```

This architecture ensures that **repeated or semantically similar queries are answered instantly** while maintaining full search capability.

---

# Core Components

| Module          | Role                                      |
| --------------- | ----------------------------------------- |
| `embeddings.py` | Loads dataset and generates embeddings    |
| `search.py`     | Handles nearest-neighbor vector retrieval |
| `clustering.py` | Performs fuzzy clustering with GMM        |
| `cache.py`      | Implements cluster-aware semantic caching |
| `main.py`       | FastAPI service exposing search endpoints |

---

# Embedding Model

The system uses **all-MiniLM-L6-v2** via the **Sentence-Transformers** library.

Key features:

* 384-dimensional embedding vectors
* Efficient inference speed
* Strong semantic understanding
* Trained on over **1 billion sentence pairs**

Compared with larger models like **all-mpnet-base-v2**, this model offers a better balance between **performance and computational cost**.

---

# Fuzzy Topic Clustering

Topic discovery is implemented using the **Gaussian Mixture Model**.

Unlike traditional clustering approaches:

### K-Means

* Hard cluster assignment
* Each document belongs to only one cluster

### Gaussian Mixture Model

* Soft probabilistic membership
* Documents can belong to multiple clusters

Example:

A document about **sports cars** might partially belong to both:

* automotive cluster
* sports cluster

This reflects **real-world topic overlap** more accurately.

---

# Cluster Selection Strategy

To determine the best number of clusters, the system evaluates models using the **Bayesian Information Criterion (BIC)**.

```
BIC = -2 × log_likelihood + (parameters × log(samples))
```

Cluster counts from **8 to 21** were tested.

The optimal configuration was:

```
Clusters: 8
BIC Score: -24,267,231
```

This provided the best balance between **model complexity and explanatory power**.

---

# Cluster-Aware Semantic Cache

A traditional cache requires scanning every stored query.

```
Lookup Complexity: O(n)
```

To improve efficiency, cached queries are grouped by **dominant cluster**.

When a query arrives:

1. Its embedding is generated
2. The dominant cluster is identified
3. Only cache entries within that cluster are checked

```
Optimized Complexity: O(n / k)
```

Similarity is determined using **cosine similarity** with a threshold of **0.85**.

This ensures cache hits only occur for **strong semantic matches**.

---

# Vector Similarity Search

Document retrieval uses **FAISS** with the **IndexFlatL2** method.

Advantages:

* Exact nearest-neighbor search
* Extremely fast for moderate datasets
* Minimal configuration required

With:

* ~18,000 documents
* 384-dimension embeddings

Search latency remains **sub-millisecond**.

For much larger datasets, approximate search techniques like **IVF** or **HNSW** would be preferable.

---

# API Endpoints

The system exposes a REST API built using **FastAPI**.

---

## POST `/query`

Search the dataset for semantically similar documents.

Request example

```json
{
  "query": "What is the best graphics card for gaming?"
}
```

Example response

```json
{
  "query": "What is the best graphics card for gaming?",
  "cache_hit": false,
  "result": [
    { "rank": 1, "document": "...", "l2_distance": 0.82 },
    { "rank": 2, "document": "...", "l2_distance": 0.87 }
  ],
  "dominant_cluster": 3,
  "cluster_distribution": [0.01, 0.02, 0.05, 0.85]
}
```

---

## GET `/cache/stats`

Returns cache usage statistics.

Example response

```json
{
  "total_entries": 12,
  "hit_count": 5,
  "miss_count": 7,
  "hit_rate": 0.417
}
```

---

## DELETE `/cache`

Clears all cached queries and resets cache statistics.

---

# Cluster Distribution

| Cluster | Documents | Notes            |
| ------- | --------- | ---------------- |
| 0       | 1,975     | Mixed topics     |
| 1       | 2,414     | Mixed topics     |
| 2       | 879       | Smallest cluster |
| 3       | 1,727     | Mixed topics     |
| 4       | 2,150     | Mixed topics     |
| 5       | 3,520     | Largest cluster  |
| 6       | 2,826     | Mixed topics     |
| 7       | 2,638     | Mixed topics     |

Some documents show **high cluster entropy**, meaning they belong to multiple clusters with similar probabilities.

These cases demonstrate the advantage of **fuzzy clustering over hard clustering**.

---

# Installation

## Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 127.0.0.1 --port 8000
```

First execution may take **approximately 15 minutes** due to:

* embedding generation
* cluster evaluation

Afterward, embeddings are loaded from disk for faster startup.

---

## Docker Deployment

```bash
docker-compose up --build
```

---

# Environment Variables

Optional variables for offline operation.

```
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

---

# Technology Stack

| Technology                | Purpose                  |
| ------------------------- | ------------------------ |
| **FastAPI**               | API framework            |
| **Sentence-Transformers** | text embeddings          |
| **FAISS**                 | vector similarity search |
| **scikit-learn**          | clustering algorithms    |
| **NumPy**                 | numerical computing      |

---

# Dataset

The system uses the **20 Newsgroups Dataset**, a classic benchmark dataset for text classification and topic modeling.

Dataset statistics:

* 18,129 cleaned documents
* 20 topic categories
* Covers technology, politics, science, sports, religion, and more

---

# Data Cleaning

The dataset was preprocessed to remove noise.

Cleaning steps included:

* Removing **email headers**
* Removing **footers and quoted replies**
* Filtering documents shorter than **30 characters**

Short documents were typically artifacts produced during metadata removal.

---

# Future Improvements

Potential enhancements include:

* Approximate vector search for large-scale datasets
* GPU acceleration for embedding generation
* Real-time query analytics dashboard
* Persistent distributed caching
* Streaming ingestion of new documents

---

# License

MIT License

---
