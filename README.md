<img width="3780" height="1890" alt="CTANE (1)" src="https://github.com/user-attachments/assets/a9a11642-685d-4545-9cc7-8d6468ff6fed" />



# OctaneDB - Lightweight & Fast Vector Database

[![PyPI version](https://badge.fury.io/py/octanedb.svg)](https://badge.fury.io/py/octanedb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OctaneDB** is a lightweight, high-performance Python vector database library built with modern Python and optimized algorithms. It's perfect for AI/ML applications requiring fast similarity search with HNSW indexing and flexible storage options.

## **Key Features**

### **Performance**
- **Fast HNSW indexing** for approximate nearest neighbor search
- **Sub-millisecond** query response times for typical workloads
- **Efficient insertion** with configurable batch sizes
- **Optimized memory usage** with HDF5 compression

### **Advanced Indexing**
- **HNSW (Hierarchical Navigable Small World)** for ultra-fast approximate search
- **FlatIndex** for exact similarity search
- **Configurable parameters** for performance tuning
- **Automatic index optimization**

### **Text Embedding Support** 

- **Automatic text-to-vector conversion** using sentence-transformers
- **Multiple embedding models** (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
- **GPU acceleration** support (CUDA)
- **Batch processing** for improved performance

### **Flexible Storage**
- **In-memory** for maximum speed
- **Persistent** file-based storage
- **Hybrid** mode for best of both worlds
- **HDF5 format** for efficient compression

### **Powerful Search**
- **Multiple distance metrics**: Cosine, Euclidean, Dot Product, Manhattan, Chebyshev, Jaccard
- **Advanced metadata filtering** with logical operators
- **Batch search** operations
- **Text-based search** with automatic embedding



### **Installation**

```bash
pip install octanedb
```

### **Basic Usage**

```python
from octanedb import OctaneDB

# Initialize with text embedding support
db = OctaneDB(
    dimension=384,  # Will be auto-set by embedding model
    embedding_model="all-MiniLM-L6-v2"
)

# Create a collection
collection = db.create_collection("documents")
db.use_collection("documents")

# Add text documents (ChromaDB-compatible!)
result = db.add(
    ids=["doc1", "doc2"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    metadatas=[
        {"category": "tropical", "color": "yellow"},
        {"category": "citrus", "color": "orange"}
    ]
)

# Search by text query
results = db.search_text(
    query_text="fruit",
    k=2,
    filter="category == 'tropical'",
    include_metadata=True
)

for doc_id, distance, metadata in results:
    print(f"Document: {db.get_document(doc_id)}")
    print(f"Distance: {distance:.4f}")
    print(f"Metadata: {metadata}")
```

## **Text Embedding Examples**

### **Working Basic Usage**

Here's a complete working example that demonstrates OctaneDB's core functionality:

```python
from octanedb import OctaneDB

# Initialize database with text embeddings
db = OctaneDB(
    dimension=384,  # sentence-transformers default dimension
    storage_mode="in-memory",
    enable_text_embeddings=True,
    embedding_model="all-MiniLM-L6-v2"  # Lightweight model
)

# Create a collection
db.create_collection("fruits")
db.use_collection("fruits")

# Add some fruit documents
fruits_data = [
    {"id": "apple", "text": "Apple is a sweet and crunchy fruit that grows on trees.", "category": "temperate"},
    {"id": "banana", "text": "Banana is a yellow tropical fruit rich in potassium.", "category": "tropical"},
    {"id": "mango", "text": "Mango is a sweet tropical fruit with a large seed.", "category": "tropical"},
    {"id": "orange", "text": "Orange is a citrus fruit with a bright orange peel.", "category": "citrus"}
]

for fruit in fruits_data:
    db.add(
        ids=[fruit["id"]],
        documents=[fruit["text"]],
        metadatas=[{"category": fruit["category"], "type": "fruit"}]
    )

# Simple text search
results = db.search_text(query_text="sweet", k=2, include_metadata=True)
print("Sweet fruits:")
for doc_id, distance, metadata in results:
    print(f"  • {doc_id}: {metadata.get('document', 'N/A')[:50]}...")

# Text search with filter
results = db.search_text(
    query_text="fruit", 
    k=2, 
    filter="category == 'tropical'",
    include_metadata=True
)
print("\nTropical fruits:")
for doc_id, distance, metadata in results:
    print(f"  • {doc_id}: {metadata.get('document', 'N/A')[:50]}...")
```


### **Advanced Text Operations**

```python
# Batch text search
query_texts = ["machine learning", "artificial intelligence", "data science"]
batch_results = db.search_text_batch(
    query_texts=query_texts,
    k=5,
    include_metadata=True
)

# Change embedding models
db.change_embedding_model("all-mpnet-base-v2")  # Higher quality, 768 dimensions

# Get available models
models = db.get_available_models()
print(f"Available models: {models}")
```

### **Custom Embeddings**

```python
# Use pre-computed embeddings
custom_embeddings = np.random.randn(100, 384).astype(np.float32)
result = db.add(
    ids=[f"vec_{i}" for i in range(100)],
    embeddings=custom_embeddings,
    metadatas=[{"source": "custom"} for _ in range(100)]
)
```

## **Advanced Usage**

### **Performance Tuning**

```python
# Optimize for speed vs. accuracy
db = OctaneDB(
    dimension=384,
    m=8,              # Fewer connections = faster, less accurate
    ef_construction=100,  # Lower = faster build
    ef_search=50      # Lower = faster search
)
```

### **Storage Management**

```python
# Persistent storage
db = OctaneDB(
    dimension=384,
    storage_path="./data",
    embedding_model="all-MiniLM-L6-v2"
)

# Save and load
db.save("./my_database.h5")
loaded_db = OctaneDB.load("./my_database.h5")
```

### **Metadata Filtering**

```python
# Complex filters
results = db.search_text(
    query_text="technology",
    k=10,
    filter={
        "$and": [
            {"category": "tech"},
            {"$or": [
                {"year": {"$gte": 2020}},
                {"priority": "high"}
            ]}
        ]
    }
)
```

## **Troubleshooting**

### **Common Issues**

1. **Empty search results**: Make sure to call `include_metadata=True` in your search methods to get metadata back.

2. **Query engine warnings**: The query engine for complex filters is under development. For now, use simple string filters like `"category == 'tropical'"`.

3. **Index not built**: The index is automatically built when needed, but you can manually trigger it with `collection._build_index()` if needed.

4. **Text embeddings not working**: Ensure you have `sentence-transformers` installed: `pip install sentence-transformers`

### **Working Example**

```python
# This will work correctly:
results = db.search_text(
    query_text="fruit", 
    k=2, 
    filter="category == 'tropical'",
    include_metadata=True  # Important!
)

# Process results correctly:
for doc_id, distance, metadata in results:
    print(f"ID: {doc_id}, Distance: {distance:.4f}")
    if metadata:
        print(f"  Document: {metadata.get('document', 'N/A')}")
        print(f"  Category: {metadata.get('category', 'N/A')}")
```

## **Performance Benchmarks**

### **OctaneDB Performance Characteristics**

**Test Environment:**
- **Hardware**: Intel i5-1300H, 16GB RAM, SSD storage
- **Dataset**: 100K vectors, 384 dimensions (float32)
- **Index Type**: HNSW with default parameters (m=16, ef_construction=200, ef_search=100)
- **Distance Metric**: Cosine similarity
- **Storage Mode**: In-memory

**Performance Results:**

| Operation | Performance | Notes |
|-----------|-------------|-------|
| **Vector Insertion** | 2,800-3,500 vectors/sec | Single-threaded insertion with metadata |
| **Index Build Time** | 45-60 seconds | HNSW index construction for 100K vectors |
| **Single Query Search** | 0.5-2.0 milliseconds | k=10 nearest neighbors |
| **Batch Search (100 queries)** | 150-200 queries/sec | k=10 per query |
| **Memory Usage** | ~1.5GB | Including vectors, metadata, and HNSW index |
| **Storage Efficiency** | ~15MB on disk | HDF5 compression for 100K vectors |

**Performance Tuning Options:**
- **Faster Build**: Reduce `ef_construction` (trades accuracy for speed)
- **Faster Search**: Reduce `ef_search` (trades accuracy for speed)
- **Memory Optimization**: Use `m=8` instead of `m=16` (fewer connections)
- **Storage Mode**: In-memory for speed, persistent for data persistence

**Benchmark Code:**
```bash
# Run performance benchmarks using CLI
octanedb benchmark --count 100000 --dimension 384

# Or use the comprehensive Python benchmarking script
python benchmark_octanedb.py --vectors 100000 --dimension 384 --runs 5

# Or use the Python API directly
from octanedb import OctaneDB
db = OctaneDB(dimension=384)
# ... run your own benchmarks
```

*Note: Performance varies based on hardware, dataset characteristics, and HNSW parameters. These numbers represent typical performance on the specified hardware configuration.*

## **Architecture**

```
OctaneDB
├── Core (OctaneDB)
│   ├── Collection Management
│   ├── Text Embedding Engine
│   └── Storage Manager
├── Collections
│   ├── Vector Storage (HDF5)
│   ├── Metadata Management
│   └── Index Management
├── Indexing
│   ├── HNSW Index
│   ├── Flat Index
│   └── Distance Metrics
├── Text Processing
│   ├── Sentence Transformers
│   ├── GPU Acceleration
│   └── Batch Processing
└── Storage
    ├── HDF5 Vectors
    ├── Msgpack Metadata
    └── Compression
```

## **Installation Options**

### **Basic Installation**
```bash
pip install octanedb
```

### **With GPU Support**
```bash
pip install octanedb[gpu]
```

### **Development Installation**
```bash
git clone https://github.com/RijinRaju/octanedb.git
cd octanedb
pip install -e .
```

##  **Requirements**

- **Python**: 3.8+
- **Core Dependencies**: NumPy, h5py, msgpack, tqdm
- **Text Embeddings**: sentence-transformers, transformers, torch
- **Optional**: CUDA for GPU acceleration, matplotlib, pandas, seaborn for benchmarking


##  **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
git clone https://github.com/RijinRaju/octanedb.git
cd octanedb
pip install -e ".[dev]"
pytest tests/
```

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Acknowledgments**

- **HNSW Algorithm**: Based on the Hierarchical Navigable Small World paper
- **Sentence Transformers**: For text embedding capabilities
- **HDF5**: For efficient vector storage
- **NumPy**: For fast numerical operations

## **Note**
A significant amount of the codebase was initially drafted using Cursor to accelerate boilerplate and some function implementations.

