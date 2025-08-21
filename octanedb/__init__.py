"""
OctaneDB - Lightning Fast Vector Database

A lightweight, high-performance Python vector database library that provides:
- Fast vector similarity search using HNSW indexing
- Automatic text embedding generation
- ChromaDB-compatible API
- Multiple storage modes (in-memory, persistent, hybrid)
- Advanced metadata filtering
- Batch operations for improved performance
"""

from .core import OctaneDB
from .collection import Collection
from .index import HNSWIndex, FlatIndex
from .storage import StorageManager
from .query import QueryEngine
from .utils import VectorUtils
from .embeddings import TextEmbedder, ChromaCompatibleEmbedder, embed_texts

__version__ = "1.0.0"
__author__ = "Rijin"
__email__ = "rijinraj856@gmail.com"

__all__ = [
    "OctaneDB",
    "Collection", 
    "HNSWIndex",
    "FlatIndex",
    "StorageManager",
    "QueryEngine",
    "VectorUtils",
    "TextEmbedder",
    "ChromaCompatibleEmbedder",
    "embed_texts"
]
