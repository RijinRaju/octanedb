"""
OctaneDB - Lightweight & Fast Vector Database

A high-performance vector database library designed for speed and efficiency.
"""

from .core import OctaneDB
from .collection import Collection
from .index import HNSWIndex, IndexType
from .storage import StorageManager
from .query import QueryEngine
from .utils import VectorUtils, DistanceMetrics

__version__ = "0.1.0"
__author__ = "OctaneDB Team"

__all__ = [
    "OctaneDB",
    "Collection", 
    "HNSWIndex",
    "IndexType",
    "StorageManager",
    "QueryEngine",
    "VectorUtils",
    "DistanceMetrics"
]
