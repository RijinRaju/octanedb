"""
Core OctaneDB class - Main interface for vector database operations.
"""

import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from .collection import Collection
from .storage import StorageManager
from .query import QueryEngine
from .utils import VectorUtils
from .index import HNSWIndex, IndexType

logger = logging.getLogger(__name__)


class OctaneDB:
    """
    Main OctaneDB class providing high-performance vector database operations.
    
    This class implements a lightweight and fast vector database with:
    - Efficient HNSW-based similarity search
    - Optimized storage and indexing
    - Fast CRUD operations
    - Collection management
    - Persistence and loading
    """
    
    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "hnsw",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        max_elements: int = 1000000,
        distance_metric: str = "cosine",
        storage_path: Optional[str] = None,
        enable_cache: bool = True,
        cache_size: int = 10000
    ):
        """
        Initialize OctaneDB.
        
        Args:
            dimension: Vector dimension
            index_type: Type of index ("hnsw", "flat", "ivf")
            m: HNSW connections per layer
            ef_construction: Construction search depth
            ef_search: Search depth
            max_elements: Maximum number of vectors
            distance_metric: Distance metric ("cosine", "euclidean", "dot")
            storage_path: Path for persistent storage
            enable_cache: Enable caching for better performance
            cache_size: Maximum cache size
        """
        self.dimension = dimension
        self.index_type = index_type
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements
        self.distance_metric = distance_metric
        self.storage_path = Path(storage_path) if storage_path else None
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        
        # Initialize components
        self._collections: Dict[str, Collection] = {}
        self._current_collection: Optional[Collection] = None
        self._storage_manager = StorageManager(
            storage_path=self.storage_path,
            enable_cache=enable_cache,
            cache_size=cache_size
        )
        self._query_engine = QueryEngine()
        self._vector_utils = VectorUtils(distance_metric)
        
        # Performance tracking
        self._stats = {
            "inserts": 0,
            "searches": 0,
            "updates": 0,
            "deletes": 0,
            "total_vectors": 0
        }
        
        logger.info(f"OctaneDB initialized with dimension {dimension}")
    
    def create_collection(
        self, 
        name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            metadata: Optional collection metadata
            
        Returns:
            Created collection
        """
        if name in self._collections:
            raise ValueError(f"Collection '{name}' already exists")
        
        collection = Collection(
            name=name,
            dimension=self.dimension,
            index_type=self.index_type,
            m=self.m,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            max_elements=self.max_elements,
            distance_metric=self.distance_metric,
            storage_manager=self._storage_manager,
            query_engine=self._query_engine,
            vector_utils=self._vector_utils
        )
        
        self._collections[name] = collection
        
        # Set as current collection if it's the first one
        if self._current_collection is None:
            self._current_collection = collection
        
        logger.info(f"Created collection: {name}")
        return collection
    
    def use_collection(self, name: str) -> None:
        """Switch to a different collection."""
        if name not in self._collections:
            raise ValueError(f"Collection '{name}' does not exist")
        self._current_collection = self._collections[name]
        logger.info(f"Switched to collection: {name}")
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        return list(self._collections.keys())
    
    def get_collection(self, name: str) -> Collection:
        """Get a collection by name."""
        if name not in self._collections:
            raise ValueError(f"Collection '{name}' does not exist")
        return self._collections[name]
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        if name not in self._collections:
            raise ValueError(f"Collection '{name}' does not exist")
        
        # Remove from storage
        self._storage_manager.delete_collection(name)
        
        # Remove from memory
        del self._collections[name]
        
        # Update current collection if needed
        if self._current_collection and self._current_collection.name == name:
            self._current_collection = self._collections.get(list(self._collections.keys())[0]) if self._collections else None
        
        logger.info(f"Deleted collection: {name}")
    
    def insert(
        self, 
        vectors: Union[np.ndarray, List], 
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None
    ) -> Union[int, List[int]]:
        """
        Insert vectors into the current collection.
        
        Args:
            vectors: Vector(s) to insert
            metadata: Optional metadata for each vector
            ids: Optional custom IDs
            
        Returns:
            Inserted vector ID(s)
        """
        if self._current_collection is None:
            raise RuntimeError("No collection selected. Create or select a collection first.")
        
        return self._current_collection.insert(vectors, metadata, ids)
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = False
    ) -> List[Tuple[int, float, Optional[Dict[str, Any]]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of (id, distance, metadata) tuples
        """
        if self._current_collection is None:
            raise RuntimeError("No collection selected. Create or select a collection first.")
        
        return self._current_collection.search(query_vector, k, filter, include_metadata)
    
    def search_batch(
        self, 
        query_vectors: np.ndarray, 
        k: int = 10, 
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = False
    ) -> List[List[Tuple[int, float, Optional[Dict[str, Any]]]]]:
        """
        Batch search for similar vectors.
        
        Args:
            query_vectors: Query vectors
            k: Number of results per query
            filter: Optional metadata filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of result lists for each query
        """
        if self._current_collection is None:
            raise RuntimeError("No collection selected. Create or select a collection first.")
        
        return self._current_collection.search_batch(query_vectors, k, filter, include_metadata)
    
    def update(self, id: int, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update a vector.
        
        Args:
            id: Vector ID to update
            vector: New vector
            metadata: New metadata
        """
        if self._current_collection is None:
            raise RuntimeError("No collection selected. Create or select a collection first.")
        
        self._current_collection.update(id, vector, metadata)
    
    def delete(self, id: int) -> None:
        """
        Delete a vector.
        
        Args:
            id: Vector ID to delete
        """
        if self._current_collection is None:
            raise RuntimeError("No collection selected. Create or select a collection first.")
        
        self._current_collection.delete(id)
    
    def delete_batch(self, ids: List[int]) -> None:
        """
        Batch delete vectors.
        
        Args:
            ids: List of vector IDs to delete
        """
        if self._current_collection is None:
            raise RuntimeError("No collection selected. Create or select a collection first.")
        
        self._current_collection.delete_batch(ids)
    
    def get_vector(self, id: int, include_metadata: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Optional[Dict[str, Any]]]]:
        """
        Get a vector by ID.
        
        Args:
            id: Vector ID
            include_metadata: Whether to include metadata
            
        Returns:
            Vector or (vector, metadata) tuple
        """
        if self._current_collection is None:
            raise RuntimeError("No collection selected. Create or select a collection first.")
        
        return self._current_collection.get_vector(id, include_metadata)
    
    def count(self) -> int:
        """Get total number of vectors in current collection."""
        if self._current_collection is None:
            return 0
        return self._current_collection.count()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = self._stats.copy()
        if self._current_collection:
            stats["current_collection"] = self._current_collection.name
            stats["collection_count"] = len(self._collections)
            stats["total_vectors"] = sum(c.count() for c in self._collections.values())
        return stats
    
    def optimize_index(self) -> None:
        """Optimize the current collection's index."""
        if self._current_collection is None:
            raise RuntimeError("No collection selected. Create or select a collection first.")
        
        self._current_collection.optimize_index()
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the database to disk.
        
        Args:
            path: Optional custom save path
        """
        save_path = Path(path) if path else self.storage_path
        if save_path is None:
            raise ValueError("No storage path specified")
        
        self._storage_manager.save_database(self._collections, save_path)
        logger.info(f"Database saved to {save_path}")
    
    @classmethod
    def load(cls, path: str, **kwargs) -> "OctaneDB":
        """
        Load a database from disk.
        
        Args:
            path: Path to saved database
            **kwargs: Additional initialization parameters
            
        Returns:
            Loaded OctaneDB instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Database file not found: {path}")
        
        # Load database metadata
        storage_manager = StorageManager(storage_path=path.parent)
        metadata = storage_manager.load_database_metadata(path)
        
        # Get dimension from first collection
        collections_metadata = metadata.get("collections", {})
        if not collections_metadata:
            raise ValueError("No collections found in database")
        
        first_collection_name = list(collections_metadata.keys())[0]
        first_collection_meta = collections_metadata[first_collection_name]
        
        # Create instance with loaded parameters
        instance = cls(
            dimension=first_collection_meta["dimension"],
            index_type=first_collection_meta.get("index_type", "hnsw"),
            m=first_collection_meta.get("m", 16),
            ef_construction=first_collection_meta.get("ef_construction", 200),
            ef_search=first_collection_meta.get("ef_search", 100),
            max_elements=first_collection_meta.get("max_elements", 1000000),
            distance_metric=first_collection_meta.get("distance_metric", "cosine"),
            storage_path=path.parent,
            **kwargs
        )
        
        # Load collections
        collections = storage_manager.load_database(path)
        instance._collections = collections
        
        # Set current collection
        if collections:
            instance._current_collection = list(collections.values())[0]
        
        logger.info(f"Database loaded from {path}")
        return instance
    
    def __len__(self) -> int:
        """Return total number of vectors across all collections."""
        return sum(c.count() for c in self._collections.values())
    
    def __contains__(self, id: int) -> bool:
        """Check if a vector ID exists in any collection."""
        return any(id in c for c in self._collections.values())
