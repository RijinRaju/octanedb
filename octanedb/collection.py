"""
Collection class - Manages vectors, metadata, and indexing for a single collection.
"""

import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict

from .index import HNSWIndex, IndexType
from .storage import StorageManager
from .query import QueryEngine
from .utils import VectorUtils

logger = logging.getLogger(__name__)


class Collection:
    """
    Collection class managing vectors, metadata, and indexing operations.
    
    Each collection is an isolated namespace for vectors with its own:
    - Vector storage and indexing
    - Metadata management
    - Search and query capabilities
    - Performance optimization
    """
    
    def __init__(
        self,
        name: str,
        dimension: int,
        index_type: str = "hnsw",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        max_elements: int = 1000000,
        distance_metric: str = "cosine",
        storage_manager: Optional[StorageManager] = None,
        query_engine: Optional[QueryEngine] = None,
        vector_utils: Optional[VectorUtils] = None
    ):
        """
        Initialize a collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            index_type: Type of index to use
            m: HNSW connections per layer
            ef_construction: Construction search depth
            ef_search: Search depth
            max_elements: Maximum number of vectors
            distance_metric: Distance metric for similarity
            storage_manager: Storage manager instance
            query_engine: Query engine instance
            vector_utils: Vector utilities instance
        """
        self.name = name
        self.dimension = dimension
        self.index_type = index_type
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements
        self.distance_metric = distance_metric
        
        # Initialize components
        self._storage_manager = storage_manager
        self._query_engine = query_engine
        self._vector_utils = vector_utils
        
        # Vector storage
        self._vectors: Dict[int, np.ndarray] = {}
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0
        
        # Index management
        self._index: Optional[HNSWIndex] = None
        self._index_built = False
        self._index_needs_rebuild = False
        
        # Performance tracking
        self._stats = {
            "inserts": 0,
            "searches": 0,
            "updates": 0,
            "deletes": 0,
            "index_builds": 0,
            "last_index_build": None
        }
        
        # Initialize index
        self._init_index()
        
        logger.info(f"Collection '{name}' initialized with dimension {dimension}")
    
    def _init_index(self) -> None:
        """Initialize the vector index."""
        if self.index_type == "hnsw":
            self._index = HNSWIndex(
                dimension=self.dimension,
                m=self.m,
                ef_construction=self.ef_construction,
                ef_search=self.ef_search,
                max_elements=self.max_elements,
                distance_metric=self.distance_metric
            )
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def insert(
        self, 
        vectors: Union[np.ndarray, List], 
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None
    ) -> Union[int, List[int]]:
        """
        Insert vectors into the collection.
        
        Args:
            vectors: Vector(s) to insert
            metadata: Optional metadata for each vector
            ids: Optional custom IDs
            
        Returns:
            Inserted vector ID(s)
        """
        # Convert to numpy array if needed
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        elif not isinstance(vectors, np.ndarray):
            vectors = np.array([vectors], dtype=np.float32)
        
        # Ensure 2D array
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # Validate dimensions
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match collection dimension {self.dimension}")
        
        # Handle metadata
        if metadata is None:
            metadata = [{} for _ in range(len(vectors))]
        elif not isinstance(metadata, list):
            metadata = [metadata]
        
        # Handle IDs
        if ids is None:
            ids = [self._next_id + i for i in range(len(vectors))]
        elif not isinstance(ids, list):
            ids = [ids]
        
        # Validate lengths
        if len(vectors) != len(metadata) or len(vectors) != len(ids):
            raise ValueError("Vectors, metadata, and IDs must have the same length")
        
        # Insert vectors
        inserted_ids = []
        for i, (vector, meta, vector_id) in enumerate(zip(vectors, metadata, ids)):
            # Check if ID already exists
            if vector_id in self._vectors:
                raise ValueError(f"Vector ID {vector_id} already exists")
            
            # Store vector and metadata
            self._vectors[vector_id] = vector.copy()
            self._metadata[vector_id] = meta.copy()
            inserted_ids.append(vector_id)
            
            # Update next_id
            self._next_id = max(self._next_id, vector_id + 1)
        
        # Mark index for rebuild
        self._index_needs_rebuild = True
        
        # Update stats
        self._stats["inserts"] += len(vectors)
        
        logger.debug(f"Inserted {len(vectors)} vectors into collection '{self.name}'")
        
        # Return single ID or list based on input
        return inserted_ids[0] if len(inserted_ids) == 1 else inserted_ids
    
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
        # Ensure index is built
        if not self._index_built or self._index_needs_rebuild:
            self._build_index()
        
        # Validate query vector
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"Query vector dimension {query_vector.shape[0]} does not match collection dimension {self.dimension}")
        
        # Search using index
        start_time = time.time()
        results = self._index.search(query_vector, k)
        search_time = time.time() - start_time
        
        # Apply filters if specified
        if filter:
            results = self._apply_filter(results, filter)
        
        # Format results
        formatted_results = []
        for vector_id, distance in results:
            metadata = self._metadata.get(vector_id) if include_metadata else None
            formatted_results.append((vector_id, distance, metadata))
        
        # Update stats
        self._stats["searches"] += 1
        
        logger.debug(f"Search completed in {search_time:.4f}s, found {len(formatted_results)} results")
        
        return formatted_results
    
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
        # Ensure index is built
        if not self._index_built or self._index_needs_rebuild:
            self._build_index()
        
        # Validate query vectors
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(f"Query vector dimension {query_vectors.shape[1]} does not match collection dimension {self.dimension}")
        
        # Batch search using index
        start_time = time.time()
        batch_results = self._index.search_batch(query_vectors, k)
        search_time = time.time() - start_time
        
        # Apply filters and format results
        formatted_batch_results = []
        for results in batch_results:
            if filter:
                results = self._apply_filter(results, filter)
            
            formatted_results = []
            for vector_id, distance in results:
                metadata = self._metadata.get(vector_id) if include_metadata else None
                formatted_results.append((vector_id, distance, metadata))
            
            formatted_batch_results.append(formatted_results)
        
        # Update stats
        self._stats["searches"] += 1
        
        logger.debug(f"Batch search completed in {search_time:.4f}s for {len(query_vectors)} queries")
        
        return formatted_batch_results
    
    def _apply_filter(self, results: List[Tuple[int, float]], filter: Dict[str, Any]) -> List[Tuple[int, float]]:
        """
        Apply metadata filter to search results.
        
        Args:
            results: Search results (id, distance) tuples
            filter: Filter criteria
            
        Returns:
            Filtered results
        """
        if not filter:
            return results
        
        filtered_results = []
        for vector_id, distance in results:
            metadata = self._metadata.get(vector_id, {})
            if self._query_engine.evaluate_filter(metadata, filter):
                filtered_results.append((vector_id, distance))
        
        return filtered_results
    
    def update(self, id: int, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update a vector.
        
        Args:
            id: Vector ID to update
            vector: New vector
            metadata: New metadata
        """
        if id not in self._vectors:
            raise ValueError(f"Vector ID {id} does not exist")
        
        # Validate vector dimension
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension {vector.shape[0]} does not match collection dimension {self.dimension}")
        
        # Update vector and metadata
        self._vectors[id] = vector.copy()
        if metadata is not None:
            self._metadata[id] = metadata.copy()
        
        # Mark index for rebuild
        self._index_needs_rebuild = True
        
        # Update stats
        self._stats["updates"] += 1
        
        logger.debug(f"Updated vector {id} in collection '{self.name}'")
    
    def delete(self, id: int) -> None:
        """
        Delete a vector.
        
        Args:
            id: Vector ID to delete
        """
        if id not in self._vectors:
            raise ValueError(f"Vector ID {id} does not exist")
        
        # Remove vector and metadata
        del self._vectors[id]
        if id in self._metadata:
            del self._metadata[id]
        
        # Mark index for rebuild
        self._index_needs_rebuild = True
        
        # Update stats
        self._stats["deletes"] += 1
        
        logger.debug(f"Deleted vector {id} from collection '{self.name}'")
    
    def delete_batch(self, ids: List[int]) -> None:
        """
        Batch delete vectors.
        
        Args:
            ids: List of vector IDs to delete
        """
        for vector_id in ids:
            self.delete(vector_id)
    
    def get_vector(self, id: int, include_metadata: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Optional[Dict[str, Any]]]]:
        """
        Get a vector by ID.
        
        Args:
            id: Vector ID
            include_metadata: Whether to include metadata
            
        Returns:
            Vector or (vector, metadata) tuple
        """
        if id not in self._vectors:
            raise ValueError(f"Vector ID {id} does not exist")
        
        vector = self._vectors[id]
        if include_metadata:
            metadata = self._metadata.get(id)
            return vector, metadata
        else:
            return vector
    
    def count(self) -> int:
        """Get total number of vectors in the collection."""
        return len(self._vectors)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        stats = self._stats.copy()
        stats["vector_count"] = len(self._vectors)
        stats["metadata_count"] = len(self._metadata)
        stats["index_built"] = self._index_built
        stats["index_needs_rebuild"] = self._index_needs_rebuild
        return stats
    
    def _build_index(self) -> None:
        """Build or rebuild the vector index."""
        if not self._vectors:
            logger.warning("No vectors to index")
            return
        
        start_time = time.time()
        
        # Convert vectors to array
        vector_ids = list(self._vectors.keys())
        vectors_array = np.array([self._vectors[vid] for vid in vector_ids], dtype=np.float32)
        
        # Build index
        self._index.build(vectors_array, vector_ids)
        
        # Update status
        self._index_built = True
        self._index_needs_rebuild = False
        
        build_time = time.time() - start_time
        
        # Update stats
        self._stats["index_builds"] += 1
        self._stats["last_index_build"] = time.time()
        
        logger.info(f"Index built for {len(vectors_array)} vectors in {build_time:.4f}s")
    
    def optimize_index(self) -> None:
        """Optimize the collection's index."""
        if not self._index_built:
            logger.warning("Index not built yet")
            return
        
        start_time = time.time()
        self._index.optimize()
        optimize_time = time.time() - start_time
        
        logger.info(f"Index optimization completed in {optimize_time:.4f}s")
    
    def clear(self) -> None:
        """Clear all vectors from the collection."""
        self._vectors.clear()
        self._metadata.clear()
        self._next_id = 0
        self._index_built = False
        self._index_needs_rebuild = False
        
        logger.info(f"Collection '{self.name}' cleared")
    
    def __len__(self) -> int:
        """Return number of vectors in the collection."""
        return len(self._vectors)
    
    def __contains__(self, id: int) -> bool:
        """Check if a vector ID exists in the collection."""
        return id in self._vectors
