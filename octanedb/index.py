"""
Index implementations for fast vector similarity search.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import time

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Supported index types."""
    HNSW = "hnsw"
    FLAT = "flat"
    IVF = "ivf"


class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) index for fast vector similarity search.
    
    HNSW is a graph-based index that provides:
    - Sub-linear search complexity
    - High accuracy for similarity search
    - Efficient construction and updates
    - Configurable search depth and connections
    """
    
    def __init__(
        self,
        dimension: int,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        max_elements: int = 1000000,
        distance_metric: str = "cosine"
    ):
        """
        Initialize HNSW index.
        
        Args:
            dimension: Vector dimension
            m: Maximum number of connections per layer
            ef_construction: Construction search depth
            ef_search: Search depth
            max_elements: Maximum number of vectors
            distance_metric: Distance metric for similarity
        """
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements
        self.distance_metric = distance_metric
        
        # Graph structure
        self._layers: List[Dict[int, List[int]]] = []  # Adjacency lists for each layer
        self._vectors: Optional[np.ndarray] = None
        self._vector_ids: Optional[List[int]] = None
        self._max_layer: int = 0
        
        # Entry point
        self._entry_point: Optional[int] = None
        self._entry_layer: int = 0
        
        # Distance computation
        self._distance_func = self._get_distance_function()
        
        logger.info(f"HNSW index initialized with m={m}, ef_construction={ef_construction}")
    
    def _get_distance_function(self):
        """Get the appropriate distance function."""
        if self.distance_metric == "cosine":
            return self._cosine_distance
        elif self.distance_metric == "euclidean":
            return self._euclidean_distance
        elif self.distance_metric == "dot":
            return self._dot_distance
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot_product / (norm_a * norm_b))
    
    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance between vectors."""
        return np.linalg.norm(a - b)
    
    def _dot_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute negative dot product distance."""
        return -np.dot(a, b)
    
    def build(self, vectors: np.ndarray, vector_ids: List[int]) -> None:
        """
        Build the HNSW index from vectors.
        
        Args:
            vectors: Array of vectors to index
            vector_ids: Corresponding vector IDs
        """
        if len(vectors) == 0:
            logger.warning("No vectors to index")
            return
        
        start_time = time.time()
        
        # Store vectors and IDs
        self._vectors = vectors.astype(np.float32)
        self._vector_ids = vector_ids
        
        # Initialize layers
        self._layers = [{} for _ in range(self._get_max_layer(len(vectors)) + 1)]
        
        # Insert vectors one by one
        for i in range(len(vectors)):
            self._insert_vector(i, vector_ids[i])
            logger.debug(f"Inserted vector {i} (ID: {vector_ids[i]}) at layer {self._layers[-1] if self._layers else 'N/A'}")
        
        build_time = time.time() - start_time
        logger.info(f"HNSW index built for {len(vectors)} vectors in {build_time:.4f}s")
        logger.info(f"Final entry point: {self._entry_point}, entry layer: {self._entry_layer}")
        logger.info(f"Number of layers: {len(self._layers)}")
    
    def _get_max_layer(self, num_vectors: int) -> int:
        """Calculate maximum layer for the given number of vectors."""
        return max(0, int(np.log(num_vectors) / np.log(self.m)))
    
    def _insert_vector(self, vector_idx: int, vector_id: int) -> None:
        """Insert a single vector into the index."""
        # Determine layer for this vector
        layer = self._get_random_layer()
        logger.debug(f"Vector {vector_idx} (ID: {vector_id}) assigned to layer {layer}")
        
        # Find nearest neighbors in current layer and lower layers
        if self._entry_point is not None:
            nearest = self._search_layer(
                self._vectors[vector_idx], 
                self._entry_point, 
                self.ef_construction, 
                layer
            )
            
            # If no neighbors found in current layer, search in lower layers
            if not nearest and layer > 0:
                for lower_layer in range(layer - 1, -1, -1):
                    lower_nearest = self._search_layer(
                        self._vectors[vector_idx],
                        self._entry_point,
                        self.ef_construction,
                        lower_layer
                    )
                    if lower_nearest:
                        nearest = lower_nearest
                        break
            
            # If still no neighbors found, create a simple connection to the entry point
            if not nearest:
                nearest = [(self._entry_point, self._distance_func(
                    self._vectors[vector_idx], 
                    self._vectors[self._entry_point]
                ))]
        else:
            nearest = []
        
        # Add connections in current layer
        self._add_connections(vector_idx, nearest, layer)
        
        # Update entry point if needed
        if self._entry_point is None or layer > self._entry_layer:
            self._entry_point = vector_idx
            self._entry_layer = layer
            logger.debug(f"Updated entry point to {vector_idx} at layer {layer}")
        
        # Add to layers
        for l in range(layer + 1):
            if l not in self._layers:
                self._layers.append({})
            self._layers[l][vector_idx] = []
        
        # Ensure the vector is also added to the bottom layer (layer 0) for searchability
        if 0 not in self._layers:
            self._layers.append({})
        if vector_idx not in self._layers[0]:
            self._layers[0][vector_idx] = []
    
    def _get_random_layer(self) -> int:
        """Get random layer based on exponential distribution."""
        # Use a smaller factor to avoid extremely high layers
        # The original used self.m which was too large
        factor = 1.0  # This gives more reasonable layer distribution
        return int(-np.log(np.random.random()) * factor)
    
    def _search_layer(
        self, 
        query: np.ndarray, 
        entry_point: int, 
        ef: int, 
        layer: int
    ) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors in a specific layer.
        
        Args:
            query: Query vector
            entry_point: Starting point for search
            ef: Search depth
            layer: Layer to search in
            
        Returns:
            List of (vector_idx, distance) tuples
        """
        if layer >= len(self._layers):
            return []
        
        # Initialize candidates and visited
        candidates = [(entry_point, self._distance_func(query, self._vectors[entry_point]))]
        visited = {entry_point}
        
        # Search loop
        while candidates:
            # Get closest candidate
            current_idx, current_dist = candidates.pop(0)
            
            # Check if we can improve
            if len(candidates) >= ef and current_dist > candidates[-1][1]:
                break
            
            # Explore neighbors
            for neighbor_idx in self._layers[layer].get(current_idx, []):
                if neighbor_idx in visited:
                    continue
                
                visited.add(neighbor_idx)
                neighbor_dist = self._distance_func(query, self._vectors[neighbor_idx])
                
                # Add to candidates if better than worst
                if len(candidates) < ef or neighbor_dist < candidates[-1][1]:
                    # Insert in sorted order
                    insert_pos = 0
                    for i, (_, dist) in enumerate(candidates):
                        if neighbor_dist < dist:
                            insert_pos = i
                            break
                    
                    candidates.insert(insert_pos, (neighbor_idx, neighbor_dist))
                    
                    # Keep only top ef candidates
                    if len(candidates) > ef:
                        candidates = candidates[:ef]
        
        return candidates
    
    def _add_connections(self, vector_idx: int, nearest: List[Tuple[int, float]], layer: int) -> None:
        """Add connections for a vector in a specific layer."""
        if layer >= len(self._layers):
            return
        
        # Sort by distance
        nearest.sort(key=lambda x: x[1])
        
        # Add bidirectional connections in the current layer
        for neighbor_idx, _ in nearest[:self.m]:
            # Add connection from vector to neighbor
            if vector_idx not in self._layers[layer]:
                self._layers[layer][vector_idx] = []
            self._layers[layer][vector_idx].append(neighbor_idx)
            
            # Add connection from neighbor to vector
            if neighbor_idx not in self._layers[layer]:
                self._layers[layer][neighbor_idx] = []
            self._layers[layer][neighbor_idx].append(vector_idx)
        
        # Also add connections in the bottom layer (layer 0) for better searchability
        if layer > 0 and 0 < len(self._layers):
            for neighbor_idx, _ in nearest[:self.m]:
                # Add connection from vector to neighbor in bottom layer
                if vector_idx not in self._layers[0]:
                    self._layers[0][vector_idx] = []
                if neighbor_idx not in self._layers[0][vector_idx]:
                    self._layers[0][vector_idx].append(neighbor_idx)
                
                # Add connection from neighbor to vector in bottom layer
                if neighbor_idx not in self._layers[0]:
                    self._layers[0][neighbor_idx] = []
                if vector_idx not in self._layers[0][neighbor_idx]:
                    self._layers[0][neighbor_idx].append(vector_idx)
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of results to return
            
        Returns:
            List of (vector_id, distance) tuples
        """
        if self._entry_point is None:
            logger.warning("HNSW search failed: entry_point is None")
            return []
        
        logger.debug(f"HNSW search: entry_point={self._entry_point}, entry_layer={self._entry_layer}, layers={len(self._layers)}")
        
        # Start from top layer
        current_layer = self._entry_layer
        current_point = self._entry_point
        
        # Search down through layers
        while current_layer > 0:
            nearest = self._search_layer(query, current_point, 1, current_layer)
            if nearest:
                current_point = nearest[0][0]
            current_layer -= 1
        
        # Search in bottom layer (layer 0) where most connections are
        nearest = self._search_layer(query, current_point, self.ef_search, 0)
        
        # If no results found in bottom layer, try searching from all vectors in bottom layer
        if not nearest and len(self._layers) > 0 and self._layers[0]:
            all_candidates = []
            for vector_idx in self._layers[0].keys():
                distance = self._distance_func(query, self._vectors[vector_idx])
                all_candidates.append((vector_idx, distance))
            
            # Sort by distance and take top k
            all_candidates.sort(key=lambda x: x[1])
            nearest = all_candidates[:k]
        
        # Convert to vector IDs and return top k
        results = []
        for vector_idx, distance in nearest[:k]:
            vector_id = self._vector_ids[vector_idx]
            results.append((vector_id, distance))
        
        return results
    
    def search_batch(self, queries: np.ndarray, k: int) -> List[List[Tuple[int, float]]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: Array of query vectors
            k: Number of results per query
            
        Returns:
            List of result lists for each query
        """
        results = []
        for query in queries:
            query_results = self.search(query, k)
            results.append(query_results)
        return results
    
    def optimize(self) -> None:
        """Optimize the index structure."""
        # Rebalance connections if needed
        for layer in range(len(self._layers)):
            for vector_idx in list(self._layers[layer].keys()):
                connections = self._layers[layer][vector_idx]
                if len(connections) > self.m * 2:
                    # Keep only closest connections
                    if self._vectors is not None:
                        distances = [
                            (neighbor_idx, self._distance_func(
                                self._vectors[vector_idx], 
                                self._vectors[neighbor_idx]
                            ))
                            for neighbor_idx in connections
                        ]
                        distances.sort(key=lambda x: x[1])
                        best_connections = [idx for idx, _ in distances[:self.m]]
                        
                        # Update connections
                        self._layers[layer][vector_idx] = best_connections
                        
                        # Remove reverse connections
                        for neighbor_idx in best_connections:
                            if neighbor_idx in self._layers[layer]:
                                if vector_idx in self._layers[layer][neighbor_idx]:
                                    self._layers[layer][neighbor_idx].remove(vector_idx)
        
        logger.info("HNSW index optimization completed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = {
            "index_type": "hnsw",
            "dimension": self.dimension,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "max_elements": self.max_elements,
            "distance_metric": self.distance_metric,
            "num_vectors": len(self._vector_ids) if self._vector_ids else 0,
            "num_layers": len(self._layers),
            "entry_point": self._entry_point,
            "entry_layer": self._entry_layer
        }
        
        # Connection statistics
        if self._layers:
            total_connections = sum(len(connections) for connections in self._layers[0].values())
            avg_connections = total_connections / len(self._layers[0]) if self._layers[0] else 0
            stats["total_connections"] = total_connections
            stats["avg_connections"] = avg_connections
        
        return stats


class FlatIndex:
    """
    Simple flat index for exact search (used for small datasets or testing).
    """
    
    def __init__(self, dimension: int, distance_metric: str = "cosine"):
        self.dimension = dimension
        self.distance_metric = distance_metric
        self._vectors: Optional[np.ndarray] = None
        self._vector_ids: Optional[List[int]] = None
        self._distance_func = self._get_distance_function()
    
    def _get_distance_function(self):
        """Get the appropriate distance function."""
        if self.distance_metric == "cosine":
            return self._cosine_distance
        elif self.distance_metric == "euclidean":
            return self._euclidean_distance
        elif self.distance_metric == "dot":
            return self._dot_distance
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot_product / (norm_a * norm_b))
    
    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance between vectors."""
        return np.linalg.norm(a - b)
    
    def _dot_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute negative dot product distance."""
        return -np.dot(a, b)
    
    def build(self, vectors: np.ndarray, vector_ids: List[int]) -> None:
        """Build the flat index."""
        self._vectors = vectors.astype(np.float32)
        self._vector_ids = vector_ids
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors using exact computation."""
        if self._vectors is None:
            return []
        
        # Compute distances to all vectors
        distances = []
        for i, vector in enumerate(self._vectors):
            distance = self._distance_func(query, vector)
            distances.append((i, distance))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        
        results = []
        for vector_idx, distance in distances[:k]:
            vector_id = self._vector_ids[vector_idx]
            results.append((vector_id, distance))
        
        return results
    
    def search_batch(self, queries: np.ndarray, k: int) -> List[List[Tuple[int, float]]]:
        """Batch search for multiple queries."""
        results = []
        for query in queries:
            query_results = self.search(query, k)
            results.append(query_results)
        return results
    
    def optimize(self) -> None:
        """No optimization needed for flat index."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "index_type": "flat",
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "num_vectors": len(self._vector_ids) if self._vector_ids else 0
        }
