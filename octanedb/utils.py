"""
Utility classes for vector operations, distance metrics, and helper functions.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)


class DistanceMetrics(Enum):
    """Supported distance metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    JACCARD = "jaccard"


class VectorUtils:
    """
    Utility class for vector operations and distance computations.
    
    Provides:
    - Fast distance calculations
    - Vector normalization
    - Batch operations
    - Performance optimizations
    """
    
    def __init__(self, distance_metric: str = "cosine"):
        """
        Initialize vector utilities.
        
        Args:
            distance_metric: Default distance metric to use
        """
        self.distance_metric = distance_metric
        self._distance_func = self._get_distance_function()
        
        logger.info(f"Vector utilities initialized with metric: {distance_metric}")
    
    def _get_distance_function(self):
        """Get the appropriate distance function."""
        if self.distance_metric == "cosine":
            return self.cosine_distance
        elif self.distance_metric == "euclidean":
            return self.euclidean_distance
        elif self.distance_metric == "dot":
            return self.dot_distance
        elif self.distance_metric == "manhattan":
            return self.manhattan_distance
        elif self.distance_metric == "chebyshev":
            return self.chebyshev_distance
        elif self.distance_metric == "jaccard":
            return self.jaccard_distance
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine distance (0 = identical, 2 = opposite)
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0
        
        cosine_similarity = dot_product / (norm_a * norm_b)
        return 1.0 - cosine_similarity
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Euclidean distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(a - b)
    
    @staticmethod
    def dot_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute negative dot product distance.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Negative dot product
        """
        return -np.dot(a, b)
    
    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Manhattan (L1) distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Manhattan distance
        """
        return np.sum(np.abs(a - b))
    
    @staticmethod
    def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Chebyshev (L∞) distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Chebyshev distance
        """
        return np.max(np.abs(a - b))
    
    @staticmethod
    def jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Jaccard distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Jaccard distance
        """
        intersection = np.sum(np.minimum(a, b))
        union = np.sum(np.maximum(a, b))
        
        if union == 0:
            return 1.0
        
        return 1.0 - (intersection / union)
    
    def compute_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute distance between two vectors using the configured metric.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Distance value
        """
        return self._distance_func(a, b)
    
    def compute_distances_batch(
        self, 
        query: np.ndarray, 
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute distances between a query vector and multiple vectors.
        
        Args:
            query: Query vector
            vectors: Array of vectors to compare against
            
        Returns:
            Array of distances
        """
        if self.distance_metric == "cosine":
            # Optimized cosine distance for batch computation
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                return np.ones(len(vectors))
            
            # Normalize query
            query_normalized = query / query_norm
            
            # Compute norms for all vectors
            vector_norms = np.linalg.norm(vectors, axis=1)
            
            # Avoid division by zero
            valid_norms = vector_norms > 0
            distances = np.ones(len(vectors))
            
            if np.any(valid_norms):
                # Compute dot products
                dot_products = np.dot(vectors[valid_norms], query_normalized)
                
                # Compute cosine similarities
                cosine_similarities = dot_products / vector_norms[valid_norms]
                
                # Convert to distances
                distances[valid_norms] = 1.0 - cosine_similarities
            
            return distances
        
        elif self.distance_metric == "euclidean":
            # Optimized Euclidean distance for batch computation
            return np.linalg.norm(vectors - query, axis=1)
        
        else:
            # Fallback to individual computation
            distances = np.zeros(len(vectors))
            for i, vector in enumerate(vectors):
                distances[i] = self._distance_func(query, vector)
            return distances
    
    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Vector to normalize
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """
        Normalize multiple vectors to unit length.
        
        Args:
            vectors: Array of vectors to normalize
            
        Returns:
            Array of normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        return vectors / norms
    
    @staticmethod
    def compute_centroid(vectors: np.ndarray) -> np.ndarray:
        """
        Compute the centroid (mean) of multiple vectors.
        
        Args:
            vectors: Array of vectors
            
        Returns:
            Centroid vector
        """
        if len(vectors) == 0:
            raise ValueError("Cannot compute centroid of empty vector set")
        
        return np.mean(vectors, axis=0)
    
    @staticmethod
    def compute_variance(vectors: np.ndarray) -> np.ndarray:
        """
        Compute the variance of multiple vectors.
        
        Args:
            vectors: Array of vectors
            
        Returns:
            Variance vector
        """
        if len(vectors) < 2:
            raise ValueError("Need at least 2 vectors to compute variance")
        
        return np.var(vectors, axis=0)
    
    @staticmethod
    def find_nearest_neighbors(
        query: np.ndarray,
        vectors: np.ndarray,
        k: int,
        distance_func: callable = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors using brute force search.
        
        Args:
            query: Query vector
            vectors: Array of vectors to search in
            k: Number of neighbors to find
            distance_func: Distance function to use
            
        Returns:
            Tuple of (indices, distances)
        """
        if distance_func is None:
            distance_func = VectorUtils.cosine_distance
        
        # Compute distances
        distances = np.array([
            distance_func(query, vector) for vector in vectors
        ])
        
        # Find k nearest neighbors
        if k >= len(vectors):
            indices = np.arange(len(vectors))
        else:
            indices = np.argpartition(distances, k)[:k]
        
        # Sort by distance
        sorted_indices = indices[np.argsort(distances[indices])]
        sorted_distances = distances[sorted_indices]
        
        return sorted_indices, sorted_distances
    
    @staticmethod
    def compute_similarity_matrix(vectors: np.ndarray, metric: str = "cosine") -> np.ndarray:
        """
        Compute similarity matrix between all pairs of vectors.
        
        Args:
            vectors: Array of vectors
            metric: Similarity metric to use
            
        Returns:
            Similarity matrix
        """
        n = len(vectors)
        similarity_matrix = np.zeros((n, n))
        
        if metric == "cosine":
            # Normalize vectors
            normalized_vectors = VectorUtils.normalize_vectors(vectors)
            
            # Compute cosine similarities
            similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
            
            # Ensure diagonal is 1.0 (self-similarity)
            np.fill_diagonal(similarity_matrix, 1.0)
        
        elif metric == "dot":
            # Compute dot products
            similarity_matrix = np.dot(vectors, vectors.T)
        
        else:
            # Compute distances and convert to similarities
            for i in range(n):
                for j in range(n):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        distance = VectorUtils.compute_distance(vectors[i], vectors[j])
                        # Convert distance to similarity (inverse relationship)
                        similarity_matrix[i, j] = 1.0 / (1.0 + distance)
        
        return similarity_matrix
    
    @staticmethod
    def compute_diversity_score(vectors: np.ndarray, metric: str = "cosine") -> float:
        """
        Compute diversity score for a set of vectors.
        
        Args:
            vectors: Array of vectors
            metric: Distance metric to use
            
        Returns:
            Diversity score (higher = more diverse)
        """
        if len(vectors) < 2:
            return 0.0
        
        # Compute pairwise distances
        total_distance = 0.0
        count = 0
        
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                distance = VectorUtils.compute_distance(vectors[i], vectors[j])
                total_distance += distance
                count += 1
        
        # Return average distance
        return total_distance / count if count > 0 else 0.0
    
    @staticmethod
    def compute_quality_score(
        query: np.ndarray,
        results: np.ndarray,
        ground_truth: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute quality metrics for search results.
        
        Args:
            query: Query vector
            results: Retrieved result vectors
            ground_truth: Ground truth relevant vectors (optional)
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Compute average distance to query
        distances = np.array([
            VectorUtils.cosine_distance(query, result) for result in results
        ])
        
        metrics["avg_distance"] = float(np.mean(distances))
        metrics["min_distance"] = float(np.min(distances))
        metrics["max_distance"] = float(np.max(distances))
        metrics["std_distance"] = float(np.std(distances))
        
        # Compute diversity of results
        if len(results) > 1:
            metrics["diversity"] = VectorUtils.compute_diversity_score(results)
        
        # Compute precision/recall if ground truth is provided
        if ground_truth is not None:
            # This is a simplified implementation
            # In practice, you'd need relevance scores or binary relevance
            metrics["num_relevant"] = len(ground_truth)
            metrics["num_retrieved"] = len(results)
        
        return metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector utilities statistics."""
        return {
            "distance_metric": self.distance_metric,
            "supported_metrics": [metric.value for metric in DistanceMetrics]
        }


class PerformanceOptimizer:
    """
    Performance optimization utilities for vector operations.
    """
    
    @staticmethod
    def optimize_batch_size(
        vector_dimension: int,
        available_memory: int = None,
        target_latency: float = 0.1
    ) -> int:
        """
        Optimize batch size for vector operations.
        
        Args:
            vector_dimension: Dimension of vectors
            available_memory: Available memory in bytes
            target_latency: Target latency in seconds
            
        Returns:
            Optimal batch size
        """
        # Estimate memory per vector (float32)
        bytes_per_vector = vector_dimension * 4
        
        if available_memory:
            # Use 80% of available memory
            usable_memory = int(available_memory * 0.8)
            max_batch_size = usable_memory // bytes_per_vector
        else:
            # Default to reasonable batch size
            max_batch_size = 10000
        
        # Adjust based on dimension
        if vector_dimension > 1000:
            max_batch_size = min(max_batch_size, 1000)
        elif vector_dimension > 100:
            max_batch_size = min(max_batch_size, 5000)
        
        return max(1, min(max_batch_size, 100000))
    
    @staticmethod
    def estimate_memory_usage(
        num_vectors: int,
        vector_dimension: int,
        include_index: bool = True
    ) -> Dict[str, int]:
        """
        Estimate memory usage for vector storage.
        
        Args:
            num_vectors: Number of vectors
            vector_dimension: Dimension of vectors
            include_index: Whether to include index memory
            
        Returns:
            Dictionary of memory usage estimates
        """
        # Vector storage (float32)
        vector_memory = num_vectors * vector_dimension * 4
        
        # Metadata storage (estimated)
        metadata_memory = num_vectors * 100  # Rough estimate
        
        # Index memory
        index_memory = 0
        if include_index:
            # HNSW index memory (rough estimate)
            index_memory = num_vectors * vector_dimension * 2
        
        total_memory = vector_memory + metadata_memory + index_memory
        
        return {
            "vectors": vector_memory,
            "metadata": metadata_memory,
            "index": index_memory,
            "total": total_memory
        }
    
    @staticmethod
    def get_optimal_threads(num_vectors: int, vector_dimension: int) -> int:
        """
        Get optimal number of threads for parallel processing.
        
        Args:
            num_vectors: Number of vectors
            vector_dimension: Dimension of vectors
            
        Returns:
            Optimal number of threads
        """
        import os
        
        # Get CPU count
        cpu_count = os.cpu_count() or 1
        
        # Adjust based on workload
        if num_vectors < 1000 or vector_dimension < 100:
            return min(cpu_count, 2)
        elif num_vectors < 10000:
            return min(cpu_count, 4)
        else:
            return min(cpu_count, 8)
