"""
OctaneDB Core - Main database interface.
Provides high-level operations for vector database management.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import numpy as np

from .collection import Collection
from .storage import StorageManager
from .embeddings import TextEmbedder, ChromaCompatibleEmbedder

logger = logging.getLogger(__name__)


class OctaneDB:
    """
    OctaneDB - Lightning Fast Vector Database.
    
    A lightweight, high-performance vector database library that provides:
    - Fast vector similarity search using HNSW indexing
    - Automatic text embedding generation
    - ChromaDB-compatible API
    - Multiple storage modes (in-memory, persistent, hybrid)
    - Advanced metadata filtering
    - Batch operations for improved performance
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "hnsw",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        max_elements: int = 1000000,
        distance_metric: str = "cosine",
        storage_path: Optional[Union[str, Path]] = None,
        embedding_model: Optional[str] = None,
        enable_text_embeddings: bool = True,
        **kwargs
    ):
        """
        Initialize OctaneDB.
        
        Args:
            dimension: Vector dimension
            index_type: Type of index to use
            m: HNSW connections per layer
            ef_construction: Construction search depth
            ef_search: Search depth
            max_elements: Maximum number of vectors
            distance_metric: Distance metric for similarity
            storage_path: Path for persistent storage
            embedding_model: Sentence-transformers model for text embeddings
            enable_text_embeddings: Whether to enable text embedding functionality
            **kwargs: Additional arguments for collections
        """
        self.dimension = dimension
        self.index_type = index_type
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements
        self.distance_metric = distance_metric
        self.embedding_model = embedding_model
        self.enable_text_embeddings = enable_text_embeddings
        
        # Storage management
        self._storage_path = Path(storage_path) if storage_path else None
        self._storage_manager = StorageManager(storage_path=self._storage_path) if self._storage_path else None
        
        # Collection management
        self._collections: Dict[str, Collection] = {}
        self._current_collection: Optional[Collection] = None
        
        # Text embedding support
        self._text_embedder: Optional[TextEmbedder] = None
        if enable_text_embeddings and embedding_model:
            try:
                self._text_embedder = TextEmbedder(embedding_model)
                # Update dimension if it doesn't match
                if self._text_embedder.dimension != dimension:
                    logger.warning(f"Embedding model dimension ({self._text_embedder.dimension}) "
                                 f"doesn't match specified dimension ({dimension})")
                    self.dimension = self._text_embedder.dimension
            except ImportError:
                logger.warning("Text embeddings disabled: sentence-transformers not available")
                self._text_embedder = None
        
        # Performance tracking
        self._stats = {
            "collections_created": 0,
            "total_vectors": 0,
            "total_documents": 0,
            "operations_performed": 0
        }
        
        logger.info(f"OctaneDB initialized with dimension {self.dimension}")
        if self._text_embedder:
            logger.info(f"Text embeddings enabled with model: {embedding_model}")
    
    def create_collection(
        self, 
        name: str, 
        **kwargs
    ) -> Collection:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            **kwargs: Additional collection parameters
            
        Returns:
            Created collection instance
        """
        if name in self._collections:
            raise ValueError(f"Collection '{name}' already exists")
        
        # Use embedding model from OctaneDB if not specified
        if "embedding_model" not in kwargs and self.embedding_model:
            kwargs["embedding_model"] = self.embedding_model
        
        # Create collection with text embedding support
        collection = Collection(
            name=name,
            dimension=self.dimension,
            index_type=self.index_type,
            m=self.m,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            max_elements=self.max_elements,
            distance_metric=self.distance_metric,
            enable_text_embeddings=self.enable_text_embeddings,
            **kwargs
        )
        
        self._collections[name] = collection
        
        # Set as current collection if it's the first one
        if self._current_collection is None:
            self._current_collection = collection
        
        # Update stats
        self._stats["collections_created"] += 1
        
        logger.info(f"Collection '{name}' created successfully")
        return collection
    
    def use_collection(self, name: str) -> Collection:
        """
        Set the current collection for operations.
        
        Args:
            name: Collection name
            
        Returns:
            Collection instance
        """
        if name not in self._collections:
            raise ValueError(f"Collection '{name}' does not exist")
        
        self._current_collection = self._collections[name]
        logger.debug(f"Switched to collection: {name}")
        return self._current_collection
    
    def get_collection(self, name: str) -> Collection:
        """
        Get a collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            Collection instance
        """
        if name not in self._collections:
            raise ValueError(f"Collection '{name}' does not exist")
        
        return self._collections[name]
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        return list(self._collections.keys())
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if deleted, False if not found
        """
        if name not in self._collections:
            return False
        
        # Clear current collection if it's the one being deleted
        if self._current_collection == self._collections[name]:
            self._current_collection = None
        
        # Delete collection
        del self._collections[name]
        
        # Set new current collection if needed
        if self._current_collection is None and self._collections:
            self._current_collection = list(self._collections.values())[0]
        
        logger.info(f"Collection '{name}' deleted")
        return True
    
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
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        result = self._current_collection.insert(vectors, metadata, ids)
        self._stats["operations_performed"] += 1
        self._update_total_stats()
        return result
    
    def add(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        documents: Union[str, List[str]] = None,
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        embeddings: Optional[Union[List[float], List[List[float]]]] = None
    ) -> Dict[str, Any]:
        """
        ChromaDB-compatible add method for text documents.
        
        Args:
            ids: Document IDs (auto-generated if not provided)
            documents: Text documents to add
            metadatas: Optional metadata for documents
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            Dictionary with 'ids' and 'embeddings' keys
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        result = self._current_collection.add(ids, documents, metadatas, embeddings)
        self._stats["operations_performed"] += 1
        self._update_total_stats()
        return result
    
    def add_text_documents(
        self,
        documents: Union[str, List[str]],
        ids: Optional[Union[str, List[str]]] = None,
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> Dict[str, Any]:
        """
        Add text documents with automatic embedding generation.
        
        Args:
            documents: Text document(s) to add
            ids: Document IDs (auto-generated if not provided)
            metadatas: Optional metadata for documents
            batch_size: Batch size for embedding generation
            show_progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary with 'ids', 'embeddings', and 'documents' keys
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        result = self._current_collection.add_text_documents(
            documents, ids, metadatas, batch_size, show_progress_bar
        )
        self._stats["operations_performed"] += 1
        self._update_total_stats()
        return result
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = False
    ) -> List[Tuple[int, float, Optional[Dict[str, Any]]]]:
        """
        Search for similar vectors in the current collection.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of (id, distance, metadata) tuples
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        result = self._current_collection.search(query_vector, k, filter, include_metadata)
        self._stats["operations_performed"] += 1
        return result
    
    def search_text(
        self,
        query_text: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = False
    ) -> List[Tuple[int, float, Optional[Dict[str, Any]]]]:
        """
        Search for similar documents using text query.
        
        Args:
            query_text: Text query to search for
            k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of (id, distance, metadata) tuples
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        result = self._current_collection.search_text(query_text, k, filter, include_metadata)
        self._stats["operations_performed"] += 1
        return result
    
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
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        result = self._current_collection.search_batch(query_vectors, k, filter, include_metadata)
        self._stats["operations_performed"] += 1
        return result
    
    def search_text_batch(
        self,
        query_texts: List[str],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = False,
        batch_size: int = 32
    ) -> List[List[Tuple[int, float, Optional[Dict[str, Any]]]]]:
        """
        Batch search for similar documents using text queries.
        
        Args:
            query_texts: List of text queries
            k: Number of results per query
            filter: Optional metadata filter
            include_metadata: Whether to include metadata in results
            batch_size: Batch size for embedding generation
            
        Returns:
            List of result lists for each query
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        result = self._current_collection.search_text_batch(
            query_texts, k, filter, include_metadata, batch_size
        )
        self._stats["operations_performed"] += 1
        return result
    
    def update(self, id: int, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update a vector in the current collection.
        
        Args:
            id: Vector ID to update
            vector: New vector
            metadata: New metadata
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        self._current_collection.update(id, vector, metadata)
        self._stats["operations_performed"] += 1
    
    def delete(self, id: int) -> None:
        """
        Delete a vector from the current collection.
        
        Args:
            id: Vector ID to delete
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        self._current_collection.delete(id)
        self._stats["operations_performed"] += 1
        self._update_total_stats()
    
    def delete_batch(self, ids: List[int]) -> None:
        """
        Batch delete vectors from the current collection.
        
        Args:
            ids: List of vector IDs to delete
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        self._current_collection.delete_batch(ids)
        self._stats["operations_performed"] += 1
        self._update_total_stats()
    
    def get_vector(self, id: int, include_metadata: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Optional[Dict[str, Any]]]]:
        """
        Get a vector from the current collection.
        
        Args:
            id: Vector ID
            include_metadata: Whether to include metadata
            
        Returns:
            Vector or (vector, metadata) tuple
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        return self._current_collection.get_vector(id, include_metadata)
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """
        Get a text document from the current collection.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document text or None if not found
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        return self._current_collection.get_document(doc_id)
    
    def list_documents(self) -> List[str]:
        """List all text document IDs in the current collection."""
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        return self._current_collection.list_documents()
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a text document from the current collection.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deleted, False if not found
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        result = self._current_collection.delete_document(doc_id)
        if result:
            self._stats["operations_performed"] += 1
            self._update_total_stats()
        return result
    
    def clear_documents(self) -> None:
        """Clear all text documents from the current collection."""
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        self._current_collection.clear_documents()
        self._stats["operations_performed"] += 1
        self._update_total_stats()
    
    def change_embedding_model(self, model_name: str, **kwargs) -> None:
        """
        Change the text embedding model for the current collection.
        
        Args:
            model_name: New sentence-transformers model name
            **kwargs: Additional arguments for TextEmbedder
        """
        if self._current_collection is None:
            raise RuntimeError("No current collection. Use create_collection() or use_collection() first.")
        
        self._current_collection.change_embedding_model(model_name, **kwargs)
        # Update OctaneDB dimension if it changed
        if self._current_collection.dimension != self.dimension:
            self.dimension = self._current_collection.dimension
            logger.info(f"OctaneDB dimension updated to: {self.dimension}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available sentence-transformers models."""
        if not self._text_embedder:
            return []
        return self._text_embedder.get_available_models()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        stats = self._stats.copy()
        stats["collection_count"] = len(self._collections)
        stats["current_collection"] = self._current_collection.name if self._current_collection else None
        stats["text_embeddings_enabled"] = self._text_embedder is not None
        stats["embedding_model"] = self._text_embedder.model_name if self._text_embedder else None
        
        # Add collection-specific stats
        if self._current_collection:
            collection_stats = self._current_collection.get_stats()
            stats.update(collection_stats)
        
        return stats
    
    def get_collection_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            name: Collection name (uses current collection if None)
            
        Returns:
            Collection information dictionary
        """
        if name is None:
            if not self._current_collection:
                raise RuntimeError("No current collection")
            return self._current_collection.get_collection_info()
        
        if name not in self._collections:
            raise ValueError(f"Collection '{name}' does not exist")
        
        return self._collections[name].get_collection_info()
    
    def _update_total_stats(self) -> None:
        """Update total statistics across all collections."""
        total_vectors = 0
        total_documents = 0
        
        for collection in self._collections.values():
            collection_stats = collection.get_stats()
            total_vectors += collection_stats.get("vector_count", 0)
            total_documents += collection_stats.get("document_count", 0)
        
        self._stats["total_vectors"] = total_vectors
        self._stats["total_documents"] = total_documents
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the database to persistent storage.
        
        Args:
            path: Optional custom save path
        """
        if not self._storage_manager:
            raise RuntimeError("Storage manager not initialized. Set storage_path during initialization.")
        
        save_path = Path(path) if path else self._storage_path
        self._storage_manager.save_database(self, save_path)
        logger.info(f"Database saved to: {save_path}")
    
    @classmethod
    def load(cls, path: str, **kwargs) -> "OctaneDB":
        """
        Load a database from persistent storage.
        
        Args:
            path: Path to the saved database
            **kwargs: Additional initialization parameters
            
        Returns:
            Loaded OctaneDB instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Database file not found: {path}")
        
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
        storage_manager.load_database(instance, path)
        
        logger.info(f"Database loaded from: {path}")
        return instance
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._storage_manager:
            self.save()
    
    def __len__(self) -> int:
        """Return total number of vectors across all collections."""
        return sum(len(collection) for collection in self._collections.values())
    
    def __contains__(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return collection_name in self._collections
