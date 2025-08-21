"""
Collection management for OctaneDB.
Handles vector storage, indexing, and operations within a collection.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

from .index import HNSWIndex
from .embeddings import TextEmbedder, ChromaCompatibleEmbedder

logger = logging.getLogger(__name__)


class Collection:
    """
    A collection of vectors with metadata and indexing capabilities.
    
    Features:
    - Vector storage and retrieval
    - Metadata management
    - Automatic indexing (HNSW)
    - Text embedding generation
    - ChromaDB-compatible API
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
        storage_manager=None,
        query_engine=None,
        vector_utils=None,
        embedding_model=None,
        enable_text_embeddings: bool = True
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
            embedding_model: Sentence-transformers model name for text embeddings
            enable_text_embeddings: Whether to enable text embedding functionality
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
        
        # Text document storage
        self._documents: Dict[int, str] = {}
        self._text_embedder = None
        
        # Initialize text embeddings if enabled
        if enable_text_embeddings and embedding_model:
            try:
                self._text_embedder = TextEmbedder(embedding_model)
                # Update dimension if it doesn't match
                if self._text_embedder.dimension != dimension:
                    logger.warning(f"Embedding model dimension ({self._text_embedder.dimension}) "
                                 f"doesn't match collection dimension ({dimension})")
                    self.dimension = self._text_embedder.dimension
            except ImportError:
                logger.warning("Text embeddings disabled: sentence-transformers not available")
                self._text_embedder = None
        
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
            "text_documents": 0,
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
        vectors, 
        metadata=None,
        ids=None
    ):
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
            
            # Update next_id (only for integer IDs)
            if isinstance(vector_id, int):
                self._next_id = max(self._next_id, vector_id + 1)
        
        # Mark index for rebuild
        self._index_needs_rebuild = True
        
        # Update stats
        self._stats["inserts"] += len(vectors)
        
        logger.debug(f"Inserted {len(vectors)} vectors into collection '{self.name}'")
        
        # Return single ID or list based on input
        return inserted_ids[0] if len(inserted_ids) == 1 else inserted_ids
    
    def add_text_documents(
        self,
        documents,
        ids=None,
        metadatas=None,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ):
        """
        Add text documents with automatic embedding generation (ChromaDB-compatible).
        
        Args:
            documents: Text document(s) to add
            ids: Document IDs (auto-generated if not provided)
            metadatas: Optional metadata for documents
            batch_size: Batch size for embedding generation
            show_progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary with 'ids', 'embeddings', and 'documents' keys
        """
        if not self._text_embedder:
            raise RuntimeError("Text embeddings not enabled. Set embedding_model during collection creation.")
        
        # Ensure documents is a list
        if isinstance(documents, str):
            documents = [documents]
        
        if not documents:
            raise ValueError("No documents provided")
        
        num_docs = len(documents)
        
        # Handle IDs
        if ids is None:
            # Auto-generate string IDs
            ids = [f"doc_{self._next_id + i}" for i in range(num_docs)]
        elif isinstance(ids, str):
            ids = [ids]
        
        if len(ids) != num_docs:
            raise ValueError("Number of IDs must match number of documents")
        
        # Handle metadata
        if metadatas is None:
            metadatas = [{} for _ in range(num_docs)]
        elif isinstance(metadatas, dict):
            metadatas = [metadatas for _ in range(num_docs)]
        
        if len(metadatas) != num_docs:
            raise ValueError("Number of metadatas must match number of documents")
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {num_docs} documents...")
        embeddings = self._text_embedder.embed_texts(
            documents, 
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )
        
        # Store documents
        for doc_id, doc_text in zip(ids, documents):
            self._documents[doc_id] = doc_text
        
        # Insert vectors into collection
        vector_ids = self.insert(
            vectors=embeddings,
            metadata=metadatas,
            ids=ids
        )
        
        # Update stats
        self._stats["text_documents"] += num_docs
        
        # Prepare result
        result = {
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
            "metadatas": metadatas,
            "vector_ids": vector_ids
        }
        
        logger.info(f"Added {num_docs} text documents with IDs: {ids}")
        return result
    
    def add(
        self,
        ids=None,
        documents=None,
        metadatas=None,
        embeddings=None
    ):
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
        if documents is not None:
            # Use text document processing
            return self.add_text_documents(documents, ids, metadatas)
        elif embeddings is not None:
            # Use pre-computed embeddings
            if isinstance(embeddings, list) and isinstance(embeddings[0], (int, float)):
                embeddings = [embeddings]
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Handle IDs
            if ids is None:
                ids = [f"vec_{self._next_id + i}" for i in range(len(embeddings))]
            elif isinstance(ids, str):
                ids = [ids]
            
            # Handle metadata
            if metadatas is None:
                metadatas = [{} for _ in range(len(embeddings))]
            elif isinstance(metadatas, dict):
                metadatas = [metadatas for _ in range(len(embeddings))]
            
            # Insert vectors
            vector_ids = self.insert(vectors=embeddings, metadata=metadatas, ids=ids)
            
            return {
                "ids": ids,
                "embeddings": embeddings,
                "vector_ids": vector_ids
            }
        else:
            raise ValueError("Either 'documents' or 'embeddings' must be provided")
    
    def get_document(self, doc_id: str):
        """Get a text document by ID."""
        return self._documents.get(doc_id)
    
    def list_documents(self):
        """List all text document IDs."""
        return list(self._documents.keys())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a text document by ID."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            # Also delete the corresponding vector if it exists
            if doc_id in self._vectors:
                del self._vectors[doc_id]
            if doc_id in self._metadata:
                del self._metadata[doc_id]
            logger.info(f"Deleted document: {doc_id}")
            return True
        return False
    
    def clear_documents(self):
        """Clear all text documents and vectors."""
        self._documents.clear()
        self._vectors.clear()
        self._metadata.clear()
        self._next_id = 0
        self._index_built = False
        self._index_needs_rebuild = False
        logger.info("Cleared all documents and vectors")
    
    def get_collection_info(self):
        """Get comprehensive information about the collection."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "num_vectors": len(self._vectors),
            "num_documents": len(self._documents),
            "index_type": self.index_type,
            "distance_metric": self.distance_metric,
            "index_built": self._index_built,
            "text_embeddings_enabled": self._text_embedder is not None,
            "embedding_model": self._text_embedder.model_name if self._text_embedder else None,
            "stats": self._stats.copy()
        }
    
    def change_embedding_model(self, model_name: str, **kwargs):
        """
        Change the text embedding model.
        
        Args:
            model_name: New sentence-transformers model name
            **kwargs: Additional arguments for TextEmbedder
        """
        if not self._text_embedder:
            raise RuntimeError("Text embeddings not enabled")
        
        old_dimension = self.dimension
        self._text_embedder.change_model(model_name, **kwargs)
        
        # Update collection dimension if it changed
        if self._text_embedder.dimension != old_dimension:
            logger.warning(f"Embedding model dimension changed from {old_dimension} to {self._text_embedder.dimension}")
            self.dimension = self._text_embedder.dimension
            # Rebuild index with new dimension
            self._index_needs_rebuild = True
        
        logger.info(f"Embedding model changed to: {model_name}")
    
    def search(
        self, 
        query_vector, 
        k: int = 10, 
        filter=None,
        include_metadata: bool = False
    ):
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
    
    def search_text(
        self,
        query_text: str,
        k: int = 10,
        filter=None,
        include_metadata: bool = False
    ):
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
        if not self._text_embedder:
            raise RuntimeError("Text embeddings not enabled")
        
        # Generate embedding for query text
        query_embedding = self._text_embedder.embed_texts(query_text)[0]
        
        # Search using the embedding
        return self.search(
            query_vector=query_embedding,
            k=k,
            filter=filter,
            include_metadata=include_metadata
        )
    
    def search_batch(
        self, 
        query_vectors, 
        k: int = 10, 
        filter=None,
        include_metadata: bool = False
    ):
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
    
    def search_text_batch(
        self,
        query_texts,
        k: int = 10,
        filter=None,
        include_metadata: bool = False,
        batch_size: int = 32
    ):
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
        if not self._text_embedder:
            raise RuntimeError("Text embeddings not enabled")
        
        # Generate embeddings for all query texts
        query_embeddings = self._text_embedder.embed_texts(
            query_texts, 
            batch_size=batch_size
        )
        
        # Search using the embeddings
        return self.search_batch(
            query_vectors=query_embeddings,
            k=k,
            filter=filter,
            include_metadata=include_metadata
        )
    
    def _apply_filter(self, results, filter):
        """
        Apply metadata filter to search results.
        
        Args:
            results: Search results (id, distance) tuples
            filter: Filter criteria
            
        Returns:
            Filtered results
        """
        if not self._query_engine:
            logger.warning("Query engine not available, returning unfiltered results")
            return results
        
        # Apply filter using query engine
        filtered_results = []
        for vector_id, distance in results:
            metadata = self._metadata.get(vector_id, {})
            if self._query_engine.evaluate(metadata, filter):
                filtered_results.append((vector_id, distance))
        
        return filtered_results
    
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
    
    def update(self, id: int, vector, metadata=None) -> None:
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
    
    def delete_batch(self, ids) -> None:
        """
        Batch delete vectors.
        
        Args:
            ids: List of vector IDs to delete
        """
        for vector_id in ids:
            self.delete(vector_id)
    
    def get_vector(self, id: int, include_metadata: bool = False):
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
    
    def get_stats(self):
        """Get collection statistics."""
        stats = self._stats.copy()
        stats["vector_count"] = len(self._vectors)
        stats["metadata_count"] = len(self._metadata)
        stats["document_count"] = len(self._documents)
        stats["index_built"] = self._index_built
        stats["index_needs_rebuild"] = self._index_needs_rebuild
        return stats
    
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
        self._documents.clear()
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
