"""
Text embedding generation for OctaneDB.
Provides automatic text-to-vector conversion using sentence-transformers.
"""

import logging
from typing import List, Union, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


class TextEmbedder:
    """
    Text embedding generator using sentence-transformers.
    
    Provides automatic text-to-vector conversion with support for:
    - Multiple embedding models
    - Batch processing
    - Custom model configuration
    - GPU acceleration (if available)
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        **kwargs
    ):
        """
        Initialize the text embedder.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to use ('cpu', 'cuda', or None for auto)
            normalize_embeddings: Whether to normalize embeddings
            **kwargs: Additional arguments for SentenceTransformer
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for text embedding. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        
        # Auto-detect device if not specified
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA detected, using GPU for embeddings")
            else:
                device = "cpu"
                logger.info("Using CPU for embeddings")
        
        self.device = device
        
        # Initialize the model
        logger.info(f"Loading sentence-transformers model: {model_name}")
        self.model = SentenceTransformer(
            model_name, 
            device=device,
            **kwargs
        )
        
        # Get model info
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
        
        # Performance tracking
        self._stats = {
            "embeddings_generated": 0,
            "total_tokens": 0,
            "total_time": 0.0
        }
    
    def embed_texts(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress bar
            
        Returns:
            numpy array of embeddings with shape (n_texts, dimension)
        """
        import time
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        start_time = time.time()
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True
            )
            
            # Ensure correct shape
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Update stats
            self._stats["embeddings_generated"] += len(texts)
            self._stats["total_tokens"] += sum(len(text.split()) for text in texts)
            self._stats["total_time"] += time.time() - start_time
            
            logger.debug(f"Generated {len(texts)} embeddings in {time.time() - start_time:.4f}s")
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy array of embedding with shape (1, dimension)
        """
        return self.embed_texts([text])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "stats": self._stats.copy()
        }
    
    def change_model(self, model_name: str, **kwargs):
        """
        Change the embedding model.
        
        Args:
            model_name: New model name
            **kwargs: Additional arguments for SentenceTransformer
        """
        logger.info(f"Changing model from {self.model_name} to {model_name}")
        
        # Save current device
        device = self.device
        
        # Initialize new model
        self.model = SentenceTransformer(
            model_name, 
            device=device,
            **kwargs
        )
        
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model changed successfully. New dimension: {self.dimension}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available sentence-transformers models."""
        # Common models that work well for most use cases
        return [
            "all-MiniLM-L6-v2",      # 384 dimensions, fast, good quality
            "all-MiniLM-L12-v2",     # 384 dimensions, better quality, slower
            "all-mpnet-base-v2",     # 768 dimensions, high quality
            "all-MiniLM-L6-v2",      # 384 dimensions, multilingual
            "paraphrase-MiniLM-L6-v2", # 384 dimensions, paraphrase-focused
            "distiluse-base-multilingual-cased-v2", # 512 dimensions, multilingual
        ]


class ChromaCompatibleEmbedder(TextEmbedder):
    """
    ChromaDB-compatible text embedder.
    
    Provides the same API as ChromaDB for easy migration:
    - add() method for documents
    - Automatic ID generation
    - Batch processing
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs
    ):
        """
        Initialize ChromaDB-compatible embedder.
        
        Args:
            model_name: Sentence-transformers model name
            **kwargs: Additional arguments for TextEmbedder
        """
        super().__init__(model_name, **kwargs)
        
        # Document storage (for ChromaDB compatibility)
        self._documents: Dict[str, str] = {}
        self._next_id = 0
    
    def add(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        documents: Union[str, List[str]] = None,
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        embeddings: Optional[Union[List[float], List[List[float]]]] = None
    ) -> Dict[str, Any]:
        """
        Add documents to the collection (ChromaDB-compatible API).
        
        Args:
            ids: Document IDs (auto-generated if not provided)
            documents: Text documents to add
            metadatas: Optional metadata for documents
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            Dictionary with 'ids' and 'embeddings' keys
        """
        # Ensure documents is a list
        if isinstance(documents, str):
            documents = [documents]
        
        if not documents:
            raise ValueError("No documents provided")
        
        num_docs = len(documents)
        
        # Handle IDs
        if ids is None:
            # Auto-generate IDs
            ids = [f"doc_{self._next_id + i}" for i in range(num_docs)]
            self._next_id += num_docs
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
        
        # Store documents
        for doc_id, doc_text in zip(ids, documents):
            self._documents[doc_id] = doc_text
        
        # Generate or use provided embeddings
        if embeddings is None:
            # Generate embeddings automatically
            embeddings = self.embed_texts(documents)
        else:
            # Convert to numpy array
            embeddings = np.array(embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
        
        # Prepare result
        result = {
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
            "metadatas": metadatas
        }
        
        logger.info(f"Added {num_docs} documents with IDs: {ids}")
        return result
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Get a document by ID."""
        return self._documents.get(doc_id)
    
    def list_documents(self) -> List[str]:
        """List all document IDs."""
        return list(self._documents.keys())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            logger.info(f"Deleted document: {doc_id}")
            return True
        return False
    
    def clear_documents(self):
        """Clear all documents."""
        self._documents.clear()
        self._next_id = 0
        logger.info("Cleared all documents")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the document collection."""
        return {
            "num_documents": len(self._documents),
            "model_info": self.get_model_info(),
            "document_ids": list(self._documents.keys())
        }


# Convenience function for quick embedding
def embed_texts(
    texts: Union[str, List[str]], 
    model_name: str = "all-MiniLM-L6-v2",
    **kwargs
) -> np.ndarray:
    """
    Quick function to generate embeddings for texts.
    
    Args:
        texts: Text(s) to embed
        model_name: Model to use
        **kwargs: Additional arguments for TextEmbedder
        
    Returns:
        numpy array of embeddings
    """
    embedder = TextEmbedder(model_name, **kwargs)
    return embedder.embed_texts(texts)
