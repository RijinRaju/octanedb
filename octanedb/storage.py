"""
Storage manager for efficient persistence and loading of vector databases.
"""

import numpy as np
import h5py
import msgpack
import logging
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Storage manager for efficient persistence and loading of vector databases.
    
    Features:
    - HDF5-based storage with compression
    - Efficient metadata serialization
    - Incremental saves and loads
    - Memory-mapped access for large datasets
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        enable_cache: bool = True,
        cache_size: int = 10000,
        compression: str = "gzip",
        compression_opts: int = 6
    ):
        """
        Initialize storage manager.
        
        Args:
            storage_path: Base path for storage
            enable_cache: Enable caching for better performance
            cache_size: Maximum cache size
            compression: Compression algorithm
            compression_opts: Compression options
        """
        self.storage_path = storage_path
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.compression = compression
        self.compression_opts = compression_opts
        
        # Cache for frequently accessed data
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Create storage directory if it doesn't exist
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Storage manager initialized at {self.storage_path}")
    
    def save_database(self, collections: Dict[str, Any], save_path: Path) -> None:
        """
        Save the entire database to disk.
        
        Args:
            collections: Dictionary of collections to save
            save_path: Path to save the database
        """
        start_time = time.time()
        
        # Create HDF5 file
        with h5py.File(save_path, 'w') as f:
            # Save database metadata
            self._save_database_metadata(f, collections)
            
            # Save each collection
            for collection_name, collection in collections.items():
                self._save_collection(f, collection_name, collection)
        
        save_time = time.time() - start_time
        logger.info(f"Database saved to {save_path} in {save_time:.4f}s")
    
    def _save_database_metadata(self, f: h5py.File, collections: Dict[str, Any]) -> None:
        """Save database-level metadata."""
        # Database info
        db_info = {
            "version": "0.1.0",
            "created_at": time.time(),
            "num_collections": len(collections),
            "total_vectors": sum(c.count() for c in collections.values())
        }
        
        # Collection metadata
        collection_metadata = {}
        for name, collection in collections.items():
            collection_metadata[name] = {
                "dimension": collection.dimension,
                "index_type": collection.index_type,
                "m": collection.m,
                "ef_construction": collection.ef_construction,
                "ef_search": collection.ef_search,
                "max_elements": collection.max_elements,
                "distance_metric": collection.distance_metric,
                "vector_count": collection.count(),
                "metadata_count": len(collection._metadata)
            }
        
        # Save as datasets to avoid HDF5 attribute limitations
        f.create_dataset('db_info', data=np.frombuffer(msgpack.packb(db_info), dtype=np.uint8))
        f.create_dataset('collection_metadata', data=np.frombuffer(msgpack.packb(collection_metadata), dtype=np.uint8))
    
    def _save_collection(self, f: h5py.File, collection_name: str, collection: Any) -> None:
        """Save a single collection."""
        # Create collection group
        collection_group = f.create_group(collection_name)
        
        # Save collection metadata
        collection_group.attrs['name'] = collection_name
        collection_group.attrs['dimension'] = collection.dimension
        collection_group.attrs['index_type'] = collection.index_type
        collection_group.attrs['distance_metric'] = collection.distance_metric
        
        # Save vectors
        if collection._vectors:
            vectors_array = np.array([v for v in collection._vectors.values()], dtype=np.float32)
            vector_ids = list(collection._vectors.keys())
            
            # Save vectors with compression
            collection_group.create_dataset(
                'vectors',
                data=vectors_array,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            
            # Save vector IDs
            collection_group.create_dataset(
                'vector_ids',
                data=vector_ids,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
        
        # Save metadata
        if collection._metadata:
            # Convert integer keys to strings for msgpack compatibility
            metadata_for_storage = {str(k): v for k, v in collection._metadata.items()}
            # Convert metadata to bytes
            metadata_bytes = msgpack.packb(metadata_for_storage)
            collection_group.create_dataset(
                'metadata',
                data=np.frombuffer(metadata_bytes, dtype=np.uint8),
                compression=self.compression,
                compression_opts=self.compression_opts
            )
        
        # Save index if built
        if collection._index and collection._index_built:
            self._save_index(collection_group, collection._index)
        
        # Save collection stats
        stats = collection.get_stats()
        collection_group.create_dataset('stats', data=np.frombuffer(msgpack.packb(stats), dtype=np.uint8))
    
    def _save_index(self, collection_group: h5py.Group, index) -> None:
        """Save index data."""
        if hasattr(index, 'get_stats'):
            index_stats = index.get_stats()
            collection_group.create_dataset('index_stats', data=np.frombuffer(msgpack.packb(index_stats), dtype=np.uint8))
    
    def load_database(self, load_path: Path) -> Dict[str, Any]:
        """
        Load the entire database from disk.
        
        Args:
            load_path: Path to load the database from
            
        Returns:
            Dictionary of loaded collections
        """
        start_time = time.time()
        
        collections = {}
        
        with h5py.File(load_path, 'r') as f:
            # Load database metadata from datasets
            db_info = msgpack.unpackb(bytes(f['db_info'][:]))
            collection_metadata = msgpack.unpackb(bytes(f['collection_metadata'][:]))
            
            # Load each collection
            for collection_name in f.keys():
                if collection_name not in ['db_info', 'collection_metadata']:
                    collection = self._load_collection(f[collection_name], collection_name)
                    collections[collection_name] = collection
        
        load_time = time.time() - start_time
        logger.info(f"Database loaded from {load_path} in {load_time:.4f}s")
        
        return collections
    
    def load_database_metadata(self, load_path: Path) -> Dict[str, Any]:
        """Load only database metadata without loading vectors."""
        with h5py.File(load_path, 'r') as f:
            db_info = msgpack.unpackb(bytes(f['db_info'][:]))
            collection_metadata = msgpack.unpackb(bytes(f['collection_metadata'][:]))
            
            # Combine metadata
            metadata = db_info.copy()
            metadata['collections'] = collection_metadata
            
            return metadata
    
    def _load_collection(self, collection_group: h5py.Group, collection_name: str) -> Any:
        """Load a single collection."""
        # Get collection metadata from attributes (primary source)
        dimension = collection_group.attrs.get('dimension')
        index_type = collection_group.attrs.get('index_type', 'hnsw')
        distance_metric = collection_group.attrs.get('distance_metric', 'cosine')
        
        # Validate that we have the essential dimension
        if dimension is None:
            raise ValueError(f"Collection '{collection_name}' missing dimension information")
        
        # Create collection using string import to avoid circular dependency
        from .collection import Collection
        collection = Collection(
            name=collection_name,
            dimension=dimension,
            index_type=index_type,
            distance_metric=distance_metric
        )
        
        # Load vectors
        if 'vectors' in collection_group and 'vector_ids' in collection_group:
            vectors = collection_group['vectors'][:]
            vector_ids = collection_group['vector_ids'][:].tolist()
            
            # Restore vectors
            for i, vector_id in enumerate(vector_ids):
                collection._vectors[vector_id] = vectors[i]
                collection._next_id = max(collection._next_id, vector_id + 1)
        
        # Load metadata
        if 'metadata' in collection_group:
            metadata_bytes = collection_group['metadata'][:].tobytes()
            loaded_metadata = msgpack.unpackb(metadata_bytes)
            # Convert string keys back to integers
            collection._metadata = {int(k): v for k, v in loaded_metadata.items()}
        
        # Load stats (already loaded above for dimension, etc.)
        if 'stats' in collection_group:
            stats_bytes = collection_group['stats'][:].tobytes()
            stats = msgpack.unpackb(stats_bytes)
            collection._stats.update(stats)
        
        # Mark index as needing rebuild
        collection._index_built = False
        collection._index_needs_rebuild = True
        
        logger.debug(f"Loaded collection '{collection_name}' with {len(collection._vectors)} vectors")
        
        return collection
    
    def save_collection(self, collection: Any, collection_name: str) -> None:
        """
        Save a single collection.
        
        Args:
            collection: Collection object to save
            collection_name: Name of the collection
        """
        if not self.storage_path:
            raise ValueError("Storage path not set")
        
        collection_path = self.storage_path / f"{collection_name}.h5"
        
        with h5py.File(collection_path, 'w') as f:
            self._save_collection(f, collection_name, collection)
        
        logger.info(f"Collection '{collection_name}' saved to {collection_path}")
    
    def load_collection(self, collection_name: str) -> Any:
        """
        Load a single collection.
        
        Args:
            collection_name: Name of the collection to load
            
        Returns:
            Loaded collection
        """
        if not self.storage_path:
            raise ValueError("Storage path not set")
        
        collection_path = self.storage_path / f"{collection_name}.h5"
        
        if not collection_path.exists():
            raise FileNotFoundError(f"Collection file not found: {collection_path}")
        
        with h5py.File(collection_path, 'r') as f:
            collection = self._load_collection(f, collection_name)
        
        logger.info(f"Collection '{collection_name}' loaded from {collection_path}")
        return collection
    
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection from storage.
        
        Args:
            collection_name: Name of the collection to delete
        """
        if not self.storage_path:
            return
        
        collection_path = self.storage_path / f"{collection_name}.h5"
        
        if collection_path.exists():
            collection_path.unlink()
            logger.info(f"Collection '{collection_name}' deleted from storage")
    
    def list_saved_collections(self) -> List[str]:
        """List all saved collections."""
        if not self.storage_path:
            return []
        
        collections = []
        for file_path in self.storage_path.glob("*.h5"):
            collections.append(file_path.stem)
        
        return collections
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a saved collection without loading it.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection information or None if not found
        """
        if not self.storage_path:
            return None
        
        collection_path = self.storage_path / f"{collection_name}.h5"
        
        if not collection_path.exists():
            return None
        
        try:
            with h5py.File(collection_path, 'r') as f:
                info = {
                    'name': collection_name,
                    'dimension': f.attrs.get('dimension'),
                    'index_type': f.attrs.get('index_type'),
                    'distance_metric': f.attrs.get('distance_metric'),
                    'vector_count': f['vectors'].shape[0] if 'vectors' in f else 0,
                    'metadata_count': len(f['metadata']) if 'metadata' in f else 0
                }
                
                if 'stats' in f.attrs:
                    stats = msgpack.unpackb(f.attrs['stats'])
                    info.update(stats)
                
                return info
        except Exception as e:
            logger.error(f"Error reading collection info for '{collection_name}': {e}")
            return None
    
    def optimize_storage(self, collection_name: str) -> None:
        """
        Optimize storage for a collection (e.g., recompress with better settings).
        
        Args:
            collection_name: Name of the collection to optimize
        """
        if not self.storage_path:
            return
        
        collection_path = self.storage_path / f"{collection_name}.h5"
        
        if not collection_path.exists():
            logger.warning(f"Collection '{collection_name}' not found for optimization")
            return
        
        # Load and resave with optimized compression
        try:
            with h5py.File(collection_path, 'r') as f:
                collection = self._load_collection(f, collection_name)
            
            # Save with optimized settings
            optimized_path = collection_path.with_suffix('.optimized.h5')
            with h5py.File(optimized_path, 'w') as f:
                self._save_collection(f, collection_name, collection)
            
            # Replace original file
            collection_path.unlink()
            optimized_path.rename(collection_path)
            
            logger.info(f"Storage optimized for collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error optimizing storage for collection '{collection_name}': {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'storage_path': str(self.storage_path) if self.storage_path else None,
            'cache_enabled': self.enable_cache,
            'cache_size': self.cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        }
        
        if self.storage_path:
            # File system stats
            total_size = 0
            file_count = 0
            for file_path in self.storage_path.glob("*.h5"):
                total_size += file_path.stat().st_size
                file_count += 1
            
            stats['total_storage_size'] = total_size
            stats['file_count'] = file_count
            stats['saved_collections'] = self.list_saved_collections()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the storage cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Storage cache cleared")
    
    def _get_cache_key(self, key: str) -> str:
        """Generate a cache key."""
        return f"storage_{key}"
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.enable_cache:
            return None
        
        cache_key = self._get_cache_key(key)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if not self.enable_cache:
            return
        
        cache_key = self._get_cache_key(key)
        
        # Implement LRU cache
        if len(self._cache) >= self.cache_size:
            # Remove oldest item (simple implementation)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = value
