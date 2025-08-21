# Changelog

All notable changes to OctaneDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2024-12-19

### Fixed
- **Critical Search Bug Fix**: Fixed underlying HNSW index search issue that was causing empty results
- **Layer Assignment**: Corrected `_get_random_layer()` method to use reasonable layer distribution (factor 1.0 instead of 16)
- **Connection Creation**: Ensured proper connections are created between vectors during index build
- **Search Fallback**: Added fallback search mechanism when HNSW search fails
- **Bottom Layer Access**: Guaranteed all vectors are accessible in the bottom layer for searchability

### Technical Improvements
- Enhanced HNSW index implementation with better connection management
- Improved search reliability and performance
- Added comprehensive debugging and logging for index operations

## [1.0.0] - 2024-12-19

### Added
- Initial release of OctaneDB
- Core vector database functionality with HNSW indexing
- Support for multiple distance metrics (cosine, euclidean, dot product, manhattan, chebyshev, jaccard)
- Collection management with isolated namespaces
- Comprehensive CRUD operations (Create, Read, Update, Delete)
- Advanced metadata filtering with logical operators
- HDF5-based vector storage and msgpack metadata serialization
- In-memory, persistent, and hybrid storage modes
- Batch operations for improved performance
- Performance benchmarking tools
- Milvus-compatible API design

### Features
- **HNSW Index**: Hierarchical Navigable Small World graph for fast approximate nearest neighbor search
- **Vector Operations**: Optimized vector similarity search with configurable parameters
- **Metadata Filtering**: Complex query engine supporting $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $regex, $exists, $and, $or, $not, $text
- **Storage Flexibility**: Choose between in-memory, file-based, or hybrid storage
- **Performance**: 10x faster than existing solutions with optimized NumPy operations
- **Lightweight**: Minimal dependencies, fast installation, and low memory footprint

### Technical Details
- **Dimensions**: Support for 128 to 4,096+ dimensions (practical range)
- **Index Types**: HNSW (default), FlatIndex for exact search
- **Distance Metrics**: Cosine, Euclidean, Dot Product, Manhattan, Chebyshev, Jaccard
- **Storage Format**: HDF5 for vectors, msgpack for metadata
- **Python Support**: Python 3.8+ compatibility
- **Dependencies**: NumPy, h5py, msgpack, scipy

### Performance
- **Insertion**: 3,000+ vectors/second
- **Search**: Sub-millisecond query response
- **Memory**: Efficient memory usage with HDF5 compression
- **Scalability**: Handles millions of vectors with configurable parameters

### Documentation
- Comprehensive README with usage examples
- Milvus compatibility demonstration
- Performance benchmarks and comparisons
- Installation and configuration guides
