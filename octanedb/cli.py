#!/usr/bin/env python3
"""
OctaneDB Command Line Interface
Provides command-line access to OctaneDB functionality
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List
import numpy as np

from . import OctaneDB


def create_collection(args):
    """Create a new collection."""
    try:
        db = OctaneDB(dimension=args.dimension)
        collection = db.create_collection(args.name)
        print(f"Collection '{args.name}' created successfully with dimension {args.dimension}")
        return 0
    except Exception as e:
        print(f" Failed to create collection: {e}")
        return 1


def insert_vectors(args):
    """Insert vectors into a collection."""
    try:
        db = OctaneDB(dimension=args.dimension)
        db.create_collection(args.collection)
        db.use_collection(args.collection)
        
        # Generate sample vectors
        vectors = np.random.randn(args.count, args.dimension).astype(np.float32)
        metadata = [{"id": i, "description": f"Vector {i}"} for i in range(args.count)]
        
        inserted_ids = db.insert(vectors=vectors, metadata=metadata)
        print(f"Successfully inserted {len(inserted_ids)} vectors into collection '{args.collection}'")
        return 0
    except Exception as e:
        print(f" Failed to insert vectors: {e}")
        return 1


def search_vectors(args):
    """Search for similar vectors."""
    try:
        db = OctaneDB(dimension=args.dimension)
        db.create_collection(args.collection)
        db.use_collection(args.collection)
        
        # Insert some sample data first
        sample_vectors = np.random.randn(10, args.dimension).astype(np.float32)
        sample_metadata = [{"id": i, "description": f"Sample {i}"} for i in range(10)]
        db.insert(vectors=sample_vectors, metadata=sample_metadata)
        
        # Create query vector
        query_vector = np.random.randn(args.dimension).astype(np.float32)
        
        # Perform search
        results = db.search(query_vector=query_vector, k=args.k, include_metadata=True)
        
        print(f"🔍 Search results for collection '{args.collection}':")
        for i, (vector_id, distance, metadata) in enumerate(results):
            print(f"  {i+1}. ID: {vector_id}, Distance: {distance:.6f}")
            if metadata:
                print(f"     Metadata: {metadata}")
        
        return 0
    except Exception as e:
        print(f"Failed to search vectors: {e}")
        return 1


def benchmark(args):
    """Run performance benchmarks."""
    try:
        print("OctaneDB Performance Benchmark")
        print("=" * 50)
        
        db = OctaneDB(dimension=args.dimension)
        collection = db.create_collection("benchmark")
        db.use_collection("benchmark")
        
        # Insert benchmark
        print(f"Inserting {args.count:,} vectors...")
        start_time = __import__('time').time()
        
        vectors = np.random.randn(args.count, args.dimension).astype(np.float32)
        metadata = [{"id": i} for i in range(args.count)]
        
        inserted_ids = db.insert(vectors=vectors, metadata=metadata)
        insert_time = __import__('time').time() - start_time
        
        print(f"Inserted {len(inserted_ids):,} vectors in {insert_time:.4f}s")
        print(f" Rate: {args.count/insert_time:.0f} vectors/second")
        
        # Search benchmark
        print(f"\n Running search benchmark...")
        start_time = __import__('time').time()
        
        query_vectors = np.random.randn(100, args.dimension).astype(np.float32)
        total_results = 0
        
        for i, query_vector in enumerate(query_vectors):
            results = db.search(query_vector=query_vector, k=10)
            total_results += len(results)
            
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/100 queries...")
        
        search_time = __import__('time').time() - start_time
        
        print(f"Search completed in {search_time:.4f}s")
        print(f"   Rate: {100/search_time:.1f} queries/second")
        print(f"   Total results: {total_results:,}")
        
        # Memory usage
        stats = db.get_stats()
        print(f"\n Database Statistics:")
        print(f"   Collections: {stats['collection_count']}")
        print(f"   Total vectors: {stats['total_vectors']}")
        print(f"   Current collection: {stats['current_collection']}")
        
        return 0
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1


def info(args):
    """Show OctaneDB information."""
    print("OctaneDB - Lightning Fast Vector Database")
    print("=" * 50)
    print(f"Version: 1.0.0")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"Author: Rijin")
    print(f"License: MIT")
    print(f"GitHub: https://github.com/yourusername/OctaneDB")
    print("\nFeatures:")
    print("   HNSW Indexing for fast similarity search")
    print("   Multiple distance metrics (cosine, euclidean, dot)")
    print("   HDF5 storage with msgpack metadata")
    print("   In-memory, persistent, and hybrid storage")
    print("   Milvus-compatible API")
    print("   HNSW indexing for fast similarity search")
    print("\nInstallation:")
    print("  pip install octanedb")
    print("\nQuick Start:")
    print("  from octanedb import OctaneDB")
    print("  db = OctaneDB(dimension=384)")
    print("  collection = db.create_collection('my_collection')")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OctaneDB - Lightning Fast Vector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  octanedb create --name documents --dimension 384
  octanedb insert --collection documents --count 1000 --dimension 384
  octanedb search --collection documents --k 5 --dimension 384
  octanedb benchmark --count 10000 --dimension 384
  octanedb info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create collection command
    create_parser = subparsers.add_parser('create', help='Create a new collection')
    create_parser.add_argument('--name', required=True, help='Collection name')
    create_parser.add_argument('--dimension', type=int, default=384, help='Vector dimension')
    create_parser.set_defaults(func=create_collection)
    
    # Insert vectors command
    insert_parser = subparsers.add_parser('insert', help='Insert vectors into a collection')
    insert_parser.add_argument('--collection', required=True, help='Collection name')
    insert_parser.add_argument('--count', type=int, default=100, help='Number of vectors to insert')
    insert_parser.add_argument('--dimension', type=int, default=384, help='Vector dimension')
    insert_parser.set_defaults(func=insert_vectors)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for similar vectors')
    search_parser.add_argument('--collection', required=True, help='Collection name')
    search_parser.add_argument('--k', type=int, default=5, help='Number of results to return')
    search_parser.add_argument('--dimension', type=int, default=384, help='Vector dimension')
    search_parser.set_defaults(func=search_vectors)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--count', type=int, default=10000, help='Number of vectors for benchmark')
    benchmark_parser.add_argument('--dimension', type=int, default=384, help='Vector dimension')
    benchmark_parser.set_defaults(func=benchmark)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show OctaneDB information')
    info_parser.set_defaults(func=info)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
