#!/usr/bin/env python3
"""
OctaneDB Performance Benchmarking Script

This script provides comprehensive performance measurements for OctaneDB
with different configurations and dataset sizes.

Usage:
    python benchmark_octanedb.py [--vectors N] [--dimension D] [--runs R]

"""

import argparse
import time
import statistics
import numpy as np
import psutil
import os
from typing import Dict, List, Tuple
import json

try:
    from octanedb import OctaneDB
except ImportError:
    print(" OctaneDB not installed. Install with: pip install octanedb")
    exit(1)


class OctaneDBBenchmark:
    """Comprehensive benchmarking for OctaneDB."""
    
    def __init__(self, dimension: int = 384, storage_mode: str = "in-memory"):
        self.dimension = dimension
        self.storage_mode = storage_mode
        self.db = None
        self.results = {}
        
    def setup_database(self):
        """Initialize the database and collection."""
        print(f" Setting up OctaneDB (dimension={self.dimension}, storage={self.storage_mode})")
        
        self.db = OctaneDB(
            dimension=self.dimension,
            storage_mode=self.storage_mode
        )
        
        # Create benchmark collection
        collection = self.db.create_collection("benchmark")
        self.db.use_collection("benchmark")
        
        print(" Database setup complete")
        
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent()
        }
        
    def benchmark_insertion(self, num_vectors: int, batch_size: int = 1000) -> Dict[str, float]:
        """Benchmark vector insertion performance."""
        print(f"Benchmarking insertion of {num_vectors:,} vectors...")
        
        # Generate random vectors
        vectors = np.random.randn(num_vectors, self.dimension).astype(np.float32)
        metadata = [{"id": i, "benchmark": True} for i in range(num_vectors)]
        
        # Measure insertion time
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        # Insert in batches
        inserted_count = 0
        for i in range(0, num_vectors, batch_size):
            end_idx = min(i + batch_size, num_vectors)
            batch_vectors = vectors[i:end_idx]
            batch_metadata = metadata[i:end_idx]
            
            self.db.insert(
                vectors=batch_vectors.tolist(),
                metadata=batch_metadata,
                ids=list(range(i, end_idx))
            )
            
            inserted_count += len(batch_vectors)
            
            # Progress update
            if (i + batch_size) % (batch_size * 10) == 0 or end_idx == num_vectors:
                elapsed = time.time() - start_time
                rate = inserted_count / elapsed
                print(f"   Inserted {inserted_count:,}/{num_vectors:,} vectors ({rate:.0f} vectors/sec)")
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        total_time = end_time - start_time
        insertion_rate = num_vectors / total_time
        
        return {
            "total_time": total_time,
            "insertion_rate": insertion_rate,
            "batch_size": batch_size,
            "start_memory_mb": start_memory["rss_mb"],
            "end_memory_mb": end_memory["rss_mb"],
            "memory_increase_mb": end_memory["rss_mb"] - start_memory["rss_mb"]
        }
        
    def benchmark_index_build(self) -> Dict[str, float]:
        """Benchmark HNSW index construction."""
        print("Benchmarking index construction...")
        
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        # Build the index
        self.db._current_collection._build_index()
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        build_time = end_time - start_time
        
        # Get index statistics
        stats = self.db._current_collection.get_stats()
        index_stats = stats.get("index_stats", {})
        
        return {
            "build_time": build_time,
            "start_memory_mb": start_memory["rss_mb"],
            "end_memory_mb": end_memory["rss_mb"],
            "memory_increase_mb": end_memory["rss_mb"] - start_memory["rss_mb"],
            "index_stats": index_stats
        }
        
    def benchmark_search(self, num_queries: int = 100, k: int = 10) -> Dict[str, float]:
        """Benchmark search performance."""
        print(f" Benchmarking search ({num_queries} queries, k={k})...")
        
        # Generate random query vectors
        query_vectors = np.random.randn(num_queries, self.dimension).astype(np.float32)
        
        search_times = []
        results_counts = []
        
        start_time = time.time()
        
        for i, query_vector in enumerate(query_vectors):
            query_start = time.time()
            
            results = self.db.search(query_vector=query_vector, k=k)
            
            query_time = time.time() - query_start
            search_times.append(query_time * 1000)  # Convert to milliseconds
            results_counts.append(len(results))
            
            # Progress update
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{num_queries} queries...")
        
        total_time = time.time() - start_time
        
        return {
            "total_time": total_time,
            "queries_per_second": num_queries / total_time,
            "avg_search_time_ms": statistics.mean(search_times),
            "median_search_time_ms": statistics.median(search_times),
            "min_search_time_ms": min(search_times),
            "max_search_time_ms": max(search_times),
            "search_time_std_ms": statistics.stdev(search_times) if len(search_times) > 1 else 0,
            "avg_results_count": statistics.mean(results_counts),
            "search_times_ms": search_times
        }
        
    def benchmark_batch_search(self, num_queries: int = 100, k: int = 10) -> Dict[str, float]:
        """Benchmark batch search performance."""
        print(f" Benchmarking batch search ({num_queries} queries, k={k})...")
        
        # Generate random query vectors
        query_vectors = np.random.randn(num_queries, self.dimension).astype(np.float32)
        
        start_time = time.time()
        
        # Perform batch search
        batch_results = self.db.search_batch(query_vectors=query_vectors, k=k)
        
        total_time = time.time() - start_time
        
        # Count total results
        total_results = sum(len(results) for results in batch_results)
        
        return {
            "total_time": total_time,
            "queries_per_second": num_queries / total_time,
            "total_results": total_results,
            "avg_results_per_query": total_results / num_queries
        }
        
    def run_comprehensive_benchmark(self, num_vectors: int, num_runs: int = 3) -> Dict:
        """Run comprehensive benchmark with multiple runs."""
        print(f" Starting comprehensive benchmark ({num_runs} runs)")
        print(f"   Dataset: {num_vectors:,} vectors, {self.dimension} dimensions")
        print(f"   Storage: {self.storage_mode}")
        print("=" * 60)
        
        all_results = {
            "insertion": [],
            "index_build": [],
            "search": [],
            "batch_search": [],
            "system_info": self._get_system_info()
        }
        
        for run in range(num_runs):
            print(f"\n Run {run + 1}/{num_runs}")
            print("-" * 40)
            
            # Reset database for each run
            self.setup_database()
            
            # Benchmark insertion
            insertion_result = self.benchmark_insertion(num_vectors)
            all_results["insertion"].append(insertion_result)
            
            # Benchmark index build
            index_result = self.benchmark_index_build()
            all_results["index_build"].append(index_result)
            
            # Benchmark search
            search_result = self.benchmark_search(100, 10)
            all_results["search"].append(search_result)
            
            # Benchmark batch search
            batch_result = self.benchmark_batch_search(100, 10)
            all_results["batch_search"].append(batch_result)
            
            print(f"Run {run + 1} completed")
        
        # Calculate aggregate results
        self.results = self._calculate_aggregate_results(all_results)
        
        return self.results
        
    def _get_system_info(self) -> Dict:
        """Get system information."""
        return {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "numpy_version": np.__version__,
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "platform": os.name
        }
        
    def _calculate_aggregate_results(self, all_results: Dict) -> Dict:
        """Calculate aggregate results from multiple runs."""
        aggregated = {}
        
        for operation, runs in all_results.items():
            if operation == "system_info":
                aggregated[operation] = runs
                continue
                
            aggregated[operation] = {}
            
            # Get all numeric values for each metric
            for metric in runs[0].keys():
                if isinstance(runs[0][metric], (int, float)):
                    values = [run[metric] for run in runs]
                    aggregated[operation][f"{metric}_mean"] = statistics.mean(values)
                    aggregated[operation][f"{metric}_median"] = statistics.median(values)
                    aggregated[operation][f"{metric}_min"] = min(values)
                    aggregated[operation][f"{metric}_max"] = max(values)
                    if len(values) > 1:
                        aggregated[operation][f"{metric}_std"] = statistics.stdev(values)
                    else:
                        aggregated[operation][f"{metric}_std"] = 0
                else:
                    # For non-numeric values, just take the first run
                    aggregated[operation][metric] = runs[0][metric]
        
        return aggregated
        
    def print_results(self):
        """Print formatted benchmark results."""
        if not self.results:
            print(" No results to display. Run benchmark first.")
            return
            
        print("\n" + "=" * 60)
        print(" OCTANEDB BENCHMARK RESULTS")
        print("=" * 60)
        
        # System Information
        print(f"\n System Information:")
        sys_info = self.results["system_info"]
        print(f"   Python: {sys_info['python_version']}")
        print(f"   NumPy: {sys_info['numpy_version']}")
        print(f"   CPU Cores: {sys_info['cpu_count']}")
        print(f"   Memory: {sys_info['memory_gb']:.1f} GB")
        print(f"   Platform: {sys_info['platform']}")
        
        # Insertion Results
        print(f"\nInsertion Performance:")
        insertion = self.results["insertion"]
        print(f"   Rate: {insertion['insertion_rate_mean']:.0f} ± {insertion['insertion_rate_std']:.0f} vectors/sec")
        print(f"   Total Time: {insertion['total_time_mean']:.2f} ± {insertion['total_time_std']:.2f} seconds")
        print(f"   Memory Increase: {insertion['memory_increase_mb_mean']:.1f} ± {insertion['memory_increase_mb_std']:.1f} MB")
        
        # Index Build Results
        print(f"\n Index Build Performance:")
        index_build = self.results["index_build"]
        print(f"   Build Time: {index_build['build_time_mean']:.2f} ± {index_build['build_time_std']:.2f} seconds")
        print(f"   Memory Increase: {index_build['memory_increase_mb_mean']:.1f} ± {index_build['memory_increase_mb_std']:.1f} MB")
        
        # Search Results
        print(f"\n Search Performance:")
        search = self.results["search"]
        print(f"   Single Query: {search['avg_search_time_ms_mean']:.2f} ± {search['search_time_std_ms_mean']:.2f} ms")
        print(f"   Queries/sec: {search['queries_per_second_mean']:.1f} ± {search['queries_per_second_std']:.1f}")
        print(f"   Min Time: {search['min_search_time_ms_mean']:.2f} ms")
        print(f"   Max Time: {search['max_search_time_ms_mean']:.2f} ms")
        
        # Batch Search Results
        print(f"\n Batch Search Performance:")
        batch_search = self.results["batch_search"]
        print(f"   Batch Queries/sec: {batch_search['queries_per_second_mean']:.1f} ± {batch_search['queries_per_second_std']:.1f}")
        print(f"   Total Time: {batch_search['total_time_mean']:.2f} ± {batch_search['total_time_std']:.2f} seconds")
        
        print("\n" + "=" * 60)
        
    def save_results(self, filename: str = "octanedb_benchmark_results.json"):
        """Save results to JSON file."""
        if not self.results:
            print(" No results to save. Run benchmark first.")
            return
            
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f" Results saved to {filename}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="OctaneDB Performance Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_octanedb.py --vectors 10000 --dimension 128
    python benchmark_octanedb.py --vectors 100000 --dimension 384 --runs 5
    python benchmark_octanedb.py --vectors 50000 --dimension 512 --storage persistent
        """
    )
    
    parser.add_argument(
        "--vectors", "-v",
        type=int,
        default=10000,
        help="Number of vectors to benchmark (default: 10000)"
    )
    
    parser.add_argument(
        "--dimension", "-d",
        type=int,
        default=384,
        help="Vector dimension (default: 384)"
    )
    
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)"
    )
    
    parser.add_argument(
        "--storage", "-s",
        choices=["in-memory", "persistent", "hybrid"],
        default="in-memory",
        help="Storage mode (default: in-memory)"
    )
    
    parser.add_argument(
        "--save", "-o",
        action="store_true",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    print(" OctaneDB Performance Benchmarking")
    print("=" * 50)
    
    # Run benchmark
    benchmark = OctaneDBBenchmark(
        dimension=args.dimension,
        storage_mode=args.storage
    )
    
    try:
        results = benchmark.run_comprehensive_benchmark(
            num_vectors=args.vectors,
            num_runs=args.runs
        )
        
        # Display results
        benchmark.print_results()
        
        # Save results if requested
        if args.save:
            benchmark.save_results()
            
    except KeyboardInterrupt:
        print("\n Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
