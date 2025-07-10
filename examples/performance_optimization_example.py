"""Example demonstrating performance optimization features.

This example shows how to:
1. Use query optimization
2. Leverage caching for better performance
3. Process data in optimized batches
4. Monitor performance metrics
5. Manage connection pools
"""
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

from mdm.performance import (
    QueryOptimizer,
    CacheManager,
    BatchOptimizer,
    ConnectionPool,
    PerformanceMonitor,
    DatasetCache,
    CachePolicy,
    BatchConfig,
    PoolConfig,
    get_monitor
)

console = Console()


def main():
    """Run performance optimization examples."""
    console.print(Panel.fit(
        "[bold cyan]Performance Optimization Examples[/bold cyan]\n\n"
        "Demonstrating MDM's performance features",
        title="Performance Demo"
    ))
    
    # Example 1: Query Optimization
    console.print("\n[bold]Example 1: Query Optimization[/bold]")
    console.print("=" * 50 + "\n")
    example_query_optimization()
    
    # Example 2: Caching
    console.print("\n[bold]Example 2: Caching Layer[/bold]")
    console.print("=" * 50 + "\n")
    example_caching()
    
    # Example 3: Batch Processing
    console.print("\n[bold]Example 3: Batch Processing[/bold]")
    console.print("=" * 50 + "\n")
    example_batch_processing()
    
    # Example 4: Connection Pooling
    console.print("\n[bold]Example 4: Connection Pooling[/bold]")
    console.print("=" * 50 + "\n")
    example_connection_pooling()
    
    # Example 5: Performance Monitoring
    console.print("\n[bold]Example 5: Performance Monitoring[/bold]")
    console.print("=" * 50 + "\n")
    example_performance_monitoring()
    
    console.print("\n[bold green]All examples completed successfully![/bold green]")


def example_query_optimization():
    """Demonstrate query optimization features."""
    console.print("Creating query optimizer...")
    
    # Create optimizer
    optimizer = QueryOptimizer(cache_query_plans=True)
    
    # Example queries
    queries = [
        "SELECT * FROM datasets WHERE size > 1000",
        "SELECT name, created_at FROM datasets ORDER BY created_at DESC",
        "SELECT COUNT(*) FROM datasets GROUP BY problem_type",
        "SELECT d.*, s.* FROM datasets d JOIN statistics s ON d.id = s.dataset_id"
    ]
    
    console.print("\nAnalyzing queries:")
    
    # Mock connection for demo
    class MockConnection:
        class dialect:
            name = "sqlite"
    
    mock_conn = MockConnection()
    
    for query in queries:
        # Optimize query
        optimized_query, plan = optimizer.optimize_query(query, mock_conn)
        
        console.print(f"\n[cyan]Query:[/cyan] {query[:50]}...")
        console.print(f"[green]Type:[/green] {plan.query_type.value}")
        console.print(f"[yellow]Uses Index:[/yellow] {plan.uses_index}")
        
        if plan.optimization_hints:
            console.print("[red]Hints:[/red]")
            for hint in plan.optimization_hints:
                console.print(f"  • {hint}")
    
    # Show cache statistics
    stats = optimizer.get_statistics()
    console.print(f"\n[bold]Optimizer Statistics:[/bold]")
    console.print(f"Total queries: {stats['total_queries']}")
    console.print(f"Cache hits: {stats['cache_hits']}")
    console.print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")


def example_caching():
    """Demonstrate caching functionality."""
    console.print("Setting up cache managers...")
    
    # General cache
    cache = CacheManager(
        max_size_mb=10,
        policy=CachePolicy.LRU,
        default_ttl=60
    )
    
    # Dataset-specific cache
    dataset_cache = DatasetCache(max_size_mb=5)
    
    # Simulate expensive operations
    def expensive_computation(x):
        """Simulate expensive computation."""
        time.sleep(0.1)  # Simulate work
        return x ** 2
    
    # Use cache decorator
    @cache.cached(ttl=30)
    def cached_computation(x):
        return expensive_computation(x)
    
    console.print("\n[yellow]Testing cache performance:[/yellow]")
    
    # First call - cache miss
    start = time.time()
    result1 = cached_computation(42)
    time1 = time.time() - start
    console.print(f"First call (cache miss): {time1:.3f}s")
    
    # Second call - cache hit
    start = time.time()
    result2 = cached_computation(42)
    time2 = time.time() - start
    console.print(f"Second call (cache hit): {time2:.3f}s")
    console.print(f"[green]Speedup: {time1/time2:.1f}x[/green]")
    
    # Dataset cache example
    console.print("\n[yellow]Testing dataset cache:[/yellow]")
    
    # Cache dataset info
    dataset_info = {
        'name': 'test_dataset',
        'size': 1000000,
        'features': ['feature1', 'feature2', 'feature3'],
        'created_at': '2024-01-01'
    }
    
    dataset_cache.set_dataset_info('test_dataset', dataset_info)
    
    # Retrieve from cache
    cached_info = dataset_cache.get_dataset_info('test_dataset')
    console.print(f"Cached dataset info retrieved: {cached_info['name']}")
    
    # Show cache statistics
    stats = cache.get_statistics()
    console.print(f"\n[bold]Cache Statistics:[/bold]")
    console.print(f"Entries: {stats['entries']}")
    console.print(f"Size: {stats['size_mb']:.2f} MB")
    console.print(f"Hit rate: {stats['hit_rate']:.2%}")


def example_batch_processing():
    """Demonstrate batch processing optimization."""
    console.print("Creating batch optimizer...")
    
    # Configure batch processing
    config = BatchConfig(
        batch_size=1000,
        max_workers=4,
        enable_parallel=True
    )
    optimizer = BatchOptimizer(config)
    
    # Create sample data
    console.print("\nGenerating sample data...")
    df = pd.DataFrame({
        'id': range(10000),
        'value': np.random.randn(10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000)
    })
    
    # Define processing function
    def process_batch(batch_df):
        """Process a batch of data."""
        # Simulate some computation
        batch_df['processed'] = batch_df['value'] * 2 + np.random.randn(len(batch_df)) * 0.1
        batch_df['category_encoded'] = batch_df['category'].map({'A': 0, 'B': 1, 'C': 2})
        return batch_df
    
    # Process with progress tracking
    console.print("\n[yellow]Processing data in batches:[/yellow]")
    
    processed_count = 0
    def progress_callback(done, total):
        nonlocal processed_count
        processed_count = done
        console.print(f"Progress: {done}/{total} batches", end='\r')
    
    start = time.time()
    result_df = optimizer.process_dataframe_batches(
        df,
        process_batch,
        progress_callback
    )
    duration = time.time() - start
    
    console.print(f"\nProcessed {len(result_df)} rows in {duration:.2f}s")
    console.print(f"Throughput: {len(result_df)/duration:.0f} rows/second")
    
    # Show batch statistics
    stats = optimizer.get_statistics()
    console.print(f"\n[bold]Batch Processing Statistics:[/bold]")
    console.print(f"Total batches: {stats['total_batches']}")
    console.print(f"Rows per second: {stats['rows_per_second']:.0f}")
    if stats['parallel_speedup'] > 0:
        console.print(f"Parallel speedup: {stats['parallel_speedup']:.2f}x")


def example_connection_pooling():
    """Demonstrate connection pooling."""
    console.print("Setting up connection pool...")
    
    # Configure pool
    pool_config = PoolConfig(
        pool_size=5,
        max_overflow=10,
        timeout=30.0,
        recycle=3600,
        pre_ping=True
    )
    
    # Create pool (using SQLite for demo)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
        connection_string = f"sqlite:///{tmp.name}"
        pool = ConnectionPool(connection_string, pool_config)
        
        console.print("\n[yellow]Testing connection pool:[/yellow]")
        
        # Simulate multiple operations
        operation_times = []
        
        for i in range(10):
            start = time.time()
            
            # Use pooled connection
            with pool.get_connection() as conn:
                # Simulate database operation
                conn.execute("SELECT 1")
            
            operation_time = time.time() - start
            operation_times.append(operation_time)
            
            if i == 0:
                console.print(f"First operation (pool init): {operation_time:.3f}s")
            elif i == 9:
                console.print(f"Last operation (pooled): {operation_time:.3f}s")
        
        avg_time = sum(operation_times[1:]) / len(operation_times[1:])
        console.print(f"[green]Average operation time: {avg_time:.3f}s[/green]")
        
        # Show pool status
        status = pool.get_pool_status()
        console.print(f"\n[bold]Pool Status:[/bold]")
        console.print(f"Active connections: {status['active_connections']}")
        console.print(f"Pool size: {status['size']}")
        console.print(f"Checked out: {status['checked_out']}")
        
        # Get statistics
        stats = pool.get_statistics()
        console.print(f"\n[bold]Pool Statistics:[/bold]")
        console.print(f"Connections created: {stats['connections_created']}")
        console.print(f"Total checkouts: {stats['total_checkouts']}")
        console.print(f"Error rate: {stats['error_rate']:.2%}")


def example_performance_monitoring():
    """Demonstrate performance monitoring."""
    console.print("Starting performance monitor...")
    
    # Get global monitor
    monitor = get_monitor()
    
    # Simulate various operations
    console.print("\n[yellow]Simulating operations:[/yellow]")
    
    # Track dataset operations
    with monitor.track_operation("dataset_registration") as timer:
        # Simulate dataset registration
        time.sleep(0.2)
        console.print(f"Dataset registration: {timer.duration:.3f}s")
    
    # Track queries
    for i in range(5):
        query_type = np.random.choice(['select', 'insert', 'update'])
        duration = np.random.uniform(0.01, 0.1)
        rows = np.random.randint(10, 1000)
        
        monitor.track_query(query_type, duration, rows)
    
    # Track cache operations
    for i in range(20):
        hit = np.random.random() > 0.3  # 70% hit rate
        monitor.track_cache("dataset_info", hit)
    
    # Track batch processing
    monitor.track_batch("feature_generation", 5000, 2.5)
    
    # Get performance report
    report = monitor.get_report()
    
    console.print("\n[bold]Performance Report:[/bold]")
    
    # Display metrics summary
    summary = report['summary']
    
    # Counters
    if 'counters' in summary:
        table = Table(title="Operation Counters")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="yellow")
        
        for metric, count in summary['counters'].items():
            if count > 0:
                table.add_row(metric, str(int(count)))
        
        console.print(table)
    
    # Timers
    if 'timers' in summary:
        console.print("\n[bold]Operation Timings:[/bold]")
        for operation, stats in summary['timers'].items():
            console.print(f"\n{operation}:")
            console.print(f"  Count: {stats['count']}")
            console.print(f"  Average: {stats['avg']:.3f}s")
            console.print(f"  Min: {stats['min']:.3f}s")
            console.print(f"  Max: {stats['max']:.3f}s")
            console.print(f"  P95: {stats['p95']:.3f}s")
    
    # Success rates
    if 'operation_dataset_registration_success_rate' in report:
        console.print(f"\n[bold]Success Rates:[/bold]")
        console.print(f"Dataset registration: {report['operation_dataset_registration_success_rate']:.2%}")
    
    console.print(f"\n[dim]Active operations: {report['active_operations']}[/dim]")


if __name__ == "__main__":
    main()