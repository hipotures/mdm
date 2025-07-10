"""Example demonstrating storage backend migration functionality.

This example shows how to:
1. Compare legacy vs new storage backends
2. Migrate datasets between backends
3. Validate storage implementations
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from mdm.core import feature_flags
from mdm.adapters import get_storage_backend
from mdm.testing import StorageComparisonTester
from mdm.migration import StorageMigrator, StorageValidator

console = Console()


def main():
    """Run storage migration examples."""
    console.print(Panel.fit(
        "[bold cyan]Storage Backend Migration Examples[/bold cyan]\n\n"
        "This demonstrates the storage backend migration functionality",
        title="MDM Storage Migration"
    ))
    
    # Example 1: Compare storage backend implementations
    console.print("\n[bold]Example 1: Comparing Storage Backends[/bold]")
    console.print("=" * 50 + "\n")
    
    # Test SQLite backend
    console.print("Testing SQLite backend comparison...")
    sqlite_tester = StorageComparisonTester("sqlite")
    sqlite_results = sqlite_tester.run_all_tests()
    
    console.print(f"\nSQLite Success Rate: {sqlite_results['success_rate']:.1f}%")
    console.print(f"Performance Ratio: {sqlite_results['performance_ratio']:.2f}x")
    
    # Example 2: Feature flag switching
    console.print("\n[bold]Example 2: Feature Flag Switching[/bold]")
    console.print("=" * 50 + "\n")
    
    # Show current backend
    console.print(f"Current feature flag: use_new_storage = {feature_flags.get('use_new_storage', False)}")
    
    # Get backend with legacy implementation
    feature_flags.set("use_new_storage", False)
    legacy_backend = get_storage_backend("sqlite")
    console.print(f"Legacy backend type: {legacy_backend.__class__.__name__}")
    
    # Get backend with new implementation
    feature_flags.set("use_new_storage", True)
    new_backend = get_storage_backend("sqlite")
    console.print(f"New backend type: {new_backend.__class__.__name__}")
    
    # Reset flag
    feature_flags.set("use_new_storage", False)
    
    # Example 3: Backend validation
    console.print("\n[bold]Example 3: Backend Validation[/bold]")
    console.print("=" * 50 + "\n")
    
    # Validate new SQLite backend
    console.print("Validating new SQLite backend...")
    feature_flags.set("use_new_storage", True)
    
    try:
        backend = get_storage_backend("sqlite")
        validator = StorageValidator(backend)
        validation_results = validator.validate_all()
        
        console.print(f"Passed tests: {len(validation_results['passed'])}")
        console.print(f"Failed tests: {len(validation_results['failed'])}")
        
        if validation_results['passed']:
            console.print("\n[green]Passed:[/green]")
            for test in validation_results['passed'][:5]:  # Show first 5
                console.print(f"  ✓ {test}")
        
        if validation_results['failed']:
            console.print("\n[red]Failed:[/red]")
            for failure in validation_results['failed']:
                console.print(f"  ✗ {failure['test']}: {failure['error']}")
                
    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
    
    finally:
        feature_flags.set("use_new_storage", False)
    
    # Example 4: Dataset migration
    console.print("\n[bold]Example 4: Dataset Migration[/bold]")
    console.print("=" * 50 + "\n")
    
    console.print("Setting up test dataset for migration...")
    
    import pandas as pd
    import numpy as np
    
    # Create a test dataset with legacy backend
    feature_flags.set("use_new_storage", False)
    backend = get_storage_backend("sqlite")
    
    test_dataset = "migration_example_dataset"
    test_data = pd.DataFrame({
        "id": range(1000),
        "name": [f"item_{i}" for i in range(1000)],
        "value": np.random.rand(1000),
        "category": np.random.choice(["A", "B", "C"], 1000)
    })
    
    try:
        # Create dataset if it doesn't exist
        if not backend.dataset_exists(test_dataset):
            backend.create_dataset(test_dataset, {"example": True})
            backend.save_data(test_dataset, test_data)
            backend.update_metadata(test_dataset, {
                "description": "Example dataset for migration",
                "created_by": "storage_migration_example"
            })
            console.print(f"Created test dataset: {test_dataset}")
        
        # Now demonstrate migration from SQLite to DuckDB
        console.print("\nMigrating from SQLite to DuckDB...")
        
        migrator = StorageMigrator("sqlite", "duckdb")
        result = migrator.migrate_dataset(test_dataset, verify=True)
        
        if result["success"]:
            console.print(f"[green]✓ Successfully migrated {test_dataset}[/green]")
            console.print(f"  - Rows migrated: {result['rows_migrated']}")
            console.print(f"  - Tables migrated: {result['tables_migrated']}")
            console.print(f"  - Duration: {result['duration_seconds']:.2f} seconds")
            
            if "verification" in result:
                console.print(f"  - Verification: {'PASSED' if result['verification']['matches'] else 'FAILED'}")
        else:
            console.print(f"[red]✗ Migration failed: {result['error']}[/red]")
        
        # Clean up
        console.print("\nCleaning up test datasets...")
        
        # Remove from SQLite
        sqlite_backend = get_storage_backend("sqlite")
        if sqlite_backend.dataset_exists(test_dataset):
            sqlite_backend.drop_dataset(test_dataset)
        
        # Remove from DuckDB
        duckdb_backend = get_storage_backend("duckdb")
        if duckdb_backend.dataset_exists(test_dataset):
            duckdb_backend.drop_dataset(test_dataset)
            
        console.print("Cleanup complete")
        
    except Exception as e:
        console.print(f"[red]Migration example error: {e}[/red]")
    
    # Example 5: Performance comparison
    console.print("\n[bold]Example 5: Performance Comparison[/bold]")
    console.print("=" * 50 + "\n")
    
    console.print("Comparing performance between backends...")
    
    # Create test data
    n_rows = 10000
    perf_data = pd.DataFrame({
        "id": range(n_rows),
        "text": [f"text_{i}" * 5 for i in range(n_rows)],
        "value": np.random.rand(n_rows),
        "category": np.random.choice(["A", "B", "C", "D", "E"], n_rows)
    })
    
    backends_to_test = ["sqlite", "duckdb"]
    perf_results = {}
    
    for backend_type in backends_to_test:
        try:
            import time
            
            backend = get_storage_backend(backend_type)
            dataset_name = f"perf_test_{backend_type}"
            
            # Time dataset creation and data loading
            start = time.time()
            
            backend.create_dataset(dataset_name, {})
            backend.save_data(dataset_name, perf_data)
            
            # Time data retrieval
            loaded = backend.load_data(dataset_name)
            
            end = time.time()
            
            perf_results[backend_type] = {
                "time": end - start,
                "rows": len(loaded)
            }
            
            # Clean up
            backend.drop_dataset(dataset_name)
            
        except Exception as e:
            console.print(f"[red]Performance test failed for {backend_type}: {e}[/red]")
    
    # Display results
    if perf_results:
        from rich.table import Table
        
        table = Table(title="Backend Performance Comparison")
        table.add_column("Backend", style="cyan")
        table.add_column("Time (seconds)", style="yellow")
        table.add_column("Rows/Second", style="green")
        
        for backend, results in perf_results.items():
            rows_per_sec = results["rows"] / results["time"] if results["time"] > 0 else 0
            table.add_row(
                backend,
                f"{results['time']:.3f}",
                f"{rows_per_sec:,.0f}"
            )
        
        console.print("\n")
        console.print(table)
    
    console.print("\n[bold green]Storage migration examples completed![/bold green]")


if __name__ == "__main__":
    main()