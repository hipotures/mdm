"""Example demonstrating dataset registration migration functionality.

This example shows how to:
1. Register datasets with old and new systems
2. Compare registration implementations
3. Migrate datasets between systems
4. Validate migration results
"""
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mdm.core import feature_flags
from mdm.adapters import (
    get_dataset_registrar,
    get_dataset_manager,
    clear_dataset_cache
)
from mdm.testing.dataset_comparison import DatasetComparisonTester
from mdm.migration.dataset_migration import DatasetMigrator, DatasetValidator

console = Console()


def main():
    """Run dataset registration migration examples."""
    console.print(Panel.fit(
        "[bold cyan]Dataset Registration Migration Examples[/bold cyan]\n\n"
        "This demonstrates the dataset registration migration functionality",
        title="MDM Dataset Migration"
    ))
    
    # Create temporary directory for examples
    temp_dir = Path(tempfile.mkdtemp(prefix="mdm_example_"))
    
    try:
        # Example 1: Basic registration comparison
        console.print("\n[bold]Example 1: Basic Registration Comparison[/bold]")
        console.print("=" * 50 + "\n")
        example_basic_registration(temp_dir)
        
        # Example 2: Kaggle dataset registration
        console.print("\n[bold]Example 2: Kaggle Dataset Registration[/bold]")
        console.print("=" * 50 + "\n")
        example_kaggle_registration(temp_dir)
        
        # Example 3: Auto-detection features
        console.print("\n[bold]Example 3: Auto-detection Features[/bold]")
        console.print("=" * 50 + "\n")
        example_auto_detection(temp_dir)
        
        # Example 4: Dataset migration
        console.print("\n[bold]Example 4: Dataset Migration[/bold]")
        console.print("=" * 50 + "\n")
        example_dataset_migration()
        
        # Example 5: Performance comparison
        console.print("\n[bold]Example 5: Performance Comparison[/bold]")
        console.print("=" * 50 + "\n")
        example_performance_comparison(temp_dir)
        
        console.print("\n[bold green]Examples completed successfully![/bold green]")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def example_basic_registration(temp_dir: Path):
    """Example of basic dataset registration."""
    # Create sample dataset
    console.print("Creating sample dataset...")
    data = pd.DataFrame({
        'customer_id': range(1000, 1500),
        'purchase_amount': np.random.exponential(50, 500),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 500),
        'is_returning': np.random.randint(0, 2, 500)
    })
    
    csv_path = temp_dir / "purchases.csv"
    data.to_csv(csv_path, index=False)
    console.print(f"Created dataset at: {csv_path}")
    
    # Register with legacy system
    console.print("\n[yellow]Registering with legacy system:[/yellow]")
    feature_flags.set("use_new_dataset_registration", False)
    
    legacy_registrar = get_dataset_registrar()
    legacy_result = legacy_registrar.register(
        name="purchases_legacy",
        path=str(csv_path),
        target="is_returning",
        id_columns=["customer_id"],
        force=True
    )
    
    console.print(f"Legacy registration completed")
    
    # Register with new system
    console.print("\n[green]Registering with new system:[/green]")
    feature_flags.set("use_new_dataset_registration", True)
    
    new_registrar = get_dataset_registrar()
    new_result = new_registrar.register(
        name="purchases_new",
        path=str(csv_path),
        target="is_returning",
        id_columns=["customer_id"],
        force=True
    )
    
    console.print(f"New registration completed")
    
    # Compare results
    table = Table(title="Registration Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Legacy", style="yellow")
    table.add_column("New", style="green")
    
    table.add_row("Backend", legacy_result.get('backend', 'N/A'), new_result.get('backend', 'N/A'))
    table.add_row("Tables", str(len(legacy_result.get('tables', []))), str(len(new_result.get('tables', []))))
    table.add_row("Features Generated", str(legacy_result.get('features_generated', 0)), str(new_result.get('features_generated', 0)))
    table.add_row("Registration Time", f"{legacy_result.get('registration_time', 0):.2f}s", f"{new_result.get('registration_time', 0):.2f}s")
    
    console.print(table)
    
    # Cleanup
    cleanup_datasets(["purchases_legacy", "purchases_new"])


def example_kaggle_registration(temp_dir: Path):
    """Example of Kaggle dataset registration."""
    # Create Kaggle-like structure
    console.print("Creating Kaggle competition structure...")
    kaggle_dir = temp_dir / "titanic"
    kaggle_dir.mkdir()
    
    # Create train.csv
    train_data = pd.DataFrame({
        'PassengerId': range(1, 892),
        'Pclass': np.random.randint(1, 4, 891),
        'Sex': np.random.choice(['male', 'female'], 891),
        'Age': np.random.randint(1, 80, 891),
        'Fare': np.random.exponential(30, 891),
        'Survived': np.random.randint(0, 2, 891)
    })
    train_data.to_csv(kaggle_dir / "train.csv", index=False)
    
    # Create test.csv (without Survived)
    test_data = train_data[['PassengerId', 'Pclass', 'Sex', 'Age', 'Fare']].iloc[700:]
    test_data['PassengerId'] = range(892, 892 + len(test_data))
    test_data.to_csv(kaggle_dir / "test.csv", index=False)
    
    # Create sample_submission.csv
    submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': 0
    })
    submission.to_csv(kaggle_dir / "sample_submission.csv", index=False)
    
    console.print(f"Created Kaggle structure at: {kaggle_dir}")
    console.print("Files: train.csv, test.csv, sample_submission.csv")
    
    # Register with new system (has better Kaggle detection)
    console.print("\n[green]Registering Kaggle dataset:[/green]")
    feature_flags.set("use_new_dataset_registration", True)
    
    registrar = get_dataset_registrar()
    result = registrar.register(
        name="titanic_competition",
        path=str(kaggle_dir),
        force=True
        # Note: Not specifying target - should auto-detect from sample_submission
    )
    
    # Show detected information
    manager = get_dataset_manager()
    info = manager.get_dataset_info("titanic_competition")
    
    console.print("\n[bold]Auto-detected Information:[/bold]")
    console.print(f"Structure Type: Kaggle Competition")
    console.print(f"Target Column: {info.get('schema', {}).get('target_column', 'Not detected')}")
    console.print(f"ID Column: {info.get('schema', {}).get('id_columns', [])}")
    console.print(f"Tables: {list(info.get('storage', {}).get('tables', {}).keys())}")
    
    # Cleanup
    cleanup_datasets(["titanic_competition"])


def example_auto_detection(temp_dir: Path):
    """Example of auto-detection features."""
    # Create dataset with obvious patterns
    console.print("Creating dataset for auto-detection...")
    
    data = pd.DataFrame({
        'order_id': [f'ORD{i:06d}' for i in range(1, 1001)],  # ID pattern
        'customer_id': np.random.randint(1000, 2000, 1000),   # ID pattern
        'order_date': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'product_name': np.random.choice(['Widget A', 'Gadget B', 'Tool C'], 1000),
        'quantity': np.random.randint(1, 10, 1000),
        'unit_price': np.random.uniform(10, 100, 1000),
        'total_amount': None,  # Will calculate
        'is_fraud': np.random.randint(0, 2, 1000)  # Target pattern
    })
    
    # Calculate total
    data['total_amount'] = data['quantity'] * data['unit_price']
    
    csv_path = temp_dir / "orders_autodetect.csv"
    data.to_csv(csv_path, index=False)
    
    # Register without specifying schema
    console.print("\n[green]Registering with auto-detection:[/green]")
    feature_flags.set("use_new_dataset_registration", True)
    
    registrar = get_dataset_registrar()
    result = registrar.register(
        name="orders_autodetect",
        path=str(csv_path),
        force=True
        # Not specifying target, id_columns, or problem_type
    )
    
    # Show what was detected
    manager = get_dataset_manager()
    info = manager.get_dataset_info("orders_autodetect")
    schema = info.get('schema', {})
    
    table = Table(title="Auto-detection Results")
    table.add_column("Feature", style="cyan")
    table.add_column("Detected Value", style="green")
    table.add_column("Correct?", style="yellow")
    
    # Check detections
    detected_ids = set(schema.get('id_columns', []))
    expected_ids = {'order_id', 'customer_id'}
    id_correct = bool(detected_ids & expected_ids)
    
    table.add_row(
        "ID Columns",
        str(schema.get('id_columns', [])),
        "✓" if id_correct else "✗"
    )
    
    table.add_row(
        "Target Column",
        str(schema.get('target_column', 'None')),
        "✓" if schema.get('target_column') == 'is_fraud' else "✗"
    )
    
    table.add_row(
        "Problem Type",
        str(schema.get('problem_type', 'None')),
        "✓" if schema.get('problem_type') == 'binary_classification' else "✗"
    )
    
    datetime_cols = info.get('metadata', {}).get('datetime_columns', [])
    table.add_row(
        "Datetime Columns",
        str(datetime_cols),
        "✓" if 'order_date' in str(datetime_cols) else "✗"
    )
    
    console.print(table)
    
    # Cleanup
    cleanup_datasets(["orders_autodetect"])


def example_dataset_migration():
    """Example of migrating datasets between systems."""
    # First, create a dataset with legacy system
    console.print("Creating dataset with legacy system...")
    
    # Create temporary data
    temp_file = Path(tempfile.mktemp(suffix=".csv"))
    data = pd.DataFrame({
        'id': range(100),
        'value': np.random.randn(100),
        'label': np.random.choice(['A', 'B', 'C'], 100)
    })
    data.to_csv(temp_file, index=False)
    
    # Register with legacy
    feature_flags.set("use_new_dataset_registration", False)
    registrar = get_dataset_registrar()
    registrar.register(
        name="migration_test",
        path=str(temp_file),
        target="label",
        problem_type="multiclass_classification",
        force=True
    )
    
    console.print("Dataset registered with legacy system")
    
    # Create migrator
    migrator = DatasetMigrator()
    
    # Dry run first
    console.print("\n[yellow]Performing dry run migration:[/yellow]")
    dry_result = migrator.migrate_dataset(
        "migration_test",
        dry_run=True
    )
    
    console.print(f"Dry run status: {dry_result['status']}")
    if 'simulation' in dry_result:
        console.print(f"Would migrate from: {dry_result['simulation']['source_path']}")
        console.print(f"Target column: {dry_result['simulation']['target']}")
    
    # Validate before migration
    console.print("\n[cyan]Validating dataset consistency:[/cyan]")
    validator = DatasetValidator()
    
    # This will show differences since new system doesn't have it yet
    validation = validator.validate_consistency(
        "migration_test",
        check_data=False,  # Skip data check for now
        check_features=False
    )
    
    console.print(f"Pre-migration validation: {'Passed' if validation['passed'] else 'Failed'}")
    console.print(f"Checks: {validation['checks']['existence']}")
    
    # Actual migration
    console.print("\n[green]Performing actual migration:[/green]")
    result = migrator.migrate_dataset(
        "migration_test",
        dry_run=False,
        preserve_features=True
    )
    
    console.print(f"Migration status: {result['status']}")
    console.print(f"Duration: {result['duration']:.2f}s")
    
    # Validate after migration
    console.print("\n[cyan]Post-migration validation:[/cyan]")
    validation = validator.validate_consistency(
        "migration_test",
        check_data=True,
        check_features=False
    )
    
    console.print(f"Post-migration validation: {'Passed' if validation['passed'] else 'Failed'}")
    
    # Show validation details
    if 'configuration' in validation['checks']:
        config_check = validation['checks']['configuration']
        console.print(f"Configuration match: {config_check['passed']}")
        if 'details' in config_check:
            for key, match in config_check['details'].items():
                console.print(f"  {key}: {'✓' if match else '✗'}")
    
    # Cleanup
    cleanup_datasets(["migration_test"])
    temp_file.unlink()


def example_performance_comparison(temp_dir: Path):
    """Example of performance comparison between systems."""
    console.print("Running performance comparison...")
    
    # Create datasets of different sizes
    sizes = [100, 1000, 10000]
    results = []
    
    for size in sizes:
        console.print(f"\nTesting with {size:,} rows...")
        
        # Create data
        data = pd.DataFrame({
            'id': range(size),
            'numeric1': np.random.randn(size),
            'numeric2': np.random.exponential(2, size),
            'category': np.random.choice(['A', 'B', 'C', 'D'], size),
            'target': np.random.uniform(0, 1, size)
        })
        
        csv_path = temp_dir / f"perf_test_{size}.csv"
        data.to_csv(csv_path, index=False)
        
        # Time legacy
        import time
        feature_flags.set("use_new_dataset_registration", False)
        clear_dataset_cache()
        
        legacy_start = time.time()
        registrar = get_dataset_registrar()
        registrar.register(
            name=f"perf_legacy_{size}",
            path=str(csv_path),
            generate_features=False,  # Disable for pure registration test
            force=True
        )
        legacy_time = time.time() - legacy_start
        
        # Time new
        feature_flags.set("use_new_dataset_registration", True)
        clear_dataset_cache()
        
        new_start = time.time()
        registrar = get_dataset_registrar()
        registrar.register(
            name=f"perf_new_{size}",
            path=str(csv_path),
            generate_features=False,
            force=True
        )
        new_time = time.time() - new_start
        
        results.append({
            'size': size,
            'legacy_time': legacy_time,
            'new_time': new_time,
            'speedup': legacy_time / new_time if new_time > 0 else 0
        })
        
        # Cleanup
        cleanup_datasets([f"perf_legacy_{size}", f"perf_new_{size}"])
    
    # Display results
    table = Table(title="Performance Comparison")
    table.add_column("Dataset Size", style="cyan")
    table.add_column("Legacy Time (s)", style="yellow")
    table.add_column("New Time (s)", style="green")
    table.add_column("Speedup", style="magenta")
    
    for result in results:
        speedup_str = f"{result['speedup']:.2f}x"
        if result['speedup'] > 1:
            speedup_str = f"[green]{speedup_str}[/green]"
        elif result['speedup'] < 1:
            speedup_str = f"[red]{speedup_str}[/red]"
        
        table.add_row(
            f"{result['size']:,}",
            f"{result['legacy_time']:.3f}",
            f"{result['new_time']:.3f}",
            speedup_str
        )
    
    console.print(table)
    
    # Overall performance
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    console.print(f"\nAverage speedup: {avg_speedup:.2f}x")
    
    if avg_speedup > 1:
        console.print("[green]New implementation is faster! ✓[/green]")
    else:
        console.print("[yellow]New implementation is slower, but provides better features[/yellow]")


def cleanup_datasets(dataset_names: list):
    """Clean up test datasets."""
    manager = get_dataset_manager()
    for name in dataset_names:
        try:
            if manager.dataset_exists(name):
                manager.remove_dataset(name, force=True)
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()
