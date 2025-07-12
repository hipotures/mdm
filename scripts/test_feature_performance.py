#!/usr/bin/env python3
"""Test feature generation performance on Kaggle datasets."""

import os
import sys
import time
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mdm.config import get_config_manager, reset_config_manager
from mdm.dataset.registrar import DatasetRegistrar

console = Console()


def setup_test_environment(temp_dir: Path) -> None:
    """Setup isolated test environment."""
    os.environ['MDM_HOME_DIR'] = str(temp_dir)
    reset_config_manager()
    
    # Create directories
    (temp_dir / "datasets").mkdir(parents=True)
    (temp_dir / "config" / "datasets").mkdir(parents=True)
    (temp_dir / "logs").mkdir(parents=True)
    
    # Create config
    config_file = temp_dir / "mdm.yaml"
    config_file.write_text("""
database:
  default_backend: sqlite
  sqlite:
    synchronous: NORMAL
    journal_mode: WAL

performance:
  batch_size: 10000

logging:
  level: WARNING
  file: mdm.log
  format: console

features:
  enabled: true
  compute_statistics: false
""")


def test_dataset_registration(dataset_path: Path, dataset_name: str, 
                            generate_features: bool = True) -> Dict[str, float]:
    """Test dataset registration and measure performance."""
    registrar = DatasetRegistrar()
    
    console.print(f"\n[bold]Testing {dataset_name}[/bold]")
    console.print(f"Path: {dataset_path}")
    
    # Check dataset size
    train_file = dataset_path / "train.csv"
    if train_file.exists():
        df_sample = pd.read_csv(train_file, nrows=1000)
        total_rows = sum(1 for _ in open(train_file)) - 1  # Subtract header
        console.print(f"Train size: {total_rows:,} rows, {len(df_sample.columns)} columns")
    
    start_time = time.time()
    
    try:
        # Register dataset
        registration_start = time.time()
        result = registrar.register(
            name=dataset_name,
            path=dataset_path,
            auto_detect=True,
            generate_features=generate_features,
            force=True
        )
        registration_time = time.time() - registration_start
        
        # Get timing breakdown
        times = {
            'total_time': registration_time,
            'data_loading': 0,
            'feature_generation': 0,
            'database_write': 0
        }
        
        # Estimate component times (rough approximation)
        if generate_features:
            times['feature_generation'] = registration_time * 0.4  # ~40% for features
            times['data_loading'] = registration_time * 0.3  # ~30% for loading
            times['database_write'] = registration_time * 0.3  # ~30% for DB writes
        else:
            times['data_loading'] = registration_time * 0.5
            times['database_write'] = registration_time * 0.5
        
        console.print(f"[green]✓ Registration completed in {registration_time:.2f}s[/green]")
        
        return times
        
    except Exception as e:
        console.print(f"[red]✗ Registration failed: {e}[/red]")
        return {'total_time': -1, 'error': str(e)}


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test feature generation performance")
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to test')
    parser.add_argument('--no-features', action='store_true', help='Disable feature generation')
    parser.add_argument('--limit', type=int, default=5, help='Limit number of datasets to test')
    args = parser.parse_args()
    
    # Find Kaggle datasets
    kaggle_base = Path("/mnt/ml/kaggle")
    pattern = "playground-series-s4*"
    
    if args.datasets:
        dataset_paths = [kaggle_base / d for d in args.datasets]
    else:
        dataset_paths = sorted(kaggle_base.glob(pattern))[:args.limit]
    
    if not dataset_paths:
        console.print("[red]No datasets found![/red]")
        return
    
    console.print(f"[bold]Feature Generation Performance Test[/bold]")
    console.print(f"Testing {len(dataset_paths)} datasets\n")
    
    # Create temporary environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        setup_test_environment(temp_path)
        
        results = []
        
        # Test each dataset
        for dataset_path in dataset_paths:
            if not dataset_path.exists():
                console.print(f"[yellow]Skipping {dataset_path.name} - not found[/yellow]")
                continue
            
            dataset_name = dataset_path.name.replace("playground-series-", "")
            
            # Test with features
            if not args.no_features:
                times_with = test_dataset_registration(
                    dataset_path, 
                    f"{dataset_name}_with_features",
                    generate_features=True
                )
                
                # Test without features
                times_without = test_dataset_registration(
                    dataset_path,
                    f"{dataset_name}_no_features", 
                    generate_features=False
                )
                
                results.append({
                    'dataset': dataset_name,
                    'with_features': times_with['total_time'],
                    'without_features': times_without['total_time'],
                    'feature_overhead': times_with['total_time'] - times_without['total_time']
                })
            else:
                times = test_dataset_registration(
                    dataset_path,
                    dataset_name,
                    generate_features=False
                )
                results.append({
                    'dataset': dataset_name,
                    'time': times['total_time']
                })
        
        # Display results
        console.print("\n[bold]Performance Summary[/bold]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Dataset", style="cyan")
        
        if not args.no_features:
            table.add_column("With Features (s)", justify="right")
            table.add_column("Without Features (s)", justify="right")
            table.add_column("Feature Overhead (s)", justify="right")
            table.add_column("Overhead %", justify="right")
            
            for result in results:
                if result['with_features'] > 0 and result['without_features'] > 0:
                    overhead_pct = (result['feature_overhead'] / result['without_features']) * 100
                    table.add_row(
                        result['dataset'],
                        f"{result['with_features']:.2f}",
                        f"{result['without_features']:.2f}",
                        f"{result['feature_overhead']:.2f}",
                        f"{overhead_pct:.1f}%"
                    )
        else:
            table.add_column("Time (s)", justify="right")
            
            for result in results:
                if result['time'] > 0:
                    table.add_row(
                        result['dataset'],
                        f"{result['time']:.2f}"
                    )
        
        console.print(table)
        
        # Summary statistics
        if not args.no_features and results:
            valid_results = [r for r in results if r['with_features'] > 0]
            if valid_results:
                avg_overhead = sum(r['feature_overhead'] for r in valid_results) / len(valid_results)
                avg_time_with = sum(r['with_features'] for r in valid_results) / len(valid_results)
                
                console.print(f"\n[bold]Average Statistics:[/bold]")
                console.print(f"Average time with features: {avg_time_with:.2f}s")
                console.print(f"Average feature overhead: {avg_overhead:.2f}s")
                console.print(f"Average overhead percentage: {(avg_overhead / (avg_time_with - avg_overhead)) * 100:.1f}%")


if __name__ == "__main__":
    main()