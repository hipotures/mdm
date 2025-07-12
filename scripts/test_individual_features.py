#!/usr/bin/env python3
"""Test individual feature generators performance."""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mdm.features.generic import (
    CategoricalFeatures,
    ClusteringFeatures,
    DistributionFeatures,
    InteractionFeatures,
    MissingDataFeatures,
    SequentialFeatures,
    StatisticalFeatures,
    TemporalFeatures,
    TextFeatures,
)

console = Console()


def load_sample_data(dataset_path: Path, sample_size: int = 10000) -> pd.DataFrame:
    """Load sample data from dataset."""
    train_file = dataset_path / "train.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    
    # Load sample
    df = pd.read_csv(train_file, nrows=sample_size)
    console.print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def test_feature_generator(generator_class, df: pd.DataFrame, name: str) -> dict:
    """Test a single feature generator."""
    console.print(f"\nTesting [bold]{name}[/bold]...")
    
    try:
        # Create instance
        generator = generator_class()
        
        # Get applicable columns
        start = time.time()
        applicable_cols = generator.get_applicable_columns(df)
        col_detection_time = time.time() - start
        
        console.print(f"  Applicable columns: {len(applicable_cols)}")
        
        # Generate features
        start = time.time()
        features, descriptions = generator.generate_features(df)
        generation_time = time.time() - start
        
        console.print(f"  Generated features: {len(features.columns)}")
        console.print(f"  [green]✓ Success[/green] - Total time: {generation_time:.2f}s")
        
        return {
            'name': name,
            'applicable_columns': len(applicable_cols),
            'features_generated': len(features.columns),
            'column_detection_time': col_detection_time,
            'generation_time': generation_time,
            'total_time': col_detection_time + generation_time,
            'status': 'success'
        }
        
    except Exception as e:
        console.print(f"  [red]✗ Failed: {str(e)[:100]}[/red]")
        return {
            'name': name,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Main test function."""
    # Test on first available dataset
    kaggle_base = Path("/mnt/ml/kaggle")
    dataset_paths = sorted(kaggle_base.glob("playground-series-s4*"))[:1]
    
    if not dataset_paths:
        console.print("[red]No datasets found![/red]")
        return
    
    dataset_path = dataset_paths[0]
    console.print(f"[bold]Feature Generator Performance Test[/bold]")
    console.print(f"Dataset: {dataset_path.name}\n")
    
    # Load sample data
    df = load_sample_data(dataset_path)
    
    # Add some missing values for testing
    for col in df.columns[:3]:
        mask = np.random.random(len(df)) < 0.1
        df.loc[mask, col] = np.nan
    
    # Test generators
    generators = [
        (MissingDataFeatures, "MissingDataFeatures"),
        (CategoricalFeatures, "CategoricalFeatures"),
        (StatisticalFeatures, "StatisticalFeatures"),
        (TextFeatures, "TextFeatures"),
        (TemporalFeatures, "TemporalFeatures"),
        (InteractionFeatures, "InteractionFeatures"),
        (SequentialFeatures, "SequentialFeatures"),
        (DistributionFeatures, "DistributionFeatures"),
        (ClusteringFeatures, "ClusteringFeatures"),
    ]
    
    results = []
    for generator_class, name in generators:
        result = test_feature_generator(generator_class, df, name)
        results.append(result)
    
    # Display results
    console.print("\n[bold]Performance Summary[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Generator", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Columns", justify="right")
    table.add_column("Features", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Features/s", justify="right")
    
    for result in results:
        if result['status'] == 'success':
            features_per_sec = result['features_generated'] / result['total_time'] if result['total_time'] > 0 else 0
            table.add_row(
                result['name'],
                "[green]✓[/green]",
                str(result['applicable_columns']),
                str(result['features_generated']),
                f"{result['total_time']:.3f}",
                f"{features_per_sec:.1f}"
            )
        else:
            table.add_row(
                result['name'],
                "[red]✗[/red]",
                "-",
                "-",
                "-",
                "-"
            )
    
    console.print(table)
    
    # Complexity analysis
    console.print("\n[bold]Complexity Analysis (10k rows)[/bold]")
    
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        # Sort by time
        successful.sort(key=lambda x: x['total_time'])
        
        console.print("\nFastest generators:")
        for r in successful[:3]:
            console.print(f"  • {r['name']}: {r['total_time']:.3f}s")
        
        console.print("\nSlowest generators:")
        for r in successful[-3:]:
            console.print(f"  • {r['name']}: {r['total_time']:.3f}s")
        
        # Total time estimate
        total_time = sum(r['total_time'] for r in successful)
        console.print(f"\nTotal time for all generators: {total_time:.2f}s")
        console.print(f"Average time per generator: {total_time/len(successful):.2f}s")


if __name__ == "__main__":
    main()