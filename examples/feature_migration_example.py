"""Example demonstrating feature engineering migration functionality.

This example shows how to:
1. Compare legacy vs new feature engineering
2. Migrate custom transformers
3. Validate feature engineering implementations
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mdm.core import feature_flags
from mdm.adapters.feature_manager import get_feature_generator
from mdm.testing.feature_comparison import FeatureComparisonTester
from mdm.migration.feature_migration import FeatureMigrator, FeatureValidator

console = Console()


def main():
    """Run feature engineering migration examples."""
    console.print(Panel.fit(
        "[bold cyan]Feature Engineering Migration Examples[/bold cyan]\n\n"
        "This demonstrates the feature engineering migration functionality",
        title="MDM Feature Migration"
    ))
    
    # Example 1: Compare feature engineering implementations
    console.print("\n[bold]Example 1: Comparing Feature Engineering[/bold]")
    console.print("=" * 50 + "\n")
    
    # Create sample dataset
    console.print("Creating sample dataset...")
    sample_data = create_sample_dataset(1000)
    console.print(f"Dataset shape: {sample_data.shape}")
    console.print(f"Columns: {', '.join(sample_data.columns)}\n")
    
    # Compare implementations
    console.print("Comparing legacy vs new feature generation...")
    
    # Legacy generation
    feature_flags.set("use_new_features", False)
    legacy_gen = get_feature_generator()
    legacy_features = legacy_gen.generate_features(
        sample_data,
        target_column="target",
        id_columns=["id"]
    )
    
    # New generation
    feature_flags.set("use_new_features", True)
    new_gen = get_feature_generator()
    new_features = new_gen.generate_features(
        sample_data,
        target_column="target",
        id_columns=["id"]
    )
    
    # Display comparison
    display_feature_comparison(sample_data, legacy_features, new_features)
    
    # Example 2: Feature flag switching
    console.print("\n[bold]Example 2: Feature Flag Switching[/bold]")
    console.print("=" * 50 + "\n")
    
    console.print(f"Current feature flag: use_new_features = {feature_flags.get('use_new_features', False)}")
    
    # Demonstrate switching
    for use_new in [False, True]:
        feature_flags.set("use_new_features", use_new)
        gen = get_feature_generator()
        impl_type = "New" if use_new else "Legacy"
        console.print(f"use_new_features={use_new} -> {impl_type} implementation ({gen.__class__.__name__})")
    
    # Example 3: Feature type-specific generation
    console.print("\n[bold]Example 3: Type-Specific Feature Generation[/bold]")
    console.print("=" * 50 + "\n")
    
    # Reset to legacy for comparison
    feature_flags.set("use_new_features", False)
    
    # Numeric features
    console.print("[yellow]Numeric Features:[/yellow]")
    numeric_data = sample_data[["value1", "value2"]]
    
    legacy_numeric = legacy_gen.generate_numeric_features(numeric_data)
    feature_flags.set("use_new_features", True)
    new_numeric = new_gen.generate_numeric_features(numeric_data)
    
    console.print(f"  Legacy: {len(legacy_numeric.columns) - len(numeric_data.columns)} features")
    console.print(f"  New: {len(new_numeric.columns) - len(numeric_data.columns)} features")
    
    # Categorical features
    console.print("\n[yellow]Categorical Features:[/yellow]")
    cat_data = sample_data[["category", "subcategory"]]
    
    feature_flags.set("use_new_features", False)
    legacy_cat = legacy_gen.generate_categorical_features(cat_data)
    feature_flags.set("use_new_features", True)
    new_cat = new_gen.generate_categorical_features(cat_data)
    
    console.print(f"  Legacy: {len(legacy_cat.columns) - len(cat_data.columns)} features")
    console.print(f"  New: {len(new_cat.columns) - len(cat_data.columns)} features")
    
    # Example 4: Feature importance
    console.print("\n[bold]Example 4: Feature Importance[/bold]")
    console.print("=" * 50 + "\n")
    
    # Use subset of features for importance
    feature_subset = sample_data[["value1", "value2", "category"]].copy()
    target = sample_data["target"]
    
    # Calculate importance with both implementations
    feature_flags.set("use_new_features", False)
    legacy_importance = legacy_gen.get_feature_importance(
        feature_subset, target, "regression"
    )
    
    feature_flags.set("use_new_features", True)
    new_importance = new_gen.get_feature_importance(
        feature_subset, target, "regression"
    )
    
    # Display top features
    display_feature_importance(legacy_importance, new_importance)
    
    # Example 5: Migration validation
    console.print("\n[bold]Example 5: Migration Validation[/bold]")
    console.print("=" * 50 + "\n")
    
    validator = FeatureValidator()
    
    # Validate consistency
    console.print("Validating feature engineering consistency...")
    validation_results = validator.validate_consistency(
        sample_data,
        feature_types=["numeric", "categorical"]
    )
    
    # Display validation results
    table = Table(title="Validation Results")
    table.add_column("Feature Type", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Match Rate", style="green")
    
    for feature_type, result in validation_results["feature_types"].items():
        if result["status"] == "tested":
            match_rate = f"{result['match_rate']:.1%}"
        else:
            match_rate = "N/A"
        
        table.add_row(
            feature_type,
            result["status"],
            match_rate
        )
    
    console.print(table)
    console.print(f"\nOverall match rate: {validation_results['overall_match_rate']:.1%}")
    console.print(f"Validation passed: {validation_results['passed']}")
    
    # Example 6: Performance comparison
    console.print("\n[bold]Example 6: Performance Comparison[/bold]")
    console.print("=" * 50 + "\n")
    
    # Test with different dataset sizes
    sizes = [100, 1000, 10000]
    performance_results = []
    
    for size in sizes:
        test_data = create_sample_dataset(size)
        
        # Time legacy
        import time
        feature_flags.set("use_new_features", False)
        legacy_start = time.time()
        legacy_gen.generate_features(test_data)
        legacy_time = time.time() - legacy_start
        
        # Time new
        feature_flags.set("use_new_features", True)
        new_start = time.time()
        new_gen.generate_features(test_data)
        new_time = time.time() - new_start
        
        performance_results.append({
            "size": size,
            "legacy_time": legacy_time,
            "new_time": new_time,
            "ratio": new_time / legacy_time if legacy_time > 0 else 1.0
        })
    
    # Display performance table
    perf_table = Table(title="Performance Comparison")
    perf_table.add_column("Dataset Size", style="cyan")
    perf_table.add_column("Legacy Time (s)", style="yellow")
    perf_table.add_column("New Time (s)", style="yellow")
    perf_table.add_column("Ratio", style="green")
    
    for result in performance_results:
        perf_table.add_row(
            f"{result['size']:,}",
            f"{result['legacy_time']:.3f}",
            f"{result['new_time']:.3f}",
            f"{result['ratio']:.2f}x"
        )
    
    console.print(perf_table)
    
    console.print("\n[bold green]Feature migration examples completed![/bold green]")


def create_sample_dataset(n_rows: int) -> pd.DataFrame:
    """Create a sample dataset for testing."""
    np.random.seed(42)
    
    return pd.DataFrame({
        "id": range(n_rows),
        "value1": np.random.randn(n_rows),
        "value2": np.random.exponential(2, n_rows),
        "category": np.random.choice(["A", "B", "C", "D"], n_rows),
        "subcategory": np.random.choice(["X", "Y", "Z"], n_rows),
        "date": pd.date_range("2024-01-01", periods=n_rows, freq="H"),
        "text": [f"Sample text with {i} words" for i in range(n_rows)],
        "target": np.random.randn(n_rows) + np.random.choice([0, 1], n_rows) * 2
    })


def display_feature_comparison(
    original: pd.DataFrame,
    legacy_features: pd.DataFrame,
    new_features: pd.DataFrame
):
    """Display feature comparison results."""
    orig_cols = set(original.columns)
    legacy_new_cols = set(legacy_features.columns) - orig_cols
    new_new_cols = set(new_features.columns) - orig_cols
    
    common_features = legacy_new_cols.intersection(new_new_cols)
    legacy_only = legacy_new_cols - new_new_cols
    new_only = new_new_cols - legacy_new_cols
    
    # Create comparison table
    table = Table(title="Feature Generation Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Legacy", style="yellow")
    table.add_column("New", style="green")
    
    table.add_row("Original columns", str(len(orig_cols)), str(len(orig_cols)))
    table.add_row("Total columns", str(len(legacy_features.columns)), str(len(new_features.columns)))
    table.add_row("New features", str(len(legacy_new_cols)), str(len(new_new_cols)))
    table.add_row("Common features", str(len(common_features)), str(len(common_features)))
    table.add_row("Unique features", str(len(legacy_only)), str(len(new_only)))
    
    console.print(table)
    
    if legacy_only:
        console.print(f"\n[yellow]Legacy-only features:[/yellow] {', '.join(list(legacy_only)[:5])}")
        if len(legacy_only) > 5:
            console.print(f"  ... and {len(legacy_only) - 5} more")
    
    if new_only:
        console.print(f"\n[green]New-only features:[/green] {', '.join(list(new_only)[:5])}")
        if len(new_only) > 5:
            console.print(f"  ... and {len(new_only) - 5} more")


def display_feature_importance(
    legacy_importance: pd.DataFrame,
    new_importance: pd.DataFrame
):
    """Display feature importance comparison."""
    table = Table(title="Top 5 Important Features")
    table.add_column("Rank", style="cyan")
    table.add_column("Legacy Feature", style="yellow")
    table.add_column("Legacy Score", style="yellow")
    table.add_column("New Feature", style="green")
    table.add_column("New Score", style="green")
    
    for i in range(min(5, len(legacy_importance), len(new_importance))):
        table.add_row(
            str(i + 1),
            legacy_importance.iloc[i]["feature"],
            f"{legacy_importance.iloc[i]['importance']:.3f}",
            new_importance.iloc[i]["feature"],
            f"{new_importance.iloc[i]['importance']:.3f}"
        )
    
    console.print(table)


if __name__ == "__main__":
    main()