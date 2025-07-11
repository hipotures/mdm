"""Time series commands for MDM CLI."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from mdm.api import MDMClient
from mdm.utils.time_series import TimeSeriesAnalyzer

app = typer.Typer(name="timeseries", help="Time series operations")
console = Console()


@app.command()
def analyze(
    dataset: str = typer.Argument(..., help="Dataset name"),
):
    """Analyze time series dataset."""
    try:
        client = MDMClient()

        # Get dataset info
        dataset_info = client.get_dataset(dataset)
        if not dataset_info:
            console.print(f"[red]Error:[/red] Dataset '{dataset}' not found")
            raise typer.Exit(1)

        if not dataset_info.time_column:
            console.print(f"[red]Error:[/red] Dataset '{dataset}' has no time column configured")
            raise typer.Exit(1)

        # Load data
        dfs = client.load_dataset_files(dataset)
        
        # Get the train dataframe
        train_df = dfs.get('train', dfs.get('data', None))
        if train_df is None:
            console.print(f"[red]Error:[/red] No train data found in dataset '{dataset}'")
            raise typer.Exit(1)

        # Analyze
        analyzer = TimeSeriesAnalyzer(dataset_info.time_column, dataset_info.target_column)
        analysis = analyzer.analyze(train_df)

        # Display results
        console.print(f"\n[bold]Time Series Analysis: {dataset}[/bold]\n")

        # Time range
        console.print("[bold]Time Range:[/bold]")
        console.print(f"  Start: {analysis['time_range']['start']}")
        console.print(f"  End: {analysis['time_range']['end']}")
        console.print(f"  Duration: {analysis['time_range']['duration_days']} days")
        console.print(f"  Frequency: {analysis['frequency']}")

        # Missing timestamps
        missing = analysis['missing_timestamps']
        if missing['count'] > 0:
            console.print(f"\n[yellow]Missing Timestamps:[/yellow] {missing['count']} ({missing['percentage']:.1f}%)")
            if missing['dates']:
                console.print("  First few: " + ", ".join(missing['dates'][:5]))

        # Seasonality
        if analysis['seasonality']:
            console.print("\n[bold]Seasonality Detected:[/bold]")
            for pattern, detected in analysis['seasonality'].items():
                if detected:
                    console.print(f"  - {pattern.capitalize()} pattern")

        # Target statistics
        if 'target_stats' in analysis:
            console.print("\n[bold]Target Statistics:[/bold]")
            console.print(f"  Mean: {analysis['target_stats']['mean']:.2f}")
            console.print(f"  Std: {analysis['target_stats']['std']:.2f}")
            console.print(f"  Trend: {analysis['target_stats']['trend']}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@app.command()
def split(
    dataset: str = typer.Argument(..., help="Dataset name"),
    test_size: float = typer.Option(0.2, "--test-size", help="Test set size (fraction or number of days)"),
    n_splits: int = typer.Option(3, "--n-splits", help="Number of splits for cross-validation"),
    gap: int = typer.Option(0, "--gap", help="Gap between train and test sets"),
    strategy: str = typer.Option("expanding", "--strategy", help="Strategy: 'expanding' or 'sliding'"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for splits"),
):
    """Split time series dataset for cross-validation."""
    try:
        client = MDMClient()

        # Get dataset info
        dataset_info = client.get_dataset(dataset)
        if not dataset_info:
            console.print(f"[red]Error:[/red] Dataset '{dataset}' not found")
            raise typer.Exit(1)

        # Split data
        console.print(f"Splitting dataset '{dataset}' for cross-validation...")
        splits = client.split_time_series(dataset, n_splits, test_size, gap, strategy)

        # Display split info
        console.print(f"\n[bold]Cross-Validation Splits ({strategy} window):[/bold]")
        console.print(f"Number of splits: {len(splits)}")
        console.print(f"Test size: {test_size}")
        if gap > 0:
            console.print(f"Gap between train/test: {gap}")
        
        for i, (train_df, test_df) in enumerate(splits):
            console.print(f"\n[bold]Split {i+1}:[/bold]")
            time_col = dataset_info.time_column
            if time_col and time_col in train_df.columns and time_col in test_df.columns:
                train_min = train_df[time_col].min()
                train_max = train_df[time_col].max()
                test_min = test_df[time_col].min()
                test_max = test_df[time_col].max()
                console.print(f"  train: {len(train_df):,} rows ({train_min} to {train_max})")
                console.print(f"  test:  {len(test_df):,} rows ({test_min} to {test_max})")
            else:
                console.print(f"  train: {len(train_df):,} rows")
                console.print(f"  test:  {len(test_df):,} rows")

        # Save splits if output directory specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"\nSaving splits to {output_dir}...")

            for i, (train_df, test_df) in enumerate(splits):
                # Save train
                train_path = output_dir / f"{dataset}_split{i+1}_train.csv"
                train_df.to_csv(train_path, index=False)
                console.print(f"  Saved split {i+1} train to {train_path}")
                
                # Save test
                test_path = output_dir / f"{dataset}_split{i+1}_test.csv"
                test_df.to_csv(test_path, index=False)
                console.print(f"  Saved split {i+1} test to {test_path}")

        console.print("\n[green]✓[/green] Time series split completed!")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@app.command()
def validate(
    dataset: str = typer.Argument(..., help="Dataset name"),
    n_folds: int = typer.Option(5, "--folds", help="Number of cross-validation folds"),
    gap_days: int = typer.Option(0, "--gap", help="Gap between train and test in days"),
):
    """Create time series cross-validation folds."""
    try:
        client = MDMClient()

        # Get dataset info
        dataset_info = client.get_dataset(dataset)
        if not dataset_info:
            console.print(f"[red]Error:[/red] Dataset '{dataset}' not found")
            raise typer.Exit(1)

        if not dataset_info.time_column:
            console.print(f"[red]Error:[/red] Dataset '{dataset}' has no time column configured")
            raise typer.Exit(1)

        # Load data
        dfs = client.load_dataset_files(dataset)
        train_df = dfs.get('train', dfs.get('data', None))
        if train_df is None:
            console.print(f"[red]Error:[/red] No train data found in dataset '{dataset}'")
            raise typer.Exit(1)

        # Create folds
        from mdm.utils.time_series import TimeSeriesSplitter
        splitter = TimeSeriesSplitter(dataset_info.time_column, dataset_info.group_column)
        folds = splitter.split_by_folds(train_df, n_folds, gap_days)

        # Display fold information
        console.print("\n[bold]Time Series Cross-Validation Folds:[/bold]")

        table = Table(title=f"Dataset: {dataset}")
        table.add_column("Fold", style="cyan")
        table.add_column("Train Period", style="green")
        table.add_column("Train Rows", justify="right")
        table.add_column("Test Period", style="yellow")
        table.add_column("Test Rows", justify="right")

        for fold_info in folds:
            train_start, train_end = fold_info['train_period']
            test_start, test_end = fold_info['test_period']

            table.add_row(
                str(fold_info['fold']),
                f"{train_start.date()} → {train_end.date()}",
                f"{len(fold_info['train']):,}",
                f"{test_start.date()} → {test_end.date()}",
                f"{len(fold_info['test']):,}"
            )

        console.print(table)

        if gap_days > 0:
            console.print(f"\n[yellow]Note:[/yellow] {gap_days}-day gap between train and test sets")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None
