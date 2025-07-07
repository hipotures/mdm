"""Main CLI entry point for MDM."""

import os
import shutil
from pathlib import Path

import typer
from rich.console import Console

from mdm.cli.batch import batch_app
from mdm.cli.dataset import dataset_app
from mdm.cli.timeseries import app as timeseries_app
from mdm.config import get_config
from mdm.dataset.manager import DatasetManager

# Create main app
app = typer.Typer(
    name="mdm",
    help="ML Data Manager - Streamline your ML data pipeline",
    pretty_exceptions_enable=False,
)

# Add subcommands
app.add_typer(dataset_app, name="dataset", help="Dataset management commands")
app.add_typer(batch_app, name="batch", help="Batch operations for multiple datasets")
app.add_typer(timeseries_app, name="timeseries", help="Time series operations")

# Create console for output
console = Console()


@app.command()
def version():
    """Show MDM version."""
    console.print("[bold green]MDM[/bold green] version 0.1.0")


@app.command()
def info():
    """Display system configuration and status."""
    config = get_config()
    manager = DatasetManager()

    # Header
    console.print("\n[bold cyan]ML Data Manager[/bold cyan] v0.1.0\n")

    # Configuration
    console.print("[bold]Configuration:[/bold]")
    config_file = Path.home() / ".mdm" / "mdm.yaml"
    console.print(f"  Config file: {config_file}")
    console.print(f"  Default backend: {config.default_backend}")

    # Storage paths
    console.print("\n[bold]Storage paths:[/bold]")
    console.print(f"  Datasets: {config.datasets_dir}")
    console.print(f"  Configs: {config.dataset_registry_dir}")
    console.print(f"  Cache: {config.cache_dir}")
    console.print(f"  Logs: {config.logs_dir}")

    # Database settings
    console.print("\n[bold]Database settings:[/bold]")
    console.print(f"  Backend: {config.default_backend}")
    console.print(f"  Chunk size: {config.chunk_size:,}")
    console.print(f"  Max workers: {config.max_workers}")

    # System status
    console.print("\n[bold]System status:[/bold]")

    # Count datasets
    datasets = manager.list_datasets()
    console.print(f"  Registered datasets: {len(datasets)}")

    # Calculate storage used
    total_size = 0
    if config.datasets_dir.exists():
        for item in config.datasets_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size

    # Get available disk space
    stat = shutil.disk_usage(config.datasets_dir)

    console.print(f"  Total storage used: {_format_size(total_size)}")
    console.print(f"  Available disk space: {_format_size(stat.free)}")

    # Environment
    console.print("\n[bold]Environment:[/bold]")
    console.print(f"  Python: {os.sys.version.split()[0]}")
    console.print(f"  Platform: {os.sys.platform}")
    console.print(f"  MDM home: {config.home_dir}")


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
