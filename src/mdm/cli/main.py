"""Main CLI entry point for MDM."""

import os
import shutil
from pathlib import Path

import typer
from rich.console import Console

from mdm.cli.batch import batch_app
from mdm.cli.dataset import dataset_app
from mdm.cli.timeseries import app as timeseries_app
from mdm.config import get_config_manager
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
    config_manager = get_config_manager()
    config = config_manager.config
    base_path = config_manager.base_path
    manager = DatasetManager()

    # Header
    console.print("\n[bold cyan]ML Data Manager[/bold cyan] v0.1.0\n")

    # Configuration
    console.print("[bold]Configuration:[/bold]")
    config_file = Path.home() / ".mdm" / "mdm.yaml"
    console.print(f"  Config file: {config_file}")
    console.print(f"  Default backend: {config.database.default_backend}")

    # Storage paths
    console.print("\n[bold]Storage paths:[/bold]")
    console.print(f"  Datasets: {base_path / config.paths.datasets_path}")
    console.print(f"  Configs: {base_path / config.paths.configs_path}")
    console.print(f"  Cache: {base_path / 'cache'}")
    console.print(f"  Logs: {base_path / config.paths.logs_path}")

    # Database settings
    console.print("\n[bold]Database settings:[/bold]")
    console.print(f"  Backend: {config.database.default_backend}")
    console.print(f"  Chunk size: {config.performance.batch_size:,}")
    console.print(f"  Max workers: {config.performance.max_concurrent_operations}")

    # System status
    console.print("\n[bold]System status:[/bold]")

    # Count datasets
    datasets = manager.list_datasets()
    console.print(f"  Registered datasets: {len(datasets)}")

    # Calculate storage used
    total_size = 0
    datasets_dir = base_path / config.paths.datasets_path
    if datasets_dir.exists():
        for item in datasets_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size

    # Get available disk space
    stat = shutil.disk_usage(datasets_dir)

    console.print(f"  Total storage used: {_format_size(total_size)}")
    console.print(f"  Available disk space: {_format_size(stat.free)}")

    # Environment
    console.print("\n[bold]Environment:[/bold]")
    console.print(f"  Python: {os.sys.version.split()[0]}")
    console.print(f"  Platform: {os.sys.platform}")
    console.print(f"  MDM home: {base_path}")


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main():
    """Main entry point."""
    import sys
    
    # If no arguments provided (just 'mdm'), show help
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    
    app()


if __name__ == "__main__":
    main()
