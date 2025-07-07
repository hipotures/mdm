"""Batch operations CLI commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from mdm.dataset.manager import DatasetManager
from mdm.dataset.operations import ExportOperation

# Create batch app
batch_app = typer.Typer(help="Batch operations for multiple datasets")
console = Console()


@batch_app.command("export")
def batch_export(
    dataset_names: list[str] = typer.Argument(..., help="Dataset names to export"),
    output_dir: Path = typer.Option(Path("exports"), "--output-dir", "-o", help="Output directory"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format (csv, parquet, json)"),
    compression: Optional[str] = typer.Option(None, "--compression", "-c", help="Compression type"),
    metadata_only: bool = typer.Option(False, "--metadata-only", help="Export only metadata"),
):
    """Export multiple datasets in batch."""
    manager = DatasetManager()
    export_op = ExportOperation()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track success and failures
    succeeded = []
    failed = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Exporting {len(dataset_names)} datasets...",
            total=len(dataset_names)
        )

        for dataset_name in dataset_names:
            try:
                # Check if dataset exists
                if not manager.dataset_exists(dataset_name):
                    console.print(f"[yellow]Warning:[/yellow] Dataset '{dataset_name}' not found, skipping")
                    failed.append((dataset_name, "Dataset not found"))
                    progress.advance(task)
                    continue

                # Create dataset-specific output directory
                dataset_output_dir = output_dir / dataset_name
                dataset_output_dir.mkdir(parents=True, exist_ok=True)

                # Export dataset
                progress.update(task, description=f"Exporting {dataset_name}...")

                exported_files = export_op.execute(
                    name=dataset_name,
                    format=format,
                    output_dir=dataset_output_dir,
                    compression=compression,
                    metadata_only=metadata_only,
                )

                succeeded.append((dataset_name, len(exported_files)))
                progress.advance(task)

            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to export '{dataset_name}': {e}")
                failed.append((dataset_name, str(e)))
                progress.advance(task)

    # Summary
    console.print("\n[bold]Export Summary:[/bold]")
    console.print(f"  Successfully exported: {len(succeeded)} datasets")
    console.print(f"  Failed: {len(failed)} datasets")

    if succeeded:
        console.print("\n[green]Successful exports:[/green]")
        for name, file_count in succeeded:
            console.print(f"  ✓ {name} ({file_count} files)")

    if failed:
        console.print("\n[red]Failed exports:[/red]")
        for name, error in failed:
            console.print(f"  ✗ {name}: {error}")

    console.print(f"\n[dim]Output directory: {output_dir.absolute()}[/dim]")


@batch_app.command("stats")
def batch_stats(
    dataset_names: list[str] = typer.Argument(..., help="Dataset names to analyze"),
    full: bool = typer.Option(False, "--full", help="Compute full statistics"),
    export: Optional[Path] = typer.Option(None, "--export", help="Export stats to directory"),
):
    """Compute statistics for multiple datasets."""
    manager = DatasetManager()
    from mdm.dataset.operations import StatsOperation

    stats_op = StatsOperation()

    # Track results
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Computing stats for {len(dataset_names)} datasets...",
            total=len(dataset_names)
        )

        for dataset_name in dataset_names:
            try:
                # Check if dataset exists
                if not manager.dataset_exists(dataset_name):
                    console.print(f"[yellow]Warning:[/yellow] Dataset '{dataset_name}' not found, skipping")
                    progress.advance(task)
                    continue

                # Compute stats
                progress.update(task, description=f"Analyzing {dataset_name}...")

                # Export path if specified
                export_path = None
                if export:
                    export.mkdir(parents=True, exist_ok=True)
                    export_path = export / f"{dataset_name}_stats.json"

                stats = stats_op.execute(
                    name=dataset_name,
                    full=full,
                    export=export_path,
                )

                results.append((dataset_name, stats))
                progress.advance(task)

            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to compute stats for '{dataset_name}': {e}")
                progress.advance(task)

    # Display summary
    console.print("\n[bold]Statistics Summary:[/bold]")

    for dataset_name, stats in results:
        summary = stats.get('summary', {})
        console.print(f"\n[cyan]{dataset_name}:[/cyan]")
        console.print(f"  Total rows: {summary.get('total_rows', 0):,}")
        console.print(f"  Total columns: {summary.get('total_columns', 0)}")
        console.print(f"  Tables: {summary.get('total_tables', 0)}")
        console.print(f"  Completeness: {summary.get('overall_completeness', 1.0):.1%}")

    if export:
        console.print(f"\n[dim]Stats exported to: {export.absolute()}[/dim]")


@batch_app.command("remove")
def batch_remove(
    dataset_names: list[str] = typer.Argument(..., help="Dataset names to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be deleted"),
):
    """Remove multiple datasets in batch."""
    manager = DatasetManager()
    from mdm.dataset.operations import RemoveOperation

    remove_op = RemoveOperation()

    # Get info about datasets to remove
    datasets_to_remove = []
    total_size = 0

    for dataset_name in dataset_names:
        if manager.dataset_exists(dataset_name):
            try:
                info = remove_op.execute(dataset_name, force=True, dry_run=True)
                datasets_to_remove.append(info)
                total_size += info.get('size', 0)
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Cannot get info for '{dataset_name}': {e}")

    if not datasets_to_remove:
        console.print("[yellow]No valid datasets to remove.[/yellow]")
        return

    # Show what will be removed
    console.print(f"\n[bold]Datasets to remove: {len(datasets_to_remove)}[/bold]")
    for info in datasets_to_remove:
        console.print(f"  - {info['name']} ({_format_size(info.get('size', 0))})")
    console.print(f"\n[bold]Total size:[/bold] {_format_size(total_size)}")

    if dry_run:
        console.print("\n[yellow]DRY RUN - No changes made[/yellow]")
        return

    # Confirm if not forced
    if not force:
        confirm = typer.confirm("\nAre you sure you want to remove these datasets?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            return

    # Remove datasets
    removed = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Removing {len(datasets_to_remove)} datasets...",
            total=len(datasets_to_remove)
        )

        for info in datasets_to_remove:
            try:
                progress.update(task, description=f"Removing {info['name']}...")
                remove_op.execute(info['name'], force=True, dry_run=False)
                removed += 1
                progress.advance(task)
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to remove '{info['name']}': {e}")
                failed += 1
                progress.advance(task)

    # Summary
    console.print(f"\n[green]✓[/green] Removed {removed} datasets")
    if failed > 0:
        console.print(f"[red]✗[/red] Failed to remove {failed} datasets")


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
