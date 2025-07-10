"""
Dataset management CLI commands using dependency injection.

This is an updated version that uses the DI container for better
testability and flexibility.
"""
from pathlib import Path
from typing import Any, Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mdm.core import inject, get_service, configure_container
from mdm.core.exceptions import DatasetError, MDMError
from mdm.interfaces import IDatasetRegistrar, IDatasetManager, IStorageBackend
from mdm.config import get_config

# Create dataset app
dataset_app = typer.Typer(help="Dataset management commands")
console = Console()

# Initialize DI container on module load
_config = get_config()
configure_container(_config.model_dump())


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


@dataset_app.command()
@inject
def register(
    name: str = typer.Argument(..., help="Dataset name"),
    path: Optional[Path] = typer.Argument(None, help="Path to dataset directory"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name"),
    id_columns: Optional[str] = typer.Option(
        None, "--id-columns", help="Comma-separated ID columns"
    ),
    problem_type: Optional[str] = typer.Option(
        None, "--problem-type", help="Problem type (classification/regression/etc)"
    ),
    datetime_columns: Optional[str] = typer.Option(
        None, "--datetime-columns", help="Comma-separated datetime columns"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Dataset description"
    ),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing dataset"),
    registrar: IDatasetRegistrar = None  # Will be injected
):
    """Register a new dataset using dependency injection."""
    try:
        # Parse list arguments
        id_cols = [c.strip() for c in id_columns.split(",")] if id_columns else None
        datetime_cols = [c.strip() for c in datetime_columns.split(",")] if datetime_columns else None
        
        # Register dataset
        result = registrar.register(
            name=name,
            path=str(path) if path else ".",
            target=target,
            problem_type=problem_type,
            id_columns=id_cols,
            datetime_columns=datetime_cols,
            force=force
        )
        
        # Display success message
        console.print(f"\n[green]✓ Dataset '{name}' registered successfully![/green]")
        
        # Display configuration panel
        config_items = []
        if result.get("target"):
            config_items.append(f"Target        {result['target']}")
        if result.get("problem_type"):
            config_items.append(f"Problem Type  {result['problem_type']}")
        if result.get("id_columns"):
            config_items.append(f"ID Columns    {', '.join(result['id_columns'])}")
        if result.get("tables"):
            config_items.append(f"Tables        {', '.join(result['tables'].keys())}")
        
        if config_items:
            config_panel = Panel(
                "\n".join(config_items),
                title="Configuration",
                border_style="blue"
            )
            console.print(config_panel)
        
        # Update metadata if provided
        if description or tags:
            manager = get_service(IDatasetManager)
            updates = {}
            if description:
                updates["description"] = description
            if tags:
                updates["tags"] = [t.strip() for t in tags.split(",")]
            manager.update_dataset(name, **updates)
        
    except DatasetError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@dataset_app.command()
@inject
def list(
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Maximum number of datasets"),
    sort_by: Optional[str] = typer.Option(
        None, "--sort-by", help="Sort by: name, registration_date, size"
    ),
    backend: Optional[str] = typer.Option(None, "--backend", help="Filter by backend type"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by tag"),
    manager: IDatasetManager = None  # Will be injected
):
    """List registered datasets using dependency injection."""
    try:
        # Get datasets
        datasets = manager.list_datasets(
            limit=limit,
            sort_by=sort_by,
            backend=backend,
            tag=tag
        )
        
        if not datasets:
            console.print("[yellow]No datasets found.[/yellow]")
            return
        
        # Create table
        table = Table(title="Registered Datasets", show_lines=True)
        table.add_column("Name", style="cyan")
        table.add_column("Problem Type", style="green")
        table.add_column("Target", style="yellow")
        table.add_column("Tables", style="blue")
        table.add_column("Total Rows", style="magenta", justify="right")
        table.add_column("MEM Size", style="red", justify="right")
        table.add_column("Backend", style="white")
        
        # Add rows
        for ds in datasets:
            table.add_row(
                ds.get("name", ""),
                ds.get("problem_type", "-"),
                ds.get("target", "-"),
                str(ds.get("table_count", "?")),
                f"{ds.get('total_rows', '?'):,}" if isinstance(ds.get('total_rows'), int) else "?",
                _format_size(ds.get("size", 0)) if ds.get("size") else "?",
                ds.get("backend", "unknown")
            )
        
        console.print(table)
        
        # Show warning for different backend datasets
        current_backend = _config.database.default_backend
        different_backend = [d for d in datasets if d.get("backend") != current_backend]
        if different_backend:
            console.print(
                f"\n[yellow]Warning: {len(different_backend)} dataset(s) use a different "
                f"backend than the current '{current_backend}' backend.[/yellow]"
            )
            console.print(
                "[yellow]To use these datasets, change the default_backend in ~/.mdm/mdm.yaml[/yellow]"
            )
        
    except Exception as e:
        console.print(f"[red]Error listing datasets: {e}[/red]")
        raise typer.Exit(1)


@dataset_app.command()
@inject
def info(
    name: str = typer.Argument(..., help="Dataset name"),
    manager: IDatasetManager = None  # Will be injected
):
    """Show detailed dataset information using dependency injection."""
    try:
        # Get dataset info
        info = manager.get_dataset_info(name)
        
        # Display basic info
        console.print(f"\n[bold]Dataset: {name}[/bold]")
        console.print(f"Configuration: {info.get('config_path', 'N/A')}")
        console.print(f"Database: {info.get('database_path', 'N/A')} ({_format_size(info.get('database_size', 0))})")
        console.print()
        
        # Display configuration
        console.print(f"Problem Type: {info.get('problem_type', 'N/A')}")
        console.print(f"Target Column: {info.get('target', 'N/A')}")
        console.print(f"ID Columns: {', '.join(info.get('id_columns', [])) or 'N/A'}")
        console.print()
        
        console.print(f"Source: {info.get('source', 'N/A')}")
        console.print(f"Backend: {info.get('backend', 'N/A')}")
        console.print()
        
        # Display tables
        tables = info.get("tables", {})
        if tables:
            console.print("Tables:")
            for table_name, table_info in tables.items():
                console.print(f"  - {table_name}: {table_info}")
        
        # Display description
        if info.get("description"):
            console.print(f"\nDescription: {info['description']}")
        
        # Display tags
        if info.get("tags"):
            console.print(f"Tags: {', '.join(info['tags'])}")
        
    except DatasetError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@dataset_app.command()
@inject
def stats(
    name: str = typer.Argument(..., help="Dataset name"),
    mode: str = typer.Option("basic", "--mode", "-m", help="Stats mode: basic, detailed"),
    tables: Optional[str] = typer.Option(None, "--tables", help="Comma-separated table names"),
    manager: IDatasetManager = None  # Will be injected
):
    """Show dataset statistics using dependency injection."""
    try:
        # Parse tables
        table_list = [t.strip() for t in tables.split(",")] if tables else None
        
        # Get statistics
        stats = manager.get_dataset_stats(name, mode=mode, tables=table_list)
        
        # Display header
        console.print(f"\n[bold]Statistics for dataset: {name}[/bold]")
        console.print(f"Computed at: {stats.get('computed_at', 'N/A')}")
        console.print(f"Mode: {stats.get('mode', 'basic')}")
        console.print()
        
        # Display summary
        summary = stats.get("summary", {})
        if summary:
            console.print("Summary:")
            console.print(f"- Total tables: {summary.get('total_tables', 0)}")
            console.print(f"- Total rows: {summary.get('total_rows', 0):,}")
            console.print(f"- Total columns: {summary.get('total_columns', 0)}")
            console.print(f"- Overall completeness: {summary.get('completeness', 0):.1f}%")
            console.print()
        
        # Display table statistics
        table_stats = stats.get("tables", {})
        if table_stats:
            console.print("Table Statistics:")
            for table_name, table_info in table_stats.items():
                console.print(f"\n  Table: {table_name}")
                console.print(f"  - Rows: {table_info.get('rows', 0):,}")
                console.print(f"  - Columns: {table_info.get('columns', 0)}")
                console.print(f"  - Missing cells: {table_info.get('missing_cells', 0):,}")
                console.print(f"  - Completeness: {table_info.get('completeness', 0):.1f}%")
        
    except DatasetError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@dataset_app.command()
@inject
def remove(
    name: str = typer.Argument(..., help="Dataset name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    manager: IDatasetManager = None  # Will be injected
):
    """Remove a dataset using dependency injection."""
    try:
        # Confirm removal
        if not force:
            confirm = typer.confirm(f"Are you sure you want to remove dataset '{name}'?")
            if not confirm:
                console.print("[yellow]Operation cancelled.[/yellow]")
                return
        
        # Remove dataset
        manager.remove_dataset(name, force=True)
        console.print(f"[green]✓ Dataset '{name}' removed successfully[/green]")
        
    except DatasetError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@dataset_app.command()
@inject
def update(
    name: str = typer.Argument(..., help="Dataset name"),
    description: Optional[str] = typer.Option(None, "--description", help="New description"),
    target: Optional[str] = typer.Option(None, "--target", help="New target column"),
    problem_type: Optional[str] = typer.Option(None, "--problem-type", help="New problem type"),
    id_columns: Optional[str] = typer.Option(None, "--id-columns", help="Comma-separated ID columns"),
    manager: IDatasetManager = None  # Will be injected
):
    """Update dataset metadata using dependency injection."""
    try:
        # Build updates
        updates = {}
        if description is not None:
            updates["description"] = description
        if target is not None:
            updates["target"] = target
        if problem_type is not None:
            updates["problem_type"] = problem_type
        if id_columns is not None:
            updates["id_columns"] = [c.strip() for c in id_columns.split(",")]
        
        if not updates:
            console.print("[yellow]No updates specified.[/yellow]")
            return
        
        # Update dataset
        result = manager.update_dataset(name, **updates)
        
        console.print(f"[green]✓ Dataset '{name}' updated successfully[/green]")
        
        if result.get("updated_fields"):
            console.print("\nUpdated fields:")
            for field, value in result["updated_fields"].items():
                console.print(f"  {field}: {value}")
        
    except DatasetError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@dataset_app.command()
@inject
def export(
    name: str = typer.Argument(..., help="Dataset name"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format: csv, parquet, json"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    compression: Optional[str] = typer.Option(None, "--compression", help="Compression: gzip, zip"),
    tables: Optional[str] = typer.Option(None, "--tables", help="Comma-separated table names"),
    manager: IDatasetManager = None  # Will be injected
):
    """Export dataset to files using dependency injection."""
    try:
        # Parse tables
        table_list = [t.strip() for t in tables.split(",")] if tables else None
        
        # Set default output directory
        if output_dir is None:
            output_dir = Path.cwd()
        
        # Export dataset
        exported_files = manager.export_dataset(
            name=name,
            output_dir=str(output_dir),
            format=format,
            compression=compression,
            tables=table_list
        )
        
        console.print(f"[green]✓ Dataset '{name}' exported successfully[/green]")
        console.print("\nExported files:")
        for file_path in exported_files:
            console.print(f"  - {file_path}")
        
    except DatasetError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@dataset_app.command()
@inject
def search(
    pattern: str = typer.Argument(..., help="Search pattern"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Maximum results"),
    in_fields: Optional[str] = typer.Option(
        "name,description,tags", "--in", help="Fields to search in"
    ),
    manager: IDatasetManager = None  # Will be injected
):
    """Search datasets by pattern using dependency injection."""
    try:
        # Parse fields
        fields = [f.strip() for f in in_fields.split(",")]
        
        # Search datasets
        results = manager.search_datasets(
            pattern=pattern,
            search_in=fields,
            limit=limit
        )
        
        if not results:
            console.print(f"[yellow]No datasets found matching '{pattern}'[/yellow]")
            return
        
        # Display results
        console.print(f"\n[bold]Found {len(results)} dataset(s) matching '{pattern}':[/bold]\n")
        
        for ds in results:
            console.print(f"[cyan]{ds['name']}[/cyan]")
            if ds.get("description"):
                console.print(f"  Description: {ds['description']}")
            if ds.get("tags"):
                console.print(f"  Tags: {', '.join(ds['tags'])}")
            console.print()
        
    except Exception as e:
        console.print(f"[red]Error searching datasets: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    dataset_app()