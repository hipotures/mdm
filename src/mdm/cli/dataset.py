"""Dataset management CLI commands."""

from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from mdm.core.exceptions import DatasetError
from mdm.dataset.operations import (
    InfoOperation,
    ListOperation,
    RemoveOperation,
)
from mdm.dataset.registrar import DatasetRegistrar
from mdm.storage.factory import BackendFactory

# Create dataset app
dataset_app = typer.Typer(help="Dataset management commands")
console = Console()


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def _display_column_summary(dataset_info: Any, manager: Any, table_type: str = "train") -> None:
    """Display a summary table of columns with types and null percentages."""
    backend = None
    try:
        # Get backend to query the data
        backend = BackendFactory.create(dataset_info.database["backend"], dataset_info.database)

        # Get database path
        if "path" in dataset_info.database:
            db_path = dataset_info.database["path"]
        else:
            db_path = f"{dataset_info.database['backend']}://{dataset_info.database['user']}:{dataset_info.database['password']}@{dataset_info.database['host']}:{dataset_info.database['port']}/{dataset_info.database['database']}"

        engine = backend.get_engine(db_path)

        # Get table name
        table_name = dataset_info.tables[table_type]

        # Query to get column info with null counts
        query = f"""
        SELECT COUNT(*) as total_rows
        FROM {table_name}
        """

        result = backend.query(query)
        total_rows = int(result.iloc[0]["total_rows"])

        # Get table info for column types
        table_info = backend.get_table_info(table_name, engine)

        # Create summary table
        table_title = f"\n[bold blue]Dataset Summary - {table_type.capitalize()} Table[/bold blue]"
        summary_table = Table(title=table_title)
        summary_table.add_column("Column", style="cyan")
        summary_table.add_column("Type", style="green")
        summary_table.add_column("% Null", style="yellow", justify="right")

        # For each column, get null count (limit to 20 columns)
        columns = table_info["columns"]
        total_columns = len(columns)
        columns_to_show = columns[:20]  # Limit to first 20 columns

        for col_info in columns_to_show:
            col_name = col_info["name"]
            col_type = col_info["type"]

            # Get null count for this column
            null_query = f"""
            SELECT COUNT(*) as null_count
            FROM {table_name}
            WHERE {col_name} IS NULL
            """

            try:
                null_result = backend.query(null_query)
                null_count = int(null_result.iloc[0]["null_count"])
                null_percentage = (null_count / total_rows) * 100 if total_rows > 0 else 0
                null_str = f"{null_percentage:.1f}%"
            except Exception:
                null_str = "?"

            # Format type name
            type_str = str(col_type).split("(")[0].upper()

            summary_table.add_row(col_name, type_str, null_str)

        # Add ... if there are more than 20 columns
        if total_columns > 20:
            summary_table.add_row(
                f"[dim]... ({total_columns - 20} more columns)[/dim]",
                "[dim]...[/dim]",
                "[dim]...[/dim]",
            )

        # Add total row count info
        console.print(summary_table)
        console.print(f"\n[dim]Total rows in {table_type} table: {total_rows:,}[/dim]")

    except Exception as e:
        # Don't fail registration if summary fails
        console.print(f"\n[yellow]Could not generate column summary: {e}[/yellow]")
    finally:
        if hasattr(backend, "close_connections"):
            backend.close_connections()


@dataset_app.command("register")
def register(
    name: str = typer.Argument(..., help="Dataset name"),
    path: Optional[Path] = typer.Argument(None, help="Path to dataset directory"),
    no_auto: bool = typer.Option(False, "--no-auto", help="Disable auto-detection"),
    train: Optional[Path] = typer.Option(None, "--train", help="Path to training file"),
    test: Optional[Path] = typer.Option(None, "--test", help="Path to test file"),
    validation: Optional[Path] = typer.Option(None, "--validation", help="Path to validation file"),
    submission: Optional[Path] = typer.Option(None, "--submission", help="Path to submission file"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name"),
    id_columns: Optional[str] = typer.Option(
        None, "--id-columns", help="Comma-separated ID columns"
    ),
    problem_type: Optional[str] = typer.Option(
        None, "--problem-type", help="Problem type (classification/regression/time_series)"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Dataset description"
    ),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    time_column: Optional[str] = typer.Option(
        None, "--time-column", help="Time column for time series datasets"
    ),
    group_column: Optional[str] = typer.Option(
        None, "--group-column", help="Group column for grouped time series"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-registration"),
    no_features: bool = typer.Option(False, "--no-features", help="Skip feature generation"),
    categorical_columns: Optional[str] = typer.Option(
        None, "--categorical-columns", help="Comma-separated columns to force as categorical"
    ),
    datetime_columns: Optional[str] = typer.Option(
        None, "--datetime-columns", help="Comma-separated columns to force as datetime"
    ),
    numeric_columns: Optional[str] = typer.Option(
        None, "--numeric-columns", help="Comma-separated columns to force as numeric"
    ),
    text_columns: Optional[str] = typer.Option(
        None, "--text-columns", help="Comma-separated columns to force as text"
    ),
):
    """Register a new dataset."""
    # Log registration start
    from loguru import logger
    logger.info(f"Starting registration of dataset '{name}'")
    
    try:
        registrar = DatasetRegistrar()

        # Prepare kwargs
        kwargs = {
            "force": force,
        }

        # Only override generate_features if user explicitly specified --no-features
        if no_features:
            kwargs["generate_features"] = False

        if target:
            kwargs["target_column"] = target
        if id_columns:
            kwargs["id_columns"] = [col.strip() for col in id_columns.split(",")]
        if problem_type:
            kwargs["problem_type"] = problem_type
        if time_column:
            kwargs["time_column"] = time_column
        if group_column:
            kwargs["group_column"] = group_column

        # Build type schema for ydata-profiling
        type_schema = {}
        if categorical_columns:
            for col in categorical_columns.split(","):
                type_schema[col.strip()] = "categorical"
        if datetime_columns:
            for col in datetime_columns.split(","):
                type_schema[col.strip()] = "datetime"
        if numeric_columns:
            for col in numeric_columns.split(","):
                type_schema[col.strip()] = "numeric"
        if text_columns:
            for col in text_columns.split(","):
                type_schema[col.strip()] = "text"

        if type_schema:
            kwargs["type_schema"] = type_schema

        # Handle manual mode
        if no_auto:
            if not train:
                console.print("[red]Error:[/red] --train is required with --no-auto")
                raise typer.Exit(1)
            if not target:
                console.print("[red]Error:[/red] --target is required with --no-auto")
                raise typer.Exit(1)
            if path:
                console.print("[red]Error:[/red] Directory path not allowed with --no-auto")
                raise typer.Exit(1)

            # Build file dictionary
            files = {"train": train}
            if test:
                files["test"] = test
            if validation:
                files["validation"] = validation
            if submission:
                files["submission"] = submission

            # TODO: Implement manual registration
            console.print("[yellow]Manual registration not yet implemented[/yellow]")
            raise typer.Exit(1)
        # Auto-detection mode
        if not path:
            # Use current directory if no path provided
            path = Path("./")

        # Register dataset
        dataset_info = registrar.register(
            name=name,
            path=path,
            auto_detect=True,
            description=description,
            tags=[tag.strip() for tag in tags.split(",")] if tags else None,
            **kwargs,
        )

        # Display success message
        console.print(f"\n[green]✓[/green] Dataset '{dataset_info.name}' registered successfully!")

        # Display dataset info in a nice table
        from rich.panel import Panel
        from rich.table import Table

        config_table = Table(show_header=False, box=None, padding=(0, 1))
        config_table.add_column(style="cyan", no_wrap=True)
        config_table.add_column(style="white")

        config_table.add_row("Target", dataset_info.target_column or "[dim]None[/dim]")
        config_table.add_row("Problem Type", dataset_info.problem_type or "[dim]Unknown[/dim]")
        config_table.add_row(
            "ID Columns",
            ", ".join(dataset_info.id_columns) if dataset_info.id_columns else "[dim]None[/dim]",
        )

        # Format tables nicely
        tables = list(dataset_info.tables.keys())
        if len(tables) <= 3:
            tables_str = ", ".join(tables)
        else:
            tables_str = ", ".join(tables[:3]) + f", [dim]...({len(tables)} total)[/dim]"
        config_table.add_row("Tables", tables_str)

        console.print(
            Panel(config_table, title="[bold blue]Configuration[/bold blue]", expand=False)
        )

        # Display column summary for train or data table
        if "train" in dataset_info.tables:
            _display_column_summary(dataset_info, registrar.manager, "train")
        elif "data" in dataset_info.tables:
            _display_column_summary(dataset_info, registrar.manager, "data")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@dataset_app.command("list")
def list_datasets(
    format: str = typer.Option(
        "rich", "--format", "-f", help="Output format (rich, text, or filename)"
    ),
    filter_str: Optional[str] = typer.Option(
        None, "--filter", help="Filter datasets (e.g., 'problem_type=classification')"
    ),
    sort_by: str = typer.Option("name", "--sort-by", help="Sort field (name, registration_date)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum number of results"),
):
    """List all registered datasets."""
    try:
        list_op = ListOperation()
        datasets = list_op.execute(
            format=format,
            filter_str=filter_str,
            sort_by=sort_by,
            limit=limit,
        )

        if not datasets:
            if format == "json":
                console.print("[]")
            else:
                console.print("[yellow]No datasets registered yet.[/yellow]")
            return

        # Handle JSON format
        if format == "json":
            import json

            console.print(json.dumps(datasets, indent=2))
            return

        # Create table
        table = Table(title="Registered Datasets")
        table.add_column("Name", style="cyan")
        table.add_column("Problem Type", style="green")
        table.add_column("Target")
        table.add_column("Tables")
        table.add_column("Total Rows")
        table.add_column("MEM Size")
        table.add_column("Backend", style="yellow")

        for dataset in datasets:
            # Format row count and size
            row_count = dataset.get("row_count")
            size = dataset.get("size")

            row_str = "?" if row_count is None else f"{row_count:,}"
            size_str = "?" if size is None else _format_size(size)

            # Format backend column
            dataset_backend = dataset.get("backend", "unknown")
            if dataset.get("backend_compatible", True):
                backend_str = dataset_backend
            else:
                # Show incompatible backend with warning
                backend_str = f"[red]Not Supported/{dataset_backend}[/red]"

            table.add_row(
                dataset["name"],
                dataset.get("problem_type") or "-",
                dataset.get("target_column") or "-",
                str(len(dataset.get("tables", {}))),
                row_str,
                size_str,
                backend_str,
            )

        console.print(table)

        # Show warning if there are incompatible datasets
        incompatible_count = sum(1 for d in datasets if not d.get("backend_compatible", True))
        if incompatible_count > 0:
            console.print(
                f"\n[yellow]Warning:[/yellow] {incompatible_count} dataset(s) use a different backend than the current '{datasets[0].get('current_backend', 'unknown')}' backend."
            )
            console.print(
                "[dim]To use these datasets, change the default_backend in ~/.mdm/mdm.yaml[/dim]"
            )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@dataset_app.command("info")
def dataset_info(
    name: str = typer.Argument(..., help="Dataset name"),
    details: bool = typer.Option(False, "--details", "-d", help="Include detailed statistics"),
):
    """Show detailed information about a dataset."""
    try:
        info_op = InfoOperation()
        info = info_op.execute(name, details=details)

        # Display dataset info
        console.print(f"\n[bold cyan]Dataset: {info.get('display_name', info['name'])}[/bold cyan]")
        console.print(f"Configuration: {info.get('file', 'N/A')}")

        if info.get("dataset_path"):
            console.print(
                f"Database: {info.get('database_file', info['dataset_path'])} ({_format_size(info.get('database_size', info.get('total_size', 0)))})"
            )

        if info.get("description"):
            console.print(f"\nDescription: {info['description']}")

        console.print(f"\nProblem Type: {info.get('problem_type') or 'Unknown'}")
        console.print(f"Target Column: {info.get('target_column') or 'None'}")
        console.print(
            f"ID Columns: {', '.join(info.get('id_columns', [])) if info.get('id_columns') else 'None'}"
        )

        if info.get("tags"):
            console.print(f"Tags: {', '.join(info['tags'])}")

        console.print(f"\nSource: {info.get('source', 'Unknown')}")
        console.print(f"Backend: {info.get('database', {}).get('backend', 'Unknown')}")

        # Display tables
        console.print("\n[bold]Tables:[/bold]")
        for table_type, table_name in info.get("tables", {}).items():
            console.print(f"  - {table_type}: {table_name}")

        # Display metadata
        if info.get("created_at"):
            console.print(f"\nRegistered: {info['created_at']}")
        if info.get("last_updated_at"):
            console.print(f"Last Modified: {info['last_updated_at']}")

        # Display statistics if available
        if details and info.get("statistics"):
            console.print("\n[bold]Statistics:[/bold]")
            console.print(info["statistics"]["note"])

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@dataset_app.command("search")
def search_datasets(
    query: Optional[str] = typer.Argument(None, help="Search query (optional when using --tag)"),
    deep: bool = typer.Option(False, "--deep", help="Search in dataset metadata (slower)"),
    pattern: bool = typer.Option(False, "--pattern", help="Use glob pattern matching"),
    case_sensitive: bool = typer.Option(False, "--case-sensitive", help="Case sensitive search"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Search for datasets with specific tag"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum results"),
) -> None:
    """Search for datasets."""
    try:
        from mdm.dataset.operations import SearchOperation

        # If no query and no tag, show error
        if not query and not tag:
            console.print("[red]Error:[/red] Either a search query or --tag must be provided")
            raise typer.Exit(1)

        # Use tag as query if no query provided
        if not query and tag:
            query = tag

        operation = SearchOperation()
        results = operation.execute(
            query=query,
            deep=deep,
            pattern=pattern,
            case_sensitive=case_sensitive,
            tag=tag,
            limit=limit,
        )

        if not results:
            if tag:
                console.print(f"[yellow]No datasets found with tag '{tag}'[/yellow]")
            else:
                console.print(f"[yellow]No datasets found matching '{query}'[/yellow]")
            return

        # Display results
        table_title = f"Datasets with tag '{tag}'" if tag else f"Search Results for '{query}'"
        table = Table(title=table_title)
        table.add_column("Name", style="cyan")
        table.add_column("Description")
        table.add_column("Tags", style="yellow")
        table.add_column("Match Location", style="green")

        for result in results:
            tags_str = ", ".join(result.get("tags", [])) if result.get("tags") else "-"
            table.add_row(
                result["name"],
                (
                    result.get("description", "-")[:50] + "..."
                    if len(result.get("description", "")) > 50
                    else result.get("description", "-")
                ),
                tags_str,
                result.get("match_location", "name"),
            )

        console.print(table)
        console.print(f"\n[green]Found {len(results)} match(es)[/green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@dataset_app.command("stats")
def show_statistics(
    name: str = typer.Argument(..., help="Dataset name"),
    full: bool = typer.Option(False, "--full", help="Show full statistics including correlations"),
    export: Optional[Path] = typer.Option(None, "--export", help="Export statistics to file"),
) -> None:
    """Display dataset statistics."""
    try:
        from mdm.dataset.operations import StatsOperation

        operation = StatsOperation()
        stats = operation.execute(name, full=full)

        if export:
            operation.export_stats(stats, export)
            console.print(f"[green]✓[/green] Statistics exported to {export}")
        else:
            # Display statistics
            console.print(f"\n[bold cyan]Statistics for dataset: {name}[/bold cyan]")
            console.print(f"Computed at: {stats['computed_at']}")
            console.print(f"Mode: {stats['mode']}\n")

            # Summary info
            summary = stats.get("summary", {})
            console.print("[bold]Summary:[/bold]")
            console.print(f"- Total tables: {summary.get('total_tables', 0)}")
            console.print(f"- Total rows: {summary.get('total_rows', 0):,}")
            console.print(f"- Total columns: {summary.get('total_columns', 0)}")
            console.print(
                f"- Overall completeness: {summary.get('overall_completeness', 0) * 100:.1f}%"
            )

            # Table statistics
            if stats.get("tables"):
                console.print("\n[bold]Table Statistics:[/bold]")
                for table_name, table_stats in stats["tables"].items():
                    if table_stats:
                        console.print(f"\n  Table: {table_name}")
                        console.print(f"  - Rows: {table_stats.get('row_count', 0):,}")
                        console.print(f"  - Columns: {table_stats.get('column_count', 0)}")
                        if "missing_values" in table_stats:
                            missing = table_stats["missing_values"]
                            console.print(
                                f"  - Missing cells: {missing.get('total_missing_cells', 0):,}"
                            )
                            completeness = 100 - missing.get("total_missing_percentage", 0)
                            console.print(f"  - Completeness: {completeness:.1f}%")

            if full:
                console.print("\n[bold]Detailed Statistics:[/bold]")
                # Show detailed stats for each table
                for table_name, table_stats in stats.get("tables", {}).items():
                    if table_stats and "columns" in table_stats:
                        console.print(f"\n  [bold]{table_name}[/bold] columns:")
                        for col_name, col_stats in table_stats["columns"].items():
                            if col_stats:
                                console.print(f"    {col_name}:")
                                console.print(f"      - Type: {col_stats.get('dtype', 'unknown')}")
                                console.print(f"      - Non-null: {col_stats.get('non_null', 0):,}")
                                console.print(f"      - Unique: {col_stats.get('unique', 0):,}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@dataset_app.command("update")
def update_dataset(
    name: str = typer.Argument(..., help="Dataset name"),
    description: Optional[str] = typer.Option(None, "--description", help="New description"),
    target: Optional[str] = typer.Option(None, "--target", help="New target column"),
    problem_type: Optional[str] = typer.Option(None, "--problem-type", help="New problem type"),
    id_columns: Optional[str] = typer.Option(
        None, "--id-columns", help="Comma-separated ID columns"
    ),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
) -> None:
    """Update dataset metadata."""
    from mdm.dataset.operations import UpdateOperation

    # Build updates dict
    updates = {}
    if description is not None:
        updates["description"] = description
    if target is not None:
        updates["target_column"] = target
    if problem_type is not None:
        # Validate problem type
        valid_problem_types = [
            "binary_classification",
            "multiclass_classification",
            "regression",
            "time_series",
            "clustering",
        ]
        if problem_type not in valid_problem_types:
            console.print(
                f"[red]Error:[/red] Invalid problem type '{problem_type}'. Valid options are: {', '.join(valid_problem_types)}"
            )
            raise typer.Exit(1)
        updates["problem_type"] = problem_type
    if id_columns is not None:
        # Validate and parse id_columns
        columns = [col.strip() for col in id_columns.split(",") if col.strip()]
        if not columns and id_columns.strip():  # Non-empty string but no valid columns
            console.print(
                "[red]Error:[/red] Invalid id_columns format. Please provide comma-separated column names."
            )
            raise typer.Exit(1)
        if columns:  # Only add if there are valid columns
            updates["id_columns"] = columns
    if tags is not None:
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        updates["tags"] = tag_list

    if not updates:
        console.print("No updates specified")
        return

    try:
        operation = UpdateOperation()
        # Unpack updates dictionary to match execute method signature
        result = operation.execute(
            name,
            description=updates.get("description"),
            target=updates.get("target_column"),
            problem_type=updates.get("problem_type"),
            id_columns=updates.get("id_columns"),
            tags=updates.get("tags")
        )

        console.print(f"[green]✓[/green] Dataset '{name}' updated successfully")
        console.print("\nUpdated fields:")
        for field, value in updates.items():
            console.print(f"  {field}: {value}")

    except DatasetError as e:
        # Dataset-specific errors are safe to show
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None
    except Exception as e:
        # Generic errors - log the actual error and show generic message
        import logging

        logging.debug(f"Update dataset error: {e}", exc_info=True)
        console.print("[red]Error:[/red] Failed to update dataset. Check logs for details.")
        raise typer.Exit(1) from None


@dataset_app.command("export")
def export_dataset(
    name: str = typer.Argument(..., help="Dataset name"),
    output_dir: Path = typer.Option(Path(), "--output-dir", "-o", help="Output directory"),
    table: Optional[str] = typer.Option(None, "--table", help="Export specific table only"),
    format: Optional[str] = typer.Option(
        None, "--format", "-f", help="Output format: csv, parquet, json (default: from config)"
    ),
    compression: Optional[str] = typer.Option(None, "--compression", help="Compression type"),
    metadata_only: bool = typer.Option(False, "--metadata-only", help="Export only metadata"),
    no_header: bool = typer.Option(False, "--no-header", help="Exclude header row (CSV only)"),
) -> None:
    """Export dataset to files."""
    try:
        from mdm.dataset.operations import ExportOperation

        operation = ExportOperation()
        exported_files = operation.execute(
            name=name,
            format=format,
            output_dir=output_dir,
            table=table,
            compression=compression,
            metadata_only=metadata_only,
            no_header=no_header,
        )

        console.print(f"[green]✓[/green] Dataset '{name}' exported successfully")
        console.print("\nExported files:")
        for file_path in exported_files:
            console.print(f"  - {file_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@dataset_app.command("remove")
def remove_dataset(
    name: str = typer.Argument(..., help="Dataset name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be deleted"),
):
    """Remove a registered dataset."""
    try:
        remove_op = RemoveOperation()

        if dry_run:
            # Preview mode
            info = remove_op.execute(name, force=True, dry_run=True)
            console.print("\n[yellow]DRY RUN - No changes will be made[/yellow]")
            console.print(f"Would remove dataset: {info['name']}")
            console.print(f"- Config: {info['config_file']}")
            if info.get("dataset_directory"):
                console.print(
                    f"- Database: {info['dataset_directory']} ({_format_size(info['size'])})"
                )
            if info.get("postgresql_db"):
                console.print(f"- PostgreSQL database: {info['postgresql_db']}")
            return

        # Get removal info for confirmation
        info = remove_op.execute(name, force=True, dry_run=True)

        if not force:
            console.print(f"\nRemoving dataset: {info['name']}")
            console.print(f"- Config: {info['config_file']}")
            if info.get("dataset_directory"):
                console.print(
                    f"- Database: {info['dataset_directory']} ({_format_size(info['size'])})"
                )

            confirm = typer.confirm("\nAre you sure?")
            if not confirm:
                console.print("[yellow]Cancelled.[/yellow]")
                return

        # Actually remove
        remove_op.execute(name, force=True, dry_run=False)
        console.print(f"[green]✓[/green] Dataset '{name}' removed successfully")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None
