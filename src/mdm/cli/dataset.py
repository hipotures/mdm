"""Dataset management CLI commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from mdm.dataset.manager import DatasetManager
from mdm.dataset.registrar import DatasetRegistrar

# Create dataset app
dataset_app = typer.Typer(help="Dataset management commands")
console = Console()


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
    id_columns: Optional[str] = typer.Option(None, "--id-columns", help="Comma-separated ID columns"),
    problem_type: Optional[str] = typer.Option(None, "--problem-type", help="Problem type"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Dataset description"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-registration"),
    no_features: bool = typer.Option(False, "--no-features", help="Skip feature generation"),
):
    """Register a new dataset."""
    try:
        registrar = DatasetRegistrar()

        # Prepare kwargs
        kwargs = {
            "generate_features": not no_features,
        }

        if target:
            kwargs["target_column"] = target
        if id_columns:
            kwargs["id_columns"] = [col.strip() for col in id_columns.split(",")]
        if problem_type:
            kwargs["problem_type"] = problem_type

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
            console.print("[red]Error:[/red] Path required for auto-detection")
            raise typer.Exit(1)

        # Register dataset
        dataset_info = registrar.register(
            name=name,
            path=path,
            auto_detect=True,
            description=description,
            tags=[tag.strip() for tag in tags.split(",")] if tags else None,
            **kwargs
        )

        # Display success message
        console.print(f"\n[green]✓[/green] Dataset '{dataset_info.name}' registered successfully!")

        # Display dataset info
        console.print("\nConfiguration:")
        console.print(f"- Target: {dataset_info.target_column or 'None'}")
        console.print(f"- Problem Type: {dataset_info.problem_type or 'Unknown'}")
        console.print(f"- ID Columns: {', '.join(dataset_info.id_columns) if dataset_info.id_columns else 'None'}")
        console.print(f"- Tables: {', '.join(dataset_info.tables.keys())}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@dataset_app.command("list")
def list_datasets():
    """List all registered datasets."""
    try:
        manager = DatasetManager()
        datasets = manager.list_datasets()

        if not datasets:
            console.print("[yellow]No datasets registered yet.[/yellow]")
            return

        # Create table
        table = Table(title="Registered Datasets")
        table.add_column("Name", style="cyan")
        table.add_column("Description")
        table.add_column("Problem Type", style="green")
        table.add_column("Target")
        table.add_column("Tables")

        for dataset in datasets:
            table.add_row(
                dataset.name,
                dataset.description[:50] + "..." if len(dataset.description) > 50 else dataset.description,
                dataset.problem_type or "-",
                dataset.target_column or "-",
                str(len(dataset.tables))
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@dataset_app.command("info")
def dataset_info(
    name: str = typer.Argument(..., help="Dataset name")
):
    """Show detailed information about a dataset."""
    try:
        manager = DatasetManager()
        dataset = manager.get_dataset(name)

        if not dataset:
            console.print(f"[red]Error:[/red] Dataset '{name}' not found")
            raise typer.Exit(1)

        # Display dataset info
        console.print(f"\n[bold cyan]Dataset: {dataset.display_name}[/bold cyan]")
        console.print(f"Internal name: {dataset.name}")

        if dataset.description:
            console.print(f"\nDescription: {dataset.description}")

        console.print(f"\nProblem Type: {dataset.problem_type or 'Unknown'}")
        console.print(f"Target Column: {dataset.target_column or 'None'}")
        console.print(f"ID Columns: {', '.join(dataset.id_columns) if dataset.id_columns else 'None'}")

        if dataset.tags:
            console.print(f"Tags: {', '.join(dataset.tags)}")

        console.print(f"\nSource: {dataset.source}")
        console.print(f"Backend: {dataset.database.get('backend', 'Unknown')}")

        # Display tables
        console.print("\n[bold]Tables:[/bold]")
        for table_type, table_name in dataset.tables.items():
            console.print(f"  - {table_type}: {table_name}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@dataset_app.command("remove")
def remove_dataset(
    name: str = typer.Argument(..., help="Dataset name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
):
    """Remove a registered dataset."""
    try:
        manager = DatasetManager()

        if not force:
            confirm = typer.confirm(f"Are you sure you want to remove dataset '{name}'?")
            if not confirm:
                console.print("[yellow]Cancelled.[/yellow]")
                return

        manager.remove_dataset(name)
        console.print(f"[green]✓[/green] Dataset '{name}' removed successfully")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
