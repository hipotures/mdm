"""Main CLI entry point for MDM."""

import typer
from rich.console import Console

from mdm.cli.dataset import dataset_app

# Create main app
app = typer.Typer(
    name="mdm",
    help="ML Data Manager - Streamline your ML data pipeline",
    pretty_exceptions_enable=False,
)

# Add subcommands
app.add_typer(dataset_app, name="dataset", help="Dataset management commands")

# Create console for output
console = Console()


@app.command()
def version():
    """Show MDM version."""
    console.print("[bold green]MDM[/bold green] version 0.1.0")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
