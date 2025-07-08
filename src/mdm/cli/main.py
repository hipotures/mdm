"""Main CLI entry point for MDM."""

import os
import shutil
import logging
from pathlib import Path

import typer
from rich.console import Console

from mdm.cli.batch import batch_app
from mdm.cli.dataset import dataset_app
from mdm.cli.timeseries import app as timeseries_app
from mdm.config import get_config_manager
from mdm.dataset.manager import DatasetManager


def setup_logging():
    """Setup logging configuration for both standard logging and loguru."""
    # Get configuration
    config_manager = get_config_manager()
    config = config_manager.config
    base_path = config_manager.base_path
    
    # Create logs directory
    logs_dir = base_path / config.paths.logs_path
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Get log file path from config or use default
    # If the file path is absolute, use it as is, otherwise relative to logs_dir
    if Path(config.logging.file).is_absolute():
        log_file = Path(config.logging.file)
    else:
        log_file = logs_dir / config.logging.file
    
    # Get log level from environment or config
    log_level = os.environ.get('MDM_LOGGING_LEVEL', config.logging.level)
    
    # Configure standard logging
    # Remove existing handlers first
    logging.root.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Console handler (only warnings and errors)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Configure root logger
    logging.root.setLevel(getattr(logging, log_level.upper()))
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    # Set specific loggers if needed
    logging.getLogger('mdm').setLevel(getattr(logging, log_level.upper()))
    
    # Configure SQLAlchemy logging based on echo setting
    sqlalchemy_echo = config.database.sqlalchemy.echo
    if sqlalchemy_echo and log_level.upper() in ['DEBUG', 'INFO']:
        # When echo is enabled and log level is DEBUG or INFO, show SQL queries
        # Create a special console handler for SQLAlchemy that shows INFO messages
        sql_console_handler = logging.StreamHandler()
        sql_console_handler.setLevel(logging.INFO)
        sql_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Configure SQLAlchemy loggers
        sql_logger = logging.getLogger('sqlalchemy.engine')
        sql_logger.setLevel(logging.INFO)
        sql_logger.addHandler(sql_console_handler)
        sql_logger.propagate = False  # Don't propagate to root logger
        
        pool_logger = logging.getLogger('sqlalchemy.pool')
        pool_logger.setLevel(logging.INFO)
        pool_logger.addHandler(sql_console_handler)
        pool_logger.propagate = False
    else:
        # When echo is disabled or log level is WARNING or higher, suppress SQLAlchemy logs
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
    
    # Configure loguru for feature modules
    try:
        from loguru import logger as loguru_logger
        # Remove default handler
        loguru_logger.remove()
        # Add file handler with format matching standard logging
        loguru_logger.add(
            log_file,
            level=log_level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss,SSS} - {name} - {level} - {message}",
            rotation=config.logging.max_bytes,  # Size in bytes
            retention=config.logging.backup_count
        )
        # Add console handler (only warnings and errors)
        loguru_logger.add(
            lambda msg: print(msg, end=''),
            level="WARNING",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    except ImportError:
        pass  # loguru not installed
    
    # Log startup
    logger = logging.getLogger(__name__)
    logger.info(f"MDM logging initialized - Level: {log_level}, File: {log_file}")


# Don't setup logging on import - let it be called when needed

# Create main app
app = typer.Typer(
    name="mdm",
    help="ML Data Manager - Streamline your ML data pipeline",
    pretty_exceptions_enable=False,
)


@app.callback()
def main_callback():
    """Setup logging before running any command."""
    setup_logging()

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
