"""Optimized main CLI entry point for MDM with true lazy loading."""

import os
import sys
from pathlib import Path

import typer
from rich.console import Console

# Create console for output
console = Console()

# Create main app
app = typer.Typer(
    name="mdm",
    help="ML Data Manager - Streamline your ML data pipeline",
    pretty_exceptions_enable=False,
    no_args_is_help=True,  # Show help when no args provided
)


def _setup_logging_if_needed(ctx: typer.Context) -> None:
    """Setup logging only when needed - skip for simple commands."""
    # Commands that don't need logging setup
    simple_commands = {'version', 'help', None}
    
    # Skip if simple command or help flag
    if ctx.invoked_subcommand in simple_commands:
        return
    if '--help' in sys.argv or '-h' in sys.argv:
        return
    
    # Setup logging for other commands
    _setup_logging()


def _setup_logging():
    """Setup logging configuration - imported only when needed."""
    import logging
    from loguru import logger
    from mdm.config import get_config_manager
    
    config_manager = get_config_manager()
    config = config_manager.config
    base_path = config_manager.base_path
    
    # Create logs directory
    logs_dir = base_path / config.paths.logs_path
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Get log file path from config or use default
    if Path(config.logging.file).is_absolute():
        log_file = Path(config.logging.file)
    else:
        log_file = logs_dir / config.logging.file
    
    # Get log level from environment or config
    log_level = os.environ.get('MDM_LOGGING_LEVEL', config.logging.level)
    
    # Get SQLAlchemy echo setting
    sqlalchemy_echo = config.database.sqlalchemy.echo
    
    # Remove default loguru handler
    logger.remove()
    
    # Determine format based on config
    log_format = config.logging.format.lower()
    if log_format == "json":
        file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message} | {extra}"
        console_format = file_format
        serialize = True
    else:  # console format
        file_format = "{time:YYYY-MM-DD HH:mm:ss,SSS} - {name} - {level} - {message}"
        console_format = "{time:HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
        serialize = False
    
    # Add file handler with rotation
    logger.add(
        log_file,
        level=log_level.upper(),
        format=file_format,
        rotation=config.logging.max_bytes,
        retention=config.logging.backup_count,
        compression="gz" if config.logging.backup_count > 0 else None,
        serialize=serialize,
        enqueue=True  # Thread-safe
    )
    
    # Add console handler (only warnings and errors for clean output)
    logger.add(
        sys.stderr,
        level="WARNING",
        format=console_format,
        colorize=True,
        filter=lambda record: "sqlalchemy" not in record["name"]
    )
    
    # Configure SQLAlchemy logging
    if sqlalchemy_echo and log_level.upper() in ['DEBUG', 'INFO']:
        logger.add(
            sys.stderr,
            level="INFO",
            format="{time:HH:mm:ss.SSS} | SQL | {message}",
            colorize=True,
            filter=lambda record: "sqlalchemy.engine" in record["name"]
        )
    
    # Intercept standard logging and redirect to loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Set up the interceptor
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Configure specific loggers
    for logger_name in ["mdm", "sqlalchemy.engine", "sqlalchemy.pool"]:
        logging.getLogger(logger_name).handlers = []
        logging.getLogger(logger_name).propagate = True
    
    # Set SQLAlchemy log level if echo is enabled
    if sqlalchemy_echo:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    
    # Log startup
    logger.info(f"MDM logging initialized - Level: {log_level}, File: {log_file}")


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """Main callback - sets up logging when needed."""
    _setup_logging_if_needed(ctx)


# Add subcommands using factory functions to enable lazy loading
@app.command()
def dataset(ctx: typer.Context):
    """Dataset management commands."""
    # Setup logging before importing heavy modules
    _setup_logging()
    
    # Import only when command is actually used
    from mdm.cli.dataset import dataset_app
    
    # Get the remaining args after 'dataset'
    import sys
    args = sys.argv[2:]  # Skip 'mdm' and 'dataset'
    
    # Invoke the dataset app directly
    try:
        dataset_app(args, standalone_mode=False)
    except SystemExit as e:
        ctx.exit(e.code)


@app.command()
def batch(ctx: typer.Context):
    """Batch operations for multiple datasets."""
    _setup_logging()
    
    from mdm.cli.batch import batch_app
    
    import sys
    args = sys.argv[2:]
    
    try:
        batch_app(args, standalone_mode=False)
    except SystemExit as e:
        ctx.exit(e.code)


@app.command()
def timeseries(ctx: typer.Context):
    """Time series operations."""
    _setup_logging()
    
    from mdm.cli.timeseries import app as timeseries_app
    
    import sys
    args = sys.argv[2:]
    
    try:
        timeseries_app(args, standalone_mode=False)
    except SystemExit as e:
        ctx.exit(e.code)


@app.command()
def stats(ctx: typer.Context):
    """View statistics and monitoring data."""
    _setup_logging()
    
    from mdm.cli.stats import app as stats_app
    
    import sys
    args = sys.argv[2:]
    
    try:
        stats_app(args, standalone_mode=False)
    except SystemExit as e:
        ctx.exit(e.code)


@app.command()
def version():
    """Show MDM version."""
    # Only import version when needed - no other imports!
    from mdm import __version__
    console.print(f"[bold green]MDM[/bold green] version {__version__}")


@app.command()
def info():
    """Display system configuration and status."""
    # Setup logging for this command
    _setup_logging()
    
    # Heavy imports only when needed
    import shutil
    from mdm import __version__
    from mdm.config import get_config_manager
    from mdm.dataset.manager import DatasetManager
    
    config_manager = get_config_manager()
    config = config_manager.config
    base_path = config_manager.base_path
    manager = DatasetManager()

    # Header
    console.print(f"\n[bold cyan]ML Data Manager[/bold cyan] v{__version__}\n")

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
    console.print(f"  Python: {sys.version.split()[0]}")
    console.print(f"  Platform: {sys.platform}")
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
    app()


if __name__ == "__main__":
    main()