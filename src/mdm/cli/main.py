"""Main CLI entry point for MDM - Optimized for fast startup."""

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
)

# Flag to track if subcommands have been added
_subcommands_added = False


def ensure_subcommands():
    """Ensure all subcommands are added to the app."""
    global _subcommands_added
    if _subcommands_added:
        return
    _subcommands_added = True
    
    # Import and add all subcommands
    from mdm.cli.dataset import dataset_app
    from mdm.cli.batch import batch_app
    from mdm.cli.timeseries import app as timeseries_app
    from mdm.cli.stats import app as stats_app
    
    app.add_typer(dataset_app, name="dataset", help="Dataset management commands")
    app.add_typer(batch_app, name="batch", help="Batch operations for multiple datasets")
    app.add_typer(timeseries_app, name="timeseries", help="Time series operations")
    app.add_typer(stats_app, name="stats", help="View statistics and monitoring data")


# For testing purposes, always ensure subcommands are available when the module is imported
# This doesn't affect performance in production because main() still uses lazy loading
if 'pytest' in sys.modules or os.environ.get('TESTING'):
    ensure_subcommands()

# Global flag to track if logging has been set up
_logging_initialized = False


def setup_logging():
    """Setup logging configuration for both standard logging and loguru."""
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True
    
    # Import heavy modules only when logging is needed
    import logging
    from loguru import logger
    from mdm.config import get_config_manager
    
    config_manager = get_config_manager()
    config = config_manager.config
    base_path = config_manager.base_path
    
    # Configure DI container
    from mdm.core.di import configure_services
    configure_services(config.model_dump())
    
    # Create logs directory
    logs_dir = base_path / config.paths.logs_path
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Get log file path from config or use default
    if config.logging.file:
        if Path(config.logging.file).is_absolute():
            log_file = Path(config.logging.file)
        else:
            log_file = logs_dir / config.logging.file
    else:
        log_file = logs_dir / "mdm.log"  # Default if no file specified
    
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
        filter=lambda record: "sqlalchemy" not in record["name"]  # Filter out SQLAlchemy by default
    )
    
    # Configure SQLAlchemy logging
    if sqlalchemy_echo and log_level.upper() in ['DEBUG', 'INFO']:
        # Add special console handler for SQLAlchemy
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
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
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
    
    # In debug mode, log full configuration
    if log_level.upper() == "DEBUG":
        logger.debug("MDM Configuration:")
        logger.debug(f"  Base path: {base_path}")
        logger.debug(f"  Config file: {base_path / 'mdm.yaml'}")
        
        # Dynamically log all configuration sections
        config_dict = config.model_dump()
        
        def log_config_section(section_name: str, section_data: dict, indent: int = 2):
            """Recursively log configuration sections."""
            prefix = "  " * indent
            for key, value in section_data.items():
                if isinstance(value, dict):
                    logger.debug(f"{prefix}{key}:")
                    log_config_section(key, value, indent + 1)
                else:
                    logger.debug(f"{prefix}{key}: {value}")
        
        for section, data in config_dict.items():
            if isinstance(data, dict):
                logger.debug(f"  {section}:")
                log_config_section(section, data)
            else:
                logger.debug(f"  {section}: {data}")


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """Setup logging before running any command."""
    # Ensure subcommands are available
    ensure_subcommands()
    
    # Skip setup for simple commands that don't need it
    if ctx.invoked_subcommand in ['version', None]:
        return
    setup_logging()


# Main commands - these are lightweight and don't need lazy loading
@app.command()
def version():
    """Show MDM version."""
    from mdm import __version__
    console.print(f"[bold green]MDM[/bold green] version {__version__}")


@app.command()
def info():
    """Display system configuration and status."""
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
    # Special fast path for version command
    if len(sys.argv) == 2 and sys.argv[1] == 'version':
        from mdm import __version__
        console.print(f"[bold green]MDM[/bold green] version {__version__}")
        sys.exit(0)
    
    # If no arguments provided (just 'mdm'), show help
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    
    # For CLI usage, we can still use lazy loading for performance
    # But ensure_subcommands will be called by the callback anyway
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        # Only load the subcommand we need for optimal performance
        if cmd == 'dataset':
            if not _subcommands_added:
                from mdm.cli.dataset import dataset_app
                app.add_typer(dataset_app, name="dataset", help="Dataset management commands")
        elif cmd == 'batch':
            if not _subcommands_added:
                from mdm.cli.batch import batch_app
                app.add_typer(batch_app, name="batch", help="Batch operations for multiple datasets")
        elif cmd == 'timeseries':
            if not _subcommands_added:
                from mdm.cli.timeseries import app as timeseries_app
                app.add_typer(timeseries_app, name="timeseries", help="Time series operations")
        elif cmd == 'stats':
            if not _subcommands_added:
                from mdm.cli.stats import app as stats_app
                app.add_typer(stats_app, name="stats", help="View statistics and monitoring data")
        elif cmd == '--help' or cmd == '-h':
            # For help, ensure all subcommands are loaded
            ensure_subcommands()
    
    # Now run the app
    app()


if __name__ == "__main__":
    main()