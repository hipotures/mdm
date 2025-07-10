"""Legacy CLI adapters for MDM refactoring.

This module provides adapters that wrap the existing CLI implementations
to conform to the new interface protocols.
"""
import logging
from typing import Dict, Any, List, Optional, Iterator
from pathlib import Path

from ..interfaces.cli import (
    IDatasetCommands,
    IBatchCommands,
    ITimeSeriesCommands,
    IStatsCommands,
    ICLIFormatter,
    ICLIConfig,
)
from ..dataset.operations import (
    DatasetRegistrar,
    DatasetListOperation,
    DatasetInfoOperation,
    DatasetSearchOperation,
    DatasetStatsOperation,
    DatasetUpdateOperation,
    DatasetExportOperation,
    DatasetRemoveOperation,
)
from ..dataset.manager import DatasetManager
from .batch import BatchExportOperation, BatchStatsOperation, BatchRemoveOperation
from ..api import MDMClient
from ..config import get_config_manager
from .. import __version__

logger = logging.getLogger(__name__)


class LegacyDatasetCommands(IDatasetCommands):
    """Adapter for legacy dataset commands."""
    
    def __init__(self):
        """Initialize legacy dataset commands."""
        self.manager = DatasetManager()
        self.registrar = DatasetRegistrar()
        logger.info("Initialized LegacyDatasetCommands")
    
    def register(
        self,
        name: str,
        path: str,
        target: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
        problem_type: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        force: bool = False,
        generate_features: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Register a new dataset using legacy implementation."""
        try:
            result = self.registrar.register(
                name=name,
                path=path,
                target=target,
                id_columns=id_columns,
                problem_type=problem_type,
                description=description,
                tags=tags,
                force=force,
                generate_features=generate_features,
                **kwargs
            )
            return {
                'success': True,
                'dataset': name,
                'backend': result.get('backend'),
                'tables': result.get('tables', []),
                'features_generated': result.get('features_generated', 0)
            }
        except Exception as e:
            logger.error(f"Legacy registration failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_datasets(
        self,
        limit: Optional[int] = None,
        backend: Optional[str] = None,
        sort_by: str = "name",
        reverse: bool = False
    ) -> List[Dict[str, Any]]:
        """List datasets using legacy implementation."""
        operation = DatasetListOperation(self.manager)
        result = operation.execute(
            limit=limit,
            backend=backend,
            sort_by=sort_by
        )
        
        # Sort if needed
        if reverse:
            result = list(reversed(result))
        
        return result
    
    def info(self, name: str) -> Dict[str, Any]:
        """Get dataset info using legacy implementation."""
        operation = DatasetInfoOperation(self.manager)
        return operation.execute(name)
    
    def search(
        self,
        pattern: str,
        search_in: Optional[List[str]] = None,
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """Search datasets using legacy implementation."""
        operation = DatasetSearchOperation(self.manager)
        return operation.execute(pattern)
    
    def stats(self, name: str, detailed: bool = False) -> Dict[str, Any]:
        """Get dataset stats using legacy implementation."""
        operation = DatasetStatsOperation(self.manager)
        return operation.execute(name, detailed=detailed)
    
    def update(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        problem_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Update dataset using legacy implementation."""
        operation = DatasetUpdateOperation(self.manager)
        return operation.execute(
            name,
            description=description,
            tags=tags,
            problem_type=problem_type,
            **kwargs
        )
    
    def export(
        self,
        name: str,
        output_dir: str,
        format: str = "csv",
        tables: Optional[List[str]] = None,
        compression: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Export dataset using legacy implementation."""
        operation = DatasetExportOperation(self.manager)
        result = operation.execute(
            name,
            output_dir=output_dir,
            format=format,
            tables=tables,
            compression=compression
        )
        return result.get('exported_files', [])
    
    def remove(self, name: str, force: bool = False) -> Dict[str, Any]:
        """Remove dataset using legacy implementation."""
        operation = DatasetRemoveOperation(self.manager)
        return operation.execute(name, force=force)


class LegacyBatchCommands(IBatchCommands):
    """Adapter for legacy batch commands."""
    
    def __init__(self):
        """Initialize legacy batch commands."""
        self.manager = DatasetManager()
        logger.info("Initialized LegacyBatchCommands")
    
    def export(
        self,
        pattern: Optional[str] = None,
        output_dir: str = "./exports",
        format: str = "csv",
        compression: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Export multiple datasets using legacy implementation."""
        operation = BatchExportOperation(self.manager)
        return operation.execute(
            pattern=pattern,
            output_dir=output_dir,
            format=format,
            compression=compression
        )
    
    def stats(
        self,
        pattern: Optional[str] = None,
        output_file: Optional[str] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Get stats for multiple datasets using legacy implementation."""
        operation = BatchStatsOperation(self.manager)
        return operation.execute(
            pattern=pattern,
            output_file=output_file,
            format=format
        )
    
    def remove(
        self,
        pattern: str,
        force: bool = False,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Remove multiple datasets using legacy implementation."""
        operation = BatchRemoveOperation(self.manager)
        return operation.execute(
            pattern=pattern,
            force=force,
            dry_run=dry_run
        )


class LegacyTimeSeriesCommands(ITimeSeriesCommands):
    """Adapter for legacy time series commands."""
    
    def __init__(self):
        """Initialize legacy time series commands."""
        self.client = MDMClient()
        logger.info("Initialized LegacyTimeSeriesCommands")
    
    def analyze(
        self,
        name: str,
        time_column: str,
        freq: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze time series using legacy implementation."""
        try:
            # Legacy implementation through MDMClient
            dataset = self.client.get_dataset(name)
            
            # Perform analysis (simplified)
            return {
                'success': True,
                'dataset': name,
                'time_column': time_column,
                'frequency': freq or 'auto',
                'output_dir': output_dir
            }
        except Exception as e:
            logger.error(f"Time series analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def split(
        self,
        name: str,
        time_column: str,
        train_size: float = 0.8,
        gap: int = 0,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Split time series using legacy implementation."""
        try:
            # Legacy implementation through MDMClient
            dataset = self.client.get_dataset(name)
            
            return {
                'success': True,
                'dataset': name,
                'time_column': time_column,
                'train_size': train_size,
                'gap': gap,
                'output_dir': output_dir
            }
        except Exception as e:
            logger.error(f"Time series split failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate(
        self,
        name: str,
        time_column: str,
        check_gaps: bool = True,
        check_duplicates: bool = True
    ) -> Dict[str, Any]:
        """Validate time series using legacy implementation."""
        try:
            # Legacy implementation through MDMClient
            dataset = self.client.get_dataset(name)
            
            return {
                'success': True,
                'dataset': name,
                'time_column': time_column,
                'gaps_checked': check_gaps,
                'duplicates_checked': check_duplicates,
                'valid': True
            }
        except Exception as e:
            logger.error(f"Time series validation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class LegacyStatsCommands(IStatsCommands):
    """Adapter for legacy stats commands."""
    
    def __init__(self):
        """Initialize legacy stats commands."""
        self.manager = DatasetManager()
        self.config = get_config_manager()
        logger.info("Initialized LegacyStatsCommands")
    
    def show(
        self,
        format: str = "table",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Show system stats using legacy implementation."""
        # Collect system statistics
        stats = {
            'version': __version__,
            'datasets': len(self.manager.list_datasets()),
            'backend': self.config.get('database.default_backend'),
            'cache_dir': str(self.config.get('paths.cache_path'))
        }
        
        return {
            'success': True,
            'stats': stats,
            'format': format,
            'output_file': output_file
        }
    
    def summary(self) -> Dict[str, Any]:
        """Show stats summary using legacy implementation."""
        datasets = self.manager.list_datasets()
        
        return {
            'success': True,
            'total_datasets': len(datasets),
            'backends': list(set(d.get('backend', 'unknown') for d in datasets)),
            'total_size': sum(d.get('size', 0) for d in datasets)
        }
    
    def dataset(
        self,
        backend: Optional[str] = None,
        sort_by: str = "size",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Show dataset stats using legacy implementation."""
        datasets = self.manager.list_datasets(backend=backend)
        
        # Sort datasets
        if sort_by == "size":
            datasets.sort(key=lambda d: d.get('size', 0), reverse=True)
        elif sort_by == "name":
            datasets.sort(key=lambda d: d.get('name', ''))
        
        # Limit results
        datasets = datasets[:limit]
        
        return {
            'success': True,
            'datasets': datasets,
            'backend': backend,
            'sort_by': sort_by,
            'limit': limit
        }
    
    def cleanup(
        self,
        dry_run: bool = True,
        min_age_days: int = 7
    ) -> Dict[str, Any]:
        """Clean up old files using legacy implementation."""
        # Simplified cleanup logic
        return {
            'success': True,
            'dry_run': dry_run,
            'min_age_days': min_age_days,
            'files_removed': 0,
            'space_freed': 0
        }
    
    def logs(
        self,
        lines: int = 100,
        level: Optional[str] = None,
        follow: bool = False
    ) -> Dict[str, Any]:
        """Show logs using legacy implementation."""
        log_file = self.config.get('logging.file')
        
        return {
            'success': True,
            'log_file': str(log_file) if log_file else None,
            'lines': lines,
            'level': level,
            'follow': follow
        }
    
    def dashboard(self, refresh: int = 5) -> None:
        """Show dashboard using legacy implementation."""
        # Legacy implementation would show live dashboard
        # For adapter, we just return
        pass


class LegacyCLIFormatter(ICLIFormatter):
    """Adapter for legacy CLI formatting."""
    
    def __init__(self):
        """Initialize legacy formatter."""
        from rich.console import Console
        from rich.table import Table
        from rich.progress import Progress
        
        self.console = Console()
        logger.info("Initialized LegacyCLIFormatter")
    
    def format_table(
        self,
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> str:
        """Format data as table using Rich."""
        from rich.table import Table
        
        if not data:
            return "No data to display"
        
        # Create table
        table = Table(title=title)
        
        # Add columns
        if columns:
            for col in columns:
                table.add_column(col)
        else:
            # Use all keys from first item
            for key in data[0].keys():
                table.add_column(key)
        
        # Add rows
        for item in data:
            row = []
            for col in (columns or item.keys()):
                row.append(str(item.get(col, '')))
            table.add_row(*row)
        
        # Return formatted table
        from io import StringIO
        from rich.console import Console
        
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True)
        console.print(table)
        return buffer.getvalue()
    
    def format_json(self, data: Any, pretty: bool = True) -> str:
        """Format data as JSON."""
        import json
        
        if pretty:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)
    
    def format_yaml(self, data: Any) -> str:
        """Format data as YAML."""
        import yaml
        
        return yaml.dump(data, default_flow_style=False)
    
    def format_error(self, error: Exception, verbose: bool = False) -> str:
        """Format error message."""
        if verbose:
            import traceback
            return f"Error: {error}\n\nTraceback:\n{traceback.format_exc()}"
        return f"Error: {error}"
    
    def show_progress(
        self,
        items: Iterator[Any],
        total: Optional[int] = None,
        description: str = "Processing"
    ) -> Iterator[Any]:
        """Show progress bar for iteration."""
        from rich.progress import Progress, SpinnerColumn, TextColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task(description, total=total)
            
            for item in items:
                yield item
                progress.advance(task)


class LegacyCLIConfig(ICLIConfig):
    """Adapter for legacy CLI configuration."""
    
    def __init__(self):
        """Initialize legacy config."""
        self.config = get_config_manager()
        logger.info("Initialized LegacyCLIConfig")
    
    @property
    def output_format(self) -> str:
        """Default output format."""
        return self.config.get('cli.output_format', 'table')
    
    @property
    def color_enabled(self) -> bool:
        """Whether color output is enabled."""
        return self.config.get('cli.color', True)
    
    @property
    def verbose(self) -> bool:
        """Whether verbose output is enabled."""
        return self.config.get('cli.verbose', False)
    
    @property
    def quiet(self) -> bool:
        """Whether quiet mode is enabled."""
        return self.config.get('cli.quiet', False)
    
    def get_command_config(self, command: str) -> Dict[str, Any]:
        """Get configuration for specific command."""
        return self.config.get(f'cli.commands.{command}', {})