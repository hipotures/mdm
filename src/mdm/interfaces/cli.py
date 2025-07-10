"""CLI interfaces for MDM refactoring.

This module defines protocol interfaces for CLI components,
enabling a clean separation between CLI and business logic.
"""
from typing import Protocol, Dict, Any, List, Optional, Iterator
from pathlib import Path


class ICommandHandler(Protocol):
    """Protocol for command handlers."""
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the command with given arguments.
        
        Args:
            **kwargs: Command arguments
            
        Returns:
            Command result dictionary
        """
        ...


class IDatasetCommands(Protocol):
    """Protocol for dataset-related CLI commands."""
    
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
        """Register a new dataset."""
        ...
    
    def list_datasets(
        self,
        limit: Optional[int] = None,
        backend: Optional[str] = None,
        sort_by: str = "name",
        reverse: bool = False
    ) -> List[Dict[str, Any]]:
        """List registered datasets."""
        ...
    
    def info(self, name: str) -> Dict[str, Any]:
        """Get dataset information."""
        ...
    
    def search(
        self,
        pattern: str,
        search_in: Optional[List[str]] = None,
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for datasets."""
        ...
    
    def stats(self, name: str, detailed: bool = False) -> Dict[str, Any]:
        """Get dataset statistics."""
        ...
    
    def update(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        problem_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Update dataset metadata."""
        ...
    
    def export(
        self,
        name: str,
        output_dir: str,
        format: str = "csv",
        tables: Optional[List[str]] = None,
        compression: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Export dataset to files."""
        ...
    
    def remove(self, name: str, force: bool = False) -> Dict[str, Any]:
        """Remove a dataset."""
        ...


class IBatchCommands(Protocol):
    """Protocol for batch operation commands."""
    
    def export(
        self,
        pattern: Optional[str] = None,
        output_dir: str = "./exports",
        format: str = "csv",
        compression: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Export multiple datasets."""
        ...
    
    def stats(
        self,
        pattern: Optional[str] = None,
        output_file: Optional[str] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Generate statistics for multiple datasets."""
        ...
    
    def remove(
        self,
        pattern: str,
        force: bool = False,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Remove multiple datasets."""
        ...


class ITimeSeriesCommands(Protocol):
    """Protocol for time series commands."""
    
    def analyze(
        self,
        name: str,
        time_column: str,
        freq: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze time series dataset."""
        ...
    
    def split(
        self,
        name: str,
        time_column: str,
        train_size: float = 0.8,
        gap: int = 0,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Split time series data."""
        ...
    
    def validate(
        self,
        name: str,
        time_column: str,
        check_gaps: bool = True,
        check_duplicates: bool = True
    ) -> Dict[str, Any]:
        """Validate time series data."""
        ...


class IStatsCommands(Protocol):
    """Protocol for statistics commands."""
    
    def show(
        self,
        format: str = "table",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Show system statistics."""
        ...
    
    def summary(self) -> Dict[str, Any]:
        """Show statistics summary."""
        ...
    
    def dataset(
        self,
        backend: Optional[str] = None,
        sort_by: str = "size",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Show dataset statistics."""
        ...
    
    def cleanup(
        self,
        dry_run: bool = True,
        min_age_days: int = 7
    ) -> Dict[str, Any]:
        """Clean up old temporary files."""
        ...
    
    def logs(
        self,
        lines: int = 100,
        level: Optional[str] = None,
        follow: bool = False
    ) -> Dict[str, Any]:
        """Show log statistics."""
        ...
    
    def dashboard(self, refresh: int = 5) -> None:
        """Show live statistics dashboard."""
        ...


class ICLIFormatter(Protocol):
    """Protocol for CLI output formatting."""
    
    def format_table(
        self,
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> str:
        """Format data as a table."""
        ...
    
    def format_json(
        self,
        data: Any,
        pretty: bool = True
    ) -> str:
        """Format data as JSON."""
        ...
    
    def format_yaml(self, data: Any) -> str:
        """Format data as YAML."""
        ...
    
    def format_error(
        self,
        error: Exception,
        verbose: bool = False
    ) -> str:
        """Format error message."""
        ...
    
    def show_progress(
        self,
        items: Iterator[Any],
        total: Optional[int] = None,
        description: str = "Processing"
    ) -> Iterator[Any]:
        """Show progress bar for iteration."""
        ...


class ICLIConfig(Protocol):
    """Protocol for CLI configuration."""
    
    @property
    def output_format(self) -> str:
        """Default output format."""
        ...
    
    @property
    def color_enabled(self) -> bool:
        """Whether color output is enabled."""
        ...
    
    @property
    def verbose(self) -> bool:
        """Whether verbose output is enabled."""
        ...
    
    @property
    def quiet(self) -> bool:
        """Whether quiet mode is enabled."""
        ...
    
    def get_command_config(self, command: str) -> Dict[str, Any]:
        """Get configuration for specific command."""
        ...


class ICLIPluginManager(Protocol):
    """Protocol for CLI plugin management."""
    
    def register_plugin(
        self,
        name: str,
        plugin: Any,
        commands: List[str]
    ) -> None:
        """Register a CLI plugin."""
        ...
    
    def get_plugin(self, name: str) -> Optional[Any]:
        """Get a registered plugin."""
        ...
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        ...
    
    def load_plugins(self, directory: Path) -> int:
        """Load plugins from directory."""
        ...