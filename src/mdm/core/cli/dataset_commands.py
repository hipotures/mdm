"""New dataset commands implementation.

This module provides a clean, modular implementation of dataset CLI commands
with improved error handling and progress tracking.
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ...interfaces.cli import IDatasetCommands
from ...adapters import get_dataset_registrar, get_dataset_manager
from ...core.exceptions import DatasetError, ValidationError
from .utils import format_size, format_datetime, validate_output_path

logger = logging.getLogger(__name__)
console = Console()


class NewDatasetCommands(IDatasetCommands):
    """New implementation of dataset commands with enhanced features."""
    
    def __init__(self):
        """Initialize dataset commands."""
        self._registrar = None
        self._manager = None
        logger.info("Initialized NewDatasetCommands")
    
    @property
    def registrar(self):
        """Lazy load registrar."""
        if self._registrar is None:
            self._registrar = get_dataset_registrar(force_new=True)
        return self._registrar
    
    @property
    def manager(self):
        """Lazy load manager."""
        if self._manager is None:
            self._manager = get_dataset_manager(force_new=True)
        return self._manager
    
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
        """Register a new dataset with enhanced progress tracking."""
        try:
            console.print(f"[bold]Registering dataset:[/bold] {name}")
            
            # Show registration options
            if any([target, id_columns, problem_type]):
                options = Table(show_header=False, box=None)
                if target:
                    options.add_row("Target:", f"[cyan]{target}[/cyan]")
                if id_columns:
                    options.add_row("ID Columns:", f"[cyan]{', '.join(id_columns)}[/cyan]")
                if problem_type:
                    options.add_row("Problem Type:", f"[cyan]{problem_type}[/cyan]")
                console.print(options)
            
            # Register dataset
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
            
            # Show success message
            console.print(f"\n[green]✓[/green] Dataset '[cyan]{name}[/cyan]' registered successfully!")
            
            # Show registration summary
            summary = Table(title="Registration Summary", show_header=False)
            summary.add_row("Backend:", f"[yellow]{result.get('backend', 'N/A')}[/yellow]")
            summary.add_row("Tables:", f"[blue]{', '.join(result.get('tables', []))}[/blue]")
            if generate_features:
                summary.add_row("Features:", f"[magenta]{result.get('features_generated', 0)}[/magenta]")
            summary.add_row("Size:", f"[dim]{format_size(result.get('size', 0))}[/dim]")
            
            console.print(summary)
            
            return {
                'success': True,
                'dataset': name,
                **result
            }
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            console.print(f"[red]✗[/red] Registration failed: {e}")
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
        """List datasets with enhanced display."""
        try:
            datasets = self.manager.list_datasets(
                limit=limit,
                backend=backend,
                sort_by=sort_by,
                reverse=reverse
            )
            
            if not datasets:
                console.print("[yellow]No datasets found[/yellow]")
                return []
            
            # Create table
            table = Table(title=f"Datasets ({len(datasets)} found)")
            table.add_column("Name", style="cyan")
            table.add_column("Backend", style="yellow")
            table.add_column("Tables", style="blue")
            table.add_column("Size", style="green")
            table.add_column("Registered", style="dim")
            
            for dataset in datasets:
                table.add_row(
                    dataset['name'],
                    dataset.get('backend', 'N/A'),
                    str(len(dataset.get('tables', []))),
                    format_size(dataset.get('size', 0)),
                    format_datetime(dataset.get('registration_date'))
                )
            
            console.print(table)
            return datasets
            
        except Exception as e:
            logger.error(f"List datasets failed: {e}")
            console.print(f"[red]Error listing datasets: {e}[/red]")
            return []
    
    def info(self, name: str) -> Dict[str, Any]:
        """Get dataset info with rich display."""
        try:
            info = self.manager.get_dataset_info(name)
            
            # Main info panel
            console.print(Panel.fit(
                f"[bold cyan]{name}[/bold cyan]",
                title="Dataset Information"
            ))
            
            # Configuration section
            config = info.get('config', {})
            config_table = Table(title="Configuration", show_header=False)
            config_table.add_row("Description:", config.get('description', 'No description'))
            config_table.add_row("Tags:", ', '.join(config.get('tags', [])) or 'None')
            config_table.add_row("Created:", format_datetime(config.get('created_at')))
            config_table.add_row("Updated:", format_datetime(config.get('updated_at')))
            console.print(config_table)
            
            # Storage section
            storage = info.get('storage', {})
            storage_table = Table(title="Storage", show_header=False)
            storage_table.add_row("Backend:", storage.get('backend', 'N/A'))
            storage_table.add_row("Path:", str(storage.get('path', 'N/A')))
            storage_table.add_row("Size:", format_size(storage.get('size', 0)))
            storage_table.add_row("Tables:", ', '.join(storage.get('tables', {}).keys()))
            console.print(storage_table)
            
            # Schema section
            schema = info.get('schema', {})
            if schema:
                schema_table = Table(title="Schema", show_header=False)
                schema_table.add_row("Target:", schema.get('target_column', 'None'))
                schema_table.add_row("ID Columns:", ', '.join(schema.get('id_columns', [])) or 'None')
                schema_table.add_row("Problem Type:", schema.get('problem_type', 'None'))
                console.print(schema_table)
            
            return info
            
        except Exception as e:
            logger.error(f"Get info failed: {e}")
            console.print(f"[red]Error getting dataset info: {e}[/red]")
            return {}
    
    def search(
        self,
        pattern: str,
        search_in: Optional[List[str]] = None,
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """Search datasets with highlighting."""
        try:
            results = self.manager.search_datasets(
                pattern=pattern,
                search_in=search_in or ['name', 'description', 'tags'],
                case_sensitive=case_sensitive
            )
            
            if not results:
                console.print(f"[yellow]No datasets found matching '{pattern}'[/yellow]")
                return []
            
            console.print(f"[green]Found {len(results)} datasets matching '{pattern}'[/green]\n")
            
            for dataset in results:
                # Highlight matching parts
                name = dataset['name']
                if not case_sensitive and pattern.lower() in name.lower():
                    name = name.replace(pattern, f"[bold red]{pattern}[/bold red]")
                
                console.print(f"• [cyan]{name}[/cyan]")
                if dataset.get('description'):
                    console.print(f"  {dataset['description']}")
                if dataset.get('tags'):
                    console.print(f"  Tags: [dim]{', '.join(dataset['tags'])}[/dim]")
                console.print()
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            console.print(f"[red]Error searching datasets: {e}[/red]")
            return []
    
    def stats(self, name: str, detailed: bool = False) -> Dict[str, Any]:
        """Get dataset statistics with visualization."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task("Computing statistics...", total=None)
                stats = self.manager.get_dataset_stats(name, detailed=detailed)
                progress.stop()
            
            # Display basic stats
            console.print(Panel.fit(
                f"[bold cyan]{name}[/bold cyan]",
                title="Dataset Statistics"
            ))
            
            # Table stats
            for table_name, table_stats in stats.get('tables', {}).items():
                table = Table(title=f"Table: {table_name}")
                table.add_column("Column", style="cyan")
                table.add_column("Type", style="yellow")
                table.add_column("Non-Null", style="green")
                table.add_column("Unique", style="blue")
                
                if detailed:
                    table.add_column("Mean", style="magenta")
                    table.add_column("Std", style="red")
                
                for col_name, col_stats in table_stats.get('columns', {}).items():
                    row = [
                        col_name,
                        col_stats.get('dtype', 'unknown'),
                        f"{col_stats.get('non_null_count', 0):,}",
                        f"{col_stats.get('unique_count', 0):,}"
                    ]
                    
                    if detailed and col_stats.get('dtype') in ['int64', 'float64']:
                        row.extend([
                            f"{col_stats.get('mean', 0):.2f}",
                            f"{col_stats.get('std', 0):.2f}"
                        ])
                    elif detailed:
                        row.extend(['N/A', 'N/A'])
                    
                    table.add_row(*row)
                
                console.print(table)
                console.print(f"Total rows: [bold]{table_stats.get('row_count', 0):,}[/bold]\n")
            
            return stats
            
        except Exception as e:
            logger.error(f"Get stats failed: {e}")
            console.print(f"[red]Error getting statistics: {e}[/red]")
            return {}
    
    def update(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        problem_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Update dataset metadata with confirmation."""
        try:
            # Show what will be updated
            updates = []
            if description is not None:
                updates.append(f"Description: [cyan]{description}[/cyan]")
            if tags is not None:
                updates.append(f"Tags: [cyan]{', '.join(tags)}[/cyan]")
            if problem_type is not None:
                updates.append(f"Problem Type: [cyan]{problem_type}[/cyan]")
            
            if not updates:
                console.print("[yellow]No updates specified[/yellow]")
                return {'success': True, 'message': 'No updates performed'}
            
            console.print(f"[bold]Updating dataset:[/bold] {name}")
            for update in updates:
                console.print(f"  • {update}")
            
            # Perform update
            result = self.manager.update_dataset(
                name=name,
                description=description,
                tags=tags,
                problem_type=problem_type,
                **kwargs
            )
            
            console.print(f"\n[green]✓[/green] Dataset updated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Update failed: {e}")
            console.print(f"[red]Error updating dataset: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def export(
        self,
        name: str,
        output_dir: str,
        format: str = "csv",
        tables: Optional[List[str]] = None,
        compression: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Export dataset with progress tracking."""
        try:
            # Validate output directory
            output_path = validate_output_path(output_dir)
            
            console.print(f"[bold]Exporting dataset:[/bold] {name}")
            console.print(f"Output directory: [cyan]{output_path}[/cyan]")
            console.print(f"Format: [yellow]{format}[/yellow]")
            if compression:
                console.print(f"Compression: [yellow]{compression}[/yellow]")
            
            # Export with progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                
                # Get dataset info to know table count
                info = self.manager.get_dataset_info(name)
                table_count = len(info.get('storage', {}).get('tables', {}))
                
                task = progress.add_task("Exporting...", total=table_count)
                
                exported_files = []
                for i, file_path in enumerate(self.manager.export_dataset(
                    name=name,
                    output_dir=str(output_path),
                    format=format,
                    tables=tables,
                    compression=compression,
                    **kwargs
                )):
                    exported_files.append(file_path)
                    progress.update(task, advance=1)
            
            console.print(f"\n[green]✓[/green] Exported {len(exported_files)} files:")
            for file_path in exported_files:
                console.print(f"  • [cyan]{file_path}[/cyan]")
            
            return exported_files
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            console.print(f"[red]Error exporting dataset: {e}[/red]")
            return []
    
    def remove(self, name: str, force: bool = False) -> Dict[str, Any]:
        """Remove dataset with confirmation."""
        try:
            # Get dataset info first
            info = self.manager.get_dataset_info(name)
            
            if not force:
                console.print(f"[bold yellow]Warning:[/bold yellow] This will remove dataset '{name}'")
                console.print(f"Backend: {info.get('storage', {}).get('backend', 'N/A')}")
                console.print(f"Size: {format_size(info.get('storage', {}).get('size', 0))}")
                console.print("\n[dim]This action cannot be undone.[/dim]")
                
                # In non-interactive mode, require force flag
                console.print("\n[red]Use --force flag to confirm removal[/red]")
                return {'success': False, 'error': 'Removal cancelled (use --force to confirm)'}
            
            # Remove dataset
            console.print(f"[bold]Removing dataset:[/bold] {name}")
            
            result = self.manager.remove_dataset(name, force=True)
            
            console.print(f"[green]✓[/green] Dataset '{name}' removed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Remove failed: {e}")
            console.print(f"[red]Error removing dataset: {e}[/red]")
            return {'success': False, 'error': str(e)}