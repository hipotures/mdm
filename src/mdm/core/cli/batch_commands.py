"""New batch commands implementation.

This module provides batch operations for multiple datasets with
enhanced progress tracking and error handling.
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ...interfaces.cli import IBatchCommands
from ...adapters import get_dataset_manager
from ...core.exceptions import DatasetError
from .utils import format_size, validate_output_path

logger = logging.getLogger(__name__)
console = Console()


class NewBatchCommands(IBatchCommands):
    """New implementation of batch commands with concurrent processing."""
    
    def __init__(self):
        """Initialize batch commands."""
        self._manager = None
        self.max_workers = 4  # Configurable parallelism
        logger.info("Initialized NewBatchCommands")
    
    @property
    def manager(self):
        """Lazy load manager."""
        if self._manager is None:
            self._manager = get_dataset_manager(force_new=True)
        return self._manager
    
    def export(
        self,
        pattern: Optional[str] = None,
        output_dir: str = "./exports",
        format: str = "csv",
        compression: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Export multiple datasets with parallel processing."""
        try:
            # Find matching datasets
            datasets = self._find_matching_datasets(pattern)
            if not datasets:
                console.print("[yellow]No datasets found matching pattern[/yellow]")
                return {'success': True, 'exported': 0, 'errors': 0}
            
            # Validate output directory
            output_path = validate_output_path(output_dir)
            
            console.print(f"[bold]Batch Export[/bold]")
            console.print(f"Datasets: [cyan]{len(datasets)}[/cyan]")
            console.print(f"Output: [cyan]{output_path}[/cyan]")
            console.print(f"Format: [yellow]{format}[/yellow]")
            if compression:
                console.print(f"Compression: [yellow]{compression}[/yellow]")
            
            # Export datasets in parallel
            results = {
                'success': True,
                'exported': 0,
                'errors': 0,
                'datasets': {},
                'total_size': 0
            }
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                
                task = progress.add_task("Exporting datasets...", total=len(datasets))
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit export jobs
                    futures = {}
                    for dataset in datasets:
                        future = executor.submit(
                            self._export_dataset,
                            dataset['name'],
                            output_path / dataset['name'],
                            format,
                            compression
                        )
                        futures[future] = dataset['name']
                    
                    # Process results
                    for future in as_completed(futures):
                        dataset_name = futures[future]
                        try:
                            exported_files = future.result()
                            results['datasets'][dataset_name] = {
                                'status': 'success',
                                'files': exported_files,
                                'count': len(exported_files)
                            }
                            results['exported'] += 1
                            
                            # Calculate total size
                            for file_path in exported_files:
                                if Path(file_path).exists():
                                    results['total_size'] += Path(file_path).stat().st_size
                                    
                        except Exception as e:
                            logger.error(f"Failed to export {dataset_name}: {e}")
                            results['datasets'][dataset_name] = {
                                'status': 'error',
                                'error': str(e)
                            }
                            results['errors'] += 1
                        
                        progress.update(task, advance=1)
            
            # Show summary
            self._show_export_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch export failed: {e}")
            console.print(f"[red]Batch export failed: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def stats(
        self,
        pattern: Optional[str] = None,
        output_file: Optional[str] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Generate statistics for multiple datasets."""
        try:
            # Find matching datasets
            datasets = self._find_matching_datasets(pattern)
            if not datasets:
                console.print("[yellow]No datasets found matching pattern[/yellow]")
                return {'success': True, 'datasets': 0}
            
            console.print(f"[bold]Generating statistics for {len(datasets)} datasets[/bold]")
            
            all_stats = {
                'summary': {
                    'total_datasets': len(datasets),
                    'total_size': 0,
                    'total_rows': 0,
                    'total_tables': 0,
                    'backends': {}
                },
                'datasets': {}
            }
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                
                task = progress.add_task("Computing statistics...", total=len(datasets))
                
                for dataset in datasets:
                    try:
                        stats = self.manager.get_dataset_stats(dataset['name'])
                        all_stats['datasets'][dataset['name']] = stats
                        
                        # Update summary
                        all_stats['summary']['total_size'] += dataset.get('size', 0)
                        all_stats['summary']['total_tables'] += len(stats.get('tables', {}))
                        
                        # Count rows
                        for table_stats in stats.get('tables', {}).values():
                            all_stats['summary']['total_rows'] += table_stats.get('row_count', 0)
                        
                        # Track backends
                        backend = dataset.get('backend', 'unknown')
                        all_stats['summary']['backends'][backend] = \
                            all_stats['summary']['backends'].get(backend, 0) + 1
                            
                    except Exception as e:
                        logger.error(f"Failed to get stats for {dataset['name']}: {e}")
                        all_stats['datasets'][dataset['name']] = {'error': str(e)}
                    
                    progress.update(task, advance=1)
            
            # Save to file if requested
            if output_file:
                self._save_stats(all_stats, output_file, format)
                console.print(f"[green]✓[/green] Statistics saved to: [cyan]{output_file}[/cyan]")
            
            # Show summary
            self._show_stats_summary(all_stats['summary'])
            
            return all_stats
            
        except Exception as e:
            logger.error(f"Batch stats failed: {e}")
            console.print(f"[red]Batch stats failed: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def remove(
        self,
        pattern: str,
        force: bool = False,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Remove multiple datasets with safety checks."""
        try:
            # Find matching datasets
            datasets = self._find_matching_datasets(pattern)
            if not datasets:
                console.print("[yellow]No datasets found matching pattern[/yellow]")
                return {'success': True, 'removed': 0}
            
            # Show what will be removed
            console.print(f"[bold]Datasets matching '{pattern}':[/bold]")
            
            total_size = 0
            table = Table()
            table.add_column("Dataset", style="cyan")
            table.add_column("Backend", style="yellow")
            table.add_column("Size", style="green")
            
            for dataset in datasets:
                table.add_row(
                    dataset['name'],
                    dataset.get('backend', 'N/A'),
                    format_size(dataset.get('size', 0))
                )
                total_size += dataset.get('size', 0)
            
            console.print(table)
            console.print(f"\nTotal: [bold]{len(datasets)}[/bold] datasets, "
                         f"[bold]{format_size(total_size)}[/bold]")
            
            if dry_run:
                console.print("\n[yellow]DRY RUN - No datasets will be removed[/yellow]")
                console.print("Use --no-dry-run to actually remove datasets")
                return {
                    'success': True,
                    'dry_run': True,
                    'would_remove': len(datasets),
                    'datasets': [d['name'] for d in datasets]
                }
            
            if not force:
                console.print("\n[red]This will permanently remove the datasets![/red]")
                console.print("Use --force flag to confirm removal")
                return {
                    'success': False,
                    'error': 'Removal cancelled (use --force to confirm)'
                }
            
            # Actually remove datasets
            results = {
                'success': True,
                'removed': 0,
                'errors': 0,
                'datasets': {}
            }
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                
                task = progress.add_task("Removing datasets...", total=len(datasets))
                
                for dataset in datasets:
                    try:
                        self.manager.remove_dataset(dataset['name'], force=True)
                        results['datasets'][dataset['name']] = 'removed'
                        results['removed'] += 1
                    except Exception as e:
                        logger.error(f"Failed to remove {dataset['name']}: {e}")
                        results['datasets'][dataset['name']] = f'error: {e}'
                        results['errors'] += 1
                    
                    progress.update(task, advance=1)
            
            # Show summary
            console.print(f"\n[green]✓[/green] Removed {results['removed']} datasets")
            if results['errors'] > 0:
                console.print(f"[red]✗[/red] Failed to remove {results['errors']} datasets")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch remove failed: {e}")
            console.print(f"[red]Batch remove failed: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def _find_matching_datasets(self, pattern: Optional[str]) -> List[Dict[str, Any]]:
        """Find datasets matching pattern."""
        all_datasets = self.manager.list_datasets()
        
        if not pattern:
            return all_datasets
        
        # Simple pattern matching (could be enhanced with glob/regex)
        pattern_lower = pattern.lower()
        matching = []
        
        for dataset in all_datasets:
            name_lower = dataset['name'].lower()
            if pattern_lower in name_lower:
                matching.append(dataset)
        
        return matching
    
    def _export_dataset(
        self,
        name: str,
        output_dir: Path,
        format: str,
        compression: Optional[str]
    ) -> List[str]:
        """Export a single dataset."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return self.manager.export_dataset(
            name=name,
            output_dir=str(output_dir),
            format=format,
            compression=compression
        )
    
    def _show_export_summary(self, results: Dict[str, Any]) -> None:
        """Show export summary."""
        console.print("\n[bold]Export Summary[/bold]")
        
        summary = Table(show_header=False)
        summary.add_row("Exported:", f"[green]{results['exported']}[/green]")
        if results['errors'] > 0:
            summary.add_row("Errors:", f"[red]{results['errors']}[/red]")
        summary.add_row("Total Size:", format_size(results['total_size']))
        
        console.print(summary)
        
        # Show errors if any
        if results['errors'] > 0:
            console.print("\n[red]Errors:[/red]")
            for name, info in results['datasets'].items():
                if info.get('status') == 'error':
                    console.print(f"  • {name}: {info['error']}")
    
    def _show_stats_summary(self, summary: Dict[str, Any]) -> None:
        """Show statistics summary."""
        console.print("\n[bold]Statistics Summary[/bold]")
        
        table = Table(show_header=False)
        table.add_row("Total Datasets:", f"{summary['total_datasets']:,}")
        table.add_row("Total Size:", format_size(summary['total_size']))
        table.add_row("Total Rows:", f"{summary['total_rows']:,}")
        table.add_row("Total Tables:", f"{summary['total_tables']:,}")
        
        console.print(table)
        
        if summary['backends']:
            console.print("\n[bold]Backends:[/bold]")
            for backend, count in summary['backends'].items():
                console.print(f"  • {backend}: {count}")
    
    def _save_stats(self, stats: Dict[str, Any], output_file: str, format: str) -> None:
        """Save statistics to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            import json
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        elif format == "yaml":
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(stats, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")