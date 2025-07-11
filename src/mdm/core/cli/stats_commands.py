"""New stats commands implementation.

This module provides system and dataset statistics with
enhanced monitoring and dashboard capabilities.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
import json
import yaml

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...interfaces.cli import IStatsCommands
from ...adapters import get_dataset_manager
from ...config import get_config_manager
from ... import __version__
from .utils import format_size, format_datetime

logger = logging.getLogger(__name__)
console = Console()


class NewStatsCommands(IStatsCommands):
    """New implementation of stats commands with live monitoring."""
    
    def __init__(self):
        """Initialize stats commands."""
        self._manager = None
        self._config = None
        logger.info("Initialized NewStatsCommands")
    
    @property
    def manager(self):
        """Lazy load manager."""
        if self._manager is None:
            self._manager = get_dataset_manager(force_new=True)
        return self._manager
    
    @property
    def config(self):
        """Lazy load config."""
        if self._config is None:
            self._config = get_config_manager()
        return self._config
    
    def show(
        self,
        format: str = "table",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Show comprehensive system statistics."""
        try:
            console.print("[bold]Gathering system statistics...[/bold]")
            
            # Collect statistics
            stats = self._collect_system_stats()
            
            # Format output
            if format == "table":
                self._display_stats_table(stats)
            elif format == "json":
                output = json.dumps(stats, indent=2, default=str)
                if output_file:
                    Path(output_file).write_text(output)
                    console.print(f"[green]✓[/green] Stats saved to: [cyan]{output_file}[/cyan]")
                else:
                    console.print(output)
            elif format == "yaml":
                output = yaml.dump(stats, default_flow_style=False)
                if output_file:
                    Path(output_file).write_text(output)
                    console.print(f"[green]✓[/green] Stats saved to: [cyan]{output_file}[/cyan]")
                else:
                    console.print(output)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Show stats failed: {e}")
            console.print(f"[red]Error showing stats: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def summary(self) -> Dict[str, Any]:
        """Show quick statistics summary."""
        try:
            datasets = self.manager.list_datasets()
            
            # Calculate summary
            summary = {
                'success': True,
                'total_datasets': len(datasets),
                'total_size': sum(d.get('size', 0) for d in datasets),
                'backends': {},
                'problem_types': {},
                'recent_activity': {
                    'last_24h': 0,
                    'last_7d': 0,
                    'last_30d': 0
                }
            }
            
            # Analyze datasets
            now = datetime.now()
            for dataset in datasets:
                # Backend distribution
                backend = dataset.get('backend', 'unknown')
                summary['backends'][backend] = summary['backends'].get(backend, 0) + 1
                
                # Problem type distribution
                problem_type = dataset.get('problem_type', 'unknown')
                summary['problem_types'][problem_type] = \
                    summary['problem_types'].get(problem_type, 0) + 1
                
                # Recent activity
                if 'registration_date' in dataset:
                    reg_date = datetime.fromisoformat(
                        dataset['registration_date'].replace('Z', '+00:00')
                    )
                    age = now - reg_date
                    
                    if age <= timedelta(days=1):
                        summary['recent_activity']['last_24h'] += 1
                    if age <= timedelta(days=7):
                        summary['recent_activity']['last_7d'] += 1
                    if age <= timedelta(days=30):
                        summary['recent_activity']['last_30d'] += 1
            
            # Display summary
            self._display_summary(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary failed: {e}")
            console.print(f"[red]Error generating summary: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def dataset(
        self,
        backend: Optional[str] = None,
        sort_by: str = "size",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Show dataset statistics sorted and filtered."""
        try:
            # Get datasets
            datasets = self.manager.list_datasets(backend=backend)
            
            if not datasets:
                console.print("[yellow]No datasets found[/yellow]")
                return {'success': True, 'datasets': []}
            
            # Enrich with statistics
            enriched = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task("Loading statistics...", total=len(datasets))
                
                for dataset in datasets:
                    try:
                        stats = self.manager.get_dataset_stats(dataset['name'])
                        dataset['stats'] = stats
                        
                        # Calculate total rows
                        total_rows = 0
                        for table_stats in stats.get('tables', {}).values():
                            total_rows += table_stats.get('row_count', 0)
                        dataset['total_rows'] = total_rows
                        
                        enriched.append(dataset)
                    except Exception as e:
                        logger.warning(f"Failed to get stats for {dataset['name']}: {e}")
                        dataset['stats'] = None
                        dataset['total_rows'] = 0
                        enriched.append(dataset)
                    
                    progress.update(task, advance=1)
            
            # Sort datasets
            if sort_by == "size":
                enriched.sort(key=lambda d: d.get('size', 0), reverse=True)
            elif sort_by == "name":
                enriched.sort(key=lambda d: d.get('name', ''))
            elif sort_by == "rows":
                enriched.sort(key=lambda d: d.get('total_rows', 0), reverse=True)
            elif sort_by == "date":
                enriched.sort(
                    key=lambda d: d.get('registration_date', ''),
                    reverse=True
                )
            
            # Limit results
            limited = enriched[:limit]
            
            # Display results
            self._display_dataset_stats(limited, sort_by)
            
            return {
                'success': True,
                'total': len(datasets),
                'shown': len(limited),
                'datasets': limited
            }
            
        except Exception as e:
            logger.error(f"Dataset stats failed: {e}")
            console.print(f"[red]Error getting dataset stats: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def cleanup(
        self,
        dry_run: bool = True,
        min_age_days: int = 7
    ) -> Dict[str, Any]:
        """Clean up old temporary files and cache."""
        try:
            console.print("[bold]Scanning for cleanup candidates...[/bold]")
            
            # Paths to check
            cache_path = Path(self.config.get('paths.cache_path'))
            temp_path = Path(self.config.get('paths.temp_path', '/tmp/mdm'))
            
            cleanup_info = {
                'success': True,
                'dry_run': dry_run,
                'candidates': [],
                'total_size': 0,
                'file_count': 0
            }
            
            # Find old files
            min_age = timedelta(days=min_age_days)
            now = datetime.now()
            
            for base_path in [cache_path, temp_path]:
                if not base_path.exists():
                    continue
                
                for path in base_path.rglob('*'):
                    if path.is_file():
                        try:
                            # Check age
                            mtime = datetime.fromtimestamp(path.stat().st_mtime)
                            age = now - mtime
                            
                            if age >= min_age:
                                size = path.stat().st_size
                                cleanup_info['candidates'].append({
                                    'path': str(path),
                                    'size': size,
                                    'age_days': age.days
                                })
                                cleanup_info['total_size'] += size
                                cleanup_info['file_count'] += 1
                        except Exception as e:
                            logger.debug(f"Error checking {path}: {e}")
            
            # Display cleanup info
            self._display_cleanup_info(cleanup_info)
            
            # Perform cleanup if not dry run
            if not dry_run and cleanup_info['file_count'] > 0:
                removed_count = 0
                removed_size = 0
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True
                ) as progress:
                    task = progress.add_task(
                        "Removing files...",
                        total=cleanup_info['file_count']
                    )
                    
                    for candidate in cleanup_info['candidates']:
                        try:
                            path = Path(candidate['path'])
                            if path.exists():
                                path.unlink()
                                removed_count += 1
                                removed_size += candidate['size']
                        except Exception as e:
                            logger.warning(f"Failed to remove {candidate['path']}: {e}")
                        
                        progress.update(task, advance=1)
                
                cleanup_info['removed_count'] = removed_count
                cleanup_info['removed_size'] = removed_size
                
                console.print(f"\n[green]✓[/green] Removed {removed_count} files, "
                            f"freed {format_size(removed_size)}")
            
            return cleanup_info
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            console.print(f"[red]Error during cleanup: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def logs(
        self,
        lines: int = 100,
        level: Optional[str] = None,
        follow: bool = False
    ) -> Dict[str, Any]:
        """Show and analyze log entries."""
        try:
            log_file = self.config.get('logging.file')
            
            if not log_file or not Path(log_file).exists():
                console.print("[yellow]No log file found[/yellow]")
                return {
                    'success': True,
                    'log_file': None,
                    'message': 'Logging to file not configured'
                }
            
            log_path = Path(log_file)
            console.print(f"[bold]Log file:[/bold] [cyan]{log_path}[/cyan]")
            console.print(f"Size: {format_size(log_path.stat().st_size)}")
            
            # Read last N lines
            with open(log_path, 'r') as f:
                all_lines = f.readlines()
            
            # Filter by level if specified
            if level:
                level_upper = level.upper()
                filtered_lines = [
                    line for line in all_lines
                    if level_upper in line
                ]
            else:
                filtered_lines = all_lines
            
            # Get last N lines
            display_lines = filtered_lines[-lines:]
            
            # Display logs
            console.print(f"\n[bold]Last {len(display_lines)} log entries:[/bold]")
            for line in display_lines:
                # Color based on level
                if 'ERROR' in line:
                    console.print(f"[red]{line.rstrip()}[/red]")
                elif 'WARNING' in line:
                    console.print(f"[yellow]{line.rstrip()}[/yellow]")
                elif 'INFO' in line:
                    console.print(f"[blue]{line.rstrip()}[/blue]")
                elif 'DEBUG' in line:
                    console.print(f"[dim]{line.rstrip()}[/dim]")
                else:
                    console.print(line.rstrip())
            
            # Follow mode (simplified - would need proper implementation)
            if follow:
                console.print("\n[yellow]Follow mode not implemented in this version[/yellow]")
            
            return {
                'success': True,
                'log_file': str(log_path),
                'lines_shown': len(display_lines),
                'total_lines': len(all_lines),
                'filtered': level is not None
            }
            
        except Exception as e:
            logger.error(f"Show logs failed: {e}")
            console.print(f"[red]Error showing logs: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def dashboard(self, refresh: int = 5) -> None:
        """Show live statistics dashboard."""
        try:
            console.print("[bold]MDM Statistics Dashboard[/bold]")
            console.print("[dim]Press Ctrl+C to exit[/dim]\n")
            
            layout = self._create_dashboard_layout()
            
            with Live(layout, refresh_per_second=1, screen=True) as live:
                while True:
                    try:
                        # Update dashboard data
                        self._update_dashboard(layout)
                        time.sleep(refresh)
                    except KeyboardInterrupt:
                        break
            
            console.print("\n[yellow]Dashboard closed[/yellow]")
            
        except Exception as e:
            logger.error(f"Dashboard failed: {e}")
            console.print(f"[red]Error running dashboard: {e}[/red]")
    
    def _collect_system_stats(self) -> Dict[str, Any]:
        """Collect comprehensive system statistics."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'version': __version__,
            'system': {
                'mdm_home': str(Path.home() / '.mdm'),
                'config_file': str(self.config._config_file),
                'backend': self.config.get('database.default_backend'),
            },
            'datasets': {
                'total': 0,
                'by_backend': {},
                'total_size': 0,
                'total_tables': 0,
                'total_rows': 0
            },
            'storage': {
                'cache_size': 0,
                'temp_size': 0,
                'datasets_size': 0
            }
        }
        
        # Dataset statistics
        datasets = self.manager.list_datasets()
        stats['datasets']['total'] = len(datasets)
        
        for dataset in datasets:
            backend = dataset.get('backend', 'unknown')
            stats['datasets']['by_backend'][backend] = \
                stats['datasets']['by_backend'].get(backend, 0) + 1
            stats['datasets']['total_size'] += dataset.get('size', 0)
            
            # Get detailed stats if available
            try:
                dataset_stats = self.manager.get_dataset_stats(dataset['name'])
                stats['datasets']['total_tables'] += len(dataset_stats.get('tables', {}))
                for table_stats in dataset_stats.get('tables', {}).values():
                    stats['datasets']['total_rows'] += table_stats.get('row_count', 0)
            except Exception:
                pass
        
        # Storage statistics
        for path_type, path_key in [
            ('cache_size', 'paths.cache_path'),
            ('temp_size', 'paths.temp_path'),
            ('datasets_size', 'paths.datasets_path')
        ]:
            path = Path(self.config.get(path_key, ''))
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                stats['storage'][path_type] = size
        
        return stats
    
    def _display_stats_table(self, stats: Dict[str, Any]) -> None:
        """Display statistics in table format."""
        # System info
        console.print(Panel.fit("[bold cyan]MDM System Statistics[/bold cyan]"))
        
        system_table = Table(title="System Information", show_header=False)
        system_table.add_row("Version:", stats.get('version', 'Unknown'))
        system_table.add_row("Backend:", stats['system']['backend'])
        system_table.add_row("Config File:", stats['system']['config_file'])
        system_table.add_row("MDM Home:", stats['system']['mdm_home'])
        console.print(system_table)
        
        # Dataset stats
        dataset_table = Table(title="Dataset Statistics", show_header=False)
        dataset_table.add_row("Total Datasets:", f"{stats['datasets']['total']:,}")
        dataset_table.add_row("Total Size:", format_size(stats['datasets']['total_size']))
        dataset_table.add_row("Total Tables:", f"{stats['datasets']['total_tables']:,}")
        dataset_table.add_row("Total Rows:", f"{stats['datasets']['total_rows']:,}")
        console.print(dataset_table)
        
        # Backend distribution
        if stats['datasets']['by_backend']:
            backend_table = Table(title="Datasets by Backend")
            backend_table.add_column("Backend", style="cyan")
            backend_table.add_column("Count", style="yellow")
            
            for backend, count in stats['datasets']['by_backend'].items():
                backend_table.add_row(backend, str(count))
            console.print(backend_table)
        
        # Storage stats
        storage_table = Table(title="Storage Usage", show_header=False)
        storage_table.add_row("Cache:", format_size(stats['storage']['cache_size']))
        storage_table.add_row("Temp:", format_size(stats['storage']['temp_size']))
        storage_table.add_row("Datasets:", format_size(stats['storage']['datasets_size']))
        storage_table.add_row(
            "Total:",
            format_size(sum(stats['storage'].values()))
        )
        console.print(storage_table)
    
    def _display_summary(self, summary: Dict[str, Any]) -> None:
        """Display quick summary."""
        console.print(Panel.fit("[bold cyan]MDM Quick Summary[/bold cyan]"))
        
        # Overview
        overview = Table(show_header=False)
        overview.add_row("Total Datasets:", f"[bold]{summary['total_datasets']}[/bold]")
        overview.add_row("Total Size:", f"[bold]{format_size(summary['total_size'])}[/bold]")
        console.print(overview)
        
        # Backends
        if summary['backends']:
            console.print("\n[bold]Storage Backends:[/bold]")
            for backend, count in summary['backends'].items():
                console.print(f"  • {backend}: {count}")
        
        # Problem types
        if summary['problem_types']:
            console.print("\n[bold]Problem Types:[/bold]")
            for ptype, count in summary['problem_types'].items():
                console.print(f"  • {ptype}: {count}")
        
        # Recent activity
        activity = summary['recent_activity']
        console.print("\n[bold]Recent Activity:[/bold]")
        console.print(f"  • Last 24 hours: {activity['last_24h']}")
        console.print(f"  • Last 7 days: {activity['last_7d']}")
        console.print(f"  • Last 30 days: {activity['last_30d']}")
    
    def _display_dataset_stats(self, datasets: List[Dict[str, Any]], sort_by: str) -> None:
        """Display dataset statistics table."""
        table = Table(title=f"Top Datasets by {sort_by.title()}")
        table.add_column("Name", style="cyan")
        table.add_column("Backend", style="yellow")
        table.add_column("Size", style="green")
        table.add_column("Rows", style="blue")
        table.add_column("Tables", style="magenta")
        table.add_column("Registered", style="dim")
        
        for dataset in datasets:
            table.add_row(
                dataset['name'],
                dataset.get('backend', 'N/A'),
                format_size(dataset.get('size', 0)),
                f"{dataset.get('total_rows', 0):,}",
                str(len(dataset.get('stats', {}).get('tables', {}))),
                format_datetime(dataset.get('registration_date'))
            )
        
        console.print(table)
    
    def _display_cleanup_info(self, cleanup_info: Dict[str, Any]) -> None:
        """Display cleanup information."""
        if cleanup_info['file_count'] == 0:
            console.print("[green]No cleanup candidates found[/green]")
            return
        
        console.print(f"\n[bold]Found {cleanup_info['file_count']} files to clean up[/bold]")
        console.print(f"Total size: [yellow]{format_size(cleanup_info['total_size'])}[/yellow]")
        
        if cleanup_info['dry_run']:
            console.print("\n[yellow]DRY RUN - No files will be removed[/yellow]")
            console.print("Use --no-dry-run to actually remove files")
        
        # Show sample of files
        console.print("\n[bold]Sample files:[/bold]")
        for candidate in cleanup_info['candidates'][:10]:
            console.print(f"  • {candidate['path']} "
                         f"({format_size(candidate['size'])}, "
                         f"{candidate['age_days']} days old)")
        
        if len(cleanup_info['candidates']) > 10:
            console.print(f"  ... and {len(cleanup_info['candidates']) - 10} more")
    
    def _create_dashboard_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="datasets"),
            Layout(name="storage")
        )
        
        layout["right"].split_column(
            Layout(name="activity"),
            Layout(name="performance")
        )
        
        return layout
    
    def _update_dashboard(self, layout: Layout) -> None:
        """Update dashboard with latest data."""
        # Header
        layout["header"].update(
            Panel(
                f"[bold cyan]MDM Statistics Dashboard[/bold cyan]\n"
                f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                style="blue"
            )
        )
        
        # Get current stats
        stats = self._collect_system_stats()
        datasets = self.manager.list_datasets()
        
        # Datasets panel
        dataset_info = Table(show_header=False)
        dataset_info.add_row("Total:", f"{len(datasets)}")
        dataset_info.add_row("Size:", format_size(stats['datasets']['total_size']))
        dataset_info.add_row("Tables:", f"{stats['datasets']['total_tables']:,}")
        dataset_info.add_row("Rows:", f"{stats['datasets']['total_rows']:,}")
        
        layout["datasets"].update(Panel(dataset_info, title="Datasets"))
        
        # Storage panel
        storage_info = Table(show_header=False)
        storage_info.add_row("Cache:", format_size(stats['storage']['cache_size']))
        storage_info.add_row("Temp:", format_size(stats['storage']['temp_size']))
        storage_info.add_row("Datasets:", format_size(stats['storage']['datasets_size']))
        
        layout["storage"].update(Panel(storage_info, title="Storage"))
        
        # Activity panel (simplified)
        activity_info = "[dim]Recent dataset activity[/dim]\n\n"
        recent = sorted(datasets, key=lambda d: d.get('registration_date', ''), reverse=True)[:5]
        for dataset in recent:
            activity_info += f"• {dataset['name']}\n"
        
        layout["activity"].update(Panel(activity_info, title="Recent Activity"))
        
        # Performance panel (placeholder)
        perf_info = "[dim]System performance metrics[/dim]\n\n"
        perf_info += "• CPU: [green]Normal[/green]\n"
        perf_info += "• Memory: [green]Normal[/green]\n"
        perf_info += "• Disk I/O: [green]Normal[/green]\n"
        
        layout["performance"].update(Panel(perf_info, title="Performance"))
        
        # Footer
        layout["footer"].update(
            Panel("[dim]Press Ctrl+C to exit[/dim]", style="dim")
        )