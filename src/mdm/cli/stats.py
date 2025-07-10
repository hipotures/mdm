"""CLI commands for viewing MDM statistics and monitoring data."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mdm.monitoring import SimpleMonitor, MetricType
from mdm.monitoring.dashboard import generate_dashboard

app = typer.Typer(help="View MDM statistics and monitoring data")
console = Console()


def format_duration(ms: Optional[float]) -> str:
    """Format duration in milliseconds to human readable."""
    if ms is None:
        return "N/A"
    if ms < 1000:
        return f"{ms:.1f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


def format_timestamp(timestamp: str) -> str:
    """Format timestamp to relative time."""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now()
        diff = now - dt
        
        if diff < timedelta(minutes=1):
            return "just now"
        elif diff < timedelta(hours=1):
            return f"{int(diff.total_seconds() / 60)}m ago"
        elif diff < timedelta(days=1):
            return f"{int(diff.total_seconds() / 3600)}h ago"
        else:
            return f"{diff.days}d ago"
    except:
        return timestamp


@app.command()
def show(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent operations to show"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Filter by dataset name"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by metric type")
):
    """Show recent operations and metrics."""
    monitor = SimpleMonitor()
    
    # Parse metric type if provided
    metric_type = None
    if type:
        try:
            metric_type = MetricType(type)
        except ValueError:
            console.print(f"[red]Invalid metric type: {type}[/red]")
            console.print(f"Valid types: {', '.join([t.value for t in MetricType])}")
            raise typer.Exit(1)
    
    # Get recent metrics
    metrics = monitor.get_recent_metrics(limit=limit, metric_type=metric_type, dataset_name=dataset)
    
    if not metrics:
        console.print("[yellow]No metrics found[/yellow]")
        return
    
    # Create table
    table = Table(title="Recent Operations", show_header=True, header_style="bold cyan")
    table.add_column("Time", style="dim", width=12)
    table.add_column("Type", style="cyan", width=20)
    table.add_column("Operation", width=30)
    table.add_column("Dataset", style="yellow", width=20)
    table.add_column("Duration", style="green", width=10)
    table.add_column("Status", justify="center", width=10)
    
    for metric in metrics:
        status = "[green]✓[/green]" if metric['success'] else "[red]✗[/red]"
        
        table.add_row(
            format_timestamp(metric['timestamp']),
            metric['metric_type'],
            metric['operation'],
            metric['dataset_name'] or "-",
            format_duration(metric['duration_ms']),
            status
        )
        
        # Show error message if failed
        if not metric['success'] and metric['error_message']:
            table.add_row(
                "",
                "",
                Text(f"  → {metric['error_message']}", style="red dim"),
                "",
                "",
                ""
            )
    
    console.print(table)


@app.command()
def summary():
    """Show summary statistics."""
    monitor = SimpleMonitor()
    stats = monitor.get_summary_stats()
    
    # Overall panel
    overall = stats['overall']
    if overall:
        success_rate = (overall['successful_operations'] / overall['total_operations'] * 100) if overall['total_operations'] > 0 else 0
        
        overall_text = f"""
[bold]Total Operations:[/bold] {overall['total_operations']}
[bold]Success Rate:[/bold] {success_rate:.1f}%
[bold]Average Duration:[/bold] {format_duration(overall['avg_duration_ms'])}
[bold]Last Operation:[/bold] {format_timestamp(overall['last_operation']) if overall['last_operation'] else 'Never'}
"""
        console.print(Panel(overall_text, title="Overall Statistics", border_style="cyan"))
    
    # Operations by type
    if stats['by_type']:
        table = Table(title="Operations by Type", show_header=True, header_style="bold cyan")
        table.add_column("Type", style="cyan", width=25)
        table.add_column("Count", justify="right", width=10)
        table.add_column("Avg Duration", justify="right", width=15)
        table.add_column("Errors", justify="right", width=10)
        
        for type_stat in stats['by_type']:
            error_style = "red" if type_stat['error_count'] > 0 else "green"
            table.add_row(
                type_stat['metric_type'],
                str(type_stat['count']),
                format_duration(type_stat['avg_duration_ms']),
                f"[{error_style}]{type_stat['error_count']}[/{error_style}]"
            )
        
        console.print(table)
    
    # Dataset statistics
    dataset_stats = stats['dataset_stats']
    if dataset_stats and dataset_stats['total_datasets'] > 0:
        dataset_text = f"""
[bold]Total Datasets:[/bold] {dataset_stats['total_datasets']}
[bold]Datasets Registered:[/bold] {dataset_stats['datasets_registered']}
"""
        console.print(Panel(dataset_text, title="Dataset Statistics", border_style="yellow"))
    
    # Recent errors
    if stats['recent_errors']:
        console.print("\n[bold red]Recent Errors:[/bold red]")
        for error in stats['recent_errors']:
            console.print(f"  • {format_timestamp(error['timestamp'])} - {error['operation']}")
            console.print(f"    [dim]{error['error_message']}[/dim]")


@app.command()
def dataset(name: str):
    """Show metrics for a specific dataset."""
    monitor = SimpleMonitor()
    metrics = monitor.get_dataset_metrics(name)
    
    if not metrics:
        console.print(f"[yellow]No metrics found for dataset '{name}'[/yellow]")
        return
    
    console.print(f"\n[bold]Metrics for dataset: {name}[/bold]\n")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Operation Type", style="cyan", width=25)
    table.add_column("Count", justify="right", width=10)
    table.add_column("Avg Duration", justify="right", width=15)
    table.add_column("Rows Processed", justify="right", width=15)
    table.add_column("Last Operation", width=20)
    
    for metric in metrics:
        table.add_row(
            metric['metric_type'],
            str(metric['count']),
            format_duration(metric['avg_duration_ms']),
            str(metric['total_rows_processed']) if metric['total_rows_processed'] else "-",
            format_timestamp(metric['last_operation']) if metric['last_operation'] else "-"
        )
    
    console.print(table)


@app.command()
def cleanup(
    days: int = typer.Option(30, "--days", "-d", help="Keep metrics for this many days")
):
    """Clean up old metrics."""
    monitor = SimpleMonitor()
    
    # Confirm before cleanup
    confirm = typer.confirm(f"Delete metrics older than {days} days?")
    if not confirm:
        console.print("[yellow]Cleanup cancelled[/yellow]")
        return
    
    deleted = monitor.cleanup_old_metrics(days)
    console.print(f"[green]Cleaned up {deleted} old metrics[/green]")


@app.command()
def logs(
    tail: int = typer.Option(50, "--tail", "-n", help="Number of recent log lines to show"),
    level: Optional[str] = typer.Option(None, "--level", "-l", help="Filter by log level"),
    grep: Optional[str] = typer.Option(None, "--grep", "-g", help="Filter by text pattern")
):
    """Show recent log entries."""
    from mdm.utils.paths import PathManager
    
    path_manager = PathManager()
    log_file = path_manager.base_path / "logs" / "mdm.log"
    
    if not log_file.exists():
        console.print("[yellow]No log file found[/yellow]")
        return
    
    # Read last N lines
    with open(log_file, 'r') as f:
        lines = f.readlines()[-tail:]
    
    # Apply filters
    if level:
        level = level.upper()
        lines = [line for line in lines if f"| {level} |" in line]
    
    if grep:
        lines = [line for line in lines if grep.lower() in line.lower()]
    
    # Color code by level
    for line in lines:
        line = line.strip()
        if "| ERROR |" in line:
            console.print(line, style="red")
        elif "| WARNING |" in line:
            console.print(line, style="yellow")
        elif "| INFO |" in line:
            console.print(line, style="green")
        elif "| DEBUG |" in line:
            console.print(line, style="dim")
        else:
            console.print(line)


@app.command()
def dashboard(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for dashboard HTML"),
    open: bool = typer.Option(True, "--open/--no-open", help="Open dashboard in browser")
):
    """Generate HTML dashboard with charts and statistics."""
    try:
        dashboard_path = generate_dashboard(output_path=output, open_browser=open)
        
        if open:
            console.print(f"[green]Dashboard generated and opened in browser[/green]")
        else:
            console.print(f"[green]Dashboard generated at: {dashboard_path}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error generating dashboard: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()