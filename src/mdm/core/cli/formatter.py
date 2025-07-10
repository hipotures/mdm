"""New CLI formatter implementation.

This module provides enhanced formatting capabilities for CLI output
with support for multiple formats and rich display features.
"""
import logging
import json
import yaml
from typing import Any, List, Dict, Optional, Iterator
from io import StringIO

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.tree import Tree
from rich import box

from ...interfaces.cli import ICLIFormatter

logger = logging.getLogger(__name__)


class NewCLIFormatter(ICLIFormatter):
    """New CLI formatter with enhanced display capabilities."""
    
    def __init__(self):
        """Initialize formatter."""
        self.console = Console()
        logger.info("Initialized NewCLIFormatter")
    
    def format_table(
        self,
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> str:
        """Format data as a rich table.
        
        Args:
            data: List of dictionaries to display
            title: Optional table title
            columns: Optional list of columns to display (uses all if None)
            
        Returns:
            Formatted table string
        """
        if not data:
            return "No data to display"
        
        # Determine columns
        if columns is None:
            # Use all unique keys from all items
            all_keys = set()
            for item in data:
                all_keys.update(item.keys())
            columns = sorted(all_keys)
        
        # Create table
        table = Table(title=title, box=box.ROUNDED)
        
        # Add columns with smart styling
        for col in columns:
            # Determine column style based on name/content
            style = self._get_column_style(col, data)
            table.add_column(col.replace('_', ' ').title(), style=style)
        
        # Add rows
        for item in data:
            row = []
            for col in columns:
                value = item.get(col, '')
                # Format special values
                formatted = self._format_cell_value(col, value)
                row.append(formatted)
            table.add_row(*row)
        
        # Render to string
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True)
        console.print(table)
        return buffer.getvalue()
    
    def format_json(self, data: Any, pretty: bool = True) -> str:
        """Format data as syntax-highlighted JSON.
        
        Args:
            data: Data to format
            pretty: Whether to pretty-print
            
        Returns:
            Formatted JSON string
        """
        # Convert to JSON
        if pretty:
            json_str = json.dumps(data, indent=2, default=str, sort_keys=True)
        else:
            json_str = json.dumps(data, default=str)
        
        # Apply syntax highlighting
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        
        # Render to string
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True)
        console.print(syntax)
        return buffer.getvalue()
    
    def format_yaml(self, data: Any) -> str:
        """Format data as syntax-highlighted YAML.
        
        Args:
            data: Data to format
            
        Returns:
            Formatted YAML string
        """
        # Convert to YAML
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=True)
        
        # Apply syntax highlighting
        syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
        
        # Render to string
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True)
        console.print(syntax)
        return buffer.getvalue()
    
    def format_error(self, error: Exception, verbose: bool = False) -> str:
        """Format error message with optional traceback.
        
        Args:
            error: Exception to format
            verbose: Whether to include full traceback
            
        Returns:
            Formatted error string
        """
        if verbose:
            import traceback
            
            # Create error panel with traceback
            tb_lines = traceback.format_exception(
                type(error), error, error.__traceback__
            )
            tb_text = ''.join(tb_lines)
            
            # Syntax highlight the traceback
            syntax = Syntax(
                tb_text,
                "python",
                theme="monokai",
                line_numbers=True
            )
            
            panel = Panel(
                syntax,
                title=f"[bold red]{type(error).__name__}[/bold red]",
                border_style="red",
                expand=False
            )
            
            # Render to string
            buffer = StringIO()
            console = Console(file=buffer, force_terminal=True)
            console.print(panel)
            return buffer.getvalue()
        else:
            # Simple error message
            return f"[red]Error: {type(error).__name__}: {error}[/red]"
    
    def show_progress(
        self,
        items: Iterator[Any],
        total: Optional[int] = None,
        description: str = "Processing"
    ) -> Iterator[Any]:
        """Show progress bar for iteration.
        
        Args:
            items: Items to iterate over
            total: Total number of items (if known)
            description: Progress bar description
            
        Yields:
            Items from the iterator
        """
        # If we don't know the total, try to get it
        if total is None and hasattr(items, '__len__'):
            try:
                total = len(items)
            except TypeError:
                pass
        
        # Create appropriate progress display
        if total is not None:
            # Use bar progress for known totals
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                console=self.console
            ) as progress:
                task = progress.add_task(description, total=total)
                
                for item in items:
                    yield item
                    progress.update(task, advance=1)
        else:
            # Use spinner for unknown totals
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TextColumn("({task.completed} items)"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task(description, total=None)
                
                count = 0
                for item in items:
                    yield item
                    count += 1
                    progress.update(task, completed=count)
    
    def format_tree(self, data: Dict[str, Any], title: Optional[str] = None) -> str:
        """Format hierarchical data as a tree.
        
        Args:
            data: Hierarchical data
            title: Optional tree title
            
        Returns:
            Formatted tree string
        """
        # Create tree
        tree = Tree(title or "Data Tree")
        
        # Build tree recursively
        self._build_tree(tree, data)
        
        # Render to string
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True)
        console.print(tree)
        return buffer.getvalue()
    
    def format_diff(self, old: Any, new: Any, title: Optional[str] = None) -> str:
        """Format a diff between two objects.
        
        Args:
            old: Old value
            new: New value
            title: Optional diff title
            
        Returns:
            Formatted diff string
        """
        # Convert to strings for comparison
        old_str = json.dumps(old, indent=2, default=str, sort_keys=True)
        new_str = json.dumps(new, indent=2, default=str, sort_keys=True)
        
        # Create simple diff display
        from difflib import unified_diff
        
        diff_lines = list(unified_diff(
            old_str.splitlines(keepends=True),
            new_str.splitlines(keepends=True),
            fromfile="old",
            tofile="new",
            n=3
        ))
        
        if not diff_lines:
            return "[green]No differences found[/green]"
        
        # Format diff with colors
        formatted_lines = []
        for line in diff_lines:
            if line.startswith('+'):
                formatted_lines.append(f"[green]{line.rstrip()}[/green]")
            elif line.startswith('-'):
                formatted_lines.append(f"[red]{line.rstrip()}[/red]")
            elif line.startswith('@'):
                formatted_lines.append(f"[blue]{line.rstrip()}[/blue]")
            else:
                formatted_lines.append(line.rstrip())
        
        diff_text = '\n'.join(formatted_lines)
        
        if title:
            panel = Panel(diff_text, title=title, expand=False)
            buffer = StringIO()
            console = Console(file=buffer, force_terminal=True)
            console.print(panel)
            return buffer.getvalue()
        
        return diff_text
    
    def _get_column_style(self, column: str, data: List[Dict[str, Any]]) -> str:
        """Determine appropriate style for a column."""
        column_lower = column.lower()
        
        # Name/ID columns
        if any(x in column_lower for x in ['name', 'id', 'key']):
            return "cyan"
        
        # Status/state columns
        if any(x in column_lower for x in ['status', 'state', 'type']):
            return "yellow"
        
        # Numeric columns
        if any(x in column_lower for x in ['size', 'count', 'total', 'rows']):
            return "green"
        
        # Date/time columns
        if any(x in column_lower for x in ['date', 'time', 'created', 'updated']):
            return "dim"
        
        # Boolean columns
        if data and len(data) > 0:
            sample_value = data[0].get(column)
            if isinstance(sample_value, bool):
                return "magenta"
        
        # Default
        return "white"
    
    def _format_cell_value(self, column: str, value: Any) -> str:
        """Format a cell value for display."""
        if value is None or value == '':
            return "[dim]—[/dim]"
        
        # Boolean values
        if isinstance(value, bool):
            return "[green]✓[/green]" if value else "[red]✗[/red]"
        
        # File sizes
        if 'size' in column.lower() and isinstance(value, (int, float)):
            from .utils import format_size
            return format_size(int(value))
        
        # Dates
        if any(x in column.lower() for x in ['date', 'time']) and isinstance(value, str):
            from .utils import format_datetime
            return format_datetime(value)
        
        # Lists
        if isinstance(value, list):
            return ', '.join(str(v) for v in value[:3]) + \
                   (f' ... ({len(value)} items)' if len(value) > 3 else '')
        
        # Dictionaries
        if isinstance(value, dict):
            return f"[dim]{{{len(value)} items}}[/dim]"
        
        # Default: convert to string
        return str(value)
    
    def _build_tree(self, tree: Tree, data: Any, key: Optional[str] = None) -> None:
        """Recursively build a tree from data."""
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    branch = tree.add(f"[cyan]{k}[/cyan]")
                    self._build_tree(branch, v, k)
                else:
                    tree.add(f"[cyan]{k}[/cyan]: {self._format_cell_value(k, v)}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    branch = tree.add(f"[dim][{i}][/dim]")
                    self._build_tree(branch, item)
                else:
                    tree.add(f"[dim][{i}][/dim] {item}")
        else:
            tree.add(str(data))