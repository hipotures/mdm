"""New time series commands implementation.

This module provides time series specific operations with
enhanced analysis and validation capabilities.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...interfaces.cli import ITimeSeriesCommands
from ...adapters import get_dataset_manager, get_storage_backend
from ...core.exceptions import DatasetError, ValidationError
from .utils import validate_output_path, format_datetime

logger = logging.getLogger(__name__)
console = Console()


class NewTimeSeriesCommands(ITimeSeriesCommands):
    """New implementation of time series commands with advanced features."""
    
    def __init__(self):
        """Initialize time series commands."""
        self._manager = None
        logger.info("Initialized NewTimeSeriesCommands")
    
    @property
    def manager(self):
        """Lazy load manager."""
        if self._manager is None:
            self._manager = get_dataset_manager(force_new=True)
        return self._manager
    
    def analyze(
        self,
        name: str,
        time_column: str,
        freq: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze time series dataset with comprehensive statistics."""
        try:
            console.print(f"[bold]Analyzing time series dataset:[/bold] {name}")
            console.print(f"Time column: [cyan]{time_column}[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task("Loading dataset...", total=None)
                
                # Get dataset
                info = self.manager.get_dataset_info(name)
                backend = get_storage_backend(info['storage']['backend'])
                
                # Load data
                tables = info['storage']['tables']
                if len(tables) > 1:
                    console.print(f"[yellow]Multiple tables found, using first: {list(tables.keys())[0]}[/yellow]")
                
                table_name = list(tables.keys())[0]
                df = backend.query(f"SELECT * FROM {table_name}")
                
                progress.update(task, description="Analyzing time series...")
                
                # Analyze time series
                analysis = self._analyze_timeseries(df, time_column, freq)
                
                progress.stop()
            
            # Display results
            self._display_analysis(analysis)
            
            # Save results if output directory provided
            if output_dir:
                output_path = validate_output_path(output_dir)
                self._save_analysis(analysis, output_path, name)
                console.print(f"\n[green]✓[/green] Analysis saved to: [cyan]{output_path}[/cyan]")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Time series analysis failed: {e}")
            console.print(f"[red]Analysis failed: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def split(
        self,
        name: str,
        time_column: str,
        train_size: float = 0.8,
        gap: int = 0,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Split time series data with optional gap."""
        try:
            console.print(f"[bold]Splitting time series dataset:[/bold] {name}")
            console.print(f"Time column: [cyan]{time_column}[/cyan]")
            console.print(f"Train size: [yellow]{train_size:.1%}[/yellow]")
            if gap > 0:
                console.print(f"Gap periods: [yellow]{gap}[/yellow]")
            
            # Validate parameters
            if not 0 < train_size < 1:
                raise ValidationError("train_size must be between 0 and 1")
            if gap < 0:
                raise ValidationError("gap must be non-negative")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task("Loading dataset...", total=None)
                
                # Get dataset
                info = self.manager.get_dataset_info(name)
                backend = get_storage_backend(info['storage']['backend'])
                
                # Load data
                tables = info['storage']['tables']
                table_name = list(tables.keys())[0]
                df = backend.query(f"SELECT * FROM {table_name} ORDER BY {time_column}")
                
                progress.update(task, description="Splitting data...")
                
                # Perform split
                split_result = self._split_timeseries(df, time_column, train_size, gap)
                
                progress.stop()
            
            # Display split information
            self._display_split_info(split_result)
            
            # Save splits if output directory provided
            if output_dir:
                output_path = validate_output_path(output_dir)
                self._save_splits(split_result, output_path, name)
                console.print(f"\n[green]✓[/green] Splits saved to: [cyan]{output_path}[/cyan]")
            
            return split_result
            
        except Exception as e:
            logger.error(f"Time series split failed: {e}")
            console.print(f"[red]Split failed: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def validate(
        self,
        name: str,
        time_column: str,
        check_gaps: bool = True,
        check_duplicates: bool = True
    ) -> Dict[str, Any]:
        """Validate time series data integrity."""
        try:
            console.print(f"[bold]Validating time series dataset:[/bold] {name}")
            console.print(f"Time column: [cyan]{time_column}[/cyan]")
            
            checks = []
            if check_gaps:
                checks.append("gaps")
            if check_duplicates:
                checks.append("duplicates")
            console.print(f"Checks: [yellow]{', '.join(checks)}[/yellow]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task("Loading dataset...", total=None)
                
                # Get dataset
                info = self.manager.get_dataset_info(name)
                backend = get_storage_backend(info['storage']['backend'])
                
                # Load data
                tables = info['storage']['tables']
                table_name = list(tables.keys())[0]
                df = backend.query(f"SELECT * FROM {table_name} ORDER BY {time_column}")
                
                progress.update(task, description="Validating time series...")
                
                # Perform validation
                validation_result = self._validate_timeseries(
                    df, time_column, check_gaps, check_duplicates
                )
                
                progress.stop()
            
            # Display validation results
            self._display_validation_results(validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Time series validation failed: {e}")
            console.print(f"[red]Validation failed: {e}[/red]")
            return {'success': False, 'error': str(e)}
    
    def _analyze_timeseries(
        self,
        df: pd.DataFrame,
        time_column: str,
        freq: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform time series analysis."""
        # Convert time column to datetime
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Sort by time
        df = df.sort_values(time_column)
        
        # Basic statistics
        analysis = {
            'success': True,
            'time_column': time_column,
            'basic_stats': {
                'start_date': df[time_column].min(),
                'end_date': df[time_column].max(),
                'duration': df[time_column].max() - df[time_column].min(),
                'total_periods': len(df),
                'unique_timestamps': df[time_column].nunique()
            }
        }
        
        # Detect frequency if not provided
        if not freq:
            freq = self._detect_frequency(df[time_column])
        analysis['frequency'] = freq
        
        # Time gaps analysis
        time_diffs = df[time_column].diff().dropna()
        if len(time_diffs) > 0:
            analysis['time_gaps'] = {
                'min_gap': time_diffs.min(),
                'max_gap': time_diffs.max(),
                'mean_gap': time_diffs.mean(),
                'std_gap': time_diffs.std(),
                'gap_count': (time_diffs != time_diffs.mode()[0]).sum() if len(time_diffs.mode()) > 0 else 0
            }
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_column in numeric_cols:
            numeric_cols.remove(time_column)
        
        if numeric_cols:
            analysis['numeric_columns'] = {}
            for col in numeric_cols[:10]:  # Limit to first 10
                col_analysis = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'missing': df[col].isna().sum(),
                    'missing_pct': df[col].isna().sum() / len(df) * 100
                }
                
                # Simple trend analysis
                if len(df) > 10:
                    col_analysis['trend'] = np.polyfit(range(len(df)), df[col].fillna(0), 1)[0]
                
                analysis['numeric_columns'][col] = col_analysis
        
        return analysis
    
    def _split_timeseries(
        self,
        df: pd.DataFrame,
        time_column: str,
        train_size: float,
        gap: int
    ) -> Dict[str, Any]:
        """Split time series data."""
        # Convert time column to datetime
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column)
        
        # Calculate split point
        n_total = len(df)
        n_train = int(n_total * train_size)
        
        # Apply gap if specified
        if gap > 0 and n_train + gap < n_total:
            n_train = n_train
            n_gap_end = n_train + gap
            train_df = df.iloc[:n_train]
            test_df = df.iloc[n_gap_end:]
            gap_df = df.iloc[n_train:n_gap_end]
        else:
            train_df = df.iloc[:n_train]
            test_df = df.iloc[n_train:]
            gap_df = pd.DataFrame()
        
        result = {
            'success': True,
            'train': {
                'rows': len(train_df),
                'start': train_df[time_column].min(),
                'end': train_df[time_column].max(),
                'data': train_df
            },
            'test': {
                'rows': len(test_df),
                'start': test_df[time_column].min() if len(test_df) > 0 else None,
                'end': test_df[time_column].max() if len(test_df) > 0 else None,
                'data': test_df
            }
        }
        
        if gap > 0 and len(gap_df) > 0:
            result['gap'] = {
                'rows': len(gap_df),
                'start': gap_df[time_column].min(),
                'end': gap_df[time_column].max()
            }
        
        return result
    
    def _validate_timeseries(
        self,
        df: pd.DataFrame,
        time_column: str,
        check_gaps: bool,
        check_duplicates: bool
    ) -> Dict[str, Any]:
        """Validate time series data."""
        # Convert time column to datetime
        df[time_column] = pd.to_datetime(df[time_column])
        
        result = {
            'success': True,
            'valid': True,
            'issues': [],
            'checks': {}
        }
        
        # Check for duplicates
        if check_duplicates:
            duplicates = df[time_column].duplicated()
            n_duplicates = duplicates.sum()
            
            result['checks']['duplicates'] = {
                'checked': True,
                'found': n_duplicates,
                'duplicate_dates': df[df[time_column].duplicated()][time_column].unique().tolist() if n_duplicates > 0 else []
            }
            
            if n_duplicates > 0:
                result['valid'] = False
                result['issues'].append(f"Found {n_duplicates} duplicate timestamps")
        
        # Check for gaps
        if check_gaps:
            df_sorted = df.sort_values(time_column)
            time_diffs = df_sorted[time_column].diff().dropna()
            
            if len(time_diffs) > 0:
                # Detect expected frequency
                freq = self._detect_frequency(df_sorted[time_column])
                
                # Find gaps (periods larger than expected)
                if freq:
                    expected_gap = self._freq_to_timedelta(freq)
                    gaps = time_diffs[time_diffs > expected_gap * 1.5]  # 50% tolerance
                    
                    gap_info = []
                    if len(gaps) > 0:
                        gap_indices = gaps.index
                        for idx in gap_indices[:10]:  # Limit to first 10
                            gap_info.append({
                                'start': df_sorted.iloc[idx-1][time_column],
                                'end': df_sorted.iloc[idx][time_column],
                                'duration': gaps[idx]
                            })
                    
                    result['checks']['gaps'] = {
                        'checked': True,
                        'found': len(gaps),
                        'expected_frequency': freq,
                        'gaps': gap_info
                    }
                    
                    if len(gaps) > 0:
                        result['issues'].append(f"Found {len(gaps)} gaps in time series")
        
        # Check time column type
        if df[time_column].dtype == 'object':
            result['issues'].append("Time column is not datetime type")
            result['valid'] = False
        
        return result
    
    def _detect_frequency(self, time_series: pd.Series) -> Optional[str]:
        """Detect the frequency of a time series."""
        if len(time_series) < 2:
            return None
        
        # Calculate time differences
        diffs = time_series.diff().dropna()
        
        # Get the most common difference
        if len(diffs) == 0:
            return None
        
        mode_diff = diffs.mode()
        if len(mode_diff) == 0:
            return None
        
        common_diff = mode_diff[0]
        
        # Map to pandas frequency strings
        if common_diff <= timedelta(seconds=1):
            return 'S'  # Second
        elif common_diff <= timedelta(minutes=1):
            return 'T'  # Minute
        elif common_diff <= timedelta(hours=1):
            return 'H'  # Hour
        elif common_diff <= timedelta(days=1):
            return 'D'  # Day
        elif common_diff <= timedelta(days=7):
            return 'W'  # Week
        elif common_diff <= timedelta(days=31):
            return 'M'  # Month
        elif common_diff <= timedelta(days=366):
            return 'Y'  # Year
        else:
            return None
    
    def _freq_to_timedelta(self, freq: str) -> timedelta:
        """Convert frequency string to timedelta."""
        freq_map = {
            'S': timedelta(seconds=1),
            'T': timedelta(minutes=1),
            'H': timedelta(hours=1),
            'D': timedelta(days=1),
            'W': timedelta(weeks=1),
            'M': timedelta(days=30),  # Approximate
            'Y': timedelta(days=365)   # Approximate
        }
        return freq_map.get(freq, timedelta(days=1))
    
    def _display_analysis(self, analysis: Dict[str, Any]) -> None:
        """Display time series analysis results."""
        console.print("\n[bold]Time Series Analysis Results[/bold]")
        
        # Basic stats
        stats = analysis['basic_stats']
        basic_table = Table(title="Basic Statistics", show_header=False)
        basic_table.add_row("Start Date:", format_datetime(stats['start_date']))
        basic_table.add_row("End Date:", format_datetime(stats['end_date']))
        basic_table.add_row("Duration:", str(stats['duration']))
        basic_table.add_row("Total Periods:", f"{stats['total_periods']:,}")
        basic_table.add_row("Unique Timestamps:", f"{stats['unique_timestamps']:,}")
        basic_table.add_row("Detected Frequency:", analysis.get('frequency', 'Unknown'))
        console.print(basic_table)
        
        # Time gaps
        if 'time_gaps' in analysis:
            gaps = analysis['time_gaps']
            gap_table = Table(title="Time Gap Analysis", show_header=False)
            gap_table.add_row("Min Gap:", str(gaps['min_gap']))
            gap_table.add_row("Max Gap:", str(gaps['max_gap']))
            gap_table.add_row("Mean Gap:", str(gaps['mean_gap']))
            gap_table.add_row("Irregular Gaps:", f"{gaps['gap_count']:,}")
            console.print(gap_table)
        
        # Numeric columns
        if 'numeric_columns' in analysis:
            console.print("\n[bold]Numeric Columns Analysis[/bold]")
            for col, stats in list(analysis['numeric_columns'].items())[:5]:
                col_table = Table(title=f"Column: {col}", show_header=False)
                col_table.add_row("Mean:", f"{stats['mean']:.2f}")
                col_table.add_row("Std Dev:", f"{stats['std']:.2f}")
                col_table.add_row("Range:", f"{stats['min']:.2f} - {stats['max']:.2f}")
                col_table.add_row("Missing:", f"{stats['missing']} ({stats['missing_pct']:.1f}%)")
                if 'trend' in stats:
                    trend = "↗" if stats['trend'] > 0 else "↘" if stats['trend'] < 0 else "→"
                    col_table.add_row("Trend:", f"{trend} ({stats['trend']:.4f})")
                console.print(col_table)
    
    def _display_split_info(self, split_result: Dict[str, Any]) -> None:
        """Display split information."""
        console.print("\n[bold]Time Series Split Results[/bold]")
        
        table = Table()
        table.add_column("Split", style="cyan")
        table.add_column("Rows", style="yellow")
        table.add_column("Start", style="green")
        table.add_column("End", style="green")
        table.add_column("Duration", style="blue")
        
        for split_name in ['train', 'test']:
            split = split_result[split_name]
            if split['rows'] > 0:
                duration = split['end'] - split['start']
                table.add_row(
                    split_name.capitalize(),
                    f"{split['rows']:,}",
                    format_datetime(split['start']),
                    format_datetime(split['end']),
                    str(duration)
                )
        
        if 'gap' in split_result:
            gap = split_result['gap']
            duration = gap['end'] - gap['start']
            table.add_row(
                "Gap",
                f"{gap['rows']:,}",
                format_datetime(gap['start']),
                format_datetime(gap['end']),
                str(duration)
            )
        
        console.print(table)
    
    def _display_validation_results(self, validation: Dict[str, Any]) -> None:
        """Display validation results."""
        if validation['valid']:
            console.print("\n[green]✓ Time series validation passed![/green]")
        else:
            console.print("\n[red]✗ Time series validation failed![/red]")
        
        if validation['issues']:
            console.print("\n[bold]Issues Found:[/bold]")
            for issue in validation['issues']:
                console.print(f"  • [red]{issue}[/red]")
        
        # Duplicates check
        if 'duplicates' in validation['checks']:
            dup_check = validation['checks']['duplicates']
            if dup_check['checked']:
                console.print(f"\n[bold]Duplicate Check:[/bold]")
                if dup_check['found'] == 0:
                    console.print("  [green]✓[/green] No duplicates found")
                else:
                    console.print(f"  [red]✗[/red] Found {dup_check['found']} duplicates")
                    if dup_check['duplicate_dates']:
                        console.print("  Duplicate dates (first 5):")
                        for dt in dup_check['duplicate_dates'][:5]:
                            console.print(f"    • {format_datetime(dt)}")
        
        # Gaps check  
        if 'gaps' in validation['checks']:
            gap_check = validation['checks']['gaps']
            if gap_check['checked']:
                console.print(f"\n[bold]Gap Check:[/bold]")
                console.print(f"  Expected frequency: [cyan]{gap_check['expected_frequency']}[/cyan]")
                if gap_check['found'] == 0:
                    console.print("  [green]✓[/green] No unexpected gaps found")
                else:
                    console.print(f"  [red]✗[/red] Found {gap_check['found']} gaps")
                    if gap_check['gaps']:
                        console.print("  Gaps (first 5):")
                        for gap in gap_check['gaps'][:5]:
                            console.print(f"    • {format_datetime(gap['start'])} → "
                                        f"{format_datetime(gap['end'])} "
                                        f"({gap['duration']})")
    
    def _save_analysis(self, analysis: Dict[str, Any], output_dir: Path, dataset_name: str) -> None:
        """Save analysis results to file."""
        import json
        
        # Convert datetime objects to strings
        def serialize(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, (timedelta, pd.Timedelta)):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        output_file = output_dir / f"{dataset_name}_timeseries_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=serialize)
    
    def _save_splits(self, split_result: Dict[str, Any], output_dir: Path, dataset_name: str) -> None:
        """Save split datasets to files."""
        # Save train set
        if split_result['train']['rows'] > 0:
            train_file = output_dir / f"{dataset_name}_train.csv"
            split_result['train']['data'].to_csv(train_file, index=False)
            console.print(f"  • Train set: [cyan]{train_file}[/cyan]")
        
        # Save test set
        if split_result['test']['rows'] > 0:
            test_file = output_dir / f"{dataset_name}_test.csv"
            split_result['test']['data'].to_csv(test_file, index=False)
            console.print(f"  • Test set: [cyan]{test_file}[/cyan]")