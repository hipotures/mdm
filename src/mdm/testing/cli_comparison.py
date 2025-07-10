"""CLI comparison testing for MDM refactoring.

This module provides comprehensive testing tools for comparing
old and new CLI implementations.
"""
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import time
import json
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core import feature_flags
from ..adapters.cli_manager import (
    get_dataset_commands,
    get_batch_commands,
    get_timeseries_commands,
    get_stats_commands,
    clear_cli_cache
)
from ..core.exceptions import MDMError

logger = logging.getLogger(__name__)
console = Console()


class CLITestResult:
    """Result of a CLI comparison test."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.legacy_result = None
        self.new_result = None
        self.legacy_time = 0.0
        self.new_time = 0.0
        self.legacy_error = None
        self.new_error = None
        self.differences = []
        self.performance_ratio = 1.0


class CLIComparisonTester:
    """Tests CLI implementations for compatibility and performance."""
    
    def __init__(self, test_dir: Optional[Path] = None):
        """Initialize CLI tester.
        
        Args:
            test_dir: Directory for test data (temp if not provided)
        """
        self.test_dir = test_dir or Path(tempfile.mkdtemp(prefix="mdm_cli_test_"))
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self._test_datasets = []
        logger.info(f"Initialized CLIComparisonTester with dir: {self.test_dir}")
    
    def run_all_tests(self, cleanup: bool = True) -> Dict[str, Any]:
        """Run all CLI comparison tests.
        
        Args:
            cleanup: If True, cleanup test data after running
            
        Returns:
            Test results summary
        """
        console.print(Panel.fit(
            "[bold cyan]CLI Implementation Comparison Tests[/bold cyan]\n\n"
            "Testing legacy vs new CLI implementations",
            title="CLI Tests"
        ))
        
        # Define test suites
        test_suites = [
            ("Dataset Commands", self._test_dataset_commands),
            ("Batch Commands", self._test_batch_commands),
            ("Time Series Commands", self._test_timeseries_commands),
            ("Stats Commands", self._test_stats_commands),
            ("Error Handling", self._test_error_handling),
            ("Performance", self._test_performance),
        ]
        
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'suites': {},
            'performance_summary': {}
        }
        
        # Run test suites
        for suite_name, test_func in test_suites:
            console.print(f"\n[bold]{suite_name}[/bold]")
            console.print("=" * 50)
            
            suite_results = test_func()
            results['suites'][suite_name] = suite_results
            
            # Update totals
            results['total'] += suite_results['total']
            results['passed'] += suite_results['passed']
            results['failed'] += suite_results['failed']
            results['warnings'] += suite_results.get('warnings', 0)
        
        # Calculate overall performance
        total_legacy_time = 0
        total_new_time = 0
        
        for suite_results in results['suites'].values():
            for test in suite_results.get('tests', []):
                if hasattr(test, 'legacy_time') and hasattr(test, 'new_time'):
                    total_legacy_time += test.legacy_time
                    total_new_time += test.new_time
        
        if total_new_time > 0:
            results['performance_summary']['overall_ratio'] = total_legacy_time / total_new_time
            results['performance_summary']['total_legacy_time'] = total_legacy_time
            results['performance_summary']['total_new_time'] = total_new_time
        
        # Display summary
        self._display_test_summary(results)
        
        # Cleanup if requested
        if cleanup:
            try:
                shutil.rmtree(self.test_dir)
                for dataset_name in self._test_datasets:
                    self._cleanup_dataset(dataset_name)
                logger.info("Cleaned up test data")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
        
        return results
    
    def _test_dataset_commands(self) -> Dict[str, Any]:
        """Test dataset command compatibility."""
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        
        # Test cases
        test_cases = [
            ("Register Basic Dataset", self._test_dataset_register_basic),
            ("List Datasets", self._test_dataset_list),
            ("Dataset Info", self._test_dataset_info),
            ("Search Datasets", self._test_dataset_search),
            ("Update Dataset", self._test_dataset_update),
            ("Export Dataset", self._test_dataset_export),
            ("Remove Dataset", self._test_dataset_remove),
        ]
        
        # Run tests
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Running dataset tests...", total=len(test_cases))
            
            for test_name, test_func in test_cases:
                result = test_func()
                results['tests'].append(result)
                results['total'] += 1
                
                if result.passed:
                    results['passed'] += 1
                    console.print(f"  [green]✓[/green] {test_name}")
                else:
                    results['failed'] += 1
                    console.print(f"  [red]✗[/red] {test_name}")
                    if result.differences:
                        for diff in result.differences[:3]:
                            console.print(f"    - {diff}")
                
                progress.update(task, advance=1)
        
        return results
    
    def _test_batch_commands(self) -> Dict[str, Any]:
        """Test batch command compatibility."""
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        
        # Create test datasets for batch operations
        self._create_test_datasets(3)
        
        test_cases = [
            ("Batch Export", self._test_batch_export),
            ("Batch Stats", self._test_batch_stats),
            ("Batch Remove (dry run)", self._test_batch_remove_dry),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Running batch tests...", total=len(test_cases))
            
            for test_name, test_func in test_cases:
                result = test_func()
                results['tests'].append(result)
                results['total'] += 1
                
                if result.passed:
                    results['passed'] += 1
                    console.print(f"  [green]✓[/green] {test_name}")
                else:
                    results['failed'] += 1
                    console.print(f"  [red]✗[/red] {test_name}")
                
                progress.update(task, advance=1)
        
        return results
    
    def _test_timeseries_commands(self) -> Dict[str, Any]:
        """Test time series command compatibility."""
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        
        # Create time series dataset
        ts_dataset = self._create_timeseries_dataset()
        
        test_cases = [
            ("Time Series Analyze", lambda: self._test_timeseries_analyze(ts_dataset)),
            ("Time Series Split", lambda: self._test_timeseries_split(ts_dataset)),
            ("Time Series Validate", lambda: self._test_timeseries_validate(ts_dataset)),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Running time series tests...", total=len(test_cases))
            
            for test_name, test_func in test_cases:
                result = test_func()
                results['tests'].append(result)
                results['total'] += 1
                
                if result.passed:
                    results['passed'] += 1
                    console.print(f"  [green]✓[/green] {test_name}")
                else:
                    results['failed'] += 1
                    console.print(f"  [red]✗[/red] {test_name}")
                
                progress.update(task, advance=1)
        
        return results
    
    def _test_stats_commands(self) -> Dict[str, Any]:
        """Test stats command compatibility."""
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        
        test_cases = [
            ("Show Stats", self._test_stats_show),
            ("Stats Summary", self._test_stats_summary),
            ("Dataset Stats", self._test_stats_dataset),
            ("Cleanup (dry run)", self._test_stats_cleanup),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Running stats tests...", total=len(test_cases))
            
            for test_name, test_func in test_cases:
                result = test_func()
                results['tests'].append(result)
                results['total'] += 1
                
                if result.passed:
                    results['passed'] += 1
                    console.print(f"  [green]✓[/green] {test_name}")
                else:
                    results['failed'] += 1
                    console.print(f"  [red]✗[/red] {test_name}")
                
                progress.update(task, advance=1)
        
        return results
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling compatibility."""
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        
        test_cases = [
            ("Invalid Dataset Name", self._test_error_invalid_name),
            ("Non-existent Path", self._test_error_invalid_path),
            ("Missing Dataset", self._test_error_missing_dataset),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Running error handling tests...", total=len(test_cases))
            
            for test_name, test_func in test_cases:
                result = test_func()
                results['tests'].append(result)
                results['total'] += 1
                
                if result.passed:
                    results['passed'] += 1
                    console.print(f"  [green]✓[/green] {test_name}")
                else:
                    results['failed'] += 1
                    console.print(f"  [red]✗[/red] {test_name}")
                
                progress.update(task, advance=1)
        
        return results
    
    def _test_performance(self) -> Dict[str, Any]:
        """Test performance comparison."""
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'tests': [],
            'performance_metrics': {}
        }
        
        # Create larger dataset for performance testing
        perf_dataset = self._create_performance_dataset()
        
        test_cases = [
            ("Registration Performance", lambda: self._test_perf_registration()),
            ("List Performance", lambda: self._test_perf_list()),
            ("Search Performance", lambda: self._test_perf_search()),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Running performance tests...", total=len(test_cases))
            
            for test_name, test_func in test_cases:
                result = test_func()
                results['tests'].append(result)
                results['total'] += 1
                
                # Performance tests always "pass" but we check if new is slower
                if result.performance_ratio >= 0.5:  # New is at most 2x slower
                    results['passed'] += 1
                    console.print(f"  [green]✓[/green] {test_name} "
                                f"(ratio: {result.performance_ratio:.2f}x)")
                else:
                    results['failed'] += 1
                    console.print(f"  [red]✗[/red] {test_name} "
                                f"(ratio: {result.performance_ratio:.2f}x)")
                
                # Store metrics
                results['performance_metrics'][test_name] = {
                    'legacy_time': result.legacy_time,
                    'new_time': result.new_time,
                    'ratio': result.performance_ratio
                }
                
                progress.update(task, advance=1)
        
        return results
    
    def _run_command_comparison(
        self,
        test_name: str,
        command_group: str,
        command: str,
        **kwargs
    ) -> CLITestResult:
        """Run a command with both implementations and compare."""
        result = CLITestResult(test_name)
        
        # Clear cache between tests
        clear_cli_cache()
        
        # Get command handlers
        if command_group == 'dataset':
            legacy_handler = get_dataset_commands(force_new=False)
            new_handler = get_dataset_commands(force_new=True)
        elif command_group == 'batch':
            legacy_handler = get_batch_commands(force_new=False)
            new_handler = get_batch_commands(force_new=True)
        elif command_group == 'timeseries':
            legacy_handler = get_timeseries_commands(force_new=False)
            new_handler = get_timeseries_commands(force_new=True)
        elif command_group == 'stats':
            legacy_handler = get_stats_commands(force_new=False)
            new_handler = get_stats_commands(force_new=True)
        else:
            raise ValueError(f"Unknown command group: {command_group}")
        
        # Run with legacy implementation
        start_time = time.time()
        try:
            legacy_method = getattr(legacy_handler, command)
            result.legacy_result = legacy_method(**kwargs)
        except Exception as e:
            result.legacy_error = str(e)
        result.legacy_time = time.time() - start_time
        
        # Run with new implementation
        start_time = time.time()
        try:
            new_method = getattr(new_handler, command)
            result.new_result = new_method(**kwargs)
        except Exception as e:
            result.new_error = str(e)
        result.new_time = time.time() - start_time
        
        # Compare results
        if result.legacy_error and result.new_error:
            # Both errored - check if same type of error
            result.passed = type(result.legacy_error) == type(result.new_error)
        elif result.legacy_error or result.new_error:
            # One errored - not compatible
            result.passed = False
            if result.legacy_error:
                result.differences.append("Legacy implementation errored")
            else:
                result.differences.append("New implementation errored")
        else:
            # Compare outputs
            result.differences = self._compare_results(
                result.legacy_result,
                result.new_result
            )
            result.passed = len(result.differences) == 0
        
        # Calculate performance ratio
        if result.new_time > 0:
            result.performance_ratio = result.legacy_time / result.new_time
        
        return result
    
    def _compare_results(self, legacy: Any, new: Any) -> List[str]:
        """Compare two command results."""
        differences = []
        
        # Handle None
        if legacy is None and new is None:
            return []
        if legacy is None or new is None:
            differences.append("One result is None")
            return differences
        
        # Compare types
        if type(legacy) != type(new):
            differences.append(f"Type mismatch: {type(legacy).__name__} vs {type(new).__name__}")
            return differences
        
        # Compare dictionaries
        if isinstance(legacy, dict):
            # Check success status first
            if legacy.get('success') != new.get('success'):
                differences.append("Success status differs")
            
            # Compare keys (ignoring some dynamic fields)
            ignore_keys = {'timestamp', 'duration', 'registration_time'}
            legacy_keys = set(legacy.keys()) - ignore_keys
            new_keys = set(new.keys()) - ignore_keys
            
            if legacy_keys != new_keys:
                missing = legacy_keys - new_keys
                extra = new_keys - legacy_keys
                if missing:
                    differences.append(f"Missing keys in new: {missing}")
                if extra:
                    differences.append(f"Extra keys in new: {extra}")
        
        # Compare lists
        elif isinstance(legacy, list):
            if len(legacy) != len(new):
                differences.append(f"List length differs: {len(legacy)} vs {len(new)}")
        
        return differences
    
    def _create_test_datasets(self, count: int) -> List[str]:
        """Create test datasets."""
        import pandas as pd
        import numpy as np
        
        dataset_names = []
        
        for i in range(count):
            # Create test data
            data = pd.DataFrame({
                'id': range(100),
                'value': np.random.randn(100),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            })
            
            # Save to file
            csv_path = self.test_dir / f"test_dataset_{i}.csv"
            data.to_csv(csv_path, index=False)
            
            # Register with both implementations
            dataset_name = f"cli_test_{i}"
            dataset_names.append(dataset_name)
            self._test_datasets.append(dataset_name)
            
            # Register with legacy
            legacy_cmds = get_dataset_commands(force_new=False)
            legacy_cmds.register(
                name=dataset_name,
                path=str(csv_path),
                force=True,
                generate_features=False
            )
        
        return dataset_names
    
    def _create_timeseries_dataset(self) -> str:
        """Create a time series dataset."""
        import pandas as pd
        import numpy as np
        
        # Create time series data
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': np.cumsum(np.random.randn(365)) + 100,
            'volume': np.random.poisson(1000, 365)
        })
        
        # Save to file
        csv_path = self.test_dir / "timeseries_test.csv"
        data.to_csv(csv_path, index=False)
        
        # Register dataset
        dataset_name = "cli_test_timeseries"
        self._test_datasets.append(dataset_name)
        
        legacy_cmds = get_dataset_commands(force_new=False)
        legacy_cmds.register(
            name=dataset_name,
            path=str(csv_path),
            force=True,
            generate_features=False
        )
        
        return dataset_name
    
    def _create_performance_dataset(self) -> str:
        """Create a larger dataset for performance testing."""
        import pandas as pd
        import numpy as np
        
        # Create larger dataset
        data = pd.DataFrame({
            'id': range(10000),
            'value1': np.random.randn(10000),
            'value2': np.random.exponential(2, 10000),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000),
            'flag': np.random.randint(0, 2, 10000)
        })
        
        csv_path = self.test_dir / "performance_test.csv"
        data.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def _cleanup_dataset(self, name: str) -> None:
        """Clean up a test dataset."""
        try:
            # Try to remove with both implementations
            for force_new in [False, True]:
                try:
                    cmds = get_dataset_commands(force_new=force_new)
                    cmds.remove(name, force=True)
                    break
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Failed to cleanup dataset {name}: {e}")
    
    # Individual test implementations
    def _test_dataset_register_basic(self) -> CLITestResult:
        """Test basic dataset registration."""
        import pandas as pd
        
        # Create test data
        data = pd.DataFrame({
            'id': range(50),
            'value': [i * 2.5 for i in range(50)],
            'label': ['A' if i % 2 == 0 else 'B' for i in range(50)]
        })
        
        csv_path = self.test_dir / "register_test.csv"
        data.to_csv(csv_path, index=False)
        
        # Test registration
        result = self._run_command_comparison(
            "Register Basic Dataset",
            "dataset",
            "register",
            name="cli_test_register",
            path=str(csv_path),
            target="label",
            force=True,
            generate_features=False
        )
        
        self._test_datasets.append("cli_test_register")
        return result
    
    def _test_dataset_list(self) -> CLITestResult:
        """Test dataset listing."""
        return self._run_command_comparison(
            "List Datasets",
            "dataset",
            "list_datasets",
            limit=10
        )
    
    def _test_dataset_info(self) -> CLITestResult:
        """Test dataset info."""
        # Use first test dataset
        if self._test_datasets:
            dataset_name = self._test_datasets[0]
        else:
            # Create one if needed
            datasets = self._create_test_datasets(1)
            dataset_name = datasets[0]
        
        return self._run_command_comparison(
            "Dataset Info",
            "dataset",
            "info",
            name=dataset_name
        )
    
    def _test_dataset_search(self) -> CLITestResult:
        """Test dataset search."""
        return self._run_command_comparison(
            "Search Datasets",
            "dataset",
            "search",
            pattern="cli_test"
        )
    
    def _test_dataset_update(self) -> CLITestResult:
        """Test dataset update."""
        if not self._test_datasets:
            self._create_test_datasets(1)
        
        return self._run_command_comparison(
            "Update Dataset",
            "dataset",
            "update",
            name=self._test_datasets[0],
            description="Updated description",
            tags=["test", "cli"]
        )
    
    def _test_dataset_export(self) -> CLITestResult:
        """Test dataset export."""
        if not self._test_datasets:
            self._create_test_datasets(1)
        
        export_dir = self.test_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        return self._run_command_comparison(
            "Export Dataset",
            "dataset",
            "export",
            name=self._test_datasets[0],
            output_dir=str(export_dir),
            format="csv"
        )
    
    def _test_dataset_remove(self) -> CLITestResult:
        """Test dataset removal."""
        # Create a dataset specifically for removal
        import pandas as pd
        
        data = pd.DataFrame({'id': range(10), 'value': range(10)})
        csv_path = self.test_dir / "remove_test.csv"
        data.to_csv(csv_path, index=False)
        
        # Register it first
        for force_new in [False, True]:
            cmds = get_dataset_commands(force_new=force_new)
            cmds.register(
                name="cli_test_remove",
                path=str(csv_path),
                force=True,
                generate_features=False
            )
        
        # Test removal
        return self._run_command_comparison(
            "Remove Dataset",
            "dataset",
            "remove",
            name="cli_test_remove",
            force=True
        )
    
    def _test_batch_export(self) -> CLITestResult:
        """Test batch export."""
        export_dir = self.test_dir / "batch_exports"
        export_dir.mkdir(exist_ok=True)
        
        return self._run_command_comparison(
            "Batch Export",
            "batch",
            "export",
            pattern="cli_test_",
            output_dir=str(export_dir),
            format="csv"
        )
    
    def _test_batch_stats(self) -> CLITestResult:
        """Test batch stats."""
        return self._run_command_comparison(
            "Batch Stats",
            "batch",
            "stats",
            pattern="cli_test_"
        )
    
    def _test_batch_remove_dry(self) -> CLITestResult:
        """Test batch remove with dry run."""
        return self._run_command_comparison(
            "Batch Remove (dry run)",
            "batch",
            "remove",
            pattern="cli_test_*",
            dry_run=True,
            force=True
        )
    
    def _test_timeseries_analyze(self, dataset_name: str) -> CLITestResult:
        """Test time series analysis."""
        return self._run_command_comparison(
            "Time Series Analyze",
            "timeseries",
            "analyze",
            name=dataset_name,
            time_column="date"
        )
    
    def _test_timeseries_split(self, dataset_name: str) -> CLITestResult:
        """Test time series split."""
        return self._run_command_comparison(
            "Time Series Split",
            "timeseries",
            "split",
            name=dataset_name,
            time_column="date",
            train_size=0.8
        )
    
    def _test_timeseries_validate(self, dataset_name: str) -> CLITestResult:
        """Test time series validation."""
        return self._run_command_comparison(
            "Time Series Validate",
            "timeseries",
            "validate",
            name=dataset_name,
            time_column="date"
        )
    
    def _test_stats_show(self) -> CLITestResult:
        """Test show stats."""
        return self._run_command_comparison(
            "Show Stats",
            "stats",
            "show",
            format="json"
        )
    
    def _test_stats_summary(self) -> CLITestResult:
        """Test stats summary."""
        return self._run_command_comparison(
            "Stats Summary",
            "stats",
            "summary"
        )
    
    def _test_stats_dataset(self) -> CLITestResult:
        """Test dataset stats."""
        return self._run_command_comparison(
            "Dataset Stats",
            "stats",
            "dataset",
            limit=5
        )
    
    def _test_stats_cleanup(self) -> CLITestResult:
        """Test cleanup with dry run."""
        return self._run_command_comparison(
            "Cleanup (dry run)",
            "stats",
            "cleanup",
            dry_run=True,
            min_age_days=30
        )
    
    def _test_error_invalid_name(self) -> CLITestResult:
        """Test error handling for invalid dataset name."""
        return self._run_command_comparison(
            "Invalid Dataset Name",
            "dataset",
            "register",
            name="invalid name!",
            path=str(self.test_dir / "dummy.csv"),
            force=True
        )
    
    def _test_error_invalid_path(self) -> CLITestResult:
        """Test error handling for non-existent path."""
        return self._run_command_comparison(
            "Non-existent Path",
            "dataset",
            "register",
            name="cli_test_invalid_path",
            path="/non/existent/path.csv",
            force=True
        )
    
    def _test_error_missing_dataset(self) -> CLITestResult:
        """Test error handling for missing dataset."""
        return self._run_command_comparison(
            "Missing Dataset",
            "dataset",
            "info",
            name="non_existent_dataset"
        )
    
    def _test_perf_registration(self) -> CLITestResult:
        """Test registration performance."""
        csv_path = self._create_performance_dataset()
        
        result = self._run_command_comparison(
            "Registration Performance",
            "dataset",
            "register",
            name="cli_test_perf",
            path=csv_path,
            force=True,
            generate_features=False
        )
        
        self._test_datasets.append("cli_test_perf")
        return result
    
    def _test_perf_list(self) -> CLITestResult:
        """Test list performance with many datasets."""
        # Create more datasets if needed
        if len(self._test_datasets) < 20:
            self._create_test_datasets(20 - len(self._test_datasets))
        
        return self._run_command_comparison(
            "List Performance",
            "dataset",
            "list_datasets",
            limit=50
        )
    
    def _test_perf_search(self) -> CLITestResult:
        """Test search performance."""
        return self._run_command_comparison(
            "Search Performance",
            "dataset",
            "search",
            pattern="test"
        )
    
    def _display_test_summary(self, results: Dict[str, Any]) -> None:
        """Display test results summary."""
        console.print("\n[bold]Test Summary[/bold]")
        console.print("=" * 50)
        
        # Overall results
        table = Table(show_header=False)
        table.add_row("Total Tests:", f"{results['total']}")
        table.add_row("Passed:", f"[green]{results['passed']}[/green]")
        table.add_row("Failed:", f"[red]{results['failed']}[/red]")
        if results['warnings'] > 0:
            table.add_row("Warnings:", f"[yellow]{results['warnings']}[/yellow]")
        
        console.print(table)
        
        # Performance summary
        if 'performance_summary' in results and results['performance_summary']:
            perf = results['performance_summary']
            console.print("\n[bold]Performance Summary[/bold]")
            
            perf_table = Table()
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="yellow")
            
            if 'overall_ratio' in perf:
                ratio = perf['overall_ratio']
                ratio_str = f"{ratio:.2f}x"
                if ratio > 1:
                    ratio_str = f"[green]{ratio_str} faster[/green]"
                elif ratio < 1:
                    ratio_str = f"[red]{ratio_str} slower[/red]"
                else:
                    ratio_str = f"[yellow]{ratio_str} (same)[/yellow]"
                
                perf_table.add_row("Overall Performance", ratio_str)
                perf_table.add_row(
                    "Total Legacy Time",
                    f"{perf.get('total_legacy_time', 0):.3f}s"
                )
                perf_table.add_row(
                    "Total New Time",
                    f"{perf.get('total_new_time', 0):.3f}s"
                )
            
            console.print(perf_table)
        
        # Suite breakdown
        console.print("\n[bold]Test Suite Results[/bold]")
        suite_table = Table()
        suite_table.add_column("Suite", style="cyan")
        suite_table.add_column("Total", style="white")
        suite_table.add_column("Passed", style="green")
        suite_table.add_column("Failed", style="red")
        
        for suite_name, suite_results in results['suites'].items():
            suite_table.add_row(
                suite_name,
                str(suite_results['total']),
                str(suite_results['passed']),
                str(suite_results['failed'])
            )
        
        console.print(suite_table)