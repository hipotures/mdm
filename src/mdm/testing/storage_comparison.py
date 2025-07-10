"""Storage backend comparison testing framework.

This module provides tools for comparing the behavior and performance of
legacy vs new storage backend implementations.
"""
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import tempfile
import time
import logging
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mdm.core import feature_flags
from mdm.adapters.storage_manager import get_storage_backend, clear_storage_cache
from mdm.interfaces.storage import IStorageBackend

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    legacy_result: Any
    new_result: Any
    legacy_time: float
    new_time: float
    matches: bool
    error: Optional[str] = None


class StorageComparisonTester:
    """Compares legacy and new storage backend implementations."""
    
    def __init__(self, backend_type: str, config: Optional[Dict[str, Any]] = None):
        """Initialize tester.
        
        Args:
            backend_type: Type of backend to test
            config: Optional backend configuration
        """
        self.backend_type = backend_type
        self.config = config or {}
        self.results: List[TestResult] = []
        
        # Create temporary directory for tests
        self.temp_dir = tempfile.mkdtemp(prefix="mdm_storage_test_")
        logger.info(f"Created test directory: {self.temp_dir}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comparison tests.
        
        Returns:
            Test results summary
        """
        console.print(f"\n[bold]Running Storage Backend Comparison Tests[/bold]")
        console.print(f"Backend Type: {self.backend_type}")
        console.print(f"Test Directory: {self.temp_dir}\n")
        
        # Define test suite
        tests = [
            ("Basic Engine Creation", self.test_engine_creation),
            ("Table Operations", self.test_table_operations),
            ("Query Execution", self.test_query_execution),
            ("Batch Processing", self.test_batch_processing),
            ("Metadata Handling", self.test_metadata_handling),
            ("Dataset Operations", self.test_dataset_operations),
            ("Error Handling", self.test_error_handling),
            ("Performance - Small Dataset", self.test_performance_small),
            ("Performance - Large Dataset", self.test_performance_large),
            ("Concurrent Access", self.test_concurrent_access),
        ]
        
        # Run each test
        for test_name, test_func in tests:
            console.print(f"Running: {test_name}...", end="")
            
            try:
                result = self._run_comparison_test(test_name, test_func)
                self.results.append(result)
                
                if result.matches:
                    console.print(f" [green]✓ PASS[/green]")
                else:
                    console.print(f" [red]✗ FAIL[/red]")
                    if result.error:
                        console.print(f"  Error: {result.error}")
                        
            except Exception as e:
                console.print(f" [red]✗ ERROR[/red]")
                console.print(f"  Exception: {str(e)}")
                self.results.append(TestResult(
                    test_name=test_name,
                    legacy_result=None,
                    new_result=None,
                    legacy_time=0,
                    new_time=0,
                    matches=False,
                    error=str(e)
                ))
        
        # Generate summary
        summary = self._generate_summary()
        self._display_summary(summary)
        
        return summary
    
    def test_engine_creation(self) -> Tuple[Any, Any]:
        """Test basic engine creation."""
        db_path = str(Path(self.temp_dir) / "test_engine.db")
        
        # Legacy
        legacy_backend = self._get_backend(use_new=False)
        legacy_engine = legacy_backend.get_engine(db_path)
        
        # New
        new_backend = self._get_backend(use_new=True)
        new_engine = new_backend.get_engine(db_path)
        
        # Verify both created engines
        return (legacy_engine is not None, new_engine is not None)
    
    def test_table_operations(self) -> Tuple[Any, Any]:
        """Test table creation and reading."""
        db_path = str(Path(self.temp_dir) / "test_tables.db")
        
        # Test data
        test_df = pd.DataFrame({
            "id": range(100),
            "name": [f"test_{i}" for i in range(100)],
            "value": np.random.rand(100),
            "timestamp": pd.date_range("2024-01-01", periods=100)
        })
        
        # Legacy operations
        legacy_backend = self._get_backend(use_new=False)
        legacy_engine = legacy_backend.get_engine(db_path + "_legacy")
        legacy_backend.create_table_from_dataframe(test_df, "test_table", legacy_engine)
        legacy_read = legacy_backend.read_table_to_dataframe("test_table", legacy_engine)
        
        # New operations
        new_backend = self._get_backend(use_new=True)
        new_engine = new_backend.get_engine(db_path + "_new")
        new_backend.create_table_from_dataframe(test_df, "test_table", new_engine)
        new_read = new_backend.read_table_to_dataframe("test_table", new_engine)
        
        # Compare
        legacy_shape = legacy_read.shape
        new_shape = new_read.shape
        
        return (legacy_shape, new_shape)
    
    def test_query_execution(self) -> Tuple[Any, Any]:
        """Test SQL query execution."""
        db_path = str(Path(self.temp_dir) / "test_queries.db")
        
        # Setup test data
        test_df = pd.DataFrame({
            "category": ["A", "B", "A", "C", "B", "A"] * 10,
            "value": range(60)
        })
        
        # Legacy
        legacy_backend = self._get_backend(use_new=False)
        legacy_engine = legacy_backend.get_engine(db_path + "_legacy")
        legacy_backend.create_table_from_dataframe(test_df, "data", legacy_engine)
        
        # Execute aggregation query
        query = "SELECT category, COUNT(*) as count, AVG(value) as avg_value FROM data GROUP BY category"
        legacy_result = legacy_backend.execute_query(query, legacy_engine)
        
        # New
        new_backend = self._get_backend(use_new=True)
        new_engine = new_backend.get_engine(db_path + "_new")
        new_backend.create_table_from_dataframe(test_df, "data", new_engine)
        new_result = new_backend.execute_query(query, new_engine)
        
        return (True, True)  # Both executed successfully
    
    def test_batch_processing(self) -> Tuple[Any, Any]:
        """Test batch data processing."""
        db_path = str(Path(self.temp_dir) / "test_batch.db")
        
        # Create large dataset
        n_rows = 50000
        test_df = pd.DataFrame({
            "id": range(n_rows),
            "value": np.random.rand(n_rows),
        })
        
        # Legacy batch processing
        legacy_backend = self._get_backend(use_new=False)
        legacy_engine = legacy_backend.get_engine(db_path + "_legacy")
        
        batch_size = 5000
        legacy_start = time.time()
        for i in range(0, n_rows, batch_size):
            batch = test_df.iloc[i:i+batch_size]
            if_exists = "replace" if i == 0 else "append"
            legacy_backend.create_table_from_dataframe(batch, "batch_data", legacy_engine, if_exists)
        legacy_time = time.time() - legacy_start
        
        # New batch processing
        new_backend = self._get_backend(use_new=True)
        new_engine = new_backend.get_engine(db_path + "_new")
        
        new_start = time.time()
        for i in range(0, n_rows, batch_size):
            batch = test_df.iloc[i:i+batch_size]
            if_exists = "replace" if i == 0 else "append"
            new_backend.create_table_from_dataframe(batch, "batch_data", new_engine, if_exists)
        new_time = time.time() - new_start
        
        # Verify counts
        legacy_count = legacy_backend.read_table_to_dataframe("batch_data", legacy_engine).shape[0]
        new_count = new_backend.read_table_to_dataframe("batch_data", new_engine).shape[0]
        
        return (legacy_count, new_count)
    
    def test_metadata_handling(self) -> Tuple[Any, Any]:
        """Test metadata operations."""
        # Create test dataset
        dataset_name = "test_metadata"
        
        # Legacy
        legacy_backend = self._get_backend(use_new=False)
        legacy_backend.create_dataset(dataset_name + "_legacy", {"test": True})
        
        test_metadata = {
            "description": "Test dataset",
            "version": "1.0",
            "created_at": datetime.now().isoformat()
        }
        legacy_backend.update_metadata(dataset_name + "_legacy", test_metadata)
        legacy_metadata = legacy_backend.get_metadata(dataset_name + "_legacy")
        
        # New
        new_backend = self._get_backend(use_new=True)
        new_backend.create_dataset(dataset_name + "_new", {"test": True})
        new_backend.update_metadata(dataset_name + "_new", test_metadata)
        new_metadata = new_backend.get_metadata(dataset_name + "_new")
        
        # Clean up
        legacy_backend.drop_dataset(dataset_name + "_legacy")
        new_backend.drop_dataset(dataset_name + "_new")
        
        return (
            legacy_metadata.get("description"), 
            new_metadata.get("description")
        )
    
    def test_dataset_operations(self) -> Tuple[Any, Any]:
        """Test dataset-level operations."""
        dataset_name = "test_dataset_ops"
        
        # Test data
        test_df = pd.DataFrame({
            "id": range(1000),
            "value": np.random.rand(1000)
        })
        
        # Legacy
        legacy_backend = self._get_backend(use_new=False)
        legacy_backend.create_dataset(dataset_name + "_legacy", {})
        legacy_backend.save_data(dataset_name + "_legacy", test_df)
        legacy_exists = legacy_backend.dataset_exists(dataset_name + "_legacy")
        legacy_data = legacy_backend.load_data(dataset_name + "_legacy")
        legacy_backend.drop_dataset(dataset_name + "_legacy")
        legacy_exists_after = legacy_backend.dataset_exists(dataset_name + "_legacy")
        
        # New
        new_backend = self._get_backend(use_new=True)
        new_backend.create_dataset(dataset_name + "_new", {})
        new_backend.save_data(dataset_name + "_new", test_df)
        new_exists = new_backend.dataset_exists(dataset_name + "_new")
        new_data = new_backend.load_data(dataset_name + "_new")
        new_backend.drop_dataset(dataset_name + "_new")
        new_exists_after = new_backend.dataset_exists(dataset_name + "_new")
        
        return (
            (legacy_exists, legacy_data.shape, legacy_exists_after),
            (new_exists, new_data.shape, new_exists_after)
        )
    
    def test_error_handling(self) -> Tuple[Any, Any]:
        """Test error handling behavior."""
        # Test non-existent dataset
        legacy_backend = self._get_backend(use_new=False)
        new_backend = self._get_backend(use_new=True)
        
        legacy_error = None
        new_error = None
        
        try:
            legacy_backend.load_data("non_existent_dataset")
        except Exception as e:
            legacy_error = type(e).__name__
        
        try:
            new_backend.load_data("non_existent_dataset")
        except Exception as e:
            new_error = type(e).__name__
        
        return (legacy_error is not None, new_error is not None)
    
    def test_performance_small(self) -> Tuple[Any, Any]:
        """Test performance with small dataset."""
        n_rows = 1000
        test_df = pd.DataFrame({
            "id": range(n_rows),
            "value": np.random.rand(n_rows)
        })
        
        # Legacy timing
        legacy_backend = self._get_backend(use_new=False)
        legacy_times = self._time_operations(legacy_backend, test_df, "perf_small_legacy")
        
        # New timing
        new_backend = self._get_backend(use_new=True)
        new_times = self._time_operations(new_backend, test_df, "perf_small_new")
        
        return (legacy_times["total"], new_times["total"])
    
    def test_performance_large(self) -> Tuple[Any, Any]:
        """Test performance with large dataset."""
        n_rows = 100000
        test_df = pd.DataFrame({
            "id": range(n_rows),
            "text": [f"text_{i}" * 10 for i in range(n_rows)],
            "value": np.random.rand(n_rows),
            "category": np.random.choice(["A", "B", "C", "D"], n_rows)
        })
        
        # Legacy timing
        legacy_backend = self._get_backend(use_new=False)
        legacy_times = self._time_operations(legacy_backend, test_df, "perf_large_legacy")
        
        # New timing
        new_backend = self._get_backend(use_new=True)
        new_times = self._time_operations(new_backend, test_df, "perf_large_new")
        
        return (legacy_times["total"], new_times["total"])
    
    def test_concurrent_access(self) -> Tuple[Any, Any]:
        """Test concurrent access handling."""
        # This is a simplified test - real concurrency testing would be more complex
        dataset_name = "test_concurrent"
        
        # Legacy
        legacy_backend = self._get_backend(use_new=False)
        legacy_backend.create_dataset(dataset_name + "_legacy", {})
        
        # Multiple operations
        for i in range(5):
            df = pd.DataFrame({"batch": [i] * 100, "value": range(100)})
            legacy_backend.save_data(
                dataset_name + "_legacy", 
                df, 
                table_name=f"batch_{i}"
            )
        
        legacy_tables = []
        try:
            # Get engine to list tables
            path = legacy_backend._get_dataset_path(
                dataset_name + "_legacy",
                Path.home() / ".mdm" / "datasets"
            )
            engine = legacy_backend.get_engine(path)
            legacy_tables = legacy_backend.get_table_names(engine)
        except:
            pass
        
        legacy_backend.drop_dataset(dataset_name + "_legacy")
        
        # New
        new_backend = self._get_backend(use_new=True)
        new_backend.create_dataset(dataset_name + "_new", {})
        
        for i in range(5):
            df = pd.DataFrame({"batch": [i] * 100, "value": range(100)})
            new_backend.save_data(
                dataset_name + "_new",
                df,
                table_name=f"batch_{i}"
            )
        
        new_tables = []
        try:
            # Get engine to list tables
            path = new_backend._get_dataset_path(
                dataset_name + "_new",
                Path.home() / ".mdm" / "datasets"
            )
            engine = new_backend.get_engine(path)
            new_tables = new_backend.get_table_names(engine)
        except:
            pass
        
        new_backend.drop_dataset(dataset_name + "_new")
        
        return (len(legacy_tables), len(new_tables))
    
    def _get_backend(self, use_new: bool) -> IStorageBackend:
        """Get storage backend with feature flag set."""
        # Clear cache to ensure fresh instance
        clear_storage_cache()
        
        # Set feature flag
        original_flag = feature_flags.get("use_new_storage", False)
        feature_flags.set("use_new_storage", use_new)
        
        try:
            backend = get_storage_backend(self.backend_type, self.config)
            return backend
        finally:
            # Restore original flag
            feature_flags.set("use_new_storage", original_flag)
    
    def _run_comparison_test(
        self, 
        test_name: str, 
        test_func: Callable
    ) -> TestResult:
        """Run a single comparison test."""
        # Run legacy version
        legacy_start = time.time()
        legacy_result = None
        legacy_error = None
        
        try:
            legacy_result = test_func()[0] if hasattr(test_func(), '__getitem__') else test_func()
        except Exception as e:
            legacy_error = str(e)
        
        legacy_time = time.time() - legacy_start
        
        # Run new version
        new_start = time.time()
        new_result = None
        new_error = None
        
        try:
            new_result = test_func()[1] if hasattr(test_func(), '__getitem__') else test_func()
        except Exception as e:
            new_error = str(e)
        
        new_time = time.time() - new_start
        
        # Compare results
        matches = (
            legacy_result == new_result and
            legacy_error == new_error
        )
        
        error = None
        if legacy_error or new_error:
            error = f"Legacy: {legacy_error}, New: {new_error}"
        elif not matches:
            error = f"Results differ - Legacy: {legacy_result}, New: {new_result}"
        
        return TestResult(
            test_name=test_name,
            legacy_result=legacy_result,
            new_result=new_result,
            legacy_time=legacy_time,
            new_time=new_time,
            matches=matches,
            error=error
        )
    
    def _time_operations(
        self, 
        backend: IStorageBackend,
        test_df: pd.DataFrame,
        dataset_name: str
    ) -> Dict[str, float]:
        """Time common operations."""
        times = {}
        
        # Create dataset
        start = time.time()
        backend.create_dataset(dataset_name, {})
        times["create"] = time.time() - start
        
        # Save data
        start = time.time()
        backend.save_data(dataset_name, test_df)
        times["save"] = time.time() - start
        
        # Load data
        start = time.time()
        loaded = backend.load_data(dataset_name)
        times["load"] = time.time() - start
        
        # Query data (if supported)
        try:
            start = time.time()
            # Simple aggregation query
            path = backend._get_dataset_path(dataset_name, Path.home() / ".mdm" / "datasets")
            engine = backend.get_engine(path)
            backend.execute_query("SELECT COUNT(*) FROM data", engine)
            times["query"] = time.time() - start
        except:
            times["query"] = 0
        
        # Drop dataset
        start = time.time()
        backend.drop_dataset(dataset_name)
        times["drop"] = time.time() - start
        
        times["total"] = sum(times.values())
        
        return times
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.matches)
        failed_tests = total_tests - passed_tests
        
        # Calculate performance metrics
        total_legacy_time = sum(r.legacy_time for r in self.results)
        total_new_time = sum(r.new_time for r in self.results)
        
        performance_ratio = (
            (total_new_time / total_legacy_time) if total_legacy_time > 0 else 1.0
        )
        
        return {
            "backend_type": self.backend_type,
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_legacy_time": total_legacy_time,
            "total_new_time": total_new_time,
            "performance_ratio": performance_ratio,
            "results": self.results,
            "failed_tests": [r for r in self.results if not r.matches]
        }
    
    def _display_summary(self, summary: Dict[str, Any]) -> None:
        """Display test summary."""
        # Create summary panel
        panel_content = f"""
[bold]Storage Backend Comparison Results[/bold]

Backend Type: [cyan]{summary['backend_type']}[/cyan]
Total Tests: {summary['total_tests']}
Passed: [green]{summary['passed']}[/green]
Failed: [red]{summary['failed']}[/red]
Success Rate: [{'green' if summary['success_rate'] >= 90 else 'yellow'}]{summary['success_rate']:.1f}%[/]

Performance Comparison:
Legacy Total Time: {summary['total_legacy_time']:.3f}s
New Total Time: {summary['total_new_time']:.3f}s
Performance Ratio: [{'green' if summary['performance_ratio'] <= 1.1 else 'yellow'}]{summary['performance_ratio']:.2f}x[/]
"""
        
        console.print(Panel(panel_content, title="Summary"))
        
        # Show failed tests if any
        if summary['failed_tests']:
            console.print("\n[red]Failed Tests:[/red]")
            for result in summary['failed_tests']:
                console.print(f"  - {result.test_name}")
                if result.error:
                    console.print(f"    Error: {result.error}")
        
        # Create detailed results table
        table = Table(title="Detailed Results")
        table.add_column("Test", style="cyan")
        table.add_column("Legacy Time", style="yellow")
        table.add_column("New Time", style="yellow")
        table.add_column("Speedup", style="green")
        table.add_column("Status", style="green")
        
        for result in self.results:
            speedup = (
                result.legacy_time / result.new_time 
                if result.new_time > 0 else 0
            )
            status = "[green]PASS[/green]" if result.matches else "[red]FAIL[/red]"
            
            table.add_row(
                result.test_name,
                f"{result.legacy_time:.3f}s",
                f"{result.new_time:.3f}s",
                f"{speedup:.2f}x",
                status
            )
        
        console.print("\n")
        console.print(table)