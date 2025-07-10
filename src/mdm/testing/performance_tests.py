"""Performance testing for MDM refactoring.

This module provides comprehensive performance benchmarking
between legacy and new implementations.
"""
import logging
import tempfile
import shutil
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
from rich.layout import Layout
from rich.chart import BarChart

from ..core import feature_flags
from ..adapters import (
    get_storage_backend,
    get_feature_generator,
    get_dataset_registrar,
    get_dataset_manager,
    get_config_manager,
    get_dataset_commands,
    get_batch_commands,
    clear_storage_cache,
    clear_feature_cache,
    clear_dataset_cache,
    clear_cli_cache
)

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""
    operation: str
    implementation: str
    duration: float
    memory_before: float
    memory_after: float
    cpu_percent: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_used(self) -> float:
        """Calculate memory used in MB."""
        return self.memory_after - self.memory_before
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'implementation': self.implementation,
            'duration': self.duration,
            'memory_used': self.memory_used,
            'cpu_percent': self.cpu_percent,
            'success': self.success,
            'error': self.error,
            'metadata': self.metadata
        }


@dataclass
class PerformanceComparison:
    """Comparison between legacy and new implementation performance."""
    operation: str
    legacy_metric: PerformanceMetric
    new_metric: PerformanceMetric
    
    @property
    def speedup(self) -> float:
        """Calculate speedup ratio (legacy/new)."""
        if self.new_metric.duration > 0:
            return self.legacy_metric.duration / self.new_metric.duration
        return 0.0
    
    @property
    def memory_ratio(self) -> float:
        """Calculate memory usage ratio (new/legacy)."""
        if self.legacy_metric.memory_used > 0:
            return self.new_metric.memory_used / self.legacy_metric.memory_used
        return 1.0
    
    @property
    def is_regression(self) -> bool:
        """Check if new implementation is a regression."""
        return self.speedup < 0.8 or self.memory_ratio > 1.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'legacy': self.legacy_metric.to_dict(),
            'new': self.new_metric.to_dict(),
            'speedup': self.speedup,
            'memory_ratio': self.memory_ratio,
            'is_regression': self.is_regression
        }


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, test_dir: Optional[Path] = None):
        """Initialize performance benchmark.
        
        Args:
            test_dir: Directory for test data (temp if not provided)
        """
        self.test_dir = test_dir or Path(tempfile.mkdtemp(prefix="mdm_perf_"))
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self._test_datasets = []
        self._process = psutil.Process()
        logger.info(f"Initialized PerformanceBenchmark with dir: {self.test_dir}")
    
    def run_all_benchmarks(self, cleanup: bool = True) -> Dict[str, Any]:
        """Run all performance benchmarks.
        
        Args:
            cleanup: If True, cleanup test data after running
            
        Returns:
            Benchmark results summary
        """
        console.print(Panel.fit(
            "[bold cyan]Performance Benchmark Suite[/bold cyan]\n\n"
            "Comparing legacy vs new implementation performance",
            title="Performance Tests"
        ))
        
        # Define benchmark suites
        benchmark_suites = [
            ("Registration Performance", self._benchmark_registration),
            ("Query Performance", self._benchmark_queries),
            ("Feature Generation", self._benchmark_features),
            ("Batch Operations", self._benchmark_batch_ops),
            ("Memory Usage", self._benchmark_memory),
            ("Concurrent Operations", self._benchmark_concurrency),
            ("Large Dataset Handling", self._benchmark_large_datasets),
            ("End-to-End Workflows", self._benchmark_workflows),
        ]
        
        results = {
            'start_time': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'suites': {},
            'comparisons': [],
            'summary': {},
            'regressions': []
        }
        
        # Run benchmark suites
        for suite_name, benchmark_func in benchmark_suites:
            console.print(f"\n[bold]{suite_name}[/bold]")
            console.print("=" * 50)
            
            suite_results = benchmark_func()
            results['suites'][suite_name] = suite_results
            results['comparisons'].extend(suite_results.get('comparisons', []))
            
            # Collect regressions
            for comparison in suite_results.get('comparisons', []):
                if comparison.is_regression:
                    results['regressions'].append({
                        'suite': suite_name,
                        'operation': comparison.operation,
                        'speedup': comparison.speedup,
                        'memory_ratio': comparison.memory_ratio
                    })
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        # Display results
        self._display_benchmark_results(results)
        
        # Save detailed report
        self._save_benchmark_report(results)
        
        # Cleanup if requested
        if cleanup:
            self._cleanup_test_data()
        
        return results
    
    def _benchmark_registration(self) -> Dict[str, Any]:
        """Benchmark dataset registration performance."""
        results = {
            'comparisons': [],
            'metrics': []
        }
        
        # Test different dataset sizes
        sizes = [100, 1000, 5000, 10000]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("Benchmarking registration...", total=len(sizes))
            
            for size in sizes:
                # Create test data
                data = pd.DataFrame({
                    'id': range(size),
                    'value1': np.random.randn(size),
                    'value2': np.random.exponential(1, size),
                    'category': np.random.choice(['A', 'B', 'C'], size)
                })
                
                csv_path = self.test_dir / f"reg_perf_{size}.csv"
                data.to_csv(csv_path, index=False)
                
                # Benchmark both implementations
                operation = f"register_{size}_rows"
                
                legacy_metric = self._benchmark_operation(
                    operation=operation,
                    implementation="legacy",
                    func=lambda: self._register_dataset(
                        f"perf_reg_legacy_{size}",
                        str(csv_path),
                        use_new=False
                    ),
                    metadata={'dataset_size': size}
                )
                
                new_metric = self._benchmark_operation(
                    operation=operation,
                    implementation="new",
                    func=lambda: self._register_dataset(
                        f"perf_reg_new_{size}",
                        str(csv_path),
                        use_new=True
                    ),
                    metadata={'dataset_size': size}
                )
                
                comparison = PerformanceComparison(operation, legacy_metric, new_metric)
                results['comparisons'].append(comparison)
                
                # Display inline results
                speedup_color = "green" if comparison.speedup >= 1 else "red"
                console.print(
                    f"  {size:,} rows: "
                    f"[{speedup_color}]{comparison.speedup:.2f}x[/{speedup_color}] "
                    f"(Legacy: {legacy_metric.duration:.2f}s, New: {new_metric.duration:.2f}s)"
                )
                
                progress.update(task, advance=1)
        
        return results
    
    def _benchmark_queries(self) -> Dict[str, Any]:
        """Benchmark query performance."""
        results = {
            'comparisons': [],
            'metrics': []
        }
        
        # Create test dataset if needed
        if not self._test_datasets:
            self._create_benchmark_datasets()
        
        # Define query operations
        query_ops = [
            ("get_info", lambda ds: self._get_dataset_info(ds)),
            ("get_stats", lambda ds: self._get_dataset_stats(ds)),
            ("list_datasets", lambda: self._list_datasets()),
            ("search_datasets", lambda: self._search_datasets("perf_"))
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Benchmarking queries...", total=len(query_ops))
            
            for op_name, op_func in query_ops:
                # Warm up caches
                self._warmup_caches()
                
                # Benchmark legacy
                legacy_metric = self._benchmark_operation(
                    operation=f"query_{op_name}",
                    implementation="legacy",
                    func=lambda: self._run_with_impl(False, op_func),
                    metadata={'query_type': op_name}
                )
                
                # Benchmark new
                new_metric = self._benchmark_operation(
                    operation=f"query_{op_name}",
                    implementation="new",
                    func=lambda: self._run_with_impl(True, op_func),
                    metadata={'query_type': op_name}
                )
                
                comparison = PerformanceComparison(f"query_{op_name}", legacy_metric, new_metric)
                results['comparisons'].append(comparison)
                
                progress.update(task, advance=1)
        
        return results
    
    def _benchmark_features(self) -> Dict[str, Any]:
        """Benchmark feature generation performance."""
        results = {
            'comparisons': [],
            'metrics': []
        }
        
        # Test different data characteristics
        test_cases = [
            {
                'name': 'small_numeric',
                'rows': 1000,
                'numeric_cols': 5,
                'categorical_cols': 0
            },
            {
                'name': 'mixed_types',
                'rows': 1000,
                'numeric_cols': 3,
                'categorical_cols': 2
            },
            {
                'name': 'large_categorical',
                'rows': 5000,
                'numeric_cols': 2,
                'categorical_cols': 5
            }
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Benchmarking features...", total=len(test_cases))
            
            for test_case in test_cases:
                # Create test data
                data = self._create_feature_test_data(
                    test_case['rows'],
                    test_case['numeric_cols'],
                    test_case['categorical_cols']
                )
                
                # Benchmark feature generation
                operation = f"features_{test_case['name']}"
                
                legacy_metric = self._benchmark_operation(
                    operation=operation,
                    implementation="legacy",
                    func=lambda: self._generate_features(data.copy(), use_new=False),
                    metadata=test_case
                )
                
                new_metric = self._benchmark_operation(
                    operation=operation,
                    implementation="new",
                    func=lambda: self._generate_features(data.copy(), use_new=True),
                    metadata=test_case
                )
                
                comparison = PerformanceComparison(operation, legacy_metric, new_metric)
                results['comparisons'].append(comparison)
                
                progress.update(task, advance=1)
        
        return results
    
    def _benchmark_batch_ops(self) -> Dict[str, Any]:
        """Benchmark batch operations performance."""
        results = {
            'comparisons': [],
            'metrics': []
        }
        
        # Create multiple datasets for batch operations
        batch_sizes = [5, 10, 20]
        
        for batch_size in batch_sizes:
            # Create datasets
            dataset_names = []
            for i in range(batch_size):
                data = pd.DataFrame({
                    'id': range(100),
                    'value': np.random.randn(100) * i
                })
                csv_path = self.test_dir / f"batch_{batch_size}_{i}.csv"
                data.to_csv(csv_path, index=False)
                
                dataset_name = f"batch_perf_{batch_size}_{i}"
                dataset_names.append(dataset_name)
                
                # Register quickly with legacy
                self._register_dataset(dataset_name, str(csv_path), use_new=False)
            
            # Benchmark batch export
            export_dir = self.test_dir / f"batch_exports_{batch_size}"
            export_dir.mkdir(exist_ok=True)
            
            operation = f"batch_export_{batch_size}"
            
            legacy_metric = self._benchmark_operation(
                operation=operation,
                implementation="legacy",
                func=lambda: self._batch_export(
                    f"batch_perf_{batch_size}_*",
                    str(export_dir),
                    use_new=False
                ),
                metadata={'batch_size': batch_size}
            )
            
            new_metric = self._benchmark_operation(
                operation=operation,
                implementation="new",
                func=lambda: self._batch_export(
                    f"batch_perf_{batch_size}_*",
                    str(export_dir),
                    use_new=True
                ),
                metadata={'batch_size': batch_size}
            )
            
            comparison = PerformanceComparison(operation, legacy_metric, new_metric)
            results['comparisons'].append(comparison)
        
        return results
    
    def _benchmark_memory(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        results = {
            'comparisons': [],
            'metrics': []
        }
        
        # Test memory usage for different operations
        memory_tests = [
            {
                'name': 'large_dataset_load',
                'func': self._test_large_dataset_memory,
                'size': 50000
            },
            {
                'name': 'feature_generation_memory',
                'func': self._test_feature_memory,
                'size': 10000
            },
            {
                'name': 'concurrent_operations_memory',
                'func': self._test_concurrent_memory,
                'threads': 5
            }
        ]
        
        for test in memory_tests:
            operation = f"memory_{test['name']}"
            
            # Force garbage collection before each test
            gc.collect()
            
            legacy_metric = self._benchmark_operation(
                operation=operation,
                implementation="legacy",
                func=lambda: test['func'](use_new=False, **{k: v for k, v in test.items() if k not in ['name', 'func']}),
                metadata=test
            )
            
            gc.collect()
            
            new_metric = self._benchmark_operation(
                operation=operation,
                implementation="new",
                func=lambda: test['func'](use_new=True, **{k: v for k, v in test.items() if k not in ['name', 'func']}),
                metadata=test
            )
            
            comparison = PerformanceComparison(operation, legacy_metric, new_metric)
            results['comparisons'].append(comparison)
        
        return results
    
    def _benchmark_concurrency(self) -> Dict[str, Any]:
        """Benchmark concurrent operation performance."""
        results = {
            'comparisons': [],
            'metrics': []
        }
        
        # Test concurrent scenarios
        concurrency_tests = [
            {
                'name': 'parallel_queries',
                'threads': 10,
                'operations': 50
            },
            {
                'name': 'concurrent_registrations',
                'threads': 3,
                'operations': 9
            }
        ]
        
        for test in concurrency_tests:
            operation = f"concurrent_{test['name']}"
            
            legacy_metric = self._benchmark_operation(
                operation=operation,
                implementation="legacy",
                func=lambda: self._test_concurrent_operations(
                    use_new=False,
                    threads=test['threads'],
                    operations=test['operations']
                ),
                metadata=test
            )
            
            new_metric = self._benchmark_operation(
                operation=operation,
                implementation="new",
                func=lambda: self._test_concurrent_operations(
                    use_new=True,
                    threads=test['threads'],
                    operations=test['operations']
                ),
                metadata=test
            )
            
            comparison = PerformanceComparison(operation, legacy_metric, new_metric)
            results['comparisons'].append(comparison)
        
        return results
    
    def _benchmark_large_datasets(self) -> Dict[str, Any]:
        """Benchmark large dataset handling."""
        results = {
            'comparisons': [],
            'metrics': []
        }
        
        # Test progressively larger datasets
        sizes = [10000, 50000, 100000]
        
        console.print("[dim]Note: Large dataset benchmarks may take several minutes...[/dim]")
        
        for size in sizes:
            # Create large dataset
            console.print(f"  Testing {size:,} rows...")
            
            data = pd.DataFrame({
                'id': range(size),
                'value1': np.random.randn(size),
                'value2': np.random.exponential(1, size),
                'value3': np.random.uniform(0, 100, size),
                'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], size)
            })
            
            csv_path = self.test_dir / f"large_{size}.csv"
            data.to_csv(csv_path, index=False)
            
            operation = f"large_dataset_{size}"
            
            # Skip feature generation for very large datasets
            generate_features = size <= 50000
            
            legacy_metric = self._benchmark_operation(
                operation=operation,
                implementation="legacy",
                func=lambda: self._register_dataset(
                    f"large_legacy_{size}",
                    str(csv_path),
                    use_new=False,
                    generate_features=generate_features
                ),
                metadata={'dataset_size': size, 'with_features': generate_features}
            )
            
            new_metric = self._benchmark_operation(
                operation=operation,
                implementation="new",
                func=lambda: self._register_dataset(
                    f"large_new_{size}",
                    str(csv_path),
                    use_new=True,
                    generate_features=generate_features
                ),
                metadata={'dataset_size': size, 'with_features': generate_features}
            )
            
            comparison = PerformanceComparison(operation, legacy_metric, new_metric)
            results['comparisons'].append(comparison)
            
            # Show results
            speedup_color = "green" if comparison.speedup >= 1 else "red"
            memory_color = "green" if comparison.memory_ratio <= 1.2 else "red"
            
            console.print(
                f"    Speed: [{speedup_color}]{comparison.speedup:.2f}x[/{speedup_color}] | "
                f"Memory: [{memory_color}]{comparison.memory_ratio:.2f}x[/{memory_color}]"
            )
        
        return results
    
    def _benchmark_workflows(self) -> Dict[str, Any]:
        """Benchmark end-to-end workflows."""
        results = {
            'comparisons': [],
            'metrics': []
        }
        
        # Define realistic workflows
        workflows = [
            {
                'name': 'data_science_workflow',
                'steps': [
                    'create_dataset',
                    'register',
                    'generate_features',
                    'compute_stats',
                    'export'
                ]
            },
            {
                'name': 'batch_processing_workflow',
                'steps': [
                    'create_multiple_datasets',
                    'batch_register',
                    'batch_stats',
                    'batch_export'
                ]
            }
        ]
        
        for workflow in workflows:
            operation = f"workflow_{workflow['name']}"
            
            legacy_metric = self._benchmark_operation(
                operation=operation,
                implementation="legacy",
                func=lambda: self._run_workflow(workflow, use_new=False),
                metadata=workflow
            )
            
            new_metric = self._benchmark_operation(
                operation=operation,
                implementation="new",
                func=lambda: self._run_workflow(workflow, use_new=True),
                metadata=workflow
            )
            
            comparison = PerformanceComparison(operation, legacy_metric, new_metric)
            results['comparisons'].append(comparison)
        
        return results
    
    # Helper methods
    def _benchmark_operation(
        self,
        operation: str,
        implementation: str,
        func: Callable,
        metadata: Dict[str, Any] = None
    ) -> PerformanceMetric:
        """Benchmark a single operation."""
        # Clear caches
        self._clear_all_caches()
        
        # Measure initial state
        gc.collect()
        memory_before = self._process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = self._process.cpu_percent(interval=0.1)
        
        # Run operation
        start_time = time.time()
        success = True
        error = None
        
        try:
            func()
        except Exception as e:
            success = False
            error = str(e)
            logger.error(f"Benchmark operation {operation} failed: {e}")
        
        # Measure final state
        duration = time.time() - start_time
        memory_after = self._process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = self._process.cpu_percent(interval=0.1) - cpu_before
        
        return PerformanceMetric(
            operation=operation,
            implementation=implementation,
            duration=duration,
            memory_before=memory_before,
            memory_after=memory_after,
            cpu_percent=cpu_percent,
            success=success,
            error=error,
            metadata=metadata or {}
        )
    
    def _register_dataset(
        self,
        name: str,
        path: str,
        use_new: bool,
        generate_features: bool = False
    ):
        """Register a dataset with specified implementation."""
        feature_flags.set("use_new_dataset", use_new)
        clear_dataset_cache()
        
        registrar = get_dataset_registrar()
        registrar.register(
            name=name,
            path=path,
            force=True,
            generate_features=generate_features
        )
        
        self._test_datasets.append(name)
    
    def _get_dataset_info(self, dataset_name: str):
        """Get dataset info."""
        manager = get_dataset_manager()
        return manager.get_dataset_info(dataset_name)
    
    def _get_dataset_stats(self, dataset_name: str):
        """Get dataset statistics."""
        manager = get_dataset_manager()
        return manager.get_dataset_stats(dataset_name)
    
    def _list_datasets(self):
        """List all datasets."""
        cli = get_dataset_commands()
        return cli.list_datasets(limit=50)
    
    def _search_datasets(self, pattern: str):
        """Search for datasets."""
        cli = get_dataset_commands()
        return cli.search(pattern=pattern)
    
    def _generate_features(self, data: pd.DataFrame, use_new: bool):
        """Generate features for data."""
        feature_flags.set("use_new_features", use_new)
        clear_feature_cache()
        
        gen = get_feature_generator()
        return gen.generate_features(data, {
            'id_columns': ['id'] if 'id' in data.columns else [],
            'categorical_columns': [col for col in data.columns if data[col].dtype == 'object']
        })
    
    def _batch_export(self, pattern: str, output_dir: str, use_new: bool):
        """Perform batch export."""
        feature_flags.set("use_new_cli", use_new)
        clear_cli_cache()
        
        batch = get_batch_commands()
        return batch.export(
            pattern=pattern,
            output_dir=output_dir,
            format="csv"
        )
    
    def _create_feature_test_data(
        self,
        rows: int,
        numeric_cols: int,
        categorical_cols: int
    ) -> pd.DataFrame:
        """Create test data for feature generation."""
        data = {'id': range(rows)}
        
        # Add numeric columns
        for i in range(numeric_cols):
            data[f'num_{i}'] = np.random.randn(rows)
        
        # Add categorical columns
        for i in range(categorical_cols):
            n_categories = min(10, rows // 10)
            data[f'cat_{i}'] = np.random.choice(
                [f'C{j}' for j in range(n_categories)],
                rows
            )
        
        return pd.DataFrame(data)
    
    def _test_large_dataset_memory(self, use_new: bool, size: int):
        """Test memory usage with large dataset."""
        data = pd.DataFrame({
            'id': range(size),
            'value': np.random.randn(size)
        })
        
        csv_path = self.test_dir / f"memory_test_{size}.csv"
        data.to_csv(csv_path, index=False)
        
        self._register_dataset(
            f"memory_test_{use_new}_{size}",
            str(csv_path),
            use_new=use_new
        )
    
    def _test_feature_memory(self, use_new: bool, size: int):
        """Test memory usage during feature generation."""
        data = self._create_feature_test_data(size, 5, 3)
        self._generate_features(data, use_new)
    
    def _test_concurrent_memory(self, use_new: bool, threads: int):
        """Test memory usage with concurrent operations."""
        def query_operation(i):
            try:
                self._list_datasets()
                return True
            except Exception:
                return False
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(query_operation, i) for i in range(threads * 5)]
            results = [f.result() for f in as_completed(futures)]
        
        return sum(results)
    
    def _test_concurrent_operations(
        self,
        use_new: bool,
        threads: int,
        operations: int
    ):
        """Test concurrent operations."""
        feature_flags.set("use_new_dataset", use_new)
        feature_flags.set("use_new_cli", use_new)
        
        def operation(i):
            try:
                # Mix of operations
                if i % 3 == 0:
                    self._list_datasets()
                elif i % 3 == 1:
                    self._search_datasets("test")
                else:
                    if self._test_datasets:
                        self._get_dataset_info(self._test_datasets[0])
                return True
            except Exception:
                return False
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(operation, i) for i in range(operations)]
            results = [f.result() for f in as_completed(futures)]
        
        return sum(results)
    
    def _run_workflow(self, workflow: Dict[str, Any], use_new: bool):
        """Run a complete workflow."""
        # Set all flags
        feature_flags.set("use_new_config", use_new)
        feature_flags.set("use_new_storage", use_new)
        feature_flags.set("use_new_features", use_new)
        feature_flags.set("use_new_dataset", use_new)
        feature_flags.set("use_new_cli", use_new)
        
        workflow_name = workflow['name']
        
        if workflow_name == 'data_science_workflow':
            # Create and process single dataset
            data = pd.DataFrame({
                'id': range(1000),
                'feature1': np.random.randn(1000),
                'feature2': np.random.exponential(1, 1000),
                'target': np.random.randint(0, 2, 1000)
            })
            
            csv_path = self.test_dir / f"workflow_{use_new}.csv"
            data.to_csv(csv_path, index=False)
            
            # Register
            registrar = get_dataset_registrar()
            registrar.register(
                name=f"workflow_{use_new}",
                path=str(csv_path),
                target="target",
                force=True,
                generate_features=True
            )
            
            # Get stats
            manager = get_dataset_manager()
            stats = manager.get_dataset_stats(f"workflow_{use_new}")
            
            # Export
            cli = get_dataset_commands()
            export_dir = self.test_dir / f"workflow_export_{use_new}"
            export_dir.mkdir(exist_ok=True)
            
            cli.export(
                name=f"workflow_{use_new}",
                output_dir=str(export_dir),
                format="parquet"
            )
            
        elif workflow_name == 'batch_processing_workflow':
            # Create multiple datasets
            for i in range(5):
                data = pd.DataFrame({
                    'id': range(200),
                    'value': np.random.randn(200) * i
                })
                csv_path = self.test_dir / f"batch_workflow_{use_new}_{i}.csv"
                data.to_csv(csv_path, index=False)
                
                registrar = get_dataset_registrar()
                registrar.register(
                    name=f"batch_workflow_{use_new}_{i}",
                    path=str(csv_path),
                    force=True
                )
            
            # Batch operations
            batch = get_batch_commands()
            batch.stats(pattern=f"batch_workflow_{use_new}_*")
    
    def _run_with_impl(self, use_new: bool, func: Callable):
        """Run function with specified implementation."""
        # Set appropriate flags
        feature_flags.set("use_new_dataset", use_new)
        feature_flags.set("use_new_cli", use_new)
        clear_dataset_cache()
        clear_cli_cache()
        
        # Handle functions with/without arguments
        if hasattr(func, '__self__'):
            # Bound method
            return func()
        else:
            # Check if function needs dataset argument
            import inspect
            sig = inspect.signature(func)
            if len(sig.parameters) > 0:
                # Needs dataset name
                if self._test_datasets:
                    return func(self._test_datasets[0])
                else:
                    # Create one quickly
                    self._create_benchmark_datasets()
                    return func(self._test_datasets[0])
            else:
                # No arguments needed
                return func()
    
    def _warmup_caches(self):
        """Warm up caches before benchmarking."""
        try:
            # Simple operations to warm up
            manager = get_dataset_manager()
            cli = get_dataset_commands()
            
            # Don't fail if no datasets
            try:
                cli.list_datasets(limit=1)
            except Exception:
                pass
        except Exception:
            pass
    
    def _create_benchmark_datasets(self):
        """Create datasets for benchmarking."""
        for i in range(3):
            data = pd.DataFrame({
                'id': range(500),
                'value': np.random.randn(500) * i
            })
            csv_path = self.test_dir / f"benchmark_{i}.csv"
            data.to_csv(csv_path, index=False)
            
            dataset_name = f"perf_benchmark_{i}"
            self._register_dataset(dataset_name, str(csv_path), use_new=False)
    
    def _clear_all_caches(self):
        """Clear all component caches."""
        clear_storage_cache()
        clear_feature_cache()
        clear_dataset_cache()
        clear_cli_cache()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': {
                'system': self._process.name(),
                'python_version': str(self._process.exe())
            }
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary."""
        comparisons = results['comparisons']
        
        if not comparisons:
            return {}
        
        # Calculate overall metrics
        speedups = [c.speedup for c in comparisons if c.speedup > 0]
        memory_ratios = [c.memory_ratio for c in comparisons if c.memory_ratio > 0]
        
        summary = {
            'total_operations': len(comparisons),
            'average_speedup': np.mean(speedups) if speedups else 0,
            'median_speedup': np.median(speedups) if speedups else 0,
            'average_memory_ratio': np.mean(memory_ratios) if memory_ratios else 1,
            'regression_count': len(results['regressions']),
            'improvement_count': sum(1 for c in comparisons if c.speedup > 1.1),
            'neutral_count': sum(1 for c in comparisons if 0.9 <= c.speedup <= 1.1)
        }
        
        # Find best and worst
        if comparisons:
            best = max(comparisons, key=lambda c: c.speedup)
            worst = min(comparisons, key=lambda c: c.speedup)
            
            summary['best_improvement'] = {
                'operation': best.operation,
                'speedup': best.speedup
            }
            summary['worst_regression'] = {
                'operation': worst.operation,
                'speedup': worst.speedup
            }
        
        return summary
    
    def _display_benchmark_results(self, results: Dict[str, Any]):
        """Display benchmark results summary."""
        console.print("\n[bold]Performance Benchmark Summary[/bold]")
        console.print("=" * 60)
        
        summary = results['summary']
        
        # Overall performance
        avg_speedup = summary.get('average_speedup', 0)
        speedup_color = "green" if avg_speedup >= 1 else "red"
        
        console.print(f"\nAverage Speedup: [{speedup_color}]{avg_speedup:.2f}x[/{speedup_color}]")
        console.print(f"Median Speedup: {summary.get('median_speedup', 0):.2f}x")
        console.print(f"Average Memory Ratio: {summary.get('average_memory_ratio', 1):.2f}x")
        
        # Summary table
        table = Table(title="Performance Overview")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="white")
        table.add_column("Percentage", style="yellow")
        
        total = summary.get('total_operations', 0)
        
        table.add_row(
            "Improvements",
            str(summary.get('improvement_count', 0)),
            f"{summary.get('improvement_count', 0) / total * 100:.1f}%" if total > 0 else "0%"
        )
        table.add_row(
            "Neutral",
            str(summary.get('neutral_count', 0)),
            f"{summary.get('neutral_count', 0) / total * 100:.1f}%" if total > 0 else "0%"
        )
        table.add_row(
            "Regressions",
            str(summary.get('regression_count', 0)),
            f"{summary.get('regression_count', 0) / total * 100:.1f}%" if total > 0 else "0%"
        )
        
        console.print(table)
        
        # Best and worst
        if 'best_improvement' in summary:
            best = summary['best_improvement']
            console.print(
                f"\n[green]Best Improvement:[/green] {best['operation']} "
                f"({best['speedup']:.2f}x faster)"
            )
        
        if 'worst_regression' in summary:
            worst = summary['worst_regression']
            console.print(
                f"[red]Worst Regression:[/red] {worst['operation']} "
                f"({worst['speedup']:.2f}x)"
            )
        
        # Regressions detail
        if results['regressions']:
            console.print("\n[bold red]Performance Regressions:[/bold red]")
            for reg in results['regressions'][:5]:
                console.print(
                    f"  • {reg['operation']}: "
                    f"{reg['speedup']:.2f}x speed, "
                    f"{reg['memory_ratio']:.2f}x memory"
                )
            if len(results['regressions']) > 5:
                console.print(f"  ... and {len(results['regressions']) - 5} more")
        
        # Detailed results by suite
        console.print("\n[bold]Results by Suite:[/bold]")
        suite_table = Table()
        suite_table.add_column("Suite", style="cyan")
        suite_table.add_column("Tests", style="white")
        suite_table.add_column("Avg Speedup", style="yellow")
        suite_table.add_column("Status", style="white")
        
        for suite_name, suite_data in results['suites'].items():
            comparisons = suite_data.get('comparisons', [])
            if comparisons:
                speedups = [c.speedup for c in comparisons if c.speedup > 0]
                avg_speedup = np.mean(speedups) if speedups else 0
                
                status_icon = "✓" if avg_speedup >= 0.9 else "✗"
                status_color = "green" if avg_speedup >= 0.9 else "red"
                
                suite_table.add_row(
                    suite_name,
                    str(len(comparisons)),
                    f"{avg_speedup:.2f}x",
                    f"[{status_color}]{status_icon}[/{status_color}]"
                )
        
        console.print(suite_table)
    
    def _save_benchmark_report(self, results: Dict[str, Any]):
        """Save detailed benchmark report."""
        report_path = self.test_dir / "performance_benchmark_report.json"
        
        # Convert to serializable format
        serializable_results = results.copy()
        
        # Convert comparisons
        serializable_results['comparisons'] = [
            c.to_dict() for c in results['comparisons']
        ]
        
        # Convert suite comparisons
        for suite_name, suite_data in serializable_results['suites'].items():
            if 'comparisons' in suite_data:
                suite_data['comparisons'] = [
                    c.to_dict() for c in suite_data['comparisons']
                ]
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        console.print(f"\n[dim]Detailed benchmark report saved to: {report_path}[/dim]")
        
        # Also save a CSV for easy analysis
        csv_path = self.test_dir / "performance_results.csv"
        
        rows = []
        for comp_dict in serializable_results['comparisons']:
            rows.append({
                'operation': comp_dict['operation'],
                'legacy_duration': comp_dict['legacy']['duration'],
                'new_duration': comp_dict['new']['duration'],
                'speedup': comp_dict['speedup'],
                'legacy_memory': comp_dict['legacy']['memory_used'],
                'new_memory': comp_dict['new']['memory_used'],
                'memory_ratio': comp_dict['memory_ratio'],
                'is_regression': comp_dict['is_regression']
            })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            console.print(f"[dim]Results CSV saved to: {csv_path}[/dim]")
    
    def _cleanup_test_data(self):
        """Clean up test data."""
        try:
            # Clean up test datasets
            for dataset_name in self._test_datasets:
                try:
                    # Try with both implementations
                    for use_new in [False, True]:
                        feature_flags.set("use_new_dataset", use_new)
                        manager = get_dataset_manager()
                        if hasattr(manager, 'remove_dataset'):
                            manager.remove_dataset(dataset_name, force=True)
                except Exception:
                    pass
            
            # Clean up test directory
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
            
            logger.info("Performance benchmark cleanup completed")
        except Exception as e:
            logger.warning(f"Performance benchmark cleanup failed: {e}")