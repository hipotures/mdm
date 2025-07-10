"""
Feature engineering comparison testing framework.

This module provides tools for comparing the behavior and performance of
legacy vs new feature engineering implementations.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from mdm.core import feature_flags
from mdm.adapters.feature_manager import get_feature_generator, clear_feature_cache
from mdm.interfaces.features import IFeatureGenerator

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class FeatureTestResult:
    """Result of a feature engineering test."""
    test_name: str
    legacy_features: int
    new_features: int
    common_features: int
    legacy_time: float
    new_time: float
    values_match: bool
    error: Optional[str] = None


class FeatureComparisonTester:
    """Compares legacy and new feature engineering implementations."""
    
    def __init__(self):
        """Initialize tester."""
        self.results: List[FeatureTestResult] = []
        self._test_data_cache = {}
        logger.info("Initialized FeatureComparisonTester")
    
    def run_all_tests(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Run all comparison tests.
        
        Args:
            sample_size: Number of rows in test data
            
        Returns:
            Test results summary
        """
        console.print(Panel.fit(
            "[bold cyan]Feature Engineering Comparison Tests[/bold cyan]\n\n"
            f"Comparing legacy vs new implementations with {sample_size} row datasets",
            title="Feature Comparison"
        ))
        
        # Define test suite
        tests = [
            ("Basic Feature Generation", self.test_basic_generation),
            ("Numeric Features", self.test_numeric_features),
            ("Categorical Features", self.test_categorical_features),
            ("Datetime Features", self.test_datetime_features),
            ("Text Features", self.test_text_features),
            ("Interaction Features", self.test_interaction_features),
            ("Feature Importance", self.test_feature_importance),
            ("Missing Data Handling", self.test_missing_data),
            ("Performance - Small Dataset", lambda: self.test_performance(100)),
            ("Performance - Medium Dataset", lambda: self.test_performance(10000)),
            ("Performance - Large Dataset", lambda: self.test_performance(100000)),
            ("Memory Efficiency", self.test_memory_efficiency),
        ]
        
        # Run tests with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for test_name, test_func in tests:
                task = progress.add_task(f"Running: {test_name}...", total=None)
                
                try:
                    result = self._run_comparison_test(test_name, test_func)
                    self.results.append(result)
                    
                    if result.values_match:
                        console.print(f"✓ {test_name} [green]PASS[/green]")
                    else:
                        console.print(f"✗ {test_name} [red]FAIL[/red]")
                        if result.error:
                            console.print(f"  Error: {result.error}")
                            
                except Exception as e:
                    console.print(f"✗ {test_name} [red]ERROR[/red]")
                    console.print(f"  Exception: {str(e)}")
                    self.results.append(FeatureTestResult(
                        test_name=test_name,
                        legacy_features=0,
                        new_features=0,
                        common_features=0,
                        legacy_time=0,
                        new_time=0,
                        values_match=False,
                        error=str(e)
                    ))
                
                progress.remove_task(task)
        
        # Generate summary
        summary = self._generate_summary()
        self._display_summary(summary)
        
        return summary
    
    def test_basic_generation(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Test basic feature generation."""
        # Create test data
        test_data = self._create_test_data(1000)
        
        # Legacy generation
        legacy_gen = self._get_generator(use_new=False)
        legacy_start = time.time()
        legacy_features = legacy_gen.generate_features(
            data=test_data,
            target_column="target",
            id_columns=["id"]
        )
        legacy_time = time.time() - legacy_start
        
        # New generation
        new_gen = self._get_generator(use_new=True)
        new_start = time.time()
        new_features = new_gen.generate_features(
            data=test_data,
            target_column="target",
            id_columns=["id"]
        )
        new_time = time.time() - new_start
        
        return (
            {"features": legacy_features, "time": legacy_time},
            {"features": new_features, "time": new_time}
        )
    
    def test_numeric_features(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Test numeric feature generation."""
        # Create numeric-heavy test data
        test_data = pd.DataFrame({
            "value1": np.random.randn(500),
            "value2": np.random.exponential(2, 500),
            "value3": np.random.uniform(0, 100, 500),
            "count": np.random.poisson(5, 500)
        })
        
        # Test both implementations
        legacy_gen = self._get_generator(use_new=False)
        legacy_result = legacy_gen.generate_numeric_features(test_data)
        
        new_gen = self._get_generator(use_new=True)
        new_result = new_gen.generate_numeric_features(test_data)
        
        return (
            {"features": legacy_result, "time": 0},
            {"features": new_result, "time": 0}
        )
    
    def test_categorical_features(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Test categorical feature generation."""
        # Create categorical test data
        test_data = pd.DataFrame({
            "category": np.random.choice(["A", "B", "C", "D"], 500),
            "subcategory": np.random.choice(["X", "Y", "Z"], 500),
            "rare_category": np.random.choice(
                ["common"] * 450 + ["rare1", "rare2", "rare3"], 500
            )
        })
        
        # Test both implementations
        legacy_gen = self._get_generator(use_new=False)
        legacy_result = legacy_gen.generate_categorical_features(test_data)
        
        new_gen = self._get_generator(use_new=True)
        new_result = new_gen.generate_categorical_features(test_data)
        
        return (
            {"features": legacy_result, "time": 0},
            {"features": new_result, "time": 0}
        )
    
    def test_datetime_features(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Test datetime feature generation."""
        # Create datetime test data
        test_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=500, freq="H"),
            "timestamp": pd.date_range("2024-01-01", periods=500, freq="15min"),
        })
        
        # Test both implementations
        legacy_gen = self._get_generator(use_new=False)
        legacy_result = legacy_gen.generate_datetime_features(test_data)
        
        new_gen = self._get_generator(use_new=True)
        new_result = new_gen.generate_datetime_features(test_data)
        
        return (
            {"features": legacy_result, "time": 0},
            {"features": new_result, "time": 0}
        )
    
    def test_text_features(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Test text feature generation."""
        # Create text test data
        test_data = pd.DataFrame({
            "description": [
                "This is a short text.",
                "A longer description with more words and special characters!",
                "Email: test@example.com, URL: https://example.com",
                "Numbers: 123-456-7890, Date: 2024-01-01",
                "Mixed CASE text WITH various Features."
            ] * 100
        })
        
        # Test both implementations
        legacy_gen = self._get_generator(use_new=False)
        legacy_result = legacy_gen.generate_text_features(test_data)
        
        new_gen = self._get_generator(use_new=True)
        new_result = new_gen.generate_text_features(test_data)
        
        return (
            {"features": legacy_result, "time": 0},
            {"features": new_result, "time": 0}
        )
    
    def test_interaction_features(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Test interaction feature generation."""
        # Create test data for interactions
        test_data = pd.DataFrame({
            "num1": np.random.randn(500),
            "num2": np.random.randn(500),
            "cat1": np.random.choice(["A", "B"], 500),
            "cat2": np.random.choice(["X", "Y"], 500)
        })
        
        # Test both implementations
        legacy_gen = self._get_generator(use_new=False)
        legacy_result = legacy_gen.generate_interaction_features(test_data)
        
        new_gen = self._get_generator(use_new=True)
        new_result = new_gen.generate_interaction_features(test_data)
        
        return (
            {"features": legacy_result, "time": 0},
            {"features": new_result, "time": 0}
        )
    
    def test_feature_importance(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Test feature importance calculation."""
        # Create test data with known relationships
        n = 1000
        test_features = pd.DataFrame({
            "correlated": np.random.randn(n),
            "noise": np.random.randn(n),
            "categorical": np.random.choice(["A", "B", "C"], n)
        })
        # Target correlated with first feature
        target = test_features["correlated"] * 2 + np.random.randn(n) * 0.1
        
        # Test both implementations
        legacy_gen = self._get_generator(use_new=False)
        legacy_importance = legacy_gen.get_feature_importance(
            test_features, target, "regression"
        )
        
        new_gen = self._get_generator(use_new=True)
        new_importance = new_gen.get_feature_importance(
            test_features, target, "regression"
        )
        
        return (
            {"features": legacy_importance, "time": 0},
            {"features": new_importance, "time": 0}
        )
    
    def test_missing_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Test handling of missing data."""
        # Create test data with missing values
        test_data = pd.DataFrame({
            "complete": np.random.randn(500),
            "partial": np.random.randn(500),
            "mostly_missing": np.random.randn(500)
        })
        
        # Add missing values
        test_data.loc[50:150, "partial"] = np.nan
        test_data.loc[100:400, "mostly_missing"] = np.nan
        
        # Test both implementations
        legacy_gen = self._get_generator(use_new=False)
        legacy_result = legacy_gen.generate_features(test_data)
        
        new_gen = self._get_generator(use_new=True)
        new_result = new_gen.generate_features(test_data)
        
        return (
            {"features": legacy_result, "time": 0},
            {"features": new_result, "time": 0}
        )
    
    def test_performance(self, n_rows: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Test performance with specified dataset size."""
        # Create test data
        test_data = self._create_test_data(n_rows)
        
        # Time legacy implementation
        legacy_gen = self._get_generator(use_new=False)
        legacy_start = time.time()
        legacy_features = legacy_gen.generate_features(test_data)
        legacy_time = time.time() - legacy_start
        
        # Time new implementation
        new_gen = self._get_generator(use_new=True)
        new_start = time.time()
        new_features = new_gen.generate_features(test_data)
        new_time = time.time() - new_start
        
        return (
            {"features": legacy_features, "time": legacy_time},
            {"features": new_features, "time": new_time}
        )
    
    def test_memory_efficiency(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Test memory efficiency of implementations."""
        import psutil
        import os
        
        # Create large test data
        test_data = self._create_test_data(50000)
        
        # Measure legacy memory usage
        process = psutil.Process(os.getpid())
        legacy_gen = self._get_generator(use_new=False)
        
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        legacy_features = legacy_gen.generate_features(test_data)
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        legacy_mem_delta = mem_after - mem_before
        
        # Clear memory
        del legacy_features
        import gc
        gc.collect()
        
        # Measure new memory usage
        new_gen = self._get_generator(use_new=True)
        
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        new_features = new_gen.generate_features(test_data)
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        new_mem_delta = mem_after - mem_before
        
        return (
            {"features": pd.DataFrame(), "time": 0, "memory_mb": legacy_mem_delta},
            {"features": pd.DataFrame(), "time": 0, "memory_mb": new_mem_delta}
        )
    
    def _create_test_data(self, n_rows: int) -> pd.DataFrame:
        """Create comprehensive test dataset."""
        if n_rows in self._test_data_cache:
            return self._test_data_cache[n_rows].copy()
        
        np.random.seed(42)  # For reproducibility
        
        data = pd.DataFrame({
            "id": range(n_rows),
            "numeric1": np.random.randn(n_rows),
            "numeric2": np.random.exponential(2, n_rows),
            "categorical1": np.random.choice(["A", "B", "C", "D"], n_rows),
            "categorical2": np.random.choice(["X", "Y", "Z"], n_rows),
            "datetime1": pd.date_range("2024-01-01", periods=n_rows, freq="H"),
            "text1": [f"Sample text {i}" for i in range(n_rows)],
            "target": np.random.randn(n_rows)
        })
        
        # Add some missing values
        data.loc[np.random.choice(n_rows, size=n_rows//10, replace=False), "numeric1"] = np.nan
        
        self._test_data_cache[n_rows] = data
        return data.copy()
    
    def _get_generator(self, use_new: bool) -> IFeatureGenerator:
        """Get feature generator with feature flag set."""
        # Clear cache to ensure fresh instance
        clear_feature_cache()
        
        # Set feature flag
        original_flag = feature_flags.get("use_new_features", False)
        feature_flags.set("use_new_features", use_new)
        
        try:
            generator = get_feature_generator()
            return generator
        finally:
            # Restore original flag
            feature_flags.set("use_new_features", original_flag)
    
    def _run_comparison_test(self, test_name: str, test_func) -> FeatureTestResult:
        """Run a single comparison test."""
        try:
            # Run test function
            legacy_result, new_result = test_func()
            
            # Extract results
            legacy_features = legacy_result.get("features", pd.DataFrame())
            new_features = new_result.get("features", pd.DataFrame())
            
            # Count features (handle both DataFrame and dict results)
            if isinstance(legacy_features, pd.DataFrame):
                legacy_count = len(legacy_features.columns)
                new_count = len(new_features.columns)
                
                # Find common features
                common_cols = set(legacy_features.columns).intersection(set(new_features.columns))
                common_count = len(common_cols)
                
                # Check if values match for common features
                values_match = True
                for col in common_cols:
                    if col in legacy_features.columns and col in new_features.columns:
                        if pd.api.types.is_numeric_dtype(legacy_features[col]):
                            if not np.allclose(
                                legacy_features[col].fillna(0),
                                new_features[col].fillna(0),
                                rtol=1e-5,
                                equal_nan=True
                            ):
                                values_match = False
                                break
                        else:
                            if not legacy_features[col].equals(new_features[col]):
                                values_match = False
                                break
            else:
                # Handle non-DataFrame results (like feature importance)
                legacy_count = len(legacy_features) if hasattr(legacy_features, "__len__") else 0
                new_count = len(new_features) if hasattr(new_features, "__len__") else 0
                common_count = 0
                values_match = True  # Can't compare values for non-DataFrame
            
            return FeatureTestResult(
                test_name=test_name,
                legacy_features=legacy_count,
                new_features=new_count,
                common_features=common_count,
                legacy_time=legacy_result.get("time", 0),
                new_time=new_result.get("time", 0),
                values_match=values_match,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            raise
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.values_match and not r.error)
        failed_tests = total_tests - passed_tests
        
        # Calculate metrics
        total_legacy_time = sum(r.legacy_time for r in self.results)
        total_new_time = sum(r.new_time for r in self.results)
        
        performance_ratio = (
            total_new_time / total_legacy_time if total_legacy_time > 0 else 1.0
        )
        
        # Feature coverage
        feature_coverage = []
        for result in self.results:
            if result.legacy_features > 0:
                coverage = result.common_features / result.legacy_features
                feature_coverage.append(coverage)
        
        avg_feature_coverage = np.mean(feature_coverage) if feature_coverage else 0
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_legacy_time": total_legacy_time,
            "total_new_time": total_new_time,
            "performance_ratio": performance_ratio,
            "avg_feature_coverage": avg_feature_coverage,
            "results": self.results,
            "failed_tests": [r for r in self.results if not r.values_match or r.error]
        }
    
    def _display_summary(self, summary: Dict[str, Any]) -> None:
        """Display test summary."""
        # Create summary panel
        panel_content = f"""
[bold]Feature Engineering Comparison Results[/bold]

Total Tests: {summary['total_tests']}
Passed: [green]{summary['passed']}[/green]
Failed: [red]{summary['failed']}[/red]
Success Rate: [{'green' if summary['success_rate'] >= 90 else 'yellow'}]{summary['success_rate']:.1f}%[/]

Performance Comparison:
Legacy Total Time: {summary['total_legacy_time']:.3f}s
New Total Time: {summary['total_new_time']:.3f}s
Performance Ratio: [{'green' if summary['performance_ratio'] <= 1.1 else 'yellow'}]{summary['performance_ratio']:.2f}x[/]

Average Feature Coverage: [{'green' if summary['avg_feature_coverage'] >= 0.9 else 'yellow'}]{summary['avg_feature_coverage']:.1%}[/]
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
        table.add_column("Legacy Features", style="yellow")
        table.add_column("New Features", style="yellow")
        table.add_column("Common", style="green")
        table.add_column("Coverage", style="green")
        table.add_column("Status", style="green")
        
        for result in self.results:
            coverage = (
                f"{result.common_features / result.legacy_features:.1%}"
                if result.legacy_features > 0 else "N/A"
            )
            status = (
                "[green]PASS[/green]" if result.values_match and not result.error
                else "[red]FAIL[/red]"
            )
            
            table.add_row(
                result.test_name,
                str(result.legacy_features),
                str(result.new_features),
                str(result.common_features),
                coverage,
                status
            )
        
        console.print("\n")
        console.print(table)