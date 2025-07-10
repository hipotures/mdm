"""Dataset registration comparison testing.

Provides tools for comparing old and new dataset registration implementations.
"""
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core import feature_flags
from ..adapters import (
    get_dataset_registrar,
    get_dataset_manager,
    clear_dataset_cache
)
from ..core.exceptions import DatasetError

logger = logging.getLogger(__name__)
console = Console()


class DatasetTestResult:
    """Result of a dataset comparison test."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.legacy_result = None
        self.new_result = None
        self.legacy_time = 0.0
        self.new_time = 0.0
        self.errors = []
        self.warnings = []
        self.details = {}


class DatasetComparisonTester:
    """Tests dataset registration implementations."""
    
    def __init__(self, test_dir: Optional[Path] = None):
        """Initialize tester.
        
        Args:
            test_dir: Directory for test datasets (temp if not provided)
        """
        self.test_dir = test_dir or Path(tempfile.mkdtemp(prefix="mdm_test_"))
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self._test_datasets_created = []
        logger.info(f"Initialized DatasetComparisonTester with dir: {self.test_dir}")
    
    def run_all_tests(self, cleanup: bool = True) -> Dict[str, Any]:
        """Run all comparison tests.
        
        Args:
            cleanup: If True, cleanup test data after running
            
        Returns:
            Test results summary
        """
        console.print(Panel.fit(
            "[bold cyan]Dataset Registration Comparison Tests[/bold cyan]\n\n"
            "Testing legacy vs new implementations",
            title="Dataset Tests"
        ))
        
        tests = [
            ("Basic Registration", self.test_basic_registration),
            ("Kaggle Structure", self.test_kaggle_structure),
            ("Auto-detection", self.test_auto_detection),
            ("Large Dataset", self.test_large_dataset),
            ("Multiple Tables", self.test_multiple_tables),
            ("Feature Generation", self.test_feature_generation),
            ("Error Handling", self.test_error_handling),
            ("Update Operations", self.test_update_operations),
            ("Export/Import", self.test_export_import),
            ("Search Operations", self.test_search_operations),
            ("Performance - Registration", lambda: self.test_registration_performance(1000)),
            ("Performance - Query", lambda: self.test_query_performance()),
        ]
        
        results = {
            'total': len(tests),
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'tests': {},
            'performance_ratio': 0.0
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Running tests...", total=len(tests))
            
            for test_name, test_func in tests:
                progress.update(task, description=f"Running: {test_name}")
                
                try:
                    result = test_func()
                    results['tests'][test_name] = result
                    
                    if result.passed:
                        results['passed'] += 1
                    else:
                        results['failed'] += 1
                    
                    if result.warnings:
                        results['warnings'] += len(result.warnings)
                    
                except Exception as e:
                    logger.error(f"Test {test_name} crashed: {e}")
                    result = DatasetTestResult(test_name)
                    result.errors.append(f"Test crashed: {str(e)}")
                    results['tests'][test_name] = result
                    results['failed'] += 1
                
                progress.update(task, advance=1)
        
        # Calculate overall performance ratio
        perf_times = {
            'legacy': [],
            'new': []
        }
        
        for result in results['tests'].values():
            if result.legacy_time > 0:
                perf_times['legacy'].append(result.legacy_time)
            if result.new_time > 0:
                perf_times['new'].append(result.new_time)
        
        if perf_times['legacy'] and perf_times['new']:
            avg_legacy = np.mean(perf_times['legacy'])
            avg_new = np.mean(perf_times['new'])
            results['performance_ratio'] = avg_new / avg_legacy if avg_legacy > 0 else 1.0
        
        # Display results
        self._display_results(results)
        
        # Cleanup if requested
        if cleanup:
            self.cleanup()
        
        return results
    
    def test_basic_registration(self) -> DatasetTestResult:
        """Test basic dataset registration."""
        result = DatasetTestResult("Basic Registration")
        
        # Create test data
        test_data = pd.DataFrame({
            'id': range(100),
            'value': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        test_file = self.test_dir / "basic_test.csv"
        test_data.to_csv(test_file, index=False)
        
        dataset_name = f"test_basic_{int(time.time())}"
        
        # Test legacy
        legacy_start = time.time()
        try:
            feature_flags.set("use_new_dataset_registration", False)
            registrar = get_dataset_registrar()
            legacy_result = registrar.register(
                name=dataset_name,
                path=str(test_file),
                target="target",
                force=True
            )
            result.legacy_result = legacy_result
            result.legacy_time = time.time() - legacy_start
            self._test_datasets_created.append(dataset_name)
        except Exception as e:
            result.errors.append(f"Legacy registration failed: {e}")
        
        # Test new
        new_dataset_name = f"{dataset_name}_new"
        new_start = time.time()
        try:
            feature_flags.set("use_new_dataset_registration", True)
            registrar = get_dataset_registrar()
            new_result = registrar.register(
                name=new_dataset_name,
                path=str(test_file),
                target="target",
                force=True
            )
            result.new_result = new_result
            result.new_time = time.time() - new_start
            self._test_datasets_created.append(new_dataset_name)
        except Exception as e:
            result.errors.append(f"New registration failed: {e}")
        
        # Compare results
        if result.legacy_result and result.new_result:
            # Both succeeded - compare
            result.passed = True
            
            # Check basic properties
            if result.legacy_result.get('backend') != result.new_result.get('backend'):
                result.warnings.append("Backend mismatch")
            
            # Performance comparison
            if result.new_time > result.legacy_time * 1.5:
                result.warnings.append(f"New implementation slower: {result.new_time/result.legacy_time:.2f}x")
        else:
            result.passed = False
        
        return result
    
    def test_kaggle_structure(self) -> DatasetTestResult:
        """Test Kaggle competition structure detection."""
        result = DatasetTestResult("Kaggle Structure")
        
        # Create Kaggle-like structure
        kaggle_dir = self.test_dir / "kaggle_test"
        kaggle_dir.mkdir(exist_ok=True)
        
        # Create train.csv
        train_data = pd.DataFrame({
            'id': range(1000),
            'feature1': np.random.randn(1000),
            'feature2': np.random.choice(['X', 'Y', 'Z'], 1000),
            'target': np.random.uniform(0, 1, 1000)
        })
        train_data.to_csv(kaggle_dir / "train.csv", index=False)
        
        # Create test.csv (without target)
        test_data = train_data[['id', 'feature1', 'feature2']].iloc[800:]
        test_data.to_csv(kaggle_dir / "test.csv", index=False)
        
        # Create sample_submission.csv
        submission = pd.DataFrame({
            'id': test_data['id'],
            'target': 0.5
        })
        submission.to_csv(kaggle_dir / "sample_submission.csv", index=False)
        
        dataset_name = f"test_kaggle_{int(time.time())}"
        
        # Test both implementations
        for use_new, prefix in [(False, "legacy"), (True, "new")]:
            feature_flags.set("use_new_dataset_registration", use_new)
            test_name = f"{dataset_name}_{prefix}"
            
            try:
                registrar = get_dataset_registrar()
                reg_result = registrar.register(
                    name=test_name,
                    path=str(kaggle_dir),
                    force=True
                )
                
                if prefix == "legacy":
                    result.legacy_result = reg_result
                else:
                    result.new_result = reg_result
                
                self._test_datasets_created.append(test_name)
                
                # Check if Kaggle structure was detected
                manager = get_dataset_manager()
                info = manager.get_dataset_info(test_name)
                
                # Should have detected target from sample_submission
                if info.get('schema', {}).get('target_column') != 'target':
                    result.warnings.append(f"{prefix}: Failed to detect target column")
                
            except Exception as e:
                result.errors.append(f"{prefix} registration failed: {e}")
        
        result.passed = len(result.errors) == 0
        return result
    
    def test_auto_detection(self) -> DatasetTestResult:
        """Test auto-detection capabilities."""
        result = DatasetTestResult("Auto-detection")
        
        # Create test data with obvious patterns
        test_data = pd.DataFrame({
            'customer_id': range(1000, 2000),  # ID column
            'order_date': pd.date_range('2024-01-01', periods=1000, freq='H'),
            'amount': np.random.exponential(100, 1000),
            'status': np.random.choice(['pending', 'completed', 'cancelled'], 1000),
            'is_fraud': np.random.randint(0, 2, 1000)  # Target column
        })
        
        test_file = self.test_dir / "auto_detect_test.csv"
        test_data.to_csv(test_file, index=False)
        
        dataset_name = f"test_auto_{int(time.time())}"
        
        # Test both implementations without specifying target/id
        for use_new, prefix in [(False, "legacy"), (True, "new")]:
            feature_flags.set("use_new_dataset_registration", use_new)
            test_name = f"{dataset_name}_{prefix}"
            
            try:
                registrar = get_dataset_registrar()
                reg_result = registrar.register(
                    name=test_name,
                    path=str(test_file),
                    force=True
                    # Not specifying target or id_columns
                )
                
                self._test_datasets_created.append(test_name)
                
                # Check detections
                manager = get_dataset_manager()
                info = manager.get_dataset_info(test_name)
                schema = info.get('schema', {})
                
                detections = {
                    'id_detected': 'customer_id' in schema.get('id_columns', []),
                    'target_detected': schema.get('target_column') == 'is_fraud',
                    'problem_type': schema.get('problem_type')
                }
                
                if prefix == "legacy":
                    result.legacy_result = detections
                else:
                    result.new_result = detections
                
            except Exception as e:
                result.errors.append(f"{prefix} auto-detection failed: {e}")
        
        # Compare detection results
        if result.legacy_result and result.new_result:
            if result.legacy_result == result.new_result:
                result.passed = True
            else:
                result.warnings.append("Detection results differ between implementations")
                result.details['legacy_detections'] = result.legacy_result
                result.details['new_detections'] = result.new_result
        
        return result
    
    def test_large_dataset(self) -> DatasetTestResult:
        """Test registration of large dataset."""
        result = DatasetTestResult("Large Dataset")
        
        # Create large dataset
        n_rows = 100000
        test_data = pd.DataFrame({
            'id': range(n_rows),
            'numeric1': np.random.randn(n_rows),
            'numeric2': np.random.exponential(2, n_rows),
            'cat1': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
            'cat2': np.random.choice(list('XYZWV'), n_rows),
            'target': np.random.uniform(0, 100, n_rows)
        })
        
        test_file = self.test_dir / "large_test.csv"
        test_data.to_csv(test_file, index=False)
        
        dataset_name = f"test_large_{int(time.time())}"
        
        # Test both implementations
        for use_new, prefix in [(False, "legacy"), (True, "new")]:
            feature_flags.set("use_new_dataset_registration", use_new)
            test_name = f"{dataset_name}_{prefix}"
            
            start_time = time.time()
            try:
                registrar = get_dataset_registrar()
                reg_result = registrar.register(
                    name=test_name,
                    path=str(test_file),
                    target="target",
                    force=True
                )
                
                elapsed = time.time() - start_time
                
                if prefix == "legacy":
                    result.legacy_time = elapsed
                else:
                    result.new_time = elapsed
                
                self._test_datasets_created.append(test_name)
                
                # Verify data integrity
                manager = get_dataset_manager()
                loaded = manager.load_dataset(test_name, limit=100)
                
                if len(loaded) != 100:
                    result.warnings.append(f"{prefix}: Loaded data size mismatch")
                
            except Exception as e:
                result.errors.append(f"{prefix} large dataset failed: {e}")
        
        result.passed = len(result.errors) == 0
        
        # Add performance details
        if result.legacy_time > 0 and result.new_time > 0:
            result.details['performance_ratio'] = result.new_time / result.legacy_time
        
        return result
    
    def test_multiple_tables(self) -> DatasetTestResult:
        """Test dataset with multiple tables."""
        result = DatasetTestResult("Multiple Tables")
        
        # Create directory with multiple files
        multi_dir = self.test_dir / "multi_table"
        multi_dir.mkdir(exist_ok=True)
        
        # Create main data
        main_data = pd.DataFrame({
            'id': range(1000),
            'value': np.random.randn(1000),
            'category_id': np.random.randint(1, 6, 1000)
        })
        main_data.to_csv(multi_dir / "main.csv", index=False)
        
        # Create category lookup
        categories = pd.DataFrame({
            'category_id': range(1, 6),
            'category_name': ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']
        })
        categories.to_csv(multi_dir / "categories.csv", index=False)
        
        # Create metadata
        metadata = pd.DataFrame({
            'key': ['version', 'created', 'author'],
            'value': ['1.0', '2024-01-01', 'test']
        })
        metadata.to_csv(multi_dir / "metadata.csv", index=False)
        
        dataset_name = f"test_multi_{int(time.time())}"
        
        # Test both implementations
        for use_new, prefix in [(False, "legacy"), (True, "new")]:
            feature_flags.set("use_new_dataset_registration", use_new)
            test_name = f"{dataset_name}_{prefix}"
            
            try:
                registrar = get_dataset_registrar()
                reg_result = registrar.register(
                    name=test_name,
                    path=str(multi_dir),
                    force=True
                )
                
                self._test_datasets_created.append(test_name)
                
                # Check tables
                manager = get_dataset_manager()
                info = manager.get_dataset_info(test_name)
                tables = list(info.get('storage', {}).get('tables', {}).keys())
                
                if len(tables) < 3:
                    result.warnings.append(f"{prefix}: Expected 3 tables, got {len(tables)}")
                
                if prefix == "legacy":
                    result.legacy_result = {'n_tables': len(tables), 'tables': tables}
                else:
                    result.new_result = {'n_tables': len(tables), 'tables': tables}
                
            except Exception as e:
                result.errors.append(f"{prefix} multi-table failed: {e}")
        
        result.passed = len(result.errors) == 0
        return result
    
    def test_feature_generation(self) -> DatasetTestResult:
        """Test feature generation during registration."""
        result = DatasetTestResult("Feature Generation")
        
        # Create test data suitable for feature generation
        test_data = pd.DataFrame({
            'id': range(500),
            'numeric1': np.random.randn(500),
            'numeric2': np.random.exponential(1, 500),
            'category': np.random.choice(['A', 'B', 'C'], 500),
            'date': pd.date_range('2024-01-01', periods=500, freq='D'),
            'text': [f"Description {i} with some text" for i in range(500)],
            'target': np.random.randint(0, 2, 500)
        })
        
        test_file = self.test_dir / "feature_test.csv"
        test_data.to_csv(test_file, index=False)
        
        dataset_name = f"test_features_{int(time.time())}"
        
        # Test both implementations
        for use_new, prefix in [(False, "legacy"), (True, "new")]:
            feature_flags.set("use_new_dataset_registration", use_new)
            test_name = f"{dataset_name}_{prefix}"
            
            try:
                registrar = get_dataset_registrar()
                reg_result = registrar.register(
                    name=test_name,
                    path=str(test_file),
                    target="target",
                    id_columns=["id"],
                    datetime_columns=["date"],
                    generate_features=True,  # Explicitly enable
                    force=True
                )
                
                self._test_datasets_created.append(test_name)
                
                # Check features
                manager = get_dataset_manager()
                info = manager.get_dataset_info(test_name)
                feature_tables = info.get('features', {})
                
                features_info = {
                    'n_feature_tables': len(feature_tables),
                    'total_features': sum(
                        ft.get('n_features', 0) for ft in feature_tables.values()
                    )
                }
                
                if prefix == "legacy":
                    result.legacy_result = features_info
                else:
                    result.new_result = features_info
                
            except Exception as e:
                result.errors.append(f"{prefix} feature generation failed: {e}")
        
        # Compare feature generation
        if result.legacy_result and result.new_result:
            result.passed = True
            
            # Check if both generated features
            if result.legacy_result['total_features'] == 0:
                result.warnings.append("Legacy didn't generate features")
            if result.new_result['total_features'] == 0:
                result.warnings.append("New didn't generate features")
        else:
            result.passed = False
        
        return result
    
    def test_error_handling(self) -> DatasetTestResult:
        """Test error handling in registration."""
        result = DatasetTestResult("Error Handling")
        
        error_cases = [
            # Invalid name
            {
                'name': 'invalid name with spaces',
                'path': str(self.test_dir),
                'expected_error': 'Invalid dataset name'
            },
            # Non-existent path
            {
                'name': 'test_nonexistent',
                'path': '/path/that/does/not/exist',
                'expected_error': 'does not exist'
            },
            # Empty directory
            {
                'name': 'test_empty',
                'path': str(self.test_dir / 'empty'),
                'expected_error': 'No data files found'
            }
        ]
        
        # Create empty directory for test
        (self.test_dir / 'empty').mkdir(exist_ok=True)
        
        for use_new, prefix in [(False, "legacy"), (True, "new")]:
            feature_flags.set("use_new_dataset_registration", use_new)
            
            errors_caught = []
            for case in error_cases:
                try:
                    registrar = get_dataset_registrar()
                    registrar.register(**case, force=True)
                    # Should not reach here
                    errors_caught.append(f"No error for: {case['name']}")
                except Exception as e:
                    # Check if appropriate error
                    if case['expected_error'].lower() not in str(e).lower():
                        errors_caught.append(
                            f"Unexpected error for {case['name']}: {str(e)}"
                        )
            
            if prefix == "legacy":
                result.legacy_result = {'errors_caught': len(errors_caught)}
            else:
                result.new_result = {'errors_caught': len(errors_caught)}
            
            if errors_caught:
                result.warnings.extend([f"{prefix}: {e}" for e in errors_caught])
        
        result.passed = len(result.warnings) == 0
        return result
    
    def test_update_operations(self) -> DatasetTestResult:
        """Test dataset update operations."""
        result = DatasetTestResult("Update Operations")
        
        # Create initial dataset
        test_data = pd.DataFrame({
            'id': range(100),
            'value': np.random.randn(100),
            'label': np.random.randint(0, 3, 100)
        })
        
        test_file = self.test_dir / "update_test.csv"
        test_data.to_csv(test_file, index=False)
        
        dataset_name = f"test_update_{int(time.time())}"
        
        for use_new, prefix in [(False, "legacy"), (True, "new")]:
            feature_flags.set("use_new_dataset_registration", use_new)
            test_name = f"{dataset_name}_{prefix}"
            
            try:
                # Register dataset
                registrar = get_dataset_registrar()
                registrar.register(
                    name=test_name,
                    path=str(test_file),
                    force=True
                )
                self._test_datasets_created.append(test_name)
                
                # Update dataset
                manager = get_dataset_manager()
                update_result = manager.update_dataset(
                    test_name,
                    description="Updated description",
                    target="label",
                    problem_type="multiclass_classification",
                    tags=["test", "updated"]
                )
                
                # Verify updates
                info = manager.get_dataset_info(test_name)
                
                update_success = (
                    info.get('description') == "Updated description" and
                    info.get('schema', {}).get('target_column') == "label" and
                    info.get('schema', {}).get('problem_type') == "multiclass_classification" and
                    set(info.get('tags', [])) == {"test", "updated"}
                )
                
                if prefix == "legacy":
                    result.legacy_result = {'update_success': update_success}
                else:
                    result.new_result = {'update_success': update_success}
                
                if not update_success:
                    result.warnings.append(f"{prefix}: Update verification failed")
                
            except Exception as e:
                result.errors.append(f"{prefix} update operations failed: {e}")
        
        result.passed = len(result.errors) == 0
        return result
    
    def test_export_import(self) -> DatasetTestResult:
        """Test dataset export and import."""
        result = DatasetTestResult("Export/Import")
        
        # Create test dataset
        test_data = pd.DataFrame({
            'id': range(200),
            'feature': np.random.randn(200),
            'target': np.random.choice([0, 1], 200)
        })
        
        test_file = self.test_dir / "export_test.csv"
        test_data.to_csv(test_file, index=False)
        
        dataset_name = f"test_export_{int(time.time())}"
        export_dir = self.test_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        for use_new, prefix in [(False, "legacy"), (True, "new")]:
            feature_flags.set("use_new_dataset_registration", use_new)
            test_name = f"{dataset_name}_{prefix}"
            
            try:
                # Register dataset
                registrar = get_dataset_registrar()
                registrar.register(
                    name=test_name,
                    path=str(test_file),
                    target="target",
                    force=True
                )
                self._test_datasets_created.append(test_name)
                
                # Export dataset
                manager = get_dataset_manager()
                exported_files = manager.export_dataset(
                    test_name,
                    str(export_dir),
                    format="csv"
                )
                
                # Verify export
                export_success = (
                    len(exported_files) > 0 and
                    all(Path(f).exists() for f in exported_files)
                )
                
                # Load exported data and compare
                if export_success and exported_files:
                    exported_data = pd.read_csv(exported_files[0])
                    data_match = (
                        len(exported_data) == len(test_data) and
                        list(exported_data.columns) == list(test_data.columns)
                    )
                else:
                    data_match = False
                
                if prefix == "legacy":
                    result.legacy_result = {
                        'export_success': export_success,
                        'data_match': data_match
                    }
                else:
                    result.new_result = {
                        'export_success': export_success,
                        'data_match': data_match
                    }
                
            except Exception as e:
                result.errors.append(f"{prefix} export/import failed: {e}")
        
        result.passed = (
            len(result.errors) == 0 and
            result.legacy_result and result.new_result and
            result.legacy_result['export_success'] and
            result.new_result['export_success']
        )
        
        return result
    
    def test_search_operations(self) -> DatasetTestResult:
        """Test dataset search functionality."""
        result = DatasetTestResult("Search Operations")
        
        # Create multiple datasets with searchable attributes
        search_datasets = []
        for i in range(3):
            test_data = pd.DataFrame({
                'id': range(50),
                'value': np.random.randn(50)
            })
            
            test_file = self.test_dir / f"search_test_{i}.csv"
            test_data.to_csv(test_file, index=False)
            
            dataset_name = f"search_test_{int(time.time())}_{i}"
            search_datasets.append(dataset_name)
        
        for use_new, prefix in [(False, "legacy"), (True, "new")]:
            feature_flags.set("use_new_dataset_registration", use_new)
            
            registered = []
            try:
                registrar = get_dataset_registrar()
                manager = get_dataset_manager()
                
                # Register datasets with different attributes
                for i, base_name in enumerate(search_datasets):
                    name = f"{base_name}_{prefix}"
                    registrar.register(
                        name=name,
                        path=str(self.test_dir / f"search_test_{i}.csv"),
                        force=True
                    )
                    
                    # Update with searchable attributes
                    tags = ["test", f"group{i%2}"]
                    if i == 0:
                        tags.append("special")
                    
                    manager.update_dataset(
                        name,
                        description=f"Test dataset {i} for search",
                        tags=tags
                    )
                    
                    registered.append(name)
                    self._test_datasets_created.append(name)
                
                # Test searches
                search_results = {
                    'by_name': len(manager.search_datasets("search_test")),
                    'by_tag': len(manager.search_datasets("special")),
                    'by_description': len(manager.search_datasets("dataset 1"))
                }
                
                if prefix == "legacy":
                    result.legacy_result = search_results
                else:
                    result.new_result = search_results
                
                # Verify search results
                if search_results['by_name'] < 3:
                    result.warnings.append(f"{prefix}: Name search found too few")
                if search_results['by_tag'] != 1:
                    result.warnings.append(f"{prefix}: Tag search incorrect")
                
            except Exception as e:
                result.errors.append(f"{prefix} search operations failed: {e}")
        
        result.passed = len(result.errors) == 0
        return result
    
    def test_registration_performance(self, n_rows: int = 10000) -> DatasetTestResult:
        """Test registration performance."""
        result = DatasetTestResult(f"Performance - Registration ({n_rows} rows)")
        
        # Create dataset of specified size
        test_data = pd.DataFrame({
            'id': range(n_rows),
            **{f'feature_{i}': np.random.randn(n_rows) for i in range(10)},
            'category': np.random.choice(list('ABCDEFGHIJ'), n_rows),
            'target': np.random.uniform(0, 1, n_rows)
        })
        
        test_file = self.test_dir / f"perf_test_{n_rows}.csv"
        test_data.to_csv(test_file, index=False)
        
        dataset_name = f"test_perf_{int(time.time())}"
        
        # Measure performance for both
        for use_new, prefix in [(False, "legacy"), (True, "new")]:
            feature_flags.set("use_new_dataset_registration", use_new)
            test_name = f"{dataset_name}_{prefix}"
            
            # Clear caches
            clear_dataset_cache()
            
            start_time = time.time()
            try:
                registrar = get_dataset_registrar()
                registrar.register(
                    name=test_name,
                    path=str(test_file),
                    target="target",
                    generate_features=False,  # Disable for pure registration test
                    force=True
                )
                
                elapsed = time.time() - start_time
                
                if prefix == "legacy":
                    result.legacy_time = elapsed
                else:
                    result.new_time = elapsed
                
                self._test_datasets_created.append(test_name)
                
            except Exception as e:
                result.errors.append(f"{prefix} performance test failed: {e}")
        
        if result.legacy_time > 0 and result.new_time > 0:
            result.passed = True
            result.details['rows_per_second'] = {
                'legacy': n_rows / result.legacy_time,
                'new': n_rows / result.new_time
            }
            result.details['speedup'] = result.legacy_time / result.new_time
        else:
            result.passed = False
        
        return result
    
    def test_query_performance(self) -> DatasetTestResult:
        """Test query performance."""
        result = DatasetTestResult("Performance - Query")
        
        # Use existing dataset or create one
        if not self._test_datasets_created:
            # Create a dataset first
            basic_result = self.test_basic_registration()
            if not basic_result.passed:
                result.errors.append("Failed to create test dataset")
                return result
        
        # Test query performance on first created dataset
        base_name = self._test_datasets_created[0].replace("_new", "").replace("_legacy", "")
        
        queries = [
            ("full_load", lambda m, n: m.load_dataset(n)),
            ("limited_load", lambda m, n: m.load_dataset(n, limit=1000)),
            ("column_load", lambda m, n: m.load_dataset(n, columns=['id', 'value'] if 'value' in m.get_dataset_info(n).get('columns', []) else None)),
            ("stats_basic", lambda m, n: m.get_dataset_stats(n, mode="basic")),
            ("info", lambda m, n: m.get_dataset_info(n)),
        ]
        
        for use_new, prefix in [(False, "legacy"), (True, "new")]:
            feature_flags.set("use_new_dataset_registration", use_new)
            test_name = f"{base_name}_{prefix}"
            
            if test_name not in self._test_datasets_created:
                continue
            
            manager = get_dataset_manager()
            query_times = {}
            
            for query_name, query_func in queries:
                try:
                    start = time.time()
                    query_func(manager, test_name)
                    elapsed = time.time() - start
                    query_times[query_name] = elapsed
                except Exception as e:
                    result.warnings.append(f"{prefix} {query_name} failed: {e}")
            
            if prefix == "legacy":
                result.legacy_result = query_times
            else:
                result.new_result = query_times
        
        result.passed = result.legacy_result and result.new_result
        
        if result.passed:
            # Calculate average performance ratio
            ratios = []
            for query in result.legacy_result:
                if query in result.new_result:
                    legacy_time = result.legacy_result[query]
                    new_time = result.new_result[query]
                    if legacy_time > 0:
                        ratios.append(new_time / legacy_time)
            
            if ratios:
                result.details['avg_query_ratio'] = np.mean(ratios)
        
        return result
    
    def cleanup(self) -> None:
        """Clean up test datasets and files."""
        console.print("\n[yellow]Cleaning up test data...[/yellow]")
        
        # Remove test datasets
        manager = get_dataset_manager()
        for dataset_name in self._test_datasets_created:
            try:
                if manager.dataset_exists(dataset_name):
                    manager.remove_dataset(dataset_name, force=True)
            except Exception as e:
                logger.warning(f"Failed to remove dataset {dataset_name}: {e}")
        
        # Remove test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        console.print("[green]Cleanup complete[/green]")
    
    def _display_results(self, results: Dict[str, Any]) -> None:
        """Display test results."""
        # Summary table
        table = Table(title="Test Results Summary")
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Legacy Time", style="white")
        table.add_column("New Time", style="white")
        table.add_column("Notes", style="grey")
        
        for test_name, result in results['tests'].items():
            if result.passed:
                status = "[green]✓ PASS[/green]"
            else:
                status = "[red]✗ FAIL[/red]"
            
            legacy_time = f"{result.legacy_time:.3f}s" if result.legacy_time > 0 else "-"
            new_time = f"{result.new_time:.3f}s" if result.new_time > 0 else "-"
            
            notes = []
            if result.errors:
                notes.append(f"[red]{len(result.errors)} errors[/red]")
            if result.warnings:
                notes.append(f"[yellow]{len(result.warnings)} warnings[/yellow]")
            
            table.add_row(
                test_name,
                status,
                legacy_time,
                new_time,
                ", ".join(notes) if notes else "OK"
            )
        
        console.print(table)
        
        # Overall summary
        console.print(Panel.fit(
            f"[bold]Overall Results[/bold]\n\n"
            f"Total Tests: {results['total']}\n"
            f"Passed: [green]{results['passed']}[/green]\n"
            f"Failed: [red]{results['failed']}[/red]\n"
            f"Warnings: [yellow]{results['warnings']}[/yellow]\n\n"
            f"Performance Ratio: {results['performance_ratio']:.2f}x "
            f"({'faster' if results['performance_ratio'] < 1 else 'slower'})",
            title="Summary"
        ))
