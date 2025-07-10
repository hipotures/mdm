#!/usr/bin/env python3
"""Post-deployment validation for MDM.

This script validates that MDM is functioning correctly after deployment.
"""
import argparse
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random
import string

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mdm.adapters import (
    get_storage_backend,
    get_dataset_manager,
    get_feature_generator,
    get_config_manager
)
from mdm.core import feature_flags
from mdm.performance import get_monitor
from mdm.rollout import RolloutMonitor


console = Console()


class PostDeploymentValidator:
    """Validates MDM functionality after deployment."""
    
    def __init__(self):
        """Initialize validator."""
        self.test_dataset_name = f"post_deploy_test_{''.join(random.choices(string.ascii_lowercase, k=6))}"
        self.validation_results = {}
        self.start_time = datetime.utcnow()
    
    def run_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """Run complete post-deployment validation.
        
        Returns:
            Tuple of (all_passed, detailed_results)
        """
        console.print(Panel.fit(
            "[bold cyan]MDM Post-Deployment Validation[/bold cyan]\n\n"
            "Validating that all MDM components are functioning correctly...",
            title="Post-Deployment Validation"
        ))
        
        # Define validation tests
        tests = [
            ("Feature Flags", self._validate_feature_flags),
            ("Storage Backends", self._validate_storage_backends),
            ("Dataset Operations", self._validate_dataset_operations),
            ("Feature Engineering", self._validate_feature_engineering),
            ("Configuration System", self._validate_configuration),
            ("Performance Metrics", self._validate_performance),
            ("API Compatibility", self._validate_api_compatibility),
            ("Data Integrity", self._validate_data_integrity),
            ("Error Handling", self._validate_error_handling),
            ("Monitoring System", self._validate_monitoring),
        ]
        
        # Run tests
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Running validation...", total=len(tests))
            
            for test_name, test_func in tests:
                progress.update(task, description=f"Validating {test_name}...")
                
                try:
                    start = time.time()
                    passed, details = test_func()
                    duration = time.time() - start
                    
                    self.validation_results[test_name] = {
                        'passed': passed,
                        'duration': duration,
                        'details': details
                    }
                except Exception as e:
                    self.validation_results[test_name] = {
                        'passed': False,
                        'duration': 0,
                        'details': {'error': str(e), 'type': type(e).__name__}
                    }
                
                progress.advance(task)
        
        # Cleanup test data
        self._cleanup()
        
        # Display results
        self._display_results()
        
        # Calculate overall result
        all_passed = all(r['passed'] for r in self.validation_results.values())
        total_duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        return all_passed, {
            'results': self.validation_results,
            'total_duration': total_duration,
            'timestamp': self.start_time.isoformat()
        }
    
    def _validate_feature_flags(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate feature flags are correctly set."""
        expected_flags = {
            'use_new_storage': True,
            'use_new_features': True,
            'use_new_dataset': True,
            'use_new_config': True,
            'use_new_cli': True
        }
        
        current_flags = feature_flags.get_all()
        mismatches = []
        
        for flag, expected in expected_flags.items():
            actual = current_flags.get(flag, False)
            if actual != expected:
                mismatches.append({
                    'flag': flag,
                    'expected': expected,
                    'actual': actual
                })
        
        # Test that flags are being respected
        from mdm.adapters.storage_manager import _manager
        test_backend = _manager.get_backend("sqlite")
        
        # Check if we're getting new implementation
        is_new_impl = not hasattr(test_backend, '_legacy')
        
        return len(mismatches) == 0 and is_new_impl, {
            'current_flags': current_flags,
            'mismatches': mismatches,
            'using_new_implementation': is_new_impl
        }
    
    def _validate_storage_backends(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate all storage backends are functional."""
        results = {}
        all_passed = True
        
        import pandas as pd
        import tempfile
        
        test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': ['a', 'b', 'c'],
            'number': [10.5, 20.3, 30.1]
        })
        
        for backend_type in ['sqlite', 'duckdb']:
            try:
                # Get backend
                backend = get_storage_backend(backend_type)
                
                # Test operations
                with tempfile.TemporaryDirectory() as tmpdir:
                    db_path = f"{tmpdir}/test.db"
                    
                    # Create engine
                    engine = backend.get_engine(db_path)
                    
                    # Create table
                    backend.create_table_from_dataframe(
                        test_df, 'test_table', engine
                    )
                    
                    # Read back
                    result_df = backend.read_table_to_dataframe(
                        'test_table', engine
                    )
                    
                    # Verify data
                    if len(result_df) != len(test_df):
                        raise ValueError("Row count mismatch")
                    
                    # Test query
                    query_result = backend.execute_query(
                        "SELECT COUNT(*) as cnt FROM test_table",
                        engine
                    )
                    
                    results[backend_type] = {
                        'status': 'OK',
                        'row_count': len(result_df),
                        'query_works': True
                    }
                    
            except Exception as e:
                results[backend_type] = {
                    'status': 'Failed',
                    'error': str(e)
                }
                all_passed = False
        
        return all_passed, results
    
    def _validate_dataset_operations(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate dataset management operations."""
        manager = get_dataset_manager()
        operations_tested = []
        
        try:
            # Create test data
            import pandas as pd
            import tempfile
            
            test_data = pd.DataFrame({
                'feature1': range(100),
                'feature2': ['A', 'B'] * 50,
                'target': [0, 1] * 50
            })
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                test_data.to_csv(f, index=False)
                test_file = f.name
            
            # Test registration
            from mdm.adapters import get_dataset_registrar
            registrar = get_dataset_registrar()
            
            dataset_info = registrar.register_dataset(
                name=self.test_dataset_name,
                path=test_file,
                target_column='target',
                problem_type='classification'
            )
            operations_tested.append('registration')
            
            # Test retrieval
            retrieved = manager.get_dataset(self.test_dataset_name)
            if not retrieved:
                raise ValueError("Failed to retrieve dataset")
            operations_tested.append('retrieval')
            
            # Test listing
            datasets = manager.list_datasets()
            found = any(d.name == self.test_dataset_name for d in datasets)
            if not found:
                raise ValueError("Dataset not in list")
            operations_tested.append('listing')
            
            # Test update
            manager.update_dataset(self.test_dataset_name, {
                'description': 'Post-deployment test dataset'
            })
            operations_tested.append('update')
            
            # Test statistics
            stats = manager.get_statistics(self.test_dataset_name)
            operations_tested.append('statistics')
            
            # Clean up test file
            Path(test_file).unlink()
            
            return True, {
                'operations_tested': operations_tested,
                'dataset_name': self.test_dataset_name,
                'row_count': 100
            }
            
        except Exception as e:
            return False, {
                'error': str(e),
                'operations_tested': operations_tested
            }
    
    def _validate_feature_engineering(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate feature engineering functionality."""
        try:
            import pandas as pd
            
            # Create test data
            test_df = pd.DataFrame({
                'numeric1': range(50),
                'numeric2': [x * 2.5 for x in range(50)],
                'categorical': ['A', 'B', 'C'] * 16 + ['A', 'B'],
                'text': ['sample text'] * 50
            })
            
            # Get feature generator
            generator = get_feature_generator()
            
            # Generate features
            result_df = generator.generate_features(
                df=test_df,
                column_types={
                    'numeric1': 'numeric',
                    'numeric2': 'numeric',
                    'categorical': 'categorical',
                    'text': 'text'
                }
            )
            
            # Validate features were generated
            new_columns = set(result_df.columns) - set(test_df.columns)
            
            # Check for expected feature types
            has_numeric_features = any('numeric' in col for col in new_columns)
            has_categorical_features = any('categorical' in col for col in new_columns)
            
            return len(new_columns) > 0, {
                'original_columns': len(test_df.columns),
                'final_columns': len(result_df.columns),
                'new_features': len(new_columns),
                'has_numeric_features': has_numeric_features,
                'has_categorical_features': has_categorical_features,
                'sample_features': list(new_columns)[:5]
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def _validate_configuration(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate configuration system."""
        try:
            config_manager = get_config_manager()
            
            # Test configuration access
            config = config_manager.config
            
            # Test configuration update
            original_batch_size = config.performance.batch_size
            
            config_manager.update_config({
                'performance': {
                    'batch_size': original_batch_size + 1000
                }
            })
            
            # Verify update
            updated_config = get_config_manager().config
            updated_batch_size = updated_config.performance.batch_size
            
            # Restore original
            config_manager.update_config({
                'performance': {
                    'batch_size': original_batch_size
                }
            })
            
            return updated_batch_size == original_batch_size + 1000, {
                'backend': config.database.default_backend,
                'original_batch_size': original_batch_size,
                'updated_batch_size': updated_batch_size,
                'update_worked': updated_batch_size != original_batch_size
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def _validate_performance(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate performance is acceptable."""
        monitor = get_monitor()
        report = monitor.get_report()
        
        issues = []
        metrics = {}
        
        # Check operation timings
        if 'summary' in report and 'timers' in report['summary']:
            for operation, stats in report['summary']['timers'].items():
                avg_time = stats.get('avg', 0)
                metrics[operation] = {
                    'avg': avg_time,
                    'count': stats.get('count', 0)
                }
                
                # Flag slow operations
                if avg_time > 2.0:  # 2 seconds threshold
                    issues.append(f"{operation} averaging {avg_time:.2f}s")
        
        # Check cache performance
        cache_stats = {}
        if 'counters' in report.get('summary', {}):
            counters = report['summary']['counters']
            cache_hits = counters.get('cache.hits', 0)
            cache_misses = counters.get('cache.misses', 0)
            
            if cache_hits + cache_misses > 0:
                hit_rate = cache_hits / (cache_hits + cache_misses)
                cache_stats = {
                    'hits': cache_hits,
                    'misses': cache_misses,
                    'hit_rate': hit_rate
                }
                
                if hit_rate < 0.5:  # Less than 50% hit rate
                    issues.append(f"Low cache hit rate: {hit_rate:.2%}")
        
        return len(issues) == 0, {
            'metrics': metrics,
            'cache_stats': cache_stats,
            'issues': issues,
            'total_operations': report.get('total_operations', 0)
        }
    
    def _validate_api_compatibility(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate API compatibility with legacy code."""
        compatibility_checks = []
        all_passed = True
        
        # Check legacy imports still work
        legacy_imports = [
            ("from mdm.storage import SQLiteBackend", "storage"),
            ("from mdm.dataset import DatasetRegistrar", "dataset"),
            ("from mdm.features import FeatureGenerator", "features"),
        ]
        
        for import_str, module in legacy_imports:
            try:
                exec(import_str)
                compatibility_checks.append({
                    'check': f"Legacy import: {module}",
                    'passed': True
                })
            except Exception as e:
                compatibility_checks.append({
                    'check': f"Legacy import: {module}",
                    'passed': False,
                    'error': str(e)
                })
                all_passed = False
        
        # Check adapter functions
        adapter_functions = [
            "get_storage_backend",
            "get_dataset_manager",
            "get_feature_generator",
            "get_config_manager"
        ]
        
        from mdm import adapters
        
        for func_name in adapter_functions:
            if hasattr(adapters, func_name):
                compatibility_checks.append({
                    'check': f"Adapter function: {func_name}",
                    'passed': True
                })
            else:
                compatibility_checks.append({
                    'check': f"Adapter function: {func_name}",
                    'passed': False,
                    'error': "Function not found"
                })
                all_passed = False
        
        return all_passed, {
            'compatibility_checks': compatibility_checks,
            'total_checks': len(compatibility_checks),
            'passed_checks': sum(1 for c in compatibility_checks if c.get('passed', False))
        }
    
    def _validate_data_integrity(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate data integrity for existing datasets."""
        manager = get_dataset_manager()
        datasets = manager.list_datasets()
        
        # Sample up to 5 datasets for validation
        sample_size = min(5, len(datasets))
        sampled = random.sample(datasets, sample_size) if datasets else []
        
        integrity_checks = []
        all_passed = True
        
        for dataset in sampled:
            try:
                # Check dataset can be loaded
                info = manager.get_dataset(dataset.name)
                if not info:
                    raise ValueError("Cannot load dataset info")
                
                # Check statistics exist
                stats = manager.get_statistics(dataset.name)
                
                # Basic integrity checks
                checks = {
                    'info_loaded': info is not None,
                    'has_stats': stats is not None,
                    'has_source_path': bool(info.source_path),
                    'has_created_at': info.created_at is not None
                }
                
                integrity_checks.append({
                    'dataset': dataset.name,
                    'passed': all(checks.values()),
                    'checks': checks
                })
                
                if not all(checks.values()):
                    all_passed = False
                    
            except Exception as e:
                integrity_checks.append({
                    'dataset': dataset.name,
                    'passed': False,
                    'error': str(e)
                })
                all_passed = False
        
        return all_passed, {
            'total_datasets': len(datasets),
            'sampled': sample_size,
            'integrity_checks': integrity_checks
        }
    
    def _validate_error_handling(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate error handling works correctly."""
        error_tests = []
        
        # Test 1: Invalid dataset name
        try:
            manager = get_dataset_manager()
            manager.get_dataset("definitely_does_not_exist_12345")
            error_tests.append({
                'test': 'Invalid dataset retrieval',
                'passed': True,
                'behavior': 'Returns None'
            })
        except Exception as e:
            error_tests.append({
                'test': 'Invalid dataset retrieval',
                'passed': True,
                'behavior': f'Raises {type(e).__name__}'
            })
        
        # Test 2: Invalid storage backend
        try:
            from mdm.adapters import get_storage_backend
            get_storage_backend("invalid_backend_type")
            error_tests.append({
                'test': 'Invalid storage backend',
                'passed': False,
                'behavior': 'Should have raised error'
            })
        except ValueError:
            error_tests.append({
                'test': 'Invalid storage backend',
                'passed': True,
                'behavior': 'Raises ValueError as expected'
            })
        except Exception as e:
            error_tests.append({
                'test': 'Invalid storage backend',
                'passed': False,
                'behavior': f'Unexpected error: {type(e).__name__}'
            })
        
        # Test 3: Configuration validation
        try:
            config_manager = get_config_manager()
            config_manager.update_config({
                'performance': {
                    'batch_size': -1  # Invalid value
                }
            })
            error_tests.append({
                'test': 'Invalid configuration',
                'passed': False,
                'behavior': 'Should have raised error'
            })
        except Exception:
            error_tests.append({
                'test': 'Invalid configuration',
                'passed': True,
                'behavior': 'Raises error as expected'
            })
        
        all_passed = all(test.get('passed', False) for test in error_tests)
        
        return all_passed, {
            'error_tests': error_tests,
            'total_tests': len(error_tests),
            'passed_tests': sum(1 for t in error_tests if t.get('passed', False))
        }
    
    def _validate_monitoring(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate monitoring system is active."""
        try:
            # Check rollout monitor
            monitor = RolloutMonitor()
            
            # Record test metric
            monitor.record_metric(
                "test.validation.metric",
                42.0,
                monitor.MetricType.GAUGE
            )
            
            # Check if metric was recorded
            if "test.validation.metric" not in monitor.metrics:
                raise ValueError("Metric not recorded")
            
            # Get health status
            health = monitor.get_health_status()
            
            # Create test alert
            alert = monitor.create_alert(
                monitor.AlertSeverity.INFO,
                "Test Alert",
                "Post-deployment validation test alert"
            )
            
            return True, {
                'health_status': health['status'],
                'metrics_count': health['metrics_count'],
                'test_metric_recorded': True,
                'alert_created': alert.id in monitor.alerts
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def _cleanup(self) -> None:
        """Clean up test data."""
        try:
            manager = get_dataset_manager()
            if manager.dataset_exists(self.test_dataset_name):
                manager.delete_dataset(self.test_dataset_name, force=True)
        except Exception:
            pass
    
    def _display_results(self) -> None:
        """Display validation results."""
        console.print("\n[bold]Post-Deployment Validation Results[/bold]\n")
        
        # Summary table
        table = Table(box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Details", style="dim")
        
        for component, result in self.validation_results.items():
            passed = result['passed']
            status = "[green]✓ PASSED[/green]" if passed else "[red]✗ FAILED[/red]"
            duration = f"{result['duration']:.2f}s"
            
            # Extract key detail
            details = result.get('details', {})
            if 'error' in details:
                detail_text = f"Error: {details['error']}"
            elif 'issues' in details and details['issues']:
                detail_text = f"{len(details['issues'])} issues"
            else:
                detail_text = "OK"
            
            table.add_row(component, status, duration, detail_text)
        
        console.print(table)
        
        # Performance summary
        total_duration = sum(r['duration'] for r in self.validation_results.values())
        passed_count = sum(1 for r in self.validation_results.values() if r['passed'])
        total_count = len(self.validation_results)
        
        summary = Panel.fit(
            f"[bold]Validation Summary[/bold]\n\n"
            f"Total Tests: {total_count}\n"
            f"Passed: [green]{passed_count}[/green]\n"
            f"Failed: [red]{total_count - passed_count}[/red]\n"
            f"Total Duration: {total_duration:.2f}s\n"
            f"Success Rate: {passed_count/total_count*100:.1f}%",
            border_style="green" if passed_count == total_count else "yellow"
        )
        
        console.print("\n")
        console.print(summary)
    
    def generate_report(self, output_file: Path) -> None:
        """Generate detailed validation report."""
        report = {
            'timestamp': self.start_time.isoformat(),
            'duration': (datetime.utcnow() - self.start_time).total_seconds(),
            'results': self.validation_results,
            'summary': {
                'total_tests': len(self.validation_results),
                'passed': sum(1 for r in self.validation_results.values() if r['passed']),
                'failed': sum(1 for r in self.validation_results.values() if not r['passed'])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate MDM deployment"
    )
    parser.add_argument(
        '--report',
        type=Path,
        help='Save validation report to file'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick validation only'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = PostDeploymentValidator()
    all_passed, results = validator.run_validation()
    
    # Save report if requested
    if args.report:
        validator.generate_report(args.report)
        console.print(f"\n[dim]Report saved to: {args.report}[/dim]")
    
    # Final status
    if all_passed:
        console.print("\n[bold green]✓ Post-deployment validation PASSED![/bold green]")
        console.print("\nMDM is functioning correctly.")
        sys.exit(0)
    else:
        failed_count = sum(1 for r in results['results'].values() if not r['passed'])
        console.print(
            f"\n[bold red]✗ Post-deployment validation FAILED. "
            f"{failed_count} tests failed.[/bold red]"
        )
        console.print("\nPlease investigate the failed tests.")
        sys.exit(1)


if __name__ == "__main__":
    main()