"""Example demonstrating integration testing functionality.

This example shows how to:
1. Run integration tests
2. Test migration scenarios
3. Benchmark performance
4. Verify data integrity
5. Generate comprehensive reports
"""
import sys
from pathlib import Path
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mdm.testing import (
    IntegrationTestFramework,
    MigrationTestSuite,
    PerformanceBenchmark
)
from mdm.core import feature_flags

console = Console()


def main():
    """Run integration testing examples."""
    console.print(Panel.fit(
        "[bold cyan]Integration Testing Examples[/bold cyan]\n\n"
        "This demonstrates comprehensive testing capabilities",
        title="MDM Integration Tests"
    ))
    
    # Create temporary directory for tests
    test_dir = Path(tempfile.mkdtemp(prefix="mdm_test_example_"))
    
    try:
        # Example 1: Run integration tests
        console.print("\n[bold]Example 1: Integration Tests[/bold]")
        console.print("=" * 50 + "\n")
        example_integration_tests(test_dir)
        
        # Example 2: Run migration tests
        console.print("\n[bold]Example 2: Migration Tests[/bold]")
        console.print("=" * 50 + "\n")
        example_migration_tests(test_dir)
        
        # Example 3: Run performance benchmarks
        console.print("\n[bold]Example 3: Performance Benchmarks[/bold]")
        console.print("=" * 50 + "\n")
        example_performance_benchmarks(test_dir)
        
        # Example 4: Custom test scenarios
        console.print("\n[bold]Example 4: Custom Test Scenarios[/bold]")
        console.print("=" * 50 + "\n")
        example_custom_tests(test_dir)
        
        console.print("\n[bold green]All examples completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error running examples: {e}[/bold red]")
        raise
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)


def example_integration_tests(test_dir: Path):
    """Example of running integration tests."""
    console.print("Running comprehensive integration tests...")
    
    # Create test framework
    framework = IntegrationTestFramework(test_dir)
    
    # Run specific test suites
    console.print("\n[dim]Running component integration tests...[/dim]")
    component_results = framework._test_component_integration()
    
    console.print(f"Component tests: {component_results['passed']}/{component_results['total']} passed")
    
    # Run end-to-end workflows
    console.print("\n[dim]Running end-to-end workflow tests...[/dim]")
    workflow_results = framework._test_end_to_end_workflows()
    
    console.print(f"Workflow tests: {workflow_results['passed']}/{workflow_results['total']} passed")
    
    # Display test summary
    display_test_summary({
        'Component Integration': component_results,
        'End-to-End Workflows': workflow_results
    })


def example_migration_tests(test_dir: Path):
    """Example of running migration tests."""
    console.print("Testing migration scenarios...")
    
    # Create migration test suite
    suite = MigrationTestSuite(test_dir)
    
    # Test specific migration scenarios
    console.print("\n[dim]Testing configuration migration...[/dim]")
    config_results = suite._test_config_migration()
    
    console.print(f"Config migration: {config_results['passed']}/{config_results['total']} passed")
    
    # Test rollback scenarios
    console.print("\n[dim]Testing rollback scenarios...[/dim]")
    rollback_results = suite._test_rollback_scenarios()
    
    console.print(f"Rollback tests: {rollback_results['passed']}/{rollback_results['total']} passed")
    
    # Test data integrity
    console.print("\n[dim]Testing data integrity...[/dim]")
    integrity_results = suite._test_data_integrity()
    
    console.print(f"Integrity tests: {integrity_results['passed']}/{integrity_results['total']} passed")
    
    # Calculate migration readiness
    total_tests = (config_results['total'] + rollback_results['total'] + 
                  integrity_results['total'])
    passed_tests = (config_results['passed'] + rollback_results['passed'] + 
                   integrity_results['passed'])
    
    readiness_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    console.print(f"\n[bold]Migration Readiness Score: {readiness_score:.1f}%[/bold]")
    
    if readiness_score >= 95:
        console.print("[green]✓ System is ready for migration[/green]")
    elif readiness_score >= 80:
        console.print("[yellow]⚠ System is mostly ready, fix critical issues[/yellow]")
    else:
        console.print("[red]✗ System is not ready for migration[/red]")


def example_performance_benchmarks(test_dir: Path):
    """Example of running performance benchmarks."""
    console.print("Running performance benchmarks...")
    
    # Create benchmark suite
    benchmark = PerformanceBenchmark(test_dir)
    
    # Run registration benchmark
    console.print("\n[dim]Benchmarking registration performance...[/dim]")
    reg_results = benchmark._benchmark_registration()
    
    # Display results
    if reg_results['comparisons']:
        table = Table(title="Registration Performance")
        table.add_column("Dataset Size", style="cyan")
        table.add_column("Legacy Time", style="yellow")
        table.add_column("New Time", style="yellow")
        table.add_column("Speedup", style="green")
        
        for comparison in reg_results['comparisons']:
            speedup_str = f"{comparison.speedup:.2f}x"
            if comparison.speedup < 1:
                speedup_str = f"[red]{speedup_str}[/red]"
            
            table.add_row(
                f"{comparison.legacy_metric.metadata.get('dataset_size', 'N/A'):,}",
                f"{comparison.legacy_metric.duration:.2f}s",
                f"{comparison.new_metric.duration:.2f}s",
                speedup_str
            )
        
        console.print(table)
    
    # Run memory benchmark
    console.print("\n[dim]Benchmarking memory usage...[/dim]")
    memory_results = benchmark._benchmark_memory()
    
    if memory_results['comparisons']:
        for comparison in memory_results['comparisons']:
            console.print(
                f"  {comparison.operation}: "
                f"Memory ratio: {comparison.memory_ratio:.2f}x "
                f"({'higher' if comparison.memory_ratio > 1 else 'lower'} than legacy)"
            )


def example_custom_tests(test_dir: Path):
    """Example of custom test scenarios."""
    console.print("Running custom test scenarios...")
    
    # Test scenario 1: Feature flag transitions
    console.print("\n[dim]Testing feature flag transitions...[/dim]")
    test_feature_flag_transitions()
    
    # Test scenario 2: Mixed mode operation
    console.print("\n[dim]Testing mixed mode operation...[/dim]")
    test_mixed_mode_operation()
    
    # Test scenario 3: Progressive rollout
    console.print("\n[dim]Testing progressive rollout...[/dim]")
    test_progressive_rollout()


def test_feature_flag_transitions():
    """Test feature flag transition scenarios."""
    # Save current flags
    original_flags = {
        'config': feature_flags.get("use_new_config"),
        'storage': feature_flags.get("use_new_storage"),
        'features': feature_flags.get("use_new_features"),
        'dataset': feature_flags.get("use_new_dataset"),
        'cli': feature_flags.get("use_new_cli")
    }
    
    try:
        # Test rapid toggling
        transitions_tested = 0
        transitions_successful = 0
        
        for i in range(5):
            # Toggle all flags
            new_state = i % 2 == 0
            feature_flags.set("use_new_config", new_state)
            feature_flags.set("use_new_storage", new_state)
            feature_flags.set("use_new_features", new_state)
            feature_flags.set("use_new_dataset", new_state)
            feature_flags.set("use_new_cli", new_state)
            
            transitions_tested += 1
            
            # Verify basic operations still work
            try:
                from mdm.adapters import get_config_manager
                config = get_config_manager()
                if hasattr(config, 'get_base_path'):
                    config.get_base_path()
                transitions_successful += 1
            except Exception as e:
                console.print(f"  [red]Transition {i} failed: {e}[/red]")
        
        console.print(
            f"  Feature flag transitions: "
            f"{transitions_successful}/{transitions_tested} successful"
        )
        
    finally:
        # Restore original flags
        for key, value in original_flags.items():
            if value is not None:
                feature_flags.set(f"use_new_{key}", value)


def test_mixed_mode_operation():
    """Test mixed mode with some new and some legacy components."""
    # Test different combinations
    mixed_modes = [
        {
            'name': 'Config only',
            'flags': {'config': True, 'storage': False, 'features': False}
        },
        {
            'name': 'Config + Storage',
            'flags': {'config': True, 'storage': True, 'features': False}
        },
        {
            'name': 'All except CLI',
            'flags': {'config': True, 'storage': True, 'features': True, 
                     'dataset': True, 'cli': False}
        }
    ]
    
    results = []
    
    for mode in mixed_modes:
        # Set flags
        for component, enabled in mode['flags'].items():
            feature_flags.set(f"use_new_{component}", enabled)
        
        # Test basic operations
        try:
            from mdm.adapters import (
                get_config_manager,
                get_storage_backend,
                clear_storage_cache
            )
            
            clear_storage_cache()
            config = get_config_manager()
            storage = get_storage_backend("sqlite")
            storage.close()
            
            results.append((mode['name'], True))
        except Exception as e:
            results.append((mode['name'], False))
    
    # Display results
    for mode_name, success in results:
        status = "[green]✓[/green]" if success else "[red]✗[/red]"
        console.print(f"  {status} {mode_name}")


def test_progressive_rollout():
    """Test progressive rollout scenarios."""
    rollout_stages = [
        (10, ['config']),
        (25, ['config', 'storage']),
        (50, ['config', 'storage', 'features']),
        (75, ['config', 'storage', 'features', 'dataset']),
        (100, ['config', 'storage', 'features', 'dataset', 'cli'])
    ]
    
    console.print("  Progressive rollout stages:")
    
    for percentage, components in rollout_stages:
        # Reset all flags
        for component in ['config', 'storage', 'features', 'dataset', 'cli']:
            feature_flags.set(f"use_new_{component}", False)
        
        # Enable components for this stage
        for component in components:
            feature_flags.set(f"use_new_{component}", True)
        
        # Test if system works
        try:
            from mdm.adapters import get_config_manager
            config = get_config_manager()
            
            console.print(
                f"    {percentage}% rollout "
                f"({', '.join(components)}): [green]✓[/green]"
            )
        except Exception as e:
            console.print(
                f"    {percentage}% rollout "
                f"({', '.join(components)}): [red]✗[/red]"
            )


def display_test_summary(results_by_suite: dict):
    """Display test results summary."""
    table = Table(title="Test Summary")
    table.add_column("Suite", style="cyan")
    table.add_column("Total", style="white")
    table.add_column("Passed", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Success Rate", style="yellow")
    
    total_all = 0
    passed_all = 0
    
    for suite_name, results in results_by_suite.items():
        total = results['total']
        passed = results['passed']
        failed = results['failed']
        rate = (passed / total * 100) if total > 0 else 0
        
        table.add_row(
            suite_name,
            str(total),
            str(passed),
            str(failed),
            f"{rate:.1f}%"
        )
        
        total_all += total
        passed_all += passed
    
    # Add total row
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_all}[/bold]",
        f"[bold]{passed_all}[/bold]",
        f"[bold]{total_all - passed_all}[/bold]",
        f"[bold]{(passed_all / total_all * 100) if total_all > 0 else 0:.1f}%[/bold]"
    )
    
    console.print("\n", table)


if __name__ == "__main__":
    main()