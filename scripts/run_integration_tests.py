#!/usr/bin/env python3
"""Run comprehensive integration tests for MDM refactoring.

This script runs all integration, migration, and performance tests
to validate the refactoring implementation.
"""
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mdm.testing import (
    IntegrationTestFramework,
    MigrationTestSuite,
    PerformanceBenchmark
)

console = Console()


def main():
    """Run integration tests based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MDM integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_integration_tests.py --all

  # Run only integration tests
  python run_integration_tests.py --integration

  # Run migration and performance tests
  python run_integration_tests.py --migration --performance

  # Run tests without cleanup
  python run_integration_tests.py --all --no-cleanup
        """
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all test suites"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests"
    )
    parser.add_argument(
        "--migration",
        action="store_true",
        help="Run migration tests"
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip cleanup after tests"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Directory for test data (temp if not specified)"
    )
    
    args = parser.parse_args()
    
    # If no specific tests requested, show help
    if not any([args.all, args.integration, args.migration, args.performance]):
        parser.print_help()
        return 1
    
    # Display header
    console.print(Panel.fit(
        "[bold cyan]MDM Integration Test Runner[/bold cyan]\n\n"
        f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        title="Test Execution"
    ))
    
    # Track overall results
    all_results = {
        'integration': None,
        'migration': None,
        'performance': None,
        'start_time': datetime.now()
    }
    
    # Determine what to run
    run_integration = args.all or args.integration
    run_migration = args.all or args.migration
    run_performance = args.all or args.performance
    cleanup = not args.no_cleanup
    
    # Run requested tests
    try:
        if run_integration:
            console.print("\n[bold]Running Integration Tests[/bold]")
            console.print("=" * 60)
            all_results['integration'] = run_integration_tests(
                test_dir=args.test_dir,
                cleanup=cleanup
            )
        
        if run_migration:
            console.print("\n[bold]Running Migration Tests[/bold]")
            console.print("=" * 60)
            all_results['migration'] = run_migration_tests(
                test_dir=args.test_dir,
                cleanup=cleanup
            )
        
        if run_performance:
            console.print("\n[bold]Running Performance Benchmarks[/bold]")
            console.print("=" * 60)
            all_results['performance'] = run_performance_benchmarks(
                test_dir=args.test_dir,
                cleanup=cleanup
            )
        
        # Display final summary
        all_results['end_time'] = datetime.now()
        display_final_summary(all_results)
        
        # Determine exit code
        return determine_exit_code(all_results)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Test execution failed: {e}[/bold red]")
        return 1


def run_integration_tests(test_dir=None, cleanup=True):
    """Run integration test suite."""
    framework = IntegrationTestFramework(test_dir)
    
    try:
        results = framework.run_all_tests(cleanup=cleanup)
        return results
    except Exception as e:
        console.print(f"[red]Integration tests failed: {e}[/red]")
        return {'error': str(e), 'passed': 0, 'total': 0}


def run_migration_tests(test_dir=None, cleanup=True):
    """Run migration test suite."""
    suite = MigrationTestSuite(test_dir)
    
    try:
        results = suite.run_all_tests(cleanup=cleanup)
        return results
    except Exception as e:
        console.print(f"[red]Migration tests failed: {e}[/red]")
        return {'error': str(e), 'passed': 0, 'total': 0}


def run_performance_benchmarks(test_dir=None, cleanup=True):
    """Run performance benchmark suite."""
    benchmark = PerformanceBenchmark(test_dir)
    
    try:
        results = benchmark.run_all_benchmarks(cleanup=cleanup)
        return results
    except Exception as e:
        console.print(f"[red]Performance benchmarks failed: {e}[/red]")
        return {'error': str(e), 'comparisons': [], 'regressions': []}


def display_final_summary(all_results):
    """Display final summary of all test results."""
    console.print("\n[bold]Final Test Summary[/bold]")
    console.print("=" * 60)
    
    # Calculate duration
    duration = all_results['end_time'] - all_results['start_time']
    console.print(f"Total duration: {duration.total_seconds():.1f} seconds")
    
    # Summary table
    table = Table(title="Test Results Overview")
    table.add_column("Test Suite", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Passed/Total", style="yellow")
    table.add_column("Success Rate", style="green")
    
    # Integration tests
    if all_results['integration']:
        int_res = all_results['integration']
        if 'error' in int_res:
            table.add_row(
                "Integration Tests",
                "[red]ERROR[/red]",
                "N/A",
                "0%"
            )
        else:
            total = int_res.get('total', 0)
            passed = int_res.get('passed', 0)
            rate = (passed / total * 100) if total > 0 else 0
            status = "[green]PASS[/green]" if rate >= 95 else "[yellow]WARN[/yellow]" if rate >= 80 else "[red]FAIL[/red]"
            
            table.add_row(
                "Integration Tests",
                status,
                f"{passed}/{total}",
                f"{rate:.1f}%"
            )
    
    # Migration tests
    if all_results['migration']:
        mig_res = all_results['migration']
        if 'error' in mig_res:
            table.add_row(
                "Migration Tests",
                "[red]ERROR[/red]",
                "N/A",
                "0%"
            )
        else:
            total = mig_res.get('total', 0)
            passed = mig_res.get('passed', 0)
            rate = (passed / total * 100) if total > 0 else 0
            readiness = mig_res.get('migration_readiness', {}).get('overall_score', 0)
            status = "[green]READY[/green]" if readiness >= 95 else "[yellow]PARTIAL[/yellow]" if readiness >= 80 else "[red]NOT READY[/red]"
            
            table.add_row(
                "Migration Tests",
                status,
                f"{passed}/{total}",
                f"{readiness:.1f}%"
            )
    
    # Performance benchmarks
    if all_results['performance']:
        perf_res = all_results['performance']
        if 'error' in perf_res:
            table.add_row(
                "Performance Tests",
                "[red]ERROR[/red]",
                "N/A",
                "N/A"
            )
        else:
            regressions = len(perf_res.get('regressions', []))
            comparisons = len(perf_res.get('comparisons', []))
            avg_speedup = perf_res.get('summary', {}).get('average_speedup', 0)
            
            if regressions == 0:
                status = "[green]GOOD[/green]"
            elif regressions <= 2:
                status = "[yellow]ACCEPTABLE[/yellow]"
            else:
                status = "[red]POOR[/red]"
            
            table.add_row(
                "Performance Tests",
                status,
                f"{regressions} regressions",
                f"{avg_speedup:.2f}x avg"
            )
    
    console.print(table)
    
    # Overall recommendation
    console.print("\n[bold]Overall Recommendation:[/bold]")
    
    if is_ready_for_migration(all_results):
        console.print("[green]✓ System is ready for migration[/green]")
    else:
        console.print("[red]✗ System is not ready for migration[/red]")
        console.print("\nIssues to address:")
        
        # List issues
        issues = get_migration_issues(all_results)
        for issue in issues:
            console.print(f"  • {issue}")


def is_ready_for_migration(all_results):
    """Determine if system is ready for migration."""
    # Check integration tests
    if all_results['integration']:
        int_res = all_results['integration']
        if 'error' in int_res:
            return False
        total = int_res.get('total', 0)
        passed = int_res.get('passed', 0)
        if total > 0 and (passed / total) < 0.95:
            return False
    
    # Check migration readiness
    if all_results['migration']:
        mig_res = all_results['migration']
        if 'error' in mig_res:
            return False
        readiness = mig_res.get('migration_readiness', {}).get('overall_score', 0)
        if readiness < 95:
            return False
    
    # Check performance regressions
    if all_results['performance']:
        perf_res = all_results['performance']
        if 'error' in perf_res:
            return False
        regressions = len(perf_res.get('regressions', []))
        if regressions > 3:  # Allow up to 3 minor regressions
            return False
    
    return True


def get_migration_issues(all_results):
    """Get list of issues preventing migration."""
    issues = []
    
    # Check integration tests
    if all_results['integration']:
        int_res = all_results['integration']
        if 'error' in int_res:
            issues.append(f"Integration tests failed: {int_res['error']}")
        else:
            total = int_res.get('total', 0)
            passed = int_res.get('passed', 0)
            if total > 0:
                rate = passed / total
                if rate < 0.95:
                    failed = total - passed
                    issues.append(f"Integration tests: {failed} tests failing")
    
    # Check migration readiness
    if all_results['migration']:
        mig_res = all_results['migration']
        if 'error' in mig_res:
            issues.append(f"Migration tests failed: {mig_res['error']}")
        else:
            readiness = mig_res.get('migration_readiness', {})
            score = readiness.get('overall_score', 0)
            if score < 95:
                # Get critical component issues
                for comp, status in readiness.get('critical_components', {}).items():
                    if status['score'] < 90:
                        issues.append(f"{comp}: {status['score']:.1f}% ready")
    
    # Check performance
    if all_results['performance']:
        perf_res = all_results['performance']
        if 'error' in perf_res:
            issues.append(f"Performance tests failed: {perf_res['error']}")
        else:
            regressions = perf_res.get('regressions', [])
            if len(regressions) > 3:
                issues.append(f"Performance: {len(regressions)} regressions found")
                # List worst regressions
                for reg in regressions[:3]:
                    issues.append(f"  - {reg['operation']}: {reg['speedup']:.2f}x")
    
    return issues


def determine_exit_code(all_results):
    """Determine appropriate exit code based on results."""
    # If ready for migration, exit 0
    if is_ready_for_migration(all_results):
        return 0
    
    # If any errors, exit 1
    for suite_results in all_results.values():
        if isinstance(suite_results, dict) and 'error' in suite_results:
            return 1
    
    # If not ready but no errors, exit 2
    return 2


if __name__ == "__main__":
    sys.exit(main())