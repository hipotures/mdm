#!/usr/bin/env python3
"""Unified test failure analyzer with GitHub integration."""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils import (
    GitHubConfig,
    GitHubIssueManager,
    check_github_availability,
    get_suggested_fix,
    GITHUB_AVAILABLE
)
from tests.utils.test_runner import BaseTestRunner, TestResult, RICH_AVAILABLE
from tests.utils.error_analyzer import ErrorAnalyzer, group_failures_by_pattern

if RICH_AVAILABLE:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
else:
    console = None


class UnifiedTestRunner(BaseTestRunner):
    """Unified test runner for all test types."""
    
    def __init__(self, scope: str = "all", project_root: Optional[Path] = None):
        """Initialize with test scope."""
        super().__init__(project_root)
        self.scope = scope
        
    def get_test_categories(self) -> List[tuple[str, str]]:
        """Get test categories based on scope."""
        categories = []
        
        # Unit tests
        if self.scope in ["unit", "all"]:
            categories.extend([
                # CLI tests
                ("tests/unit/cli/test_dataset_commands.py", "Dataset Commands"),
                ("tests/unit/cli/test_batch_commands.py", "Batch Commands"),
                ("tests/unit/cli/test_cli_final_coverage.py", "CLI Final Coverage"),
                ("tests/unit/cli/test_cli_direct_90.py", "CLI Direct Tests"),
                ("tests/unit/cli/test_dataset_update_comprehensive.py", "Dataset Update Comprehensive"),
                
                # API tests
                ("tests/unit/test_api_simple.py", "API Simple Tests"),
                ("tests/unit/test_api_complete.py", "API Complete Tests"),
                ("tests/unit/test_api_comprehensive.py", "API Comprehensive Tests"),
                ("tests/unit/api/test_api_error_handling.py", "API Error Handling"),
                
                # Dataset tests
                ("tests/unit/dataset/test_dataset_config.py", "Dataset Config"),
                ("tests/unit/dataset/test_manager_complete.py", "Dataset Manager"),
                ("tests/unit/dataset/test_manager_comprehensive.py", "Dataset Manager Comprehensive"),
                ("tests/unit/dataset/test_registrar_final.py", "Dataset Registrar"),
                ("tests/unit/dataset/test_registrar_comprehensive.py", "Dataset Registrar Comprehensive"),
                ("tests/unit/dataset/test_registrar_90_coverage.py", "Dataset Registrar 90% Coverage"),
                ("tests/unit/dataset/test_registrar_coverage.py", "Dataset Registrar Coverage"),
                ("tests/unit/dataset/test_registrar_enhanced.py", "Dataset Registrar Enhanced"),
                ("tests/unit/dataset/test_metadata_comprehensive.py", "Metadata Comprehensive"),
                ("tests/unit/dataset/test_metadata_90_coverage.py", "Metadata 90% Coverage"),
                ("tests/unit/dataset/test_operations_comprehensive.py", "Operations Comprehensive"),
                ("tests/unit/dataset/test_utils_complete.py", "Dataset Utils Complete"),
                ("tests/unit/dataset/test_utils_comprehensive.py", "Dataset Utils Comprehensive"),
                
                # Storage tests
                ("tests/unit/storage/test_stateless_backends.py", "Stateless Storage Backends"),
                
                # Repository tests
                ("tests/unit/repositories/test_dataset_manager.py", "Dataset Manager Repository"),
                ("tests/unit/repositories/test_feature_registry.py", "Feature Registry"),
                
                # Service tests
                ("tests/unit/services/test_dataset_service.py", "Dataset Service"),
                ("tests/unit/services/features/test_feature_engine.py", "Feature Engine"),
                ("tests/unit/services/features/test_feature_generator.py", "Feature Generator"),
                ("tests/unit/services/export/test_dataset_exporter.py", "Dataset Exporter"),
                ("tests/unit/services/batch/test_batch_export.py", "Batch Export"),
                ("tests/unit/services/batch/test_batch_stats.py", "Batch Stats"),
                ("tests/unit/services/batch/test_batch_remove.py", "Batch Remove"),
                ("tests/unit/services/operations/test_export_operation.py", "Export Operation"),
                ("tests/unit/services/operations/test_info_operation.py", "Info Operation"),
                ("tests/unit/services/operations/test_list_operation.py", "List Operation"),
                ("tests/unit/services/operations/test_remove_operation.py", "Remove Operation"),
                ("tests/unit/services/operations/test_search_operation.py", "Search Operation"),
                ("tests/unit/services/operations/test_stats_operation.py", "Stats Operation"),
                ("tests/unit/services/operations/test_update_operation.py", "Update Operation"),
                ("tests/unit/services/registration/test_auto_detect.py", "Auto Detect"),
                ("tests/unit/services/registration/test_dataset_registrar.py", "Dataset Registrar Service"),
                
                # Feature tests
                ("tests/unit/features/test_engine_complete.py", "Feature Engine Complete"),
                
                # Utils tests
                ("tests/unit/utils/test_paths.py", "Utils Paths"),
                ("tests/unit/utils/test_paths_comprehensive.py", "Utils Paths Comprehensive"),
                ("tests/unit/utils/test_time_series_utils.py", "Time Series Utils"),
                
                # Other unit tests
                ("tests/unit/test_data_integrity.py", "Data Integrity"),
                ("tests/unit/test_edge_cases.py", "Edge Cases"),
                ("tests/unit/test_large_files.py", "Large Files"),
                ("tests/unit/test_mdm_models.py", "MDM Models"),
                ("tests/unit/test_security.py", "Security"),
                ("tests/unit/test_serialization.py", "Serialization"),
                ("tests/unit/test_system_resources.py", "System Resources"),
                ("tests/unit/test_time_series.py", "Time Series"),
            ])
        
        # Integration tests
        if self.scope in ["integration", "all"]:
            categories.extend([
                ("tests/integration/cli/test_cli_integration.py", "CLI Integration"),
                ("tests/integration/cli/test_cli_real_coverage.py", "CLI Real Coverage"),
                ("tests/integration/test_dataset_lifecycle.py", "Dataset Lifecycle"),
                ("tests/integration/test_dataset_update.py", "Dataset Update"),
                ("tests/integration/test_statistics_computation.py", "Statistics Computation"),
                ("tests/integration/test_storage_backends.py", "Storage Backends"),
                ("tests/test_integration_testing.py", "Integration Testing Framework"),
            ])
        
        # E2E tests
        if self.scope in ["e2e", "all"]:
            categories.extend([
                ("tests/e2e/test_01_config/test_11_yaml.py", "YAML Configuration"),
                ("tests/e2e/test_01_config/test_12_env.py", "Environment Variables"),
                ("tests/e2e/test_01_config/test_13_backends.py", "Database Backends"),
                ("tests/e2e/test_01_config/test_14_logging.py", "Logging Configuration"),
                ("tests/e2e/test_01_config/test_15_perf.py", "Performance Configuration"),
                ("tests/e2e/test_02_dataset/test_21_register.py", "Dataset Registration"),
                ("tests/e2e/test_02_dataset/test_22_list.py", "Dataset Listing"),
                ("tests/e2e/test_02_dataset/test_23_info.py", "Dataset Info/Stats"),
                ("tests/e2e/test_isolation.py", "E2E Isolation"),
                ("tests/e2e/test_summary.py", "E2E Summary"),
            ])
        
        return categories


def create_github_issues(
    failures: List[TestResult],
    github_manager: GitHubIssueManager,
    dry_run: bool = True,
    limit: int = 100
) -> Dict[str, int]:
    """Create GitHub issues for test failures."""
    stats = {
        "created": 0,
        "updated": 0,
        "skipped": 0,
        "errors": 0,
        "rate_limited": 0
    }
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Creating GitHub issues (limit: {limit})...[/bold]")
    else:
        print(f"\nCreating GitHub issues (limit: {limit})...")
    
    for i, failure in enumerate(failures):
        if i >= limit:
            remaining = len(failures) - i
            if RICH_AVAILABLE:
                console.print(f"\n[yellow]Reached limit of {limit} issues. {remaining} failures not processed.[/yellow]")
            else:
                print(f"\nReached limit of {limit} issues. {remaining} failures not processed.")
            break
        
        # Get suggested fix
        suggested_fix = get_suggested_fix(failure.error_type or "Unknown")
        
        # Format issue
        issue_data = github_manager.format_test_failure_issue(
            test_name=failure.test_name,
            error_type=failure.error_type or "Unknown",
            error_message=failure.error_message or "No error message",
            category=failure.category,
            file_path=failure.file_path,
            additional_info={
                "suggested_fix": suggested_fix,
                "test_output": failure.output[:1000] if failure.output else None
            }
        )
        
        # Create or update issue
        result = github_manager.create_or_update_issue(
            title=issue_data["title"],
            body=issue_data["body"],
            labels=issue_data["labels"],
            issue_id=issue_data["issue_id"],
            dry_run=dry_run
        )
        
        # Update stats
        if result["rate_limited"]:
            stats["rate_limited"] += 1
            if RICH_AVAILABLE:
                console.print(f"[yellow]Rate limited: {result['message']}[/yellow]")
            else:
                print(f"Rate limited: {result['message']}")
            break
        elif result["action"] == "created" or result["action"] == "would_create":
            stats["created"] += 1
            if RICH_AVAILABLE:
                console.print(f"[green]{failure.category} - {failure.test_name}: {result['message']}[/green]")
            else:
                print(f"{failure.category} - {failure.test_name}: {result['message']}")
        elif result["action"] == "updated" or result["action"] == "would_update":
            stats["updated"] += 1
            if RICH_AVAILABLE:
                console.print(f"[blue]{failure.category} - {failure.test_name}: {result['message']}[/blue]")
            else:
                print(f"{failure.category} - {failure.test_name}: {result['message']}")
        elif result["action"] == "error":
            stats["errors"] += 1
            if RICH_AVAILABLE:
                console.print(f"[red]{failure.category} - {failure.test_name}: {result['message']}[/red]")
            else:
                print(f"{failure.category} - {failure.test_name}: {result['message']}")
        else:
            stats["skipped"] += 1
    
    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Unified test failure analyzer with GitHub integration',
        epilog='Examples:\n'
               '  %(prog)s                                    # Analyze all tests\n'
               '  %(prog)s --scope unit                       # Analyze only unit tests\n'
               '  %(prog)s --scope e2e --github               # Analyze E2E tests and create issues (dry run)\n'
               '  %(prog)s --github --no-dry-run --limit 50   # Create up to X GitHub issues\n'
               '  %(prog)s --category "CLI*"                  # Analyze only CLI-related tests\n'
               '  %(prog)s --scope e2e --parallel 4           # Run E2E tests with 4 parallel workers\n'
               '  %(prog)s --parallel 8                       # Run all tests with 8 parallel workers\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Scope and filtering
    parser.add_argument('--scope', choices=['unit', 'integration', 'e2e', 'all'],
                       default='all', help='Test scope to analyze')
    parser.add_argument('--category', help='Filter by category pattern (supports wildcards)')
    
    # GitHub integration
    parser.add_argument('--github', action='store_true',
                       help='Enable GitHub issue creation')
    parser.add_argument('--github-token', help='GitHub token (overrides GITHUB_TOKEN env)')
    parser.add_argument('--github-repo', help='GitHub repository (default: from .env)')
    parser.add_argument('--github-limit', type=int, default=50,
                       help='Maximum issues per run (default: 50)')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be created without creating (default: True)')
    parser.add_argument('--no-dry-run', dest='dry_run', action='store_false',
                       help='Actually create issues')
    
    # Output options
    parser.add_argument('--output', '-o', help='Save report to file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    # Parallel execution
    parser.add_argument('--parallel', '-p', type=int, default=1, metavar='N',
                       help='Number of parallel test workers (default: 1, sequential)')
    
    args = parser.parse_args()
    
    # Display mode information
    if args.github:
        if not check_github_availability():
            if RICH_AVAILABLE:
                console.print("[red]Error:[/red] GitHub integration not available. Check GITHUB_TOKEN in .env")
            else:
                print("Error: GitHub integration not available. Check GITHUB_TOKEN in .env")
            return 1
        
        if args.dry_run:
            if RICH_AVAILABLE and not args.quiet:
                console.print(Panel.fit(
                    "[bold yellow]DRY RUN MODE[/bold yellow]\n"
                    "Showing what would be created without actually creating issues.\n"
                    "Use --no-dry-run to actually create issues.",
                    title="Mode: DRY RUN",
                    border_style="yellow"
                ))
            elif not args.quiet:
                print("\n" + "="*80)
                print("DRY RUN MODE - No issues will be created")
                print("Use --no-dry-run to actually create issues")
                print("="*80 + "\n")
        else:
            if RICH_AVAILABLE and not args.quiet:
                console.print(Panel.fit(
                    "[bold red]LIVE MODE[/bold red]\n"
                    "Issues will be created on GitHub!",
                    title="Mode: LIVE",
                    border_style="red"
                ))
            elif not args.quiet:
                print("\n" + "="*80)
                print("LIVE MODE - Issues WILL be created on GitHub!")
                print("="*80 + "\n")
    
    # Run tests
    runner = UnifiedTestRunner(scope=args.scope)
    test_suites = runner.run_all_tests(show_progress=not args.quiet, max_workers=args.parallel)
    
    # Get all failures
    all_failures = []
    for category, suite in test_suites.items():
        # Apply category filter if specified
        if args.category:
            import fnmatch
            if not fnmatch.fnmatch(category, args.category):
                continue
        
        all_failures.extend(suite.get_failures())
    
    # Display summary
    if not args.quiet:
        runner.display_summary()
        
        if all_failures and RICH_AVAILABLE:
            # Show failure details
            console.print("\n[bold]Failure Details[/bold]")
            console.rule()
            
            # Group by error type
            error_analyzer = ErrorAnalyzer()
            grouped = group_failures_by_pattern([{
                "test": f.test_name,
                "error_type": f.error_type,
                "error_message": f.error_message,
                "category": f.category
            } for f in all_failures])
            
            for group_name, group_data in grouped.items():
                console.print(f"\n[bold]{group_data['title']}[/bold]")
                console.print(f"[dim]{group_data['description']}[/dim]")
                console.print(f"Count: {len(group_data['failures'])}")
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        runner.save_report(output_path)
    
    # Create GitHub issues if requested
    if args.github and all_failures:
        # Override config if provided
        config = GitHubConfig.from_env()
        if args.github_token:
            config.token = args.github_token
        if args.github_repo:
            config.repo = args.github_repo
        config.rate_limit = args.github_limit
        
        try:
            github_manager = GitHubIssueManager(config)
            stats = create_github_issues(
                all_failures,
                github_manager,
                dry_run=args.dry_run,
                limit=args.github_limit
            )
            
            # Display GitHub summary
            if RICH_AVAILABLE and not args.quiet:
                console.print("\n[bold]GitHub Issue Summary:[/bold]")
                console.rule()
                
                table = Table(show_header=False)
                table.add_column("Metric", style="cyan")
                table.add_column("Count", justify="right")
                
                if args.dry_run:
                    table.add_row("Would create", str(stats["created"]))
                    table.add_row("Would update", str(stats["updated"]))
                else:
                    table.add_row("Created", str(stats["created"]))
                    table.add_row("Updated", str(stats["updated"]))
                
                if stats["errors"] > 0:
                    table.add_row("Errors", str(stats["errors"]))
                if stats["rate_limited"] > 0:
                    table.add_row("Rate limited", str(stats["rate_limited"]))
                
                console.print(table)
            elif not args.quiet:
                print("\n" + "="*40)
                print("GitHub Issue Summary:")
                if args.dry_run:
                    print(f"Would create: {stats['created']}")
                    print(f"Would update: {stats['updated']}")
                else:
                    print(f"Created: {stats['created']}")
                    print(f"Updated: {stats['updated']}")
                if stats["errors"] > 0:
                    print(f"Errors: {stats['errors']}")
                if stats["rate_limited"] > 0:
                    print(f"Rate limited: {stats['rate_limited']}")
                    
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]Error:[/red] {e}")
            else:
                print(f"Error: {e}")
            return 1
    
    # Return exit code based on failures
    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())
