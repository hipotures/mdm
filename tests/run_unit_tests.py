#!/usr/bin/env python3
"""Run unit tests with optional GitHub issue creation."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils import GitHubConfig, GitHubIssueManager, check_github_availability
from tests.utils.test_runner import BaseTestRunner
from tests.analyze_test_failures import create_github_issues


class UnitTestRunner(BaseTestRunner):
    """Runner specifically for unit tests."""
    
    def get_test_categories(self) -> list[tuple[str, str]]:
        """Get unit test categories."""
        return [
            # CLI tests
            ("tests/unit/cli/test_main.py", "CLI Main Module"),
            ("tests/unit/cli/test_dataset_commands.py", "Dataset Commands"),
            ("tests/unit/cli/test_batch_commands.py", "Batch Commands"),
            ("tests/unit/cli/test_timeseries_commands.py", "Timeseries Commands"),
            ("tests/unit/cli/test_cli_90_coverage.py", "CLI Coverage Tests"),
            ("tests/unit/cli/test_cli_improved_coverage.py", "CLI Improved Coverage"),
            ("tests/unit/cli/test_cli_final_coverage.py", "CLI Final Coverage"),
            ("tests/unit/cli/test_cli_direct_90.py", "CLI Direct Tests"),
            ("tests/unit/cli/test_cli_final_90.py", "CLI Final Tests"),
            
            # API tests
            ("tests/unit/test_api_simple.py", "API Simple Tests"),
            ("tests/unit/test_api_complete.py", "API Complete Tests"),
            ("tests/unit/test_api_comprehensive.py", "API Comprehensive Tests"),
            ("tests/unit/api/test_api_error_handling.py", "API Error Handling"),
            
            # Dataset tests
            ("tests/unit/dataset/test_dataset_config.py", "Dataset Config"),
            ("tests/unit/dataset/test_manager_complete.py", "Dataset Manager Complete"),
            ("tests/unit/dataset/test_manager_comprehensive.py", "Dataset Manager Comprehensive"),
            ("tests/unit/dataset/test_registrar_final.py", "Dataset Registrar"),
            ("tests/unit/dataset/test_registrar_comprehensive.py", "Dataset Registrar Comprehensive"),
            ("tests/unit/dataset/test_metadata_comprehensive.py", "Dataset Metadata"),
            ("tests/unit/dataset/test_operations_comprehensive.py", "Dataset Operations"),
            ("tests/unit/dataset/test_utils_comprehensive.py", "Dataset Utils"),
            
            # Storage tests
            ("tests/unit/storage/test_sqlite_comprehensive.py", "SQLite Storage"),
            ("tests/unit/storage/test_duckdb_complete.py", "DuckDB Storage"),
            ("tests/unit/storage/test_postgresql_complete.py", "PostgreSQL Storage"),
            ("tests/unit/storage/test_backend_compatibility.py", "Backend Compatibility"),
            ("tests/unit/storage/test_cross_backend_compatibility.py", "Cross-Backend Compatibility"),
            
            # Repository tests
            ("tests/unit/repositories/test_backend_factory.py", "Backend Factory"),
            ("tests/unit/repositories/test_dataset_manager.py", "Dataset Manager Repository"),
            ("tests/unit/repositories/test_feature_registry.py", "Feature Registry"),
            ("tests/unit/repositories/test_storage_backend.py", "Storage Backend"),
            
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
            
            # Utility tests
            ("tests/unit/utils/test_paths.py", "Path Utils"),
            ("tests/unit/utils/test_paths_comprehensive.py", "Path Utils Comprehensive"),
            ("tests/unit/utils/test_time_series_utils.py", "Time Series Utils"),
            
            # Other tests
            ("tests/unit/test_mdm_models.py", "MDM Models"),
            ("tests/unit/test_data_integrity.py", "Data Integrity"),
            ("tests/unit/test_edge_cases.py", "Edge Cases"),
            ("tests/unit/test_large_files.py", "Large Files"),
            ("tests/unit/test_security.py", "Security"),
            ("tests/unit/test_serialization.py", "Serialization"),
            ("tests/unit/test_system_resources.py", "System Resources"),
            ("tests/unit/test_time_series.py", "Time Series"),
        ]


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run MDM unit tests with optional GitHub issue creation',
        epilog='Examples:\n'
               '  %(prog)s                                    # Run all unit tests\n'
               '  %(prog)s --github                           # Run tests and create issues (dry run)\n'
               '  %(prog)s --github --no-dry-run --limit 5    # Create up to 5 GitHub issues\n'
               '  %(prog)s --output report.json               # Save test report\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # GitHub integration
    parser.add_argument('--github', action='store_true',
                       help='Enable GitHub issue creation')
    parser.add_argument('--github-token', help='GitHub token (overrides GITHUB_TOKEN env)')
    parser.add_argument('--github-repo', help='GitHub repository (default: from .env)')
    parser.add_argument('--github-limit', type=int, default=30,
                       help='Maximum issues per run (default: 30)')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be created without creating (default: True)')
    parser.add_argument('--no-dry-run', dest='dry_run', action='store_false',
                       help='Actually create issues')
    
    # Output options
    parser.add_argument('--output', '-o', help='Save report to file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Run tests
    runner = UnitTestRunner()
    test_suites = runner.run_all_tests(show_progress=not args.quiet)
    
    # Get all failures
    all_failures = []
    for category, suite in test_suites.items():
        all_failures.extend(suite.get_failures())
    
    # Display summary
    if not args.quiet:
        runner.display_summary()
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        runner.save_report(output_path)
    
    # Create GitHub issues if requested
    if args.github and all_failures:
        if not check_github_availability():
            print("Error: GitHub integration not available. Check GITHUB_TOKEN in .env")
            return 1
        
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
            
            if not args.quiet:
                print(f"\nGitHub Issues: Created={stats['created']}, Updated={stats['updated']}, Errors={stats['errors']}")
                
        except Exception as e:
            print(f"Error creating GitHub issues: {e}")
            return 1
    
    # Return exit code based on failures
    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())