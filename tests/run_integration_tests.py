#!/usr/bin/env python3
"""Run integration tests with optional GitHub issue creation."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils import GitHubConfig, GitHubIssueManager, check_github_availability
from tests.utils.test_runner import BaseTestRunner
from tests.analyze_test_failures import create_github_issues


class IntegrationTestRunner(BaseTestRunner):
    """Runner specifically for integration tests."""
    
    def get_test_categories(self) -> list[tuple[str, str]]:
        """Get integration test categories."""
        return [
            ("tests/integration/cli/test_cli_integration.py", "CLI Integration"),
            ("tests/integration/cli/test_cli_real_coverage.py", "CLI Real Coverage"),
            ("tests/integration/test_dataset_lifecycle.py", "Dataset Lifecycle"),
            ("tests/integration/test_dataset_update.py", "Dataset Update"),
            ("tests/integration/test_statistics_computation.py", "Statistics Computation"),
            ("tests/integration/test_storage_backends.py", "Storage Backends"),
        ]


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run MDM integration tests with optional GitHub issue creation',
        epilog='Examples:\n'
               '  %(prog)s                                    # Run all integration tests\n'
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
    runner = IntegrationTestRunner()
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