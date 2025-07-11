#!/usr/bin/env python3
"""Enhanced E2E test runner with GitHub integration."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils import GitHubConfig, GitHubIssueManager, check_github_availability
from tests.utils.test_runner import BaseTestRunner
from tests.analyze_test_failures import create_github_issues


class E2ETestRunner(BaseTestRunner):
    """Runner specifically for E2E tests."""
    
    def get_test_categories(self) -> list[tuple[str, str]]:
        """Get E2E test categories."""
        return [
            ("tests/e2e/test_01_config/test_11_yaml.py", "YAML Configuration"),
            ("tests/e2e/test_01_config/test_12_env.py", "Environment Variables"),
            ("tests/e2e/test_01_config/test_13_backends.py", "Database Backends"),
            ("tests/e2e/test_01_config/test_14_logging.py", "Logging Configuration"),
            ("tests/e2e/test_01_config/test_15_perf.py", "Performance Configuration"),
            ("tests/e2e/test_02_dataset/test_21_register.py", "Dataset Registration"),
            ("tests/e2e/test_02_dataset/test_22_list.py", "Dataset Listing"),
            ("tests/e2e/test_02_dataset/test_23_info.py", "Dataset Info/Stats"),
        ]


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run MDM E2E tests with optional GitHub issue creation',
        epilog='Examples:\n'
               '  %(prog)s                                    # Run all E2E tests\n'
               '  %(prog)s --github                           # Run tests and create issues (dry run)\n'
               '  %(prog)s --github --no-dry-run --limit 5    # Create up to 5 GitHub issues\n'
               '  %(prog)s --output report.json               # Save test report\n'
               '  %(prog)s 1.1                                # Run specific test category (legacy)\n'
               '  %(prog)s 1.1.1                              # Run specific test (legacy)\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Test selection (for backward compatibility)
    parser.add_argument('test_id', nargs='?', help='Test ID or category to run (e.g., 1.1.1 or 1.1)')
    
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
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # If test_id is provided, use the legacy runner
    if args.test_id:
        # Import the legacy runner
        from tests.e2e.runner import MDMTestRunner
        runner = MDMTestRunner()
        
        # Determine what to run
        if '.' in args.test_id and args.test_id.count('.') == 2:
            # Run single test
            result = runner.run_test(args.test_id, verbose=args.verbose)
            print(f"Test {result.test_id}: {result.status}")
            if result.error:
                print(f"Error: {result.error}")
        elif '.' in args.test_id:
            # Run category
            category = runner.run_category(args.test_id, verbose=args.verbose)
            print(f"Category {category.id}: {category.passed_tests}/{category.total_tests} passed")
        else:
            # Run all in category
            category = runner.run_category(args.test_id, verbose=args.verbose)
            print(f"Category {category.id}: {category.passed_tests}/{category.total_tests} passed")
        
        if args.output:
            runner.generate_report(Path(args.output))
        
        return 0
    
    # Otherwise, use the new runner with GitHub integration
    runner = E2ETestRunner()
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