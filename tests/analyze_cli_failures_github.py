#!/usr/bin/env python3
"""Analyze CLI test failures and create GitHub issues."""

import subprocess
import sys
import re
import os
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not available, will use system environment

try:
    from github import Github, GithubException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    print("Warning: PyGithub not installed. GitHub issue creation will be disabled.")
    print("Install with: uv pip install PyGithub")


class TestFailure:
    """Represents a single test failure."""
    def __init__(self, test_name: str, file_path: str, error_type: str, 
                 error_message: str, category: str):
        self.test_name = test_name
        self.file_path = file_path
        self.error_type = error_type
        self.error_message = error_message
        self.category = category
        self.full_output = ""
        
    def get_fingerprint(self) -> str:
        """Get unique fingerprint for this failure."""
        # Create fingerprint from test name and error type
        content = f"{self.test_name}:{self.error_type}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


def run_tests_and_collect_failures():
    """Run all CLI tests and collect failure information."""
    project_root = Path(__file__).parent.parent
    failures = defaultdict(list)
    
    # Test categories for CLI
    test_categories = [
        # Unit tests
        ("tests/unit/cli/test_main.py", "CLI Main Module"),
        ("tests/unit/cli/test_dataset_commands.py", "Dataset Commands"),
        ("tests/unit/cli/test_batch_commands.py", "Batch Commands"),
        ("tests/unit/cli/test_timeseries_commands.py", "Timeseries Commands"),
        ("tests/unit/cli/test_cli_90_coverage.py", "CLI Coverage Tests"),
        ("tests/unit/cli/test_cli_improved_coverage.py", "CLI Improved Coverage"),
        ("tests/unit/cli/test_cli_final_coverage.py", "CLI Final Coverage"),
        ("tests/unit/cli/test_cli_direct_90.py", "CLI Direct Tests"),
        ("tests/unit/cli/test_cli_final_90.py", "CLI Final Tests"),
        # Integration tests
        ("tests/integration/cli/test_cli_integration.py", "CLI Integration"),
        ("tests/integration/cli/test_cli_real_coverage.py", "CLI Real Coverage"),
    ]
    
    print("Analyzing CLI test failures...")
    print("=" * 80)
    
    for test_file, category_name in test_categories:
        test_path = project_root / test_file
        if not test_path.exists():
            print(f"\nSkipping {test_file} (not found)")
            continue
            
        print(f"\nTesting {category_name}...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_path),
            "-v", "--tb=short",
            "--no-header"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse pytest output
        if result.returncode != 0:
            parse_pytest_output(result.stdout + result.stderr, test_file, 
                              category_name, failures)
    
    return failures


def parse_pytest_output(output: str, test_file: str, category: str, 
                       failures: Dict[str, List[TestFailure]]):
    """Parse pytest output to extract failure information."""
    lines = output.split('\n')
    
    current_test = None
    current_failure = None
    in_failure = False
    
    for line in lines:
        # Match test failure line
        if line.startswith('FAILED'):
            match = re.match(r'FAILED (.+) - (.+)', line)
            if match:
                test_name = match.group(1).split('::')[-1]
                error_msg = match.group(2)
                
                # Extract error type
                error_type = "Unknown"
                if "AssertionError" in error_msg:
                    error_type = "AssertionError"
                elif "AttributeError" in error_msg:
                    error_type = "AttributeError"
                elif "ModuleNotFoundError" in error_msg:
                    error_type = "ModuleNotFoundError"
                elif "TypeError" in error_msg:
                    error_type = "TypeError"
                elif "KeyError" in error_msg:
                    error_type = "KeyError"
                elif "ValueError" in error_msg:
                    error_type = "ValueError"
                elif "SystemExit" in error_msg:
                    error_type = "SystemExit"
                elif "Exception" in error_msg:
                    error_type = "Exception"
                
                current_failure = TestFailure(
                    test_name=test_name,
                    file_path=test_file,
                    error_type=error_type,
                    error_message=error_msg[:200],  # Truncate long messages
                    category=category
                )
                failures[category].append(current_failure)


def create_individual_issue(repo, category: str, failure: TestFailure, existing_issues: set, dry_run: bool = False) -> Optional[str]:
    """Create a GitHub issue for an individual test failure."""
    # Create title
    title = f"[CLI Test] {failure.test_name} - {failure.error_type}"
    
    # Check if similar issue exists
    for issue in existing_issues:
        if failure.test_name in issue.title:
            return f"Similar issue already exists: #{issue.number}"
    
    # Create issue body
    body = f"""## Test Failure: {failure.test_name}

**Category:** {category}
**File:** `{failure.file_path}`
**Error Type:** {failure.error_type}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Error Details
```
{failure.error_message}
```

### How to Reproduce
```bash
pytest {failure.file_path}::{failure.test_name}
```

### Suggested Fix
"""
    
    # Add specific suggestions based on error type
    if failure.error_type == "AssertionError":
        body += "- Check if test assertions need updating to match current behavior\n"
        body += "- Verify expected output format hasn't changed\n"
    elif failure.error_type == "AttributeError":
        body += "- Check if mocked objects have all required attributes\n"
        body += "- Verify API hasn't changed\n"
    elif failure.error_type == "TypeError":
        body += "- Check function signatures and parameter types\n"
        body += "- Verify mock configurations\n"
    elif failure.error_type == "KeyError":
        body += "- Check if required keys exist in dictionaries\n"
        body += "- Verify data structures haven't changed\n"
    else:
        body += "- Investigate the specific error and root cause\n"
    
    body += "\n---\n*This issue was automatically created by the CLI test analyzer.*\n"
    
    # Create labels based on category
    labels = ["bug", "cli", "test-failure"]
    
    # Add category-specific label
    category_label = category.lower().replace(" ", "-").replace("/", "-")
    labels.append(f"category-{category_label}")
    
    # Add error type label
    error_label = failure.error_type.lower().replace(" ", "-")
    labels.append(f"error-{error_label}")
    
    # Add test group label
    if "setup_logging" in failure.test_name.lower():
        labels.append("group-logging")
    elif "display_column_summary" in failure.test_name.lower():
        labels.append("group-column-summary")
    elif "batch" in failure.test_name.lower():
        labels.append("group-batch")
    elif "timeseries" in failure.test_name.lower():
        labels.append("group-timeseries")
    elif "integration" in category.lower():
        labels.append("group-integration")
    elif "coverage" in category.lower():
        labels.append("group-coverage")
    
    if dry_run:
        print(f"\n[DRY RUN] Would create issue:")
        print(f"Title: {title}")
        print(f"Labels: {', '.join(labels)}")
        return "[DRY RUN] Issue would be created"
    
    try:
        issue = repo.create_issue(
            title=title,
            body=body,
            labels=labels
        )
        return f"Created issue: #{issue.number}"
    except GithubException as e:
        return f"Failed to create issue: {e}"


def create_grouped_issue(repo, group: Dict, existing_issues: set) -> Optional[str]:
    """Create a GitHub issue for a group of failures."""
    title = group['title']
    
    # Check if similar issue exists
    for issue in existing_issues:
        if group['name'] in issue.title.lower() and 'CLI Tests' in issue.title:
            return f"Similar issue already exists: #{issue.number}"
    
    # Create issue body
    body = f"""## Test Failure Group: {group['name']}

**Group Type:** {group['type']}
**Total Failures:** {len(group['failures'])}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Failed Tests

| Category | Test Name | Error Type | Error Message |
|----------|-----------|------------|---------------|
"""
    
    # Add each failure to the table
    for cat, fail in group['failures']:
        error_msg = fail.error_message[:100] + "..." if len(fail.error_message) > 100 else fail.error_message
        body += f"| {cat} | `{fail.test_name}` | {fail.error_type} | {error_msg} |\n"
    
    # Add reproduction steps
    body += "\n### How to Reproduce\n\n"
    body += "Run the following commands to reproduce these failures:\n\n```bash\n"
    
    # Group files for reproduction
    files_to_test = set()
    for cat, fail in group['failures']:
        files_to_test.add(fail.file_path)
    
    for file_path in sorted(files_to_test):
        body += f"pytest {file_path}\n"
    
    body += "```\n\n"
    
    # Add analysis
    body += "### Analysis\n\n"
    if group['type'] == 'error_type':
        body += f"All these tests are failing with `{group['name']}` errors. "
        if group['name'] == 'AssertionError':
            body += "This typically indicates that test assertions need to be updated to match current behavior.\n"
        elif group['name'] == 'AttributeError':
            body += "This suggests missing attributes or incorrect mocking. Check if APIs have changed.\n"
        elif group['name'] == 'TypeError':
            body += "This indicates type mismatches or incorrect function signatures.\n"
        elif group['name'] == 'Unknown':
            body += "These failures need investigation to determine the root cause.\n"
    else:
        body += f"These tests are all related to `{group['name'].replace('_', ' ')}` functionality.\n"
    
    body += "\n### Suggested Actions\n\n"
    body += "1. Review recent changes to the affected modules\n"
    body += "2. Check if test mocks need updating\n"
    body += "3. Verify that test assertions match current expected behavior\n"
    body += "4. Consider if the implementation has bugs that need fixing\n"
    
    body += "\n### Labels\n- bug\n- cli\n- test-failure\n"
    body += f"- {group['type']}-group\n"
    
    body += "\n---\n*This issue was automatically created by the CLI test analyzer with intelligent grouping.*\n"
    
    # Truncate if too long
    if len(body) > 65000:
        body = body[:64900] + "\n\n... (truncated)"
    
    try:
        labels = ["bug", "cli", "test-failure", f"{group['type']}-group"]
        issue = repo.create_issue(
            title=title,
            body=body,
            labels=labels
        )
        return f"Created grouped issue: #{issue.number}"
    except GithubException as e:
        return f"Failed to create issue: {e}"


def main():
    """Main function to analyze failures and optionally create issues."""
    parser = argparse.ArgumentParser(description='Analyze CLI test failures')
    parser.add_argument('--create-issues', action='store_true',
                       help='Create GitHub issues for failures')
    parser.add_argument('--limit', type=int, default=10,
                       help='Maximum number of issues to create (default: 10)')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be created without creating (default: True)')
    parser.add_argument('--no-dry-run', dest='dry_run', action='store_false',
                       help='Actually create issues (use with caution)')
    args = parser.parse_args()
    
    # Collect failures
    failures = run_tests_and_collect_failures()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_failures = sum(len(f) for f in failures.values())
    print(f"Total failures: {total_failures}")
    
    if total_failures == 0:
        print("No failures found! All CLI tests are passing.")
        return
    
    # Display failures by category
    for category, category_failures in failures.items():
        if category_failures:
            print(f"\n{category}: {len(category_failures)} failures")
            for failure in category_failures:
                print(f"  - {failure.test_name}: {failure.error_type}")
    
    # Create GitHub issues if requested
    if args.create_issues and GITHUB_AVAILABLE:
        token = os.environ.get('GITHUB_TOKEN')
        if not token:
            print("\nError: GITHUB_TOKEN environment variable not set")
            print("Please set it in your .env file or environment")
            return
        
        print(f"\nCreating GitHub issues with intelligent grouping (limit: {args.limit})...")
        
        try:
            g = Github(token)
            # Get repo name from environment or use default (same as E2E script)
            repo_name = os.environ.get('GITHUB_REPO', 'hipotures/mdm')
            print(f"Using repository: {repo_name}")
            
            repo = g.get_repo(repo_name)
            
            # Get existing issues
            existing_issues = list(repo.get_issues(state='all'))
            
            print(f"\nTotal failures to process: {total_failures}")
            
            # Create individual issues
            created = 0
            for category, category_failures in failures.items():
                for failure in category_failures:
                    if created >= args.limit:
                        print(f"\nReached limit of {args.limit} issues")
                        return
                    
                    result = create_individual_issue(repo, category, failure, existing_issues, dry_run=args.dry_run)
                    print(f"{category} - {failure.test_name}: {result}")
                    if "Created" in result or "[DRY RUN]" in result:
                        created += 1
                    
        except Exception as e:
            print(f"\nError accessing GitHub: {e}")
    
    elif args.create_issues and not GITHUB_AVAILABLE:
        print("\nError: PyGithub not installed. Cannot create issues.")
        print("Install with: uv pip install PyGithub")


if __name__ == "__main__":
    main()