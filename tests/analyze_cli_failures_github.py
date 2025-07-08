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


def create_github_issue(repo, failure: TestFailure, existing_issues: set) -> Optional[str]:
    """Create a GitHub issue for a test failure."""
    # Check if issue already exists
    fingerprint = failure.get_fingerprint()
    issue_title = f"[CLI Test] {failure.category}: {failure.test_name} - {failure.error_type}"
    
    # Truncate title if too long
    if len(issue_title) > 200:
        issue_title = issue_title[:197] + "..."
    
    # Check existing issues by title
    for issue in existing_issues:
        if failure.test_name in issue.title and failure.error_type in issue.title:
            return f"Issue already exists: #{issue.number}"
    
    # Create issue body
    body = f"""## Test Failure Details

**Test Category:** {failure.category}
**Test File:** `{failure.file_path}`
**Test Name:** `{failure.test_name}`
**Error Type:** `{failure.error_type}`
**Fingerprint:** `{fingerprint}`

### Error Message
```
{failure.error_message}
```

### How to Reproduce
```bash
pytest {failure.file_path}::{failure.test_name.split('[')[0]}
```

### Environment
- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Test Type:** CLI Test
- **MDM Version:** 0.1.0

### Labels
- bug
- cli
- test-failure

---
*This issue was automatically created by the CLI test analyzer.*
"""
    
    # Truncate body if too long (GitHub limit is 65536)
    if len(body) > 65000:
        body = body[:64900] + "\n\n... (truncated)"
    
    try:
        issue = repo.create_issue(
            title=issue_title,
            body=body,
            labels=["bug", "cli", "test-failure"]
        )
        return f"Created issue: #{issue.number}"
    except GithubException as e:
        return f"Failed to create issue: {e}"


def main():
    """Main function to analyze failures and optionally create issues."""
    parser = argparse.ArgumentParser(description='Analyze CLI test failures')
    parser.add_argument('--create-issues', action='store_true',
                       help='Create GitHub issues for failures')
    parser.add_argument('--limit', type=int, default=10,
                       help='Maximum number of issues to create (default: 10)')
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
        
        print(f"\nCreating GitHub issues (limit: {args.limit})...")
        
        try:
            g = Github(token)
            repo = g.get_repo("anthropics/mdm")  # Adjust repo name as needed
            
            # Get existing issues
            existing_issues = set(repo.get_issues(state='all'))
            
            # Create issues
            created = 0
            for category, category_failures in failures.items():
                for failure in category_failures:
                    if created >= args.limit:
                        print(f"\nReached limit of {args.limit} issues")
                        return
                    
                    result = create_github_issue(repo, failure, existing_issues)
                    print(f"  {failure.test_name}: {result}")
                    created += 1
                    
        except Exception as e:
            print(f"\nError accessing GitHub: {e}")
    
    elif args.create_issues and not GITHUB_AVAILABLE:
        print("\nError: PyGithub not installed. Cannot create issues.")
        print("Install with: uv pip install PyGithub")


if __name__ == "__main__":
    main()