#!/usr/bin/env python3
"""Analyze repository unit test failures and create GitHub issues."""

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

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
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


def run_tests_and_collect_failures():
    """Run all repository unit tests and collect failure information."""
    test_dir = Path(__file__).parent
    failures = defaultdict(list)
    
    # Test files in the repositories directory
    test_files = [
        ("test_feature_registry.py", "FeatureRegistry"),
        ("test_backend_factory.py", "BackendFactory"),
        ("test_dataset_manager.py", "DatasetManager"),
        ("test_storage_backend.py", "StorageBackend"),
        # test_metadata_repository.py doesn't exist - removed
    ]
    
    if RICH_AVAILABLE:
        console.print("\n[bold]Analyzing Repository Unit Test failures...[/bold]")
        console.rule()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for test_file, category_name in test_files:
                test_path = test_dir / test_file
                if not test_path.exists():
                    task = progress.add_task(f"[yellow]Skipping {category_name} (not found)[/yellow]", total=1)
                    progress.update(task, completed=1)
                    continue
                
                task = progress.add_task(f"Testing {category_name}...", total=1)
                
                cmd = [
                    sys.executable, "-m", "pytest",
                    str(test_path),
                    "-v", "--tb=short",
                    "--no-header"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Parse output for failures
                category_failure_count = 0
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if "FAILED" in line:
                        # Extract test name
                        match = re.search(r'(test_\w+\.py)::(\S+)\s+FAILED', line)
                        if match:
                            test_name = f"{match.group(1)}::{match.group(2)}"
                            category_failure_count += 1
                            
                            # Look for error details in subsequent lines
                            error_info = ""
                            for j in range(i+1, min(i+20, len(lines))):
                                if lines[j].strip():
                                    if "short test summary" in lines[j]:
                                        break
                                    error_info += lines[j] + "\n"
                            
                            # Extract error type
                            error_type = classify_error(error_info)
                            
                            failures[category_name].append({
                                "test": test_name,
                                "error_type": error_type,
                                "error_info": error_info.strip()
                            })
                    elif "ERROR" in line and "::" in line:
                        # Handle setup/teardown errors
                        match = re.search(r'(test_\w+\.py)::(\S+)\s+ERROR', line)
                        if match:
                            test_name = f"{match.group(1)}::{match.group(2)}"
                            category_failure_count += 1
                            
                            # Look for error details
                            error_info = ""
                            for j in range(i+1, min(i+20, len(lines))):
                                if lines[j].strip():
                                    if "short test summary" in lines[j]:
                                        break
                                    error_info += lines[j] + "\n"
                            
                            error_type = classify_error(error_info)
                            
                            failures[category_name].append({
                                "test": test_name,
                                "error_type": error_type,
                                "error_info": error_info.strip()
                            })
                
                # Update progress with result
                if category_failure_count > 0:
                    progress.update(task, description=f"[red]✗[/red] {category_name} ({category_failure_count} failures)", completed=1)
                else:
                    progress.update(task, description=f"[green]✓[/green] {category_name}", completed=1)
    else:
        print("Analyzing repository unit test failures...")
        print("=" * 80)
        
        for test_file, category_name in test_files:
            test_path = test_dir / test_file
            if not test_path.exists():
                print(f"\nSkipping {category_name} (file not found)")
                continue
                
            print(f"\nTesting {category_name}...")
            
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_path),
                "-v", "--tb=short",
                "--no-header"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse output for failures
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if "FAILED" in line or ("ERROR" in line and "::" in line):
                    # Extract test name
                    match = re.search(r'(test_\w+\.py)::(\S+)\s+(FAILED|ERROR)', line)
                    if match:
                        test_name = f"{match.group(1)}::{match.group(2)}"
                        
                        # Look for error details in subsequent lines
                        error_info = ""
                        for j in range(i+1, min(i+20, len(lines))):
                            if lines[j].strip():
                                if "short test summary" in lines[j]:
                                    break
                                error_info += lines[j] + "\n"
                        
                        # Extract error type
                        error_type = classify_error(error_info)
                        
                        failures[category_name].append({
                            "test": test_name,
                            "error_type": error_type,
                            "error_info": error_info.strip()
                        })
    
    return failures


def classify_error(error_info: str) -> str:
    """Classify the error type based on error information."""
    if "ImportError" in error_info:
        if "TableType" in error_info:
            return "Import error - TableType"
        else:
            return "Import error"
    elif "AttributeError" in error_info:
        if "__truediv__" in error_info:
            return "Path mocking error"
        elif "read-only" in error_info:
            return "Read-only attribute error"
        elif "does not have the attribute" in error_info:
            return "Missing attribute error"
        else:
            return "Attribute error"
    elif "FileNotFoundError" in error_info:
        return "File/Directory not found"
    elif "AssertionError" in error_info:
        return "Assertion failed"
    elif "TypeError" in error_info:
        if "missing 1 required positional argument" in error_info:
            return "Mock setup error"
        else:
            return "Type error"
    elif "KeyError" in error_info:
        return "Key error"
    elif "ValidationError" in error_info:
        return "Pydantic validation error"
    elif "ModuleNotFoundError" in error_info:
        return "Module not found"
    else:
        return "Unknown error"


def generate_failure_report(failures):
    """Generate a markdown report of current failures."""
    report = []
    report.append("# Current MDM Repository Unit Test Failures")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary
    total_failures = sum(len(tests) for tests in failures.values())
    report.append(f"\n## Summary: {total_failures} failures across {len(failures)} test suites")
    
    # Group by error type
    error_types = defaultdict(list)
    for category, tests in failures.items():
        for test in tests:
            error_types[test["error_type"]].append({
                "category": category,
                "test": test["test"]
            })
    
    report.append("\n## Failures by Error Type")
    for error_type, tests in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
        report.append(f"\n### {error_type} ({len(tests)} failures)")
        for test in tests[:5]:  # Show first 5
            report.append(f"- **{test['category']}**: {test['test']}")
        if len(tests) > 5:
            report.append(f"- ... and {len(tests) - 5} more")
    
    # Detailed failures by category
    report.append("\n## Detailed Failures by Test Suite")
    for category, tests in failures.items():
        if tests:
            report.append(f"\n### {category} ({len(tests)} failures)")
            for test in tests[:5]:  # Limit to first 5 for readability
                test_name = test["test"].split("::")[-1]
                report.append(f"\n#### {test_name}")
                report.append(f"- **Error Type**: {test['error_type']}")
                
                # Extract key error message
                error_lines = test["error_info"].split('\n')
                for line in error_lines:
                    if "Error:" in line or "assert" in line:
                        report.append(f"- **Message**: {line.strip()[:200]}")
                        break
    
    # Common issues and fixes
    report.append("\n## Common Issues and Recommended Fixes")
    
    if "Missing attribute error" in error_types:
        report.append("\n### Missing Attribute Error")
        report.append("- **Issue**: Tests are trying to patch attributes that don't exist")
        report.append("- **Fix**: Check import paths and update patches to correct locations")
    
    if "Import error - TableType" in error_types:
        report.append("\n### Import Error - TableType")
        report.append("- **Issue**: TableType enum doesn't exist in mdm.models.enums")
        report.append("- **Fix**: Replace TableType references with string values")
    
    if "Path mocking error" in error_types:
        report.append("\n### Path Mocking Issues")
        report.append("- **Issue**: Cannot mock PosixPath object attributes")
        report.append("- **Fix**: Use global Path method mocking or alternative approaches")
    
    if "Mock setup error" in error_types:
        report.append("\n### Mock Setup Errors")
        report.append("- **Issue**: Mocks not configured with required methods/attributes")
        report.append("- **Fix**: Add missing methods to mock specs or configure return values")
    
    return "\n".join(report)


class GitHubIssueManager:
    """Manage GitHub issues for test failures."""
    
    def __init__(self, token: str, owner: str = None, repo: str = None, dry_run: bool = False):
        # Get repo from environment variable or use defaults
        repo_name = os.environ.get('GITHUB_REPO', 'hipotures/mdm')
        if '/' in repo_name:
            owner_from_env, repo_from_env = repo_name.split('/', 1)
            self.owner = owner or owner_from_env
            self.repo = repo or repo_from_env
        else:
            self.owner = owner or "hipotures"
            self.repo = repo or "mdm"
            
        self.dry_run = dry_run
        self.created_issues = []
        self.updated_issues = []
        
        if GITHUB_AVAILABLE and token:
            self.github = Github(token)
            try:
                self.repo_obj = self.github.get_repo(f"{self.owner}/{self.repo}")
            except GithubException as e:
                print(f"Error accessing repository: {e}")
                self.repo_obj = None
        else:
            self.github = None
            self.repo_obj = None
    
    def generate_issue_id(self, category: str, error_type: str, test_names: List[str]) -> str:
        """Generate deterministic ID for issue deduplication."""
        sorted_tests = sorted(test_names)
        content = f"{category}:{error_type}:{','.join(sorted_tests)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def find_existing_issue(self, category: str, error_type: str, issue_id: str) -> Optional[object]:
        """Search for existing open issues matching the pattern."""
        if not self.repo_obj:
            return None
        
        try:
            # Search for issues with our labels
            query = f"repo:{self.owner}/{self.repo} is:issue is:open label:test-failure label:repository-test in:body {issue_id}"
            issues = self.github.search_issues(query)
            
            for issue in issues:
                # Check if this is our issue by looking for the ID in the body
                if f"Issue ID: {issue_id}" in issue.body:
                    return issue
            
            return None
        except Exception as e:
            print(f"Error searching issues: {e}")
            return None
    
    def get_priority(self, failure_count: int) -> str:
        """Determine priority based on failure count."""
        if failure_count >= 10:
            return "high"
        elif failure_count >= 5:
            return "medium"
        else:
            return "low"
    
    def create_issue_body(self, category: str, error_type: str, failures: List[dict], issue_id: str) -> str:
        """Generate formatted issue body."""
        body = f"""## Summary
Automated repository unit test failure report for {category} tests.

**Error Type**: {error_type}
**Failed Tests**: {len(failures)}
**Test Run Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Issue ID**: {issue_id}

## Failed Tests
"""
        
        for i, failure in enumerate(failures[:10]):  # Limit to first 10 tests
            test_name = failure['test'].split('::')[-1]
            body += f"\n### {i+1}. {test_name}"
            body += f"\n- **Full Test Path**: `{failure['test']}`"
            body += f"\n- **Error Type**: {failure['error_type']}"
            
            # Extract key error message
            error_lines = failure['error_info'].split('\n')
            for line in error_lines:
                if "Error:" in line or "assert" in line:
                    body += f"\n- **Error Message**: `{line.strip()[:200]}`"  # Limit length
                    break
            
            body += "\n"
        
        if len(failures) > 10:
            body += f"\n... and {len(failures) - 10} more failures\n"
        
        # Add common patterns if identifiable
        body += "\n## Common Pattern\n"
        if error_type == "Missing attribute error":
            body += "Tests are trying to patch attributes that don't exist. Check import paths.\n"
        elif error_type == "Import error - TableType":
            body += "Tests are trying to import TableType enum that doesn't exist in the current codebase.\n"
        elif error_type == "Path mocking error":
            body += "Path object attributes cannot be mocked directly. Need alternative mocking strategy.\n"
        elif error_type == "Mock setup error":
            body += "Mock objects are missing required methods or attributes.\n"
        else:
            body += "Multiple tests are failing with similar error patterns.\n"
        
        body += "\n## Suggested Fix\n"
        if error_type == "Missing attribute error":
            body += "- Update patch targets to correct import locations\n"
            body += "- Verify that patched functions exist in the target modules\n"
        elif error_type == "Import error - TableType":
            body += "- Replace TableType.TRAIN with 'train', TableType.TEST with 'test', etc.\n"
            body += "- Remove TableType imports from test files\n"
        elif error_type == "Path mocking error":
            body += "- Use `@patch('pathlib.Path.exists')` at method level\n"
            body += "- Consider using temporary directories instead of mocking\n"
        elif error_type == "Mock setup error":
            body += "- Add missing methods to mock.spec or configure return_value\n"
            body += "- Check actual implementation for required interfaces\n"
        
        body += "\n---\n*This issue was automatically generated by MDM repository test failure analysis*\n"
        body += f"*Run ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}*"
        
        return body
    
    def create_or_update_issue(self, category: str, error_type: str, failures: List[dict]) -> bool:
        """Create a new issue or update existing one."""
        if not self.repo_obj:
            return False
        
        # Generate issue ID for deduplication
        test_names = [f['test'].split('::')[-1] for f in failures]
        issue_id = self.generate_issue_id(category, error_type, test_names)
        
        # Generate labels
        priority = self.get_priority(len(failures))
        labels = [
            "test-failure",
            "repository-test",
            "unit-test",
            f"priority-{priority}",
            f"suite-{category.lower().replace(' ', '-')}",
            f"error-{error_type.lower().replace(' ', '-').replace('/', '')}"
        ]
        
        # Check for existing issue
        existing_issue = self.find_existing_issue(category, error_type, issue_id)
        
        if existing_issue:
            # Update existing issue
            if self.dry_run:
                print(f"[DRY RUN] Would update issue #{existing_issue.number}: {existing_issue.title}")
                return True
            
            try:
                comment = f"""## Test Run Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Still failing with {len(failures)} test failures.

Latest failure details:
"""
                for failure in failures[:3]:  # Show first 3
                    test_name = failure['test'].split('::')[-1]
                    comment += f"\n- **{test_name}**: {failure['error_type']}"
                
                existing_issue.create_comment(comment)
                self.updated_issues.append(existing_issue.number)
                print(f"Updated issue #{existing_issue.number}: {existing_issue.title}")
                return True
            except Exception as e:
                print(f"Error updating issue: {e}")
                return False
        else:
            # Create new issue
            title = f"[Repository Test] {category}: {error_type} - {len(failures)} failures"
            body = self.create_issue_body(category, error_type, failures, issue_id)
            
            if self.dry_run:
                print(f"\n[DRY RUN] Would create issue:")
                print(f"Title: {title}")
                print(f"Labels: {', '.join(labels)}")
                print(f"Body preview (first 500 chars):\n{body[:500]}...")
                return True
            
            try:
                issue = self.repo_obj.create_issue(
                    title=title,
                    body=body,
                    labels=labels
                )
                self.created_issues.append(issue.number)
                print(f"Created issue #{issue.number}: {title}")
                return True
            except Exception as e:
                print(f"Error creating issue: {e}")
                return False
    
    def process_failures(self, failures: Dict[str, List[dict]], max_issues: int = 10) -> None:
        """Process test failures and create/update GitHub issues."""
        if not GITHUB_AVAILABLE:
            print("\nGitHub integration not available. Install PyGithub to enable.")
            return
        
        if not self.repo_obj:
            print("\nGitHub repository not accessible. Check token and repository settings.")
            return
        
        print("\n" + "=" * 80)
        print("GITHUB ISSUE CREATION")
        print("=" * 80)
        
        # Group failures by error type within each category
        issue_count = 0
        for category, tests in failures.items():
            if not tests:
                continue
            
            # Group by error type
            error_groups = defaultdict(list)
            for test in tests:
                error_groups[test['error_type']].append(test)
            
            # Create issues for each error group
            for error_type, error_tests in error_groups.items():
                if issue_count >= max_issues:
                    print(f"\nReached maximum issue limit ({max_issues}). Stopping.")
                    return
                
                if self.create_or_update_issue(category, error_type, error_tests):
                    issue_count += 1
        
        # Summary
        print("\n" + "-" * 40)
        print("GitHub Issue Summary:")
        if self.created_issues:
            print(f"Created {len(self.created_issues)} new issues: {', '.join(f'#{i}' for i in self.created_issues)}")
        if self.updated_issues:
            print(f"Updated {len(self.updated_issues)} existing issues: {', '.join(f'#{i}' for i in self.updated_issues)}")
        if not self.created_issues and not self.updated_issues:
            print("No issues created or updated.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze MDM repository unit test failures and optionally create GitHub issues")
    parser.add_argument(
        "--github-token",
        help="GitHub personal access token for issue creation",
        default=os.environ.get("GITHUB_TOKEN")
    )
    parser.add_argument(
        "--create-issues",
        action="store_true",
        help="Create GitHub issues for failures"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what issues would be created without actually creating them (default: True)"
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Actually create issues (use with caution)"
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=10,
        help="Maximum number of issues to create in one run (default: 10)"
    )
    parser.add_argument(
        "--owner",
        default="hipotures",
        help="GitHub repository owner (default: hipotures)"
    )
    parser.add_argument(
        "--repo",
        default="mdm",
        help="GitHub repository name (default: mdm)"
    )
    
    args = parser.parse_args()
    
    # Display mode information
    if args.create_issues:
        if args.dry_run:
            if RICH_AVAILABLE:
                console.print(Panel.fit(
                    "[bold yellow]DRY RUN MODE[/bold yellow]\n"
                    "Showing what would be created without actually creating issues.\n"
                    "Use --no-dry-run to actually create issues.",
                    title="Mode: DRY RUN",
                    border_style="yellow"
                ))
            else:
                print("\n" + "="*80)
                print("DRY RUN MODE - No issues will be created")
                print("Use --no-dry-run to actually create issues")
                print("="*80 + "\n")
        else:
            if RICH_AVAILABLE:
                console.print(Panel.fit(
                    "[bold red]LIVE MODE[/bold red]\n"
                    "Issues will be created on GitHub!",
                    title="Mode: LIVE",
                    border_style="red"
                ))
            else:
                print("\n" + "="*80)
                print("LIVE MODE - Issues WILL be created on GitHub!")
                print("="*80 + "\n")
    
    # Run tests and collect failures
    failures = run_tests_and_collect_failures()
    
    if not any(failures.values()):
        if RICH_AVAILABLE:
            console.print("\n[green]✓[/green] No failures found! All repository unit tests are passing.")
        else:
            print("\n\nNo failures found! All repository unit tests are passing.")
        return
    
    # Generate report
    report = generate_failure_report(failures)
    
    # Save report
    report_file = Path(__file__).parent / "CURRENT_REPOSITORY_TEST_FAILURES.md"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"\n\nFailure report saved to: {report_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    total = sum(len(tests) for tests in failures.values())
    print(f"Total failures: {total}")
    
    # Calculate pass rate
    total_tests = 0
    expected_test_counts = {
        "FeatureRegistry": 12,
        "BackendFactory": 16,
        "DatasetManager": 23,  # Estimated from existing tests
        "StorageBackend": 15,  # Estimated from existing tests
        # MetadataRepository removed - file doesn't exist
    }
    
    for category, count in expected_test_counts.items():
        if category in failures or any(category in str(f) for f in failures):
            total_tests += count
    
    passing_tests = total_tests - total
    pass_rate = (passing_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTest Statistics:")
    print(f"- Total test cases: {total_tests}")
    print(f"- Passing tests: {passing_tests}")
    print(f"- Failing tests: {total}")
    print(f"- Pass rate: {pass_rate:.1f}%")
    
    print("\nFailures by test suite:")
    for category, tests in failures.items():
        if tests:
            print(f"- {category}: {len(tests)} failures")
    
    # Create GitHub issues if requested
    if args.create_issues:
        if not args.github_token:
            print("\nError: GitHub token required for issue creation.")
            print("Set GITHUB_TOKEN environment variable or use --github-token")
            return
        
        github_manager = GitHubIssueManager(
            token=args.github_token,
            owner=args.owner,
            repo=args.repo,
            dry_run=args.dry_run
        )
        
        github_manager.process_failures(failures, max_issues=args.max_issues)


if __name__ == "__main__":
    main()