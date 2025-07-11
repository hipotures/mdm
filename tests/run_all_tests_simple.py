#!/usr/bin/env python3
"""Simple test runner that actually executes tests and reports failures."""

import subprocess
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils import GitHubConfig, GitHubIssueManager, check_github_availability


def run_pytest_on_directory(test_dir: str, test_name: str) -> Dict[str, Any]:
    """Run pytest on a directory and return results."""
    cmd = [
        sys.executable, "-m", "pytest",
        test_dir,
        "-v",
        "--tb=short",
        "--no-header",
        "-q"
    ]
    
    print(f"\nRunning {test_name} tests...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output for failures
    failures = []
    lines = result.stdout.split('\n') + result.stderr.split('\n')
    
    for line in lines:
        if "FAILED" in line and "::" in line:
            failures.append(line.strip())
    
    # Get summary
    passed = failed = 0
    for line in lines:
        if "passed" in line and "failed" not in line:
            match = re.search(r'(\d+) passed', line)
            if match:
                passed = int(match.group(1))
        if "failed" in line:
            match = re.search(r'(\d+) failed', line)
            if match:
                failed = int(match.group(1))
    
    return {
        "name": test_name,
        "directory": test_dir,
        "passed": passed,
        "failed": failed,
        "failures": failures,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple test runner with GitHub integration')
    parser.add_argument('--github', action='store_true', help='Create GitHub issues')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually create issues')
    parser.add_argument('--limit', type=int, default=10, help='Max issues to create')
    args = parser.parse_args()
    
    # Test directories to run
    test_suites = [
        ("tests/unit/cli", "Unit CLI"),
        ("tests/unit/api", "Unit API"),
        ("tests/unit/dataset", "Unit Dataset"),
        ("tests/unit/storage", "Unit Storage"),
        ("tests/unit/services", "Unit Services"),
        ("tests/integration", "Integration"),
        ("tests/e2e", "E2E")
    ]
    
    all_results = []
    total_passed = 0
    total_failed = 0
    
    print("MDM Test Runner")
    print("=" * 60)
    
    for test_dir, test_name in test_suites:
        if Path(test_dir).exists():
            result = run_pytest_on_directory(test_dir, test_name)
            all_results.append(result)
            total_passed += result["passed"]
            total_failed += result["failed"]
            
            if result["failed"] > 0:
                print(f"✗ {test_name}: {result['failed']} failures")
            else:
                print(f"✓ {test_name}: {result['passed']} passed")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    
    # Create GitHub issues if requested
    if args.github and total_failed > 0:
        if not check_github_availability():
            print("\nError: GitHub integration not available")
            return 1
        
        print(f"\n{'Creating' if not args.no_dry_run else 'Would create'} GitHub issues...")
        
        config = GitHubConfig.from_env()
        github_manager = GitHubIssueManager(config)
        
        issues_created = 0
        for result in all_results:
            for failure in result["failures"][:args.limit]:
                if issues_created >= args.limit:
                    break
                
                # Parse failure
                parts = failure.split("::")
                if len(parts) >= 2:
                    file_path = parts[0]
                    test_name = parts[1].split()[0]
                else:
                    file_path = result["directory"]
                    test_name = failure
                
                # Create issue
                issue_data = github_manager.format_test_failure_issue(
                    test_name=test_name,
                    error_type="Test Failure",
                    error_message=failure,
                    category=result["name"],
                    file_path=file_path
                )
                
                result = github_manager.create_or_update_issue(
                    title=issue_data["title"],
                    body=issue_data["body"],
                    labels=issue_data["labels"],
                    issue_id=issue_data["issue_id"],
                    dry_run=not args.no_dry_run
                )
                
                print(f"  {test_name}: {result['message']}")
                issues_created += 1
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_passed": total_passed,
        "total_failed": total_failed,
        "test_suites": all_results
    }
    
    report_file = f"test-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_file}")
    
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())