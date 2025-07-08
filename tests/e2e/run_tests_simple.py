#!/usr/bin/env python3
"""
Simple test runner for MDM E2E tests.
Runs tests and generates reports.
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def find_all_tests():
    """Find all MDM tests and their IDs."""
    test_dir = Path(__file__).parent
    tests = {}
    
    for test_file in test_dir.rglob("test_*.py"):
        if "__pycache__" in str(test_file):
            continue
            
        with open(test_file) as f:
            content = f.read()
            
        # Find all test IDs in this file
        import re
        for match in re.finditer(r'@pytest\.mark\.mdm_id\("([^"]+)"\)', content):
            test_id = match.group(1)
            tests[test_id] = str(test_file)
    
    return tests


def run_tests(test_pattern=None):
    """Run tests and return results."""
    test_dir = Path(__file__).parent
    
    if test_pattern:
        # Run specific test or category
        if '.' in test_pattern:
            # Specific test ID
            tests = find_all_tests()
            if test_pattern in tests:
                cmd = [sys.executable, "-m", "pytest", tests[test_pattern], 
                       "-k", test_pattern.replace('.', '_'), "-v", "--tb=short"]
            else:
                print(f"Test {test_pattern} not found")
                return None
        else:
            # Category (e.g., "1" or "2")
            cmd = [sys.executable, "-m", "pytest", 
                   str(test_dir / f"test_0{test_pattern}_*"), "-v", "--tb=short"]
    else:
        # Run all tests
        cmd = [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"]
    
    # Add JSON report
    report_file = Path("/tmp/mdm_test_report.json")
    cmd.extend(["--json-report", f"--json-report-file={report_file}"])
    
    # Run tests
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results
    if report_file.exists():
        with open(report_file) as f:
            return json.load(f)
    else:
        return {"summary": {"total": 0, "passed": 0, "failed": 0}}


def generate_report(results):
    """Generate markdown report from test results."""
    report = []
    report.append("# MDM E2E Test Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary
    summary = results.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    
    report.append("\n## Summary")
    report.append(f"- **Total tests:** {total}")
    if total > 0:
        report.append(f"- **Passed:** {passed} ({passed/total*100:.1f}%)")
        report.append(f"- **Failed:** {failed} ({failed/total*100:.1f}%)")
        report.append(f"- **Skipped:** {skipped} ({skipped/total*100:.1f}%)")
    
    # Failed tests
    if failed > 0 and "tests" in results:
        report.append("\n## Failed Tests")
        for test in results["tests"]:
            if test.get("outcome") == "failed":
                report.append(f"\n### {test.get('nodeid', 'Unknown test')}")
                if "call" in test and "longrepr" in test["call"]:
                    report.append("```")
                    report.append(test["call"]["longrepr"])
                    report.append("```")
    
    return "\n".join(report)


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Run MDM E2E tests")
    parser.add_argument("test_id", nargs="?", help="Test ID or category to run")
    parser.add_argument("--output", "-o", help="Output report file")
    
    args = parser.parse_args()
    
    # Run tests
    print(f"Running tests{' for ' + args.test_id if args.test_id else ''}...")
    results = run_tests(args.test_id)
    
    if results:
        # Generate report
        report = generate_report(results)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")
        else:
            print("\n" + report)
    
    # Exit with test status
    sys.exit(0 if results and results.get("summary", {}).get("failed", 0) == 0 else 1)


if __name__ == "__main__":
    main()