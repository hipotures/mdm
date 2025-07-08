#!/usr/bin/env python3
"""Simple test runner that generates a detailed report."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import re


def parse_pytest_output(output):
    """Parse pytest output to extract test results."""
    results = {
        "passed": [],
        "failed": [],
        "skipped": [],
        "errors": {}
    }
    
    # Parse test results
    for line in output.split('\n'):
        if "PASSED" in line:
            match = re.search(r'(test_\w+\.py::\S+)', line)
            if match:
                results["passed"].append(match.group(1))
        elif "FAILED" in line:
            match = re.search(r'(test_\w+\.py::\S+)', line)
            if match:
                results["failed"].append(match.group(1))
        elif "SKIPPED" in line:
            match = re.search(r'(test_\w+\.py::\S+)', line)
            if match:
                results["skipped"].append(match.group(1))
    
    # Extract failure details
    current_test = None
    in_failure = False
    failure_lines = []
    
    for line in output.split('\n'):
        if line.startswith("_"):
            in_failure = True
            # Extract test name from failure header
            match = re.search(r'_+ (\S+) _+', line)
            if match:
                current_test = match.group(1)
                failure_lines = []
        elif in_failure and line.startswith("="):
            # End of failure section
            if current_test and failure_lines:
                results["errors"][current_test] = '\n'.join(failure_lines)
            in_failure = False
            current_test = None
        elif in_failure:
            failure_lines.append(line)
    
    return results


def analyze_error_patterns(errors):
    """Analyze error patterns and categorize them."""
    categories = {
        "file_not_found": [],
        "assertion_failed": [],
        "command_failed": [],
        "output_mismatch": [],
        "already_exists": [],
        "other": []
    }
    
    for test, error in errors.items():
        if "exists()" in error and "assert False" in error:
            categories["file_not_found"].append(test)
        elif "already exists" in error:
            categories["already_exists"].append(test)
        elif "CalledProcessError" in error:
            categories["command_failed"].append(test)
        elif "not in result.stdout" in error or "not found" in error:
            categories["output_mismatch"].append(test)
        elif "AssertionError" in error:
            categories["assertion_failed"].append(test)
        else:
            categories["other"].append(test)
    
    return {k: v for k, v in categories.items() if v}


def run_test_category(category_path):
    """Run tests for a specific category."""
    print(f"\nRunning tests in {category_path.name}...")
    print("-" * 60)
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(category_path),
        "-v",
        "--tb=short",
        "--no-header"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results
    parsed = parse_pytest_output(result.stdout)
    
    # Add error analysis
    if parsed["errors"]:
        parsed["error_categories"] = analyze_error_patterns(parsed["errors"])
    
    return parsed


def generate_report(all_results):
    """Generate comprehensive test report."""
    report = []
    report.append("# MDM E2E Test Analysis Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate totals
    total_passed = sum(len(r["passed"]) for r in all_results.values())
    total_failed = sum(len(r["failed"]) for r in all_results.values())
    total_skipped = sum(len(r["skipped"]) for r in all_results.values())
    total_tests = total_passed + total_failed + total_skipped
    
    report.append("\n## Summary")
    report.append(f"- **Total tests:** {total_tests}")
    if total_tests > 0:
        report.append(f"- **Passed:** {total_passed} ({total_passed/total_tests*100:.1f}%)")
        report.append(f"- **Failed:** {total_failed} ({total_failed/total_tests*100:.1f}%)")
        report.append(f"- **Skipped:** {total_skipped} ({total_skipped/total_tests*100:.1f}%)")
    
    # Error categories across all tests
    all_error_categories = {}
    for results in all_results.values():
        for category, tests in results.get("error_categories", {}).items():
            if category not in all_error_categories:
                all_error_categories[category] = []
            all_error_categories[category].extend(tests)
    
    if all_error_categories:
        report.append("\n## Error Categories")
        for category, tests in sorted(all_error_categories.items(), key=lambda x: len(x[1]), reverse=True):
            report.append(f"\n### {category.replace('_', ' ').title()} ({len(tests)} tests)")
            for test in tests[:3]:  # Show first 3
                report.append(f"- {test}")
            if len(tests) > 3:
                report.append(f"- ... and {len(tests) - 3} more")
    
    # Detailed results by directory
    report.append("\n## Detailed Results")
    for path, results in all_results.items():
        report.append(f"\n### {path.name}")
        report.append(f"- Passed: {len(results['passed'])}")
        report.append(f"- Failed: {len(results['failed'])}")
        report.append(f"- Skipped: {len(results['skipped'])}")
        
        # Show some failures
        if results["failed"]:
            report.append("\n**Failed tests:**")
            for test in results["failed"][:5]:
                report.append(f"- {test}")
                # Show error snippet if available
                test_name = test.split("::")[-1]
                if test_name in results["errors"]:
                    error_lines = results["errors"][test_name].strip().split('\n')
                    for line in error_lines[:3]:
                        if line.strip():
                            report.append(f"  > {line.strip()}")
    
    # Recommendations
    report.append("\n## Recommendations")
    if "already_exists" in all_error_categories:
        report.append("- **Dataset Already Exists**: Tests are not properly cleaning up between runs")
        report.append("  - Solution: Ensure each test uses a unique dataset name or force cleanup")
    if "file_not_found" in all_error_categories:
        report.append("- **Files Not Created**: Expected files/directories are not being created")
        report.append("  - Solution: Check if MDM is creating files in the expected location")
        report.append("  - Verify MDM_HOME_DIR is being respected")
    if "output_mismatch" in all_error_categories:
        report.append("- **Output Format Changed**: Command output doesn't match expected format")
        report.append("  - Solution: Update tests to match current output format")
    
    return "\n".join(report)


def main():
    """Main entry point."""
    test_dir = Path(__file__).parent
    
    print("MDM E2E Test Runner")
    print("=" * 80)
    
    # Find test directories
    test_dirs = [
        test_dir / "test_01_config",
        test_dir / "test_02_dataset"
    ]
    
    all_results = {}
    
    # Run tests for each directory
    for test_path in test_dirs:
        if test_path.exists():
            results = run_test_category(test_path)
            all_results[test_path] = results
            
            # Print immediate feedback
            print(f"\nResults for {test_path.name}:")
            print(f"  Passed: {len(results['passed'])}")
            print(f"  Failed: {len(results['failed'])}")
            print(f"  Skipped: {len(results['skipped'])}")
            
            if results.get("error_categories"):
                print("  Error types:")
                for cat, tests in results["error_categories"].items():
                    print(f"    - {cat}: {len(tests)} tests")
    
    # Generate report
    print("\n" + "=" * 80)
    print("Generating report...")
    
    report = generate_report(all_results)
    
    # Save report
    report_file = test_dir / "test_analysis_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    lines = report.split('\n')
    for line in lines[:20]:  # Print first 20 lines
        print(line)


if __name__ == "__main__":
    main()