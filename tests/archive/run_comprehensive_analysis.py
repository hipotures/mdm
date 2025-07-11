#!/usr/bin/env python3
"""
Comprehensive test analyzer - runs all tests in batches and generates detailed report.
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re


def find_all_test_ids():
    """Find all test IDs in the test directory."""
    test_dir = Path(__file__).parent
    test_ids = []
    
    for test_file in test_dir.rglob("test_*.py"):
        if "__pycache__" in str(test_file):
            continue
            
        with open(test_file) as f:
            content = f.read()
        
        # Find all test IDs
        for match in re.finditer(r'@pytest\.mark\.mdm_id\("([^"]+)"\)', content):
            test_ids.append(match.group(1))
    
    return sorted(test_ids)


def run_test_batch(test_category):
    """Run all tests in a category."""
    test_dir = Path(__file__).parent
    
    # Find test directory for category
    if test_category.startswith("1"):
        test_path = test_dir / "test_01_config"
    elif test_category.startswith("2"):
        test_path = test_dir / "test_02_dataset"
    else:
        return None
    
    # Run tests
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_path),
        "-v",
        "--tb=short",
        "--json-report",
        "--json-report-file=/tmp/mdm_batch_report.json"
    ]
    
    print(f"Running category {test_category} tests...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results
    report_file = Path("/tmp/mdm_batch_report.json")
    if report_file.exists():
        with open(report_file) as f:
            return json.load(f)
    
    return {
        "summary": {"total": 0, "passed": 0, "failed": 0},
        "tests": [],
        "exitcode": result.returncode
    }


def analyze_failures(report_data):
    """Analyze test failures and categorize them."""
    error_categories = defaultdict(list)
    
    for test in report_data.get("tests", []):
        if test.get("outcome") == "failed":
            test_name = test.get("nodeid", "")
            
            # Extract test ID from nodeid
            match = re.search(r'test_(\d+)_', test_name)
            if match:
                test_id_prefix = match.group(1)
                
                # Get error details
                if "call" in test and "longrepr" in test["call"]:
                    error_text = test["call"]["longrepr"]
                    
                    # Categorize error
                    if "AssertionError" in error_text:
                        if "exists()" in error_text:
                            error_categories["missing_file_or_directory"].append(test_name)
                        elif "not in result.stdout" in error_text:
                            error_categories["output_mismatch"].append(test_name)
                        else:
                            error_categories["assertion_failed"].append(test_name)
                    elif "subprocess.CalledProcessError" in error_text:
                        error_categories["command_failed"].append(test_name)
                    elif "ModuleNotFoundError" in error_text:
                        error_categories["import_error"].append(test_name)
                    else:
                        error_categories["other"].append(test_name)
    
    return dict(error_categories)


def generate_summary_report(all_results):
    """Generate a comprehensive summary report."""
    report = []
    report.append("# MDM E2E Test Comprehensive Analysis")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overall summary
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for category, data in all_results.items():
        summary = data.get("summary", {})
        total_tests += summary.get("total", 0)
        total_passed += summary.get("passed", 0)
        total_failed += summary.get("failed", 0)
        total_skipped += summary.get("skipped", 0)
    
    report.append("\n## Overall Summary")
    report.append(f"- **Total tests:** {total_tests}")
    if total_tests > 0:
        report.append(f"- **Passed:** {total_passed} ({total_passed/total_tests*100:.1f}%)")
        report.append(f"- **Failed:** {total_failed} ({total_failed/total_tests*100:.1f}%)")
        report.append(f"- **Skipped:** {total_skipped} ({total_skipped/total_tests*100:.1f}%)")
    
    # Results by category
    report.append("\n## Results by Category")
    for category, data in sorted(all_results.items()):
        summary = data.get("summary", {})
        report.append(f"\n### Category {category}")
        report.append(f"- Total: {summary.get('total', 0)}")
        report.append(f"- Passed: {summary.get('passed', 0)}")
        report.append(f"- Failed: {summary.get('failed', 0)}")
        report.append(f"- Skipped: {summary.get('skipped', 0)}")
        
        # Show error categories
        if "error_categories" in data and data["error_categories"]:
            report.append("\n**Error Categories:**")
            for error_type, tests in data["error_categories"].items():
                report.append(f"- {error_type}: {len(tests)} tests")
    
    # Common issues
    report.append("\n## Common Issues Summary")
    all_errors = defaultdict(int)
    for category, data in all_results.items():
        for error_type, tests in data.get("error_categories", {}).items():
            all_errors[error_type] += len(tests)
    
    if all_errors:
        for error_type, count in sorted(all_errors.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- **{error_type.replace('_', ' ').title()}**: {count} tests")
    
    # Recommendations
    report.append("\n## Recommendations")
    if all_errors.get("missing_file_or_directory", 0) > 0:
        report.append("- Check dataset directory structure - tests expect datasets in `datasets/` subdirectory")
    if all_errors.get("output_mismatch", 0) > 0:
        report.append("- Review command output format - may have changed since tests were written")
    if all_errors.get("command_failed", 0) > 0:
        report.append("- Some MDM commands are failing - check for duplicate dataset names or missing features")
    
    return "\n".join(report)


def main():
    """Main entry point."""
    print("MDM E2E Test Comprehensive Analysis")
    print("=" * 80)
    print("This will run all tests by category and analyze results.")
    print()
    
    # Get all test IDs to determine categories
    test_ids = find_all_test_ids()
    categories = set(test_id.split('.')[0] for test_id in test_ids)
    
    print(f"Found {len(test_ids)} tests across {len(categories)} categories")
    print()
    
    all_results = {}
    
    # Run tests by category
    for category in sorted(categories):
        print(f"\n{'='*60}")
        print(f"Running Category {category} tests...")
        print(f"{'='*60}")
        
        start_time = time.time()
        report_data = run_test_batch(category)
        elapsed = time.time() - start_time
        
        if report_data:
            # Analyze failures
            error_categories = analyze_failures(report_data)
            report_data["error_categories"] = error_categories
            
            # Store results
            all_results[category] = report_data
            
            # Print immediate feedback
            summary = report_data.get("summary", {})
            print(f"\nCategory {category} Results:")
            print(f"- Total: {summary.get('total', 0)}")
            print(f"- Passed: {summary.get('passed', 0)}")
            print(f"- Failed: {summary.get('failed', 0)}")
            print(f"- Time: {elapsed:.2f}s")
            
            if error_categories:
                print("\nError types found:")
                for error_type, tests in error_categories.items():
                    print(f"  - {error_type}: {len(tests)} tests")
    
    # Generate and save comprehensive report
    print("\n" + "="*80)
    print("Generating comprehensive report...")
    
    report = generate_summary_report(all_results)
    
    # Save report
    report_file = Path(__file__).parent / "comprehensive_test_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_file}")
    
    # Save raw data
    data_file = Path(__file__).parent / "test_results_data.json"
    with open(data_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Raw data saved to: {data_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY:")
    print(report.split("## Results by Category")[0])


if __name__ == "__main__":
    main()