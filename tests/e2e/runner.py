#!/usr/bin/env python3
"""Test runner for MDM end-to-end tests with hierarchical selection and reporting."""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest


@dataclass
class TestResult:
    """Result of a single test."""
    test_id: str
    name: str
    status: str  # passed, failed, skipped, error
    duration: float = 0.0
    output: str = ""
    error: str = ""


@dataclass
class TestCategory:
    """Test category with results."""
    id: str
    name: str
    tests: List[TestResult] = field(default_factory=list)
    subcategories: Dict[str, 'TestCategory'] = field(default_factory=dict)
    
    @property
    def total_tests(self) -> int:
        """Total number of tests in this category and subcategories."""
        count = len(self.tests)
        for subcat in self.subcategories.values():
            count += subcat.total_tests
        return count
    
    @property
    def passed_tests(self) -> int:
        """Number of passed tests."""
        count = sum(1 for t in self.tests if t.status == "passed")
        for subcat in self.subcategories.values():
            count += subcat.passed_tests
        return count
    
    @property
    def failed_tests(self) -> int:
        """Number of failed tests."""
        count = sum(1 for t in self.tests if t.status == "failed")
        for subcat in self.subcategories.values():
            count += subcat.failed_tests
        return count


class MDMTestRunner:
    """Runner for MDM end-to-end tests."""
    
    def __init__(self, test_dir: Path = None):
        """Initialize test runner."""
        self.test_dir = test_dir or Path(__file__).parent
        self.results: Dict[str, TestCategory] = {}
        
    def run_test(self, test_id: str, verbose: bool = False) -> TestResult:
        """
        Run a single test by ID (e.g., '1.1.1').
        
        Args:
            test_id: Test ID from manual checklist
            verbose: Show test output
            
        Returns:
            TestResult object
        """
        # Find test file based on ID pattern
        test_file = self._find_test_file(test_id)
        if not test_file:
            return TestResult(
                test_id=test_id,
                name=f"Test {test_id}",
                status="error",
                error=f"Test file not found for ID {test_id}"
            )
        
        # Run pytest with marker - use custom expression
        # We'll select tests that have the mdm_id marker with the specific value
        args = [
            str(test_file),
            "-m", f"mdm_id",
            "-k", test_id,
            "--tb=short",
            "-q"
        ]
        
        if verbose:
            args.append("-v")
        
        result = subprocess.run(
            [sys.executable, "-m", "pytest"] + args,
            capture_output=True,
            text=True
        )
        
        # Parse result
        return self._parse_test_result(test_id, result)
    
    def run_category(self, category_id: str, verbose: bool = False) -> TestCategory:
        """
        Run all tests in a category (e.g., '1.1' or '2').
        
        Args:
            category_id: Category ID
            verbose: Show test output
            
        Returns:
            TestCategory with results
        """
        # Determine test directory based on category
        parts = category_id.split('.')
        if len(parts) == 1:
            # Top level category (e.g., '1')
            test_pattern = f"test_0{parts[0]}_*"
        elif len(parts) == 2:
            # Subcategory (e.g., '1.2')
            test_pattern = f"test_{parts[0]}{parts[1]}_*.py"
        else:
            raise ValueError(f"Invalid category ID: {category_id}")
        
        # Find matching test files
        test_files = list(self.test_dir.glob(f"**/{test_pattern}"))
        if not test_files:
            return TestCategory(
                id=category_id,
                name=f"Category {category_id}",
                tests=[TestResult(
                    test_id=category_id,
                    name=f"Category {category_id}",
                    status="error",
                    error=f"No test files found for category {category_id}"
                )]
            )
        
        # Run tests
        args = ["-q", "--tb=short"]
        if verbose:
            args.append("-v")
        
        args.extend([str(f) for f in test_files])
        
        result = subprocess.run(
            [sys.executable, "-m", "pytest"] + args,
            capture_output=True,
            text=True
        )
        
        # Parse results
        return self._parse_category_results(category_id, result)
    
    def run_all(self, verbose: bool = False) -> Dict[str, TestCategory]:
        """
        Run all end-to-end tests.
        
        Args:
            verbose: Show test output
            
        Returns:
            Dictionary of test results by category
        """
        args = [
            str(self.test_dir),
            "-q",
            "--tb=short",
            "--json-report",
            "--json-report-file=/tmp/mdm_test_report.json"
        ]
        
        if verbose:
            args.append("-v")
        
        result = subprocess.run(
            [sys.executable, "-m", "pytest"] + args,
            capture_output=True,
            text=True
        )
        
        # Parse JSON report if available
        report_file = Path("/tmp/mdm_test_report.json")
        if report_file.exists():
            with open(report_file) as f:
                report_data = json.load(f)
            self._parse_json_report(report_data)
        else:
            # Fallback to text parsing
            self._parse_text_output(result.stdout)
        
        return self.results
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """
        Generate test report in Markdown format.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Markdown formatted report
        """
        report = []
        report.append("# MDM Automated Test Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate totals
        total_tests = sum(cat.total_tests for cat in self.results.values())
        total_passed = sum(cat.passed_tests for cat in self.results.values())
        total_failed = sum(cat.failed_tests for cat in self.results.values())
        total_skipped = total_tests - total_passed - total_failed
        
        # Summary
        report.append("\n## Summary")
        report.append(f"- **Total tests:** {total_tests}")
        report.append(f"- **Passed:** {total_passed} ({total_passed/total_tests*100:.1f}%)")
        report.append(f"- **Failed:** {total_failed} ({total_failed/total_tests*100:.1f}%)")
        report.append(f"- **Skipped:** {total_skipped} ({total_skipped/total_tests*100:.1f}%)")
        
        # Detailed results by category
        report.append("\n## Detailed Results")
        
        for cat_id in sorted(self.results.keys()):
            category = self.results[cat_id]
            self._format_category_report(category, report, level=3)
        
        # Failed tests details
        failed_tests = []
        for category in self.results.values():
            failed_tests.extend([t for t in category.tests if t.status == "failed"])
        
        if failed_tests:
            report.append("\n## Failed Tests Details")
            for test in failed_tests:
                report.append(f"\n### {test.test_id}: {test.name}")
                report.append(f"- **Error:** {test.error}")
                if test.output:
                    report.append("- **Output:**")
                    report.append("```")
                    report.append(test.output)
                    report.append("```")
        
        # Join report
        report_text = "\n".join(report)
        
        # Save if output file specified
        if output_file:
            output_file.write_text(report_text)
        
        return report_text
    
    def _find_test_file(self, test_id: str) -> Optional[Path]:
        """Find test file for given test ID."""
        # Extract category from test ID (e.g., 1.1.1 -> test_11_*.py)
        parts = test_id.split('.')
        if len(parts) < 2:
            return None
        
        pattern = f"test_{parts[0]}{parts[1]}_*.py"
        matches = list(self.test_dir.glob(f"**/{pattern}"))
        
        return matches[0] if matches else None
    
    def _parse_test_result(self, test_id: str, result: subprocess.CompletedProcess) -> TestResult:
        """Parse test result from subprocess output."""
        status = "passed" if result.returncode == 0 else "failed"
        
        # Extract test name from output if possible
        name_match = re.search(r"test_\w+\s*\[(.*?)\]", result.stdout)
        name = name_match.group(1) if name_match else f"Test {test_id}"
        
        return TestResult(
            test_id=test_id,
            name=name,
            status=status,
            output=result.stdout,
            error=result.stderr
        )
    
    def _parse_category_results(self, category_id: str, result: subprocess.CompletedProcess) -> TestCategory:
        """Parse category test results."""
        category = TestCategory(id=category_id, name=f"Category {category_id}")
        
        # Simple parsing - count passed/failed
        lines = result.stdout.split('\n')
        for line in lines:
            if ' PASSED' in line:
                # Extract test info
                match = re.search(r"(\w+\.py)::(test_\w+)", line)
                if match:
                    test_id = f"{category_id}.{len(category.tests)+1}"
                    category.tests.append(TestResult(
                        test_id=test_id,
                        name=match.group(2),
                        status="passed"
                    ))
            elif ' FAILED' in line:
                match = re.search(r"(\w+\.py)::(test_\w+)", line)
                if match:
                    test_id = f"{category_id}.{len(category.tests)+1}"
                    category.tests.append(TestResult(
                        test_id=test_id,
                        name=match.group(2),
                        status="failed"
                    ))
        
        return category
    
    def _parse_json_report(self, report_data: dict):
        """Parse pytest JSON report."""
        # TODO: Implement JSON report parsing
        pass
    
    def _parse_text_output(self, output: str):
        """Parse pytest text output."""
        # TODO: Implement text output parsing
        pass
    
    def _format_category_report(self, category: TestCategory, report: List[str], level: int):
        """Format category report recursively."""
        status_icon = "✅" if category.failed_tests == 0 else "⚠️" if category.failed_tests < category.total_tests else "❌"
        
        report.append(f"\n{'#' * level} {category.name} [{category.passed_tests}/{category.total_tests}] {status_icon}")
        
        # List tests
        for test in category.tests:
            icon = "✅" if test.status == "passed" else "❌" if test.status == "failed" else "⏭️"
            report.append(f"- [{icon}] **{test.test_id}**: {test.name}")
        
        # Recursively format subcategories
        for subcat in category.subcategories.values():
            self._format_category_report(subcat, report, level + 1)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="MDM End-to-End Test Runner")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        help="Test target: 'all', category ID (e.g., '1.2'), or test ID (e.g., '1.2.3')"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-o", "--output", type=Path, help="Output file for report")
    parser.add_argument("--html", action="store_true", help="Generate HTML report (requires pytest-html)")
    
    args = parser.parse_args()
    
    runner = MDMTestRunner()
    
    # Run tests based on target
    if args.target == "all":
        print("Running all tests...")
        runner.run_all(verbose=args.verbose)
    elif '.' in args.target:
        # Check if it's a specific test or category
        parts = args.target.split('.')
        if len(parts) == 3:
            # Specific test
            print(f"Running test {args.target}...")
            result = runner.run_test(args.target, verbose=args.verbose)
            runner.results[args.target] = TestCategory(
                id=args.target,
                name=f"Test {args.target}",
                tests=[result]
            )
        else:
            # Category
            print(f"Running category {args.target}...")
            category = runner.run_category(args.target, verbose=args.verbose)
            runner.results[args.target] = category
    else:
        # Top-level category
        print(f"Running category {args.target}...")
        category = runner.run_category(args.target, verbose=args.verbose)
        runner.results[args.target] = category
    
    # Generate report
    report = runner.generate_report(args.output)
    
    if not args.output:
        print("\n" + report)
    else:
        print(f"\nReport saved to: {args.output}")
    
    # Exit with appropriate code
    total_failed = sum(cat.failed_tests for cat in runner.results.values())
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()