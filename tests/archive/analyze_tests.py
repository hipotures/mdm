#!/usr/bin/env python3
"""
Automated test analyzer that runs tests individually and analyzes errors.
"""

import subprocess
import sys
import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class TestAnalyzer:
    """Analyzes MDM E2E tests by running them individually."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results = []
        self.error_categories = defaultdict(list)
        
    def find_all_tests(self) -> Dict[str, Tuple[Path, str, str]]:
        """Find all test files and their test IDs."""
        tests = {}
        
        for test_file in self.test_dir.rglob("test_*.py"):
            if "__pycache__" in str(test_file):
                continue
                
            with open(test_file) as f:
                content = f.read()
            
            # Find all test functions with mdm_id markers
            pattern = r'@pytest\.mark\.mdm_id\("([^"]+)"\)\s*\n\s*def\s+(test_\w+)\([^)]*\):\s*\n\s*"""([^"]+)"""'
            
            for match in re.finditer(pattern, content, re.MULTILINE):
                test_id = match.group(1)
                test_func = match.group(2)
                test_desc = match.group(3)
                
                # Find class name if any
                class_pattern = r'class\s+(\w+).*?:\s*\n(?:.*\n)*?' + re.escape(match.group(0))
                class_match = re.search(class_pattern, content, re.MULTILINE | re.DOTALL)
                class_name = class_match.group(1) if class_match else None
                
                tests[test_id] = (test_file, class_name, test_func, test_desc)
        
        return tests
    
    def run_single_test(self, test_file: Path, class_name: Optional[str], test_func: str) -> Dict:
        """Run a single test and capture results."""
        # Build test path
        if class_name:
            test_path = f"{test_file}::{class_name}::{test_func}"
        else:
            test_path = f"{test_file}::{test_func}"
        
        # Run test with detailed output
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "-vv",
            "--tb=short",
            "--no-header",
            "--no-summary",
            "-q"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0 and "PASSED" in result.stdout,
            "skipped": "SKIPPED" in result.stdout,
            "failed": result.returncode != 0 or "FAILED" in result.stdout
        }
    
    def analyze_error(self, test_id: str, result: Dict) -> Dict:
        """Analyze test error and categorize it."""
        analysis = {
            "test_id": test_id,
            "status": "passed" if result["passed"] else "skipped" if result["skipped"] else "failed",
            "error_type": None,
            "error_message": None,
            "error_details": None,
            "suggested_fix": None
        }
        
        if result["failed"]:
            output = result["stdout"] + "\n" + result["stderr"]
            
            # Common error patterns
            if "ModuleNotFoundError" in output:
                analysis["error_type"] = "import_error"
                match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", output)
                if match:
                    analysis["error_message"] = f"Missing module: {match.group(1)}"
                    analysis["suggested_fix"] = f"Install missing module: pip install {match.group(1)}"
                    
            elif "AssertionError" in output:
                analysis["error_type"] = "assertion_failed"
                # Extract assertion details
                match = re.search(r"AssertionError: (.+?)(?:\n|$)", output)
                if match:
                    analysis["error_message"] = match.group(1)
                else:
                    # Look for assert statements
                    match = re.search(r"assert (.+?)(?:\n|$)", output)
                    if match:
                        analysis["error_message"] = f"Assertion failed: {match.group(1)}"
                
            elif "FileNotFoundError" in output:
                analysis["error_type"] = "file_not_found"
                match = re.search(r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']+)'", output)
                if match:
                    analysis["error_message"] = f"File not found: {match.group(1)}"
                    analysis["suggested_fix"] = "Check if MDM is properly installed and in PATH"
                    
            elif "not found in output" in output or "not in result.stdout" in output:
                analysis["error_type"] = "output_mismatch"
                analysis["error_message"] = "Expected output not found"
                analysis["suggested_fix"] = "Check if MDM command output format has changed"
                
            elif "Dataset.*not found" in output:
                analysis["error_type"] = "dataset_not_found"
                analysis["error_message"] = "Dataset not found"
                analysis["suggested_fix"] = "Ensure dataset is properly registered before accessing"
                
            elif "subprocess.CalledProcessError" in output:
                analysis["error_type"] = "command_failed"
                match = re.search(r"Command '(.+?)' returned non-zero exit status (\d+)", output)
                if match:
                    analysis["error_message"] = f"Command failed with exit code {match.group(2)}"
                    analysis["suggested_fix"] = "Check if MDM command syntax is correct"
            
            else:
                # Generic error
                analysis["error_type"] = "unknown"
                # Try to extract error message
                lines = output.strip().split('\n')
                for line in lines:
                    if "Error:" in line or "error:" in line:
                        analysis["error_message"] = line.strip()
                        break
                
            # Store full error details
            analysis["error_details"] = output
        
        return analysis
    
    def categorize_errors(self):
        """Categorize all errors found."""
        for result in self.results:
            if result["status"] == "failed":
                error_type = result["error_type"]
                self.error_categories[error_type].append(result)
    
    def run_all_tests(self):
        """Run all tests individually and analyze results."""
        tests = self.find_all_tests()
        total = len(tests)
        
        print(f"Found {total} tests to analyze")
        print("=" * 80)
        
        for i, (test_id, (test_file, class_name, test_func, test_desc)) in enumerate(tests.items(), 1):
            print(f"\n[{i}/{total}] Running test {test_id}: {test_desc}")
            print("-" * 40)
            
            # Run test
            result = self.run_single_test(test_file, class_name, test_func)
            
            # Analyze result
            analysis = self.analyze_error(test_id, result)
            analysis["description"] = test_desc
            analysis["file"] = str(test_file.relative_to(self.test_dir))
            analysis["function"] = test_func
            
            # Print immediate feedback
            status_symbol = "✓" if analysis["status"] == "passed" else "⊘" if analysis["status"] == "skipped" else "✗"
            print(f"{status_symbol} Status: {analysis['status'].upper()}")
            
            if analysis["error_message"]:
                print(f"  Error: {analysis['error_message']}")
                if analysis["suggested_fix"]:
                    print(f"  Fix: {analysis['suggested_fix']}")
            
            self.results.append(analysis)
        
        # Categorize errors
        self.categorize_errors()
    
    def generate_report(self) -> str:
        """Generate detailed analysis report."""
        report = []
        report.append("# MDM E2E Test Analysis Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Total tests analyzed:** {len(self.results)}")
        
        # Summary
        passed = sum(1 for r in self.results if r["status"] == "passed")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        skipped = sum(1 for r in self.results if r["status"] == "skipped")
        
        report.append("\n## Summary")
        report.append(f"- ✓ Passed: {passed} ({passed/len(self.results)*100:.1f}%)")
        report.append(f"- ✗ Failed: {failed} ({failed/len(self.results)*100:.1f}%)")
        report.append(f"- ⊘ Skipped: {skipped} ({skipped/len(self.results)*100:.1f}%)")
        
        # Error categories
        if self.error_categories:
            report.append("\n## Error Categories")
            for error_type, tests in sorted(self.error_categories.items(), key=lambda x: len(x[1]), reverse=True):
                report.append(f"\n### {error_type.replace('_', ' ').title()} ({len(tests)} tests)")
                for test in tests[:5]:  # Show first 5
                    report.append(f"- **{test['test_id']}**: {test['description']}")
                    if test['error_message']:
                        report.append(f"  - {test['error_message']}")
                if len(tests) > 5:
                    report.append(f"  - ... and {len(tests) - 5} more")
        
        # Detailed results by category
        report.append("\n## Detailed Results by Category")
        
        # Group by test category
        categories = defaultdict(list)
        for result in self.results:
            cat = result["test_id"].split('.')[0]
            categories[cat].append(result)
        
        for cat in sorted(categories.keys()):
            cat_results = categories[cat]
            cat_passed = sum(1 for r in cat_results if r["status"] == "passed")
            cat_total = len(cat_results)
            
            report.append(f"\n### Category {cat} ({cat_passed}/{cat_total} passed)")
            
            for result in sorted(cat_results, key=lambda x: x["test_id"]):
                status_symbol = "✓" if result["status"] == "passed" else "⊘" if result["status"] == "skipped" else "✗"
                report.append(f"- {status_symbol} **{result['test_id']}**: {result['description']}")
                
                if result["status"] == "failed" and result["error_message"]:
                    report.append(f"  - Error: {result['error_message']}")
                    if result["suggested_fix"]:
                        report.append(f"  - Fix: {result['suggested_fix']}")
        
        # Common issues and recommendations
        report.append("\n## Common Issues and Recommendations")
        
        if "import_error" in self.error_categories:
            report.append("\n### Import Errors")
            report.append("- MDM may not be properly installed in the test environment")
            report.append("- Run: `pip install -e .` from the project root")
        
        if "file_not_found" in self.error_categories:
            report.append("\n### File Not Found Errors")
            report.append("- MDM command may not be in PATH")
            report.append("- Check if `mdm` command is available: `which mdm`")
        
        if "assertion_failed" in self.error_categories:
            report.append("\n### Assertion Failures")
            report.append("- Command output format may have changed")
            report.append("- Database backend behavior may differ")
            report.append("- Check for recent code changes affecting output")
        
        return "\n".join(report)
    
    def save_detailed_results(self):
        """Save detailed results to JSON for further analysis."""
        output_file = self.test_dir / "test_analysis_results.json"
        
        # Remove error_details from JSON (too verbose)
        json_results = []
        for result in self.results:
            r = result.copy()
            if "error_details" in r:
                del r["error_details"]
            json_results.append(r)
        
        with open(output_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": len(self.results),
                    "passed": sum(1 for r in self.results if r["status"] == "passed"),
                    "failed": sum(1 for r in self.results if r["status"] == "failed"),
                    "skipped": sum(1 for r in self.results if r["status"] == "skipped")
                },
                "error_categories": {k: len(v) for k, v in self.error_categories.items()},
                "results": json_results
            }, f, indent=2)
        
        print(f"\n\nDetailed results saved to: {output_file}")


def main():
    """Main entry point."""
    analyzer = TestAnalyzer()
    
    print("Starting automated test analysis...")
    print("This will run each test individually and analyze errors.")
    print("=" * 80)
    
    # Run all tests
    analyzer.run_all_tests()
    
    # Generate and print report
    report = analyzer.generate_report()
    
    # Save report
    report_file = analyzer.test_dir / "test_analysis_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"\n\nAnalysis complete!")
    print(f"Report saved to: {report_file}")
    
    # Save detailed results
    analyzer.save_detailed_results()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(report.split("## Common Issues")[0])


if __name__ == "__main__":
    main()