"""Base test runner with GitHub integration support."""

import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_name: str
    file_path: str
    category: str
    passed: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    output: str = ""
    duration: float = 0.0
    
    @property
    def full_test_path(self) -> str:
        """Get full test path for pytest."""
        return f"{self.file_path}::{self.test_name}"


@dataclass 
class TestSuite:
    """Collection of test results."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def total_tests(self) -> int:
        return len(self.results)
    
    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed_tests(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100
    
    def get_failures(self) -> List[TestResult]:
        """Get list of failed tests."""
        return [r for r in self.results if not r.passed]


class BaseTestRunner(ABC):
    """Base class for test runners."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize test runner."""
        self.project_root = project_root or Path.cwd()
        self.test_suites: Dict[str, TestSuite] = {}
        
    @abstractmethod
    def get_test_categories(self) -> List[Tuple[str, str]]:
        """Return list of (test_path, category_name) tuples."""
        pass
    
    def run_pytest(self, test_path: Path, extra_args: List[str] = None) -> subprocess.CompletedProcess:
        """Run pytest on specified path."""
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_path),
            "-v", "--tb=short",
            "--no-header"
        ]
        
        # Only add json report if plugin is available
        try:
            import pytest_json_report
            cmd.extend([
                "--json-report",
                "--json-report-file=/tmp/pytest_report.json"
            ])
        except ImportError:
            pass
        
        if extra_args:
            cmd.extend(extra_args)
        
        return subprocess.run(cmd, capture_output=True, text=True)
    
    def parse_pytest_output(self, result: subprocess.CompletedProcess, test_file: str, category: str) -> TestSuite:
        """Parse pytest output and create TestSuite."""
        suite = TestSuite(name=category)
        
        # Try to parse JSON report first
        json_report_path = Path("/tmp/pytest_report.json")
        if json_report_path.exists():
            try:
                with open(json_report_path, 'r') as f:
                    report = json.load(f)
                
                for test in report.get('tests', []):
                    test_name = test['nodeid'].split('::')[-1]
                    passed = test['outcome'] == 'passed'
                    
                    error_type = None
                    error_message = None
                    
                    if not passed:
                        if 'call' in test and 'longrepr' in test['call']:
                            error_info = test['call']['longrepr']
                            error_type = self._extract_error_type(str(error_info))
                            error_message = self._extract_error_message(str(error_info))
                    
                    result = TestResult(
                        test_name=test_name,
                        file_path=test_file,
                        category=category,
                        passed=passed,
                        error_type=error_type,
                        error_message=error_message,
                        duration=test.get('duration', 0.0)
                    )
                    suite.results.append(result)
                
                return suite
                
            except (json.JSONDecodeError, KeyError):
                pass  # Fall back to text parsing
        
        # Fall back to parsing text output
        lines = (result.stdout + result.stderr).split('\n')
        
        # First check if any tests were collected
        for line in lines:
            if "collected 0 items" in line:
                # No tests found in this file
                return suite
            
        # Parse test results
        for i, line in enumerate(lines):
            # Match both file::test format and just test format
            if " FAILED" in line or " PASSED" in line or " ERROR" in line:
                # Try different patterns
                patterns = [
                    r'(\S+\.py)::(\S+)\s+(PASSED|FAILED|ERROR)',  # Full path
                    r'(\S+)::(\S+)\s+(PASSED|FAILED|ERROR)',      # Module::test
                    r'(\S+)\s+(PASSED|FAILED|ERROR)'               # Just test name
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        if len(match.groups()) == 3:
                            test_name = match.group(2)
                            status = match.group(3)
                        else:
                            test_name = match.group(1)
                            status = match.group(2)
                        
                        passed = status == "PASSED"
                        
                        error_type = None
                        error_message = None
                        
                        if not passed:
                            # Look for error details
                            error_info = self._extract_error_info(lines, i)
                            error_type = self._extract_error_type(error_info)
                            error_message = self._extract_error_message(error_info)
                        
                        test_result = TestResult(
                            test_name=test_name,
                            file_path=test_file,
                            category=category,
                            passed=passed,
                            error_type=error_type,
                            error_message=error_message,
                            output=result.stdout if not passed else ""
                        )
                        suite.results.append(test_result)
                        break
        
        # If still no results found, check for summary line
        if len(suite.results) == 0:
            for line in lines:
                # Look for pytest summary
                if "passed" in line or "failed" in line or "error" in line:
                    match = re.search(r'(\d+)\s+passed', line)
                    if match and int(match.group(1)) > 0:
                        # Tests passed but we couldn't parse individual results
                        # Add a placeholder
                        test_result = TestResult(
                            test_name="All tests",
                            file_path=test_file,
                            category=category,
                            passed=True,
                            output=""
                        )
                        suite.results.append(test_result)
        
        return suite
    
    def _extract_error_info(self, lines: List[str], start_index: int) -> str:
        """Extract error information from pytest output."""
        error_info = ""
        for j in range(start_index + 1, min(start_index + 20, len(lines))):
            if lines[j].strip():
                if "short test summary" in lines[j]:
                    break
                error_info += lines[j] + "\n"
        return error_info
    
    def _extract_error_type(self, error_info: str) -> str:
        """Extract error type from error information."""
        if "AssertionError" in error_info:
            if "assert False" in error_info and "exists()" in error_info:
                return "File/Directory not found"
            elif "not in result.stdout" in error_info:
                return "Output mismatch"
            else:
                return "AssertionError"
        elif "AttributeError" in error_info:
            return "AttributeError"
        elif "TypeError" in error_info:
            return "TypeError"
        elif "KeyError" in error_info:
            return "KeyError"
        elif "ValueError" in error_info:
            return "ValueError"
        elif "ModuleNotFoundError" in error_info:
            return "ModuleNotFoundError"
        elif "CalledProcessError" in error_info:
            return "Command failed"
        elif "SystemExit" in error_info:
            return "SystemExit"
        else:
            return "Unknown"
    
    def _extract_error_message(self, error_info: str) -> str:
        """Extract error message from error information."""
        # Get first meaningful line
        lines = error_info.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('>') and not line.startswith('_'):
                return line[:200]  # Truncate long messages
        return error_info[:200] if error_info else "No error message"
    
    def run_all_tests(self, show_progress: bool = True) -> Dict[str, TestSuite]:
        """Run all test categories and return results."""
        test_categories = self.get_test_categories()
        
        if show_progress and RICH_AVAILABLE:
            console.print(f"\n[bold]Running {self.__class__.__name__} tests...[/bold]")
            console.rule()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                for test_file, category_name in test_categories:
                    test_path = self.project_root / test_file
                    
                    if not test_path.exists():
                        task = progress.add_task(
                            f"[yellow]Skipping {category_name} (not found)[/yellow]", 
                            total=1
                        )
                        progress.update(task, completed=1)
                        continue
                    
                    task = progress.add_task(f"Testing {category_name}...", total=1)
                    
                    result = self.run_pytest(test_path)
                    suite = self.parse_pytest_output(result, test_file, category_name)
                    self.test_suites[category_name] = suite
                    
                    # Debug: print if no tests found
                    if suite.total_tests == 0 and result.returncode != 0:
                        # There was an error running tests
                        suite.results.append(TestResult(
                            test_name="pytest_error",
                            file_path=test_file,
                            category=category_name,
                            passed=False,
                            error_type="Test Execution Error",
                            error_message=result.stderr[:200] if result.stderr else "Unknown error",
                            output=result.stdout + "\n" + result.stderr
                        ))
                    
                    # Update progress with result
                    if suite.failed_tests > 0:
                        progress.update(
                            task, 
                            description=f"[red]✗[/red] {category_name} ({suite.failed_tests} failures)",
                            completed=1
                        )
                    else:
                        progress.update(
                            task,
                            description=f"[green]✓[/green] {category_name}",
                            completed=1
                        )
        else:
            # Non-rich output
            print(f"Running {self.__class__.__name__} tests...")
            print("=" * 80)
            
            for test_file, category_name in test_categories:
                test_path = self.project_root / test_file
                
                if not test_path.exists():
                    print(f"\nSkipping {test_file} (not found)")
                    continue
                
                print(f"\nTesting {category_name}...")
                
                result = self.run_pytest(test_path)
                suite = self.parse_pytest_output(result, test_file, category_name)
                self.test_suites[category_name] = suite
                
                if suite.failed_tests > 0:
                    print(f"✗ {category_name}: {suite.failed_tests} failures")
                else:
                    print(f"✓ {category_name}: All tests passed")
        
        return self.test_suites
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all test runs."""
        total_tests = sum(suite.total_tests for suite in self.test_suites.values())
        passed_tests = sum(suite.passed_tests for suite in self.test_suites.values())
        failed_tests = sum(suite.failed_tests for suite in self.test_suites.values())
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "categories": len(self.test_suites),
            "failed_categories": sum(1 for suite in self.test_suites.values() if suite.failed_tests > 0)
        }
    
    def display_summary(self):
        """Display test summary."""
        stats = self.get_summary_stats()
        
        if RICH_AVAILABLE:
            console.print("\n[bold]SUMMARY[/bold]")
            console.rule()
            
            if stats["failed_tests"] == 0:
                console.print("[bold green]✓ All tests passed![/bold green]")
            else:
                console.print(f"[bold red]Total failures: {stats['failed_tests']}[/bold red]")
            
            # Create summary table
            table = Table(title="Test Results by Category")
            table.add_column("Category", style="cyan")
            table.add_column("Total", justify="right")
            table.add_column("Passed", justify="right", style="green")
            table.add_column("Failed", justify="right", style="red")
            table.add_column("Pass Rate", justify="right")
            
            for category, suite in self.test_suites.items():
                table.add_row(
                    category,
                    str(suite.total_tests),
                    str(suite.passed_tests),
                    str(suite.failed_tests),
                    f"{suite.pass_rate:.1f}%"
                )
            
            console.print(table)
            
            # Overall stats
            console.print(f"\nOverall pass rate: [bold]{stats['pass_rate']:.1f}%[/bold]")
            
        else:
            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            
            if stats["failed_tests"] == 0:
                print("✓ All tests passed!")
            else:
                print(f"Total failures: {stats['failed_tests']}")
            
            print(f"\nTest Results by Category:")
            for category, suite in self.test_suites.items():
                print(f"{category}: {suite.passed_tests}/{suite.total_tests} passed ({suite.pass_rate:.1f}%)")
            
            print(f"\nOverall pass rate: {stats['pass_rate']:.1f}%")
    
    def save_report(self, output_file: Path):
        """Save test report to file."""
        report = {
            "runner": self.__class__.__name__,
            "summary": self.get_summary_stats(),
            "categories": {}
        }
        
        for category, suite in self.test_suites.items():
            report["categories"][category] = {
                "total": suite.total_tests,
                "passed": suite.passed_tests,
                "failed": suite.failed_tests,
                "failures": [
                    {
                        "test": result.test_name,
                        "file": result.file_path,
                        "error_type": result.error_type,
                        "error_message": result.error_message
                    }
                    for result in suite.get_failures()
                ]
            }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        if RICH_AVAILABLE:
            console.print(f"\n[green]Report saved to:[/green] {output_file}")
        else:
            print(f"\nReport saved to: {output_file}")