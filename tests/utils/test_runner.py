"""Base test runner with GitHub integration support."""

import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    subtests: Dict[str, float] = field(default_factory=dict)  # subtest_name -> duration
    
    @property
    def full_test_path(self) -> str:
        """Get full test path for pytest."""
        return f"{self.file_path}::{self.test_name}"


@dataclass 
class TestSuite:
    """Collection of test results."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    total_duration: float = 0.0
    
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
    
    def run_pytest(self, test_path: Path, extra_args: List[str] = None, show_timing: bool = False) -> subprocess.CompletedProcess:
        """Run pytest on specified path."""
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_path),
            "-v", "--tb=short",
            "--no-header",
            "--durations=0"  # Show all test durations
        ]
        
        if show_timing:
            # Add verbose timing output
            cmd.extend(["-vv", "--durations=0", "--durations-min=0.001"])
        
        # For E2E tests, ensure proper isolation
        env = None
        if 'e2e' in str(test_path):
            import os
            env = os.environ.copy()
            # Each test gets its own isolated temp directory
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="mdm_test_")
            env['MDM_HOME_DIR'] = temp_dir
        
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
        
        return subprocess.run(cmd, capture_output=True, text=True, env=env)
    
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
                    
                    # Add subtest durations if available
                    if hasattr(self, 'show_timing') and self.show_timing:
                        # duration_map will be populated later
                        pass
                    
                    suite.results.append(result)
                
                return suite
                
            except (json.JSONDecodeError, KeyError):
                pass  # Fall back to text parsing
        
        # Parse durations from pytest output
        duration_map = self._parse_durations(result.stdout)
        
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
                        
                        # Add timing information from duration_map
                        if hasattr(self, 'show_timing') and self.show_timing and duration_map:
                            # Look for exact match first
                            if test_name in duration_map:
                                test_result.duration = duration_map[test_name]
                            # Also check for class::method format
                            for duration_key, duration_value in duration_map.items():
                                if duration_key.endswith(f"::{test_name}"):
                                    test_result.duration = duration_value
                                    break
                        
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
    
    def _parse_durations(self, output: str) -> Dict[str, float]:
        """Parse test durations from pytest output."""
        durations = {}
        lines = output.split('\n')
        
        # Look for duration section in pytest output
        in_durations = False
        for line in lines:
            if "slowest durations" in line:
                in_durations = True
                continue
            
            if in_durations:
                # Match lines like "1.23s call     tests/test_file.py::TestClass::test_method"
                match = re.match(r'(\d+\.\d+)s\s+call\s+(.+?)::(.+?)::(.+?)(?:\[|$)', line)
                if match:
                    duration = float(match.group(1))
                    test_class = match.group(3)
                    test_method = match.group(4)
                    full_name = f"{test_class}::{test_method}"
                    durations[full_name] = duration
                # Also match simpler format "1.23s call     tests/test_file.py::test_function"
                else:
                    match = re.match(r'(\d+\.\d+)s\s+call\s+(.+?)::(.+?)(?:\[|$)', line)
                    if match:
                        duration = float(match.group(1))
                        test_name = match.group(3)
                        durations[test_name] = duration
                
                # Stop when we hit the summary section
                if "=" in line and "passed" in line:
                    break
        
        return durations
    
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
    
    def _format_duration(self, duration: float) -> str:
        """Format duration in seconds to human readable format."""
        if duration < 1:
            return f"{duration:.1f}s"
        elif duration < 60:
            return f"{duration:.0f}s"
        else:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            return f"{minutes}m{seconds}s"
    
    def run_single_test(self, test_file: str, category_name: str) -> TestSuite:
        """Run a single test category."""
        test_path = self.project_root / test_file
        
        # Track timing
        start_time = time.time()
        
        if not test_path.exists():
            # Return empty suite for non-existent tests
            suite = TestSuite(name=category_name)
            suite.total_duration = 0.0
            return suite
        
        result = self.run_pytest(test_path, show_timing=getattr(self, 'show_timing', False))
        suite = self.parse_pytest_output(result, test_file, category_name)
        
        # Calculate total duration
        total_duration = time.time() - start_time
        suite.total_duration = total_duration
        
        # Add error result if no tests found but pytest failed
        if suite.total_tests == 0 and result.returncode != 0:
            suite.results.append(TestResult(
                test_name="pytest_error",
                file_path=test_file,
                category=category_name,
                passed=False,
                error_type="Test Execution Error",
                error_message=result.stderr[:200] if result.stderr else "Unknown error",
                output=result.stdout + "\n" + result.stderr
            ))
        
        return suite
    
    def run_all_tests(self, show_progress: bool = True, max_workers: int = 1, show_timing: bool = False) -> Dict[str, TestSuite]:
        """Run all test categories and return results.
        
        Args:
            show_progress: Whether to show progress
            max_workers: Maximum number of parallel workers (default 1 for sequential)
            show_timing: Whether to collect detailed timing information
        """
        test_categories = self.get_test_categories()
        self.show_timing = show_timing
        
        # Sequential execution (original behavior)
        if max_workers == 1:
            return self._run_tests_sequential(test_categories, show_progress)
        
        # Parallel execution
        return self._run_tests_parallel(test_categories, show_progress, max_workers)
    
    def _run_tests_sequential(self, test_categories: List[Tuple[str, str]], show_progress: bool) -> Dict[str, TestSuite]:
        """Run tests sequentially (original implementation)."""
        
        if show_progress and RICH_AVAILABLE:
            console.print(f"\n[bold]Running {self.__class__.__name__} tests...[/bold]")
            console.rule()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                for test_file, category_name in test_categories:
                    task = progress.add_task(f"[dim]⋯[/dim] Testing {category_name}...", total=1)
                    
                    suite = self.run_single_test(test_file, category_name)
                    self.test_suites[category_name] = suite
                    
                    # Format duration with coloring
                    duration_str = self._format_duration(suite.total_duration)
                    if suite.total_duration >= 60:
                        duration_display = f"[yellow][{duration_str}][/yellow]"
                    else:
                        duration_display = f"[{duration_str}]"
                    
                    # Update progress with result
                    if suite.total_tests == 0:
                        progress.update(
                            task, 
                            description=f"[yellow]⊘[/yellow] Skipping {category_name} (not found)",
                            completed=1
                        )
                    elif suite.failed_tests > 0:
                        progress.update(
                            task, 
                            description=f"[red]✗[/red] {category_name} {duration_display} ({suite.failed_tests} failures)",
                            completed=1
                        )
                    else:
                        progress.update(
                            task,
                            description=f"[green]✓[/green] {category_name} {duration_display}",
                            completed=1
                        )
        else:
            # Non-rich output
            print(f"Running {self.__class__.__name__} tests...")
            print("=" * 80)
            
            for test_file, category_name in test_categories:
                suite = self.run_single_test(test_file, category_name)
                self.test_suites[category_name] = suite
                
                duration_str = self._format_duration(suite.total_duration)
                
                if suite.total_tests == 0:
                    print(f"\nSkipping {test_file} (not found)")
                elif suite.failed_tests > 0:
                    print(f"✗ {category_name} [{duration_str}]: {suite.failed_tests} failures")
                else:
                    print(f"✓ {category_name} [{duration_str}]: All tests passed")
        
        return self.test_suites
    
    def _run_tests_parallel(self, test_categories: List[Tuple[str, str]], show_progress: bool, max_workers: int) -> Dict[str, TestSuite]:
        """Run tests in parallel using ThreadPoolExecutor."""
        import signal
        import shutil
        
        lock = threading.Lock()
        interrupted = threading.Event()
        
        def signal_handler(signum, frame):
            """Handle Ctrl+C gracefully."""
            interrupted.set()
            if show_progress and RICH_AVAILABLE:
                console.print("\n[yellow]Interrupt received. Cleaning up...[/yellow]")
        
        # Set up signal handler
        old_handler = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            if show_progress and RICH_AVAILABLE:
                from rich.layout import Layout
                from rich.live import Live
                from rich.panel import Panel
                from rich.text import Text
                
                console.print(f"\n[bold]Running {self.__class__.__name__} tests in parallel (workers: {max_workers})...[/bold]")
                console.rule()
                
                # Initialize data structures
                completed_count = 0
                running_tests = {}  # slot_id -> test_name
                completed_tests = []  # list of completed test results
                test_queue = list(test_categories)
                futures_to_slot = {}  # future -> slot_id
                slot_to_future = {}  # slot_id -> future
                futures_to_test = {}  # future -> (test_file, category_name)
                
                def create_display():
                    """Create the display layout with running and completed tests."""
                    from rich.table import Table
                    from rich.spinner import Spinner
                    
                    # Running tests panel with spinners
                    running_table = Table.grid(padding=0)
                    running_table.add_column(width=3)  # For spinner
                    running_table.add_column()  # For test name
                    
                    for i in range(max_workers):
                        if i in running_tests:
                            spinner = Spinner("dots", style="cyan")
                            running_table.add_row(spinner, f" {running_tests[i]}")
                        else:
                            running_table.add_row("", f"[dim]Worker {i+1}: idle[/dim]")
                    
                    running_panel = Panel(
                        running_table,
                        title=f"[bold cyan]Running ({len(running_tests)}/{max_workers} workers)[/bold cyan]",
                        border_style="cyan"
                    )
                    
                    # Completed tests panel - show ALL tests
                    completed_text = "\n".join(completed_tests) if completed_tests else "[dim]No tests completed yet[/dim]"
                    completed_panel = Panel(
                        completed_text,
                        title=f"[bold green]Completed ({completed_count}/{len(test_categories)})[/bold green]",
                        border_style="green"
                    )
                    
                    # Create layout
                    layout = Layout()
                    layout.split_row(
                        Layout(running_panel, name="running"),
                        Layout(completed_panel, name="completed")
                    )
                    
                    # Add overall progress at the top
                    progress_text = Text(f"Progress: {completed_count}/{len(test_categories)} tests", 
                                       style="bold", justify="center")
                    
                    final_layout = Layout()
                    final_layout.split_column(
                        Layout(Panel(progress_text, border_style="blue"), size=3),
                        layout
                    )
                    
                    return final_layout
                
                # Run tests with live display
                with Live(create_display(), refresh_per_second=10, console=console) as live:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit initial batch
                        for slot_id in range(min(max_workers, len(test_queue))):
                            if test_queue:
                                test_file, category_name = test_queue.pop(0)
                                future = executor.submit(self.run_single_test, test_file, category_name)
                                futures_to_slot[future] = slot_id
                                slot_to_future[slot_id] = future
                                futures_to_test[future] = (test_file, category_name)
                                running_tests[slot_id] = category_name
                                live.update(create_display())
                        
                        # Process completions and submit new tests
                        while futures_to_slot and not interrupted.is_set():
                            # Wait for any test to complete
                            done, pending = concurrent.futures.wait(
                                futures_to_slot.keys(), 
                                return_when=concurrent.futures.FIRST_COMPLETED,
                                timeout=0.1  # Check for interrupts every 100ms
                            )
                            
                            if not done:
                                continue  # Timeout, check interrupted flag
                            
                            for future in done:
                                slot_id = futures_to_slot[future]
                                test_file, category_name = futures_to_test[future]
                                
                                try:
                                    suite = future.result()
                                    
                                    with lock:
                                        self.test_suites[category_name] = suite
                                        completed_count += 1
                                    
                                    # Format result
                                    duration_str = self._format_duration(suite.total_duration)
                                    if suite.total_duration >= 60:
                                        duration_display = f"[yellow][{duration_str}][/yellow]"
                                    else:
                                        duration_display = f"[{duration_str}]"
                                    
                                    if suite.total_tests == 0:
                                        result_msg = f"[yellow]⊘[/yellow] {category_name} (not found)"
                                    elif suite.failed_tests > 0:
                                        result_msg = f"[red]✗[/red] {category_name} {duration_display} ({suite.failed_tests} failures)"
                                    else:
                                        result_msg = f"[green]✓[/green] {category_name} {duration_display}"
                                    
                                except Exception as e:
                                    with lock:
                                        completed_count += 1
                                    result_msg = f"[red]✗[/red] {category_name} (error: {str(e)})"
                                
                                # Add to completed list
                                completed_tests.append(result_msg)
                                
                                # Remove from tracking
                                del futures_to_slot[future]
                                del slot_to_future[slot_id]
                                del futures_to_test[future]
                                del running_tests[slot_id]
                                
                                # Submit next test if available
                                if test_queue:
                                    test_file, category_name = test_queue.pop(0)
                                    new_future = executor.submit(self.run_single_test, test_file, category_name)
                                    futures_to_slot[new_future] = slot_id
                                    slot_to_future[slot_id] = new_future
                                    futures_to_test[new_future] = (test_file, category_name)
                                    running_tests[slot_id] = category_name
                                
                                # Update display
                                live.update(create_display())
                        
                        # If interrupted, cancel remaining futures
                        if interrupted.is_set():
                            for future in futures_to_slot:
                                future.cancel()
                            
                            # Wait for running tests to complete
                            if futures_to_slot:
                                console.print("[yellow]Waiting for running tests to complete...[/yellow]")
                                concurrent.futures.wait(futures_to_slot.keys(), timeout=5)
                
                # Show final summary
                if not interrupted.is_set():
                    console.print("\n[bold]All Tests Completed![/bold]")
                else:
                    console.print("\n[yellow]Testing interrupted by user.[/yellow]")
                console.rule()
            else:
                # Non-rich parallel output
                print(f"Running {self.__class__.__name__} tests in parallel (workers: {max_workers})...")
                print("=" * 80)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tests
                    futures = {
                        executor.submit(self.run_single_test, test_file, category_name): (test_file, category_name)
                        for test_file, category_name in test_categories
                    }
                    
                    # Process completed tests
                    try:
                        for future in as_completed(futures):
                            if interrupted.is_set():
                                print("\nInterrupted by user. Cancelling remaining tests...")
                                for f in futures:
                                    f.cancel()
                                break
                                
                            test_file, category_name = futures[future]
                            try:
                                suite = future.result()
                                
                                with lock:
                                    self.test_suites[category_name] = suite
                                
                                duration_str = self._format_duration(suite.total_duration)
                                
                                if suite.total_tests == 0:
                                    print(f"\nSkipping {test_file} (not found)")
                                elif suite.failed_tests > 0:
                                    print(f"✗ {category_name} [{duration_str}]: {suite.failed_tests} failures")
                                else:
                                    print(f"✓ {category_name} [{duration_str}]: All tests passed")
                            except Exception as e:
                                print(f"✗ {category_name}: Error - {str(e)}")
                    except KeyboardInterrupt:
                        interrupted.set()
                        print("\nKeyboard interrupt received...")
        
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, old_handler)
            
            # Clean up temp directories
            if interrupted.is_set():
                import glob
                import os
                # Clean up MDM test temp directories
                for temp_dir in glob.glob("/tmp/mdm_test_*"):
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass
                
                if show_progress and RICH_AVAILABLE:
                    console.print("[green]Cleanup completed.[/green]")
        
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
            table.add_column("Duration", justify="right")
            
            for category, suite in self.test_suites.items():
                duration_str = self._format_duration(suite.total_duration)
                # Color duration if > 60 seconds
                if suite.total_duration >= 60:
                    duration_display = f"[yellow]{duration_str}[/yellow]"
                else:
                    duration_display = duration_str
                    
                table.add_row(
                    category,
                    str(suite.total_tests),
                    str(suite.passed_tests),
                    str(suite.failed_tests),
                    f"{suite.pass_rate:.1f}%",
                    duration_display
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
                duration_str = self._format_duration(suite.total_duration)
                print(f"{category}: {suite.passed_tests}/{suite.total_tests} passed ({suite.pass_rate:.1f}%) [{duration_str}]")
            
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
                "duration": suite.total_duration,
                "duration_formatted": self._format_duration(suite.total_duration),
                "failures": [
                    {
                        "test": result.test_name,
                        "file": result.file_path,
                        "error_type": result.error_type,
                        "error_message": result.error_message,
                        "duration": result.duration
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