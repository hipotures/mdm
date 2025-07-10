"""
Comparison testing framework for MDM refactoring.

This module provides tools to compare old and new implementations
to ensure functional equivalence and measure performance differences.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import traceback
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two implementations."""
    test_name: str
    passed: bool
    old_result: Any
    new_result: Any
    old_duration: float
    new_duration: float
    old_memory: float
    new_memory: float
    differences: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def performance_delta(self) -> float:
        """Performance difference as percentage."""
        if self.old_duration == 0:
            return 0
        return ((self.new_duration - self.old_duration) / self.old_duration) * 100
    
    @property
    def memory_delta(self) -> float:
        """Memory difference as percentage."""
        if self.old_memory == 0:
            return 0
        return ((self.new_memory - self.old_memory) / self.old_memory) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat(),
            "performance_delta": self.performance_delta,
            "memory_delta": self.memory_delta,
            "old_duration": self.old_duration,
            "new_duration": self.new_duration,
            "old_memory": self.old_memory,
            "new_memory": self.new_memory,
            "differences": self.differences,
            "error": self.error
        }


class ComparisonTester:
    """Framework for comparing old and new implementations."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path.home() / ".mdm" / "comparison_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ComparisonResult] = []
        logger.info(f"Initialized ComparisonTester with results dir: {self.results_dir}")
    
    def compare(self, 
                test_name: str,
                old_impl: Callable,
                new_impl: Callable,
                args: tuple = (),
                kwargs: dict = None,
                compare_func: Optional[Callable] = None,
                timeout: Optional[float] = None) -> ComparisonResult:
        """
        Compare two implementations.
        
        Args:
            test_name: Name of the test
            old_impl: Old implementation function
            new_impl: New implementation function
            args: Positional arguments for functions
            kwargs: Keyword arguments for functions
            compare_func: Optional custom comparison function
            timeout: Optional timeout in seconds
            
        Returns:
            ComparisonResult with comparison details
        """
        kwargs = kwargs or {}
        logger.info(f"Starting comparison test: {test_name}")
        
        # Run old implementation
        try:
            old_result, old_duration, old_memory = self._run_with_metrics(
                old_impl, args, kwargs, timeout
            )
            old_error = None
        except Exception as e:
            old_result = None
            old_duration = 0
            old_memory = 0
            old_error = f"Old implementation error: {str(e)}\n{traceback.format_exc()}"
            logger.error(old_error)
        
        # Run new implementation
        try:
            new_result, new_duration, new_memory = self._run_with_metrics(
                new_impl, args, kwargs, timeout
            )
            new_error = None
        except Exception as e:
            new_result = None
            new_duration = 0
            new_memory = 0
            new_error = f"New implementation error: {str(e)}\n{traceback.format_exc()}"
            logger.error(new_error)
        
        # Handle errors
        if old_error or new_error:
            result = ComparisonResult(
                test_name=test_name,
                passed=False,
                old_result=old_result,
                new_result=new_result,
                old_duration=old_duration,
                new_duration=new_duration,
                old_memory=old_memory,
                new_memory=new_memory,
                error=old_error or new_error
            )
        else:
            # Compare results
            if compare_func:
                passed = compare_func(old_result, new_result)
                differences = None if passed else {"custom_comparison": "Failed"}
            else:
                differences = self._deep_compare(old_result, new_result)
                passed = not bool(differences)
            
            result = ComparisonResult(
                test_name=test_name,
                passed=passed,
                old_result=old_result,
                new_result=new_result,
                old_duration=old_duration,
                new_duration=new_duration,
                old_memory=old_memory,
                new_memory=new_memory,
                differences=differences
            )
        
        self.results.append(result)
        self._save_result(result)
        
        logger.info(f"Comparison test '{test_name}' completed: "
                   f"{'PASSED' if result.passed else 'FAILED'} "
                   f"(perf delta: {result.performance_delta:+.2f}%)")
        
        return result
    
    def _run_with_metrics(self, func: Callable, args: tuple, 
                          kwargs: dict, timeout: Optional[float] = None) -> Tuple[Any, float, float]:
        """Run function and collect metrics."""
        import tracemalloc
        import signal
        
        # Setup timeout if specified
        if timeout:
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function exceeded timeout of {timeout}s")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
        
        try:
            # Measure memory
            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            
            # Measure time
            start_time = time.perf_counter()
            
            # Run function
            result = func(*args, **kwargs)
            
            end_time = time.perf_counter()
            
            # Get memory peak
            peak_memory = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            
            duration = end_time - start_time
            memory_used = (peak_memory - start_memory) / 1024 / 1024  # MB
            
            return result, duration, memory_used
            
        finally:
            # Reset timeout
            if timeout:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    
    def _deep_compare(self, old_result: Any, new_result: Any) -> Optional[Dict[str, Any]]:
        """Deep comparison of results."""
        differences = {}
        
        # Handle None comparison
        if old_result is None and new_result is None:
            return None
        if old_result is None or new_result is None:
            return {"type_mismatch": f"old={type(old_result).__name__}, new={type(new_result).__name__}"}
        
        # Handle DataFrame comparison
        if isinstance(old_result, pd.DataFrame) and isinstance(new_result, pd.DataFrame):
            try:
                # Check shape
                if old_result.shape != new_result.shape:
                    differences["shape"] = f"old={old_result.shape}, new={new_result.shape}"
                    return differences
                
                # Check columns
                if not old_result.columns.equals(new_result.columns):
                    differences["columns"] = {
                        "old": list(old_result.columns),
                        "new": list(new_result.columns)
                    }
                    return differences
                
                # Check data
                pd.testing.assert_frame_equal(
                    old_result.sort_index(axis=1), 
                    new_result.sort_index(axis=1),
                    check_dtype=False,  # Allow minor dtype differences
                    check_column_type=False,
                    rtol=1e-5,  # Relative tolerance for floats
                    atol=1e-8   # Absolute tolerance for floats
                )
                return None
            except AssertionError as e:
                return {"dataframe_diff": str(e)}
        
        # Handle numpy array comparison
        if isinstance(old_result, np.ndarray) and isinstance(new_result, np.ndarray):
            try:
                np.testing.assert_allclose(old_result, new_result, rtol=1e-5, atol=1e-8)
                return None
            except AssertionError as e:
                return {"array_diff": str(e)}
        
        # Handle dict comparison
        if isinstance(old_result, dict) and isinstance(new_result, dict):
            # Check keys
            old_keys = set(old_result.keys())
            new_keys = set(new_result.keys())
            if old_keys != new_keys:
                differences["keys"] = {
                    "only_in_old": list(old_keys - new_keys),
                    "only_in_new": list(new_keys - old_keys)
                }
            
            # Check values for common keys
            for key in old_keys & new_keys:
                sub_diff = self._deep_compare(old_result[key], new_result[key])
                if sub_diff:
                    differences[f"value[{key}]"] = sub_diff
            
            return differences if differences else None
        
        # Handle list/tuple comparison
        if isinstance(old_result, (list, tuple)) and isinstance(new_result, (list, tuple)):
            if len(old_result) != len(new_result):
                differences["length"] = f"old={len(old_result)}, new={len(new_result)}"
                return differences
            
            for i, (old_item, new_item) in enumerate(zip(old_result, new_result)):
                sub_diff = self._deep_compare(old_item, new_item)
                if sub_diff:
                    differences[f"item[{i}]"] = sub_diff
            
            return differences if differences else None
        
        # Handle primitive comparison
        if old_result != new_result:
            return {
                "value_diff": f"old={repr(old_result)}, new={repr(new_result)}",
                "type": f"old={type(old_result).__name__}, new={type(new_result).__name__}"
            }
        
        return None
    
    def _save_result(self, result: ComparisonResult) -> None:
        """Save result to disk."""
        filename = f"{result.test_name}_{result.timestamp:%Y%m%d_%H%M%S}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.debug(f"Saved comparison result to: {filepath}")
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate summary report of all comparison tests."""
        if not self.results:
            return "No comparison results available"
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        # Calculate averages
        perf_deltas = [r.performance_delta for r in self.results if not r.error]
        mem_deltas = [r.memory_delta for r in self.results if not r.error]
        
        avg_perf_delta = sum(perf_deltas) / len(perf_deltas) if perf_deltas else 0
        avg_mem_delta = sum(mem_deltas) / len(mem_deltas) if mem_deltas else 0
        
        report = f"""
Comparison Test Report
======================
Generated: {datetime.now():%Y-%m-%d %H:%M:%S}
Total Tests: {total}
Passed: {passed} ({passed/total*100:.1f}%)
Failed: {total - passed}

Performance Summary:
  Average Performance Delta: {avg_perf_delta:+.2f}%
  Average Memory Delta: {avg_mem_delta:+.2f}%
  
  Best Performance Improvement: {min(perf_deltas):+.2f}% if perf_deltas else 'N/A'
  Worst Performance Regression: {max(perf_deltas):+.2f}% if perf_deltas else 'N/A'

Failed Tests:
"""
        for result in self.results:
            if not result.passed:
                report += f"\n- {result.test_name}:"
                if result.error:
                    report += f"\n  Error: {result.error.split(chr(10))[0]}"
                elif result.differences:
                    report += f"\n  Differences: {json.dumps(result.differences, indent=4)}"
        
        report += "\n\nDetailed Results:\n"
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report += f"\n{status} {result.test_name}:"
            report += f"\n  Performance: {result.performance_delta:+.2f}% "
            report += f"({result.old_duration:.3f}s → {result.new_duration:.3f}s)"
            report += f"\n  Memory: {result.memory_delta:+.2f}% "
            report += f"({result.old_memory:.2f}MB → {result.new_memory:.2f}MB)"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_file}")
        
        return report
    
    def load_results(self, pattern: str = "*.json") -> None:
        """Load previous results from disk."""
        for file_path in self.results_dir.glob(pattern):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    # Reconstruct ComparisonResult (simplified)
                    result = ComparisonResult(
                        test_name=data["test_name"],
                        passed=data["passed"],
                        old_result=None,  # Not serialized
                        new_result=None,  # Not serialized
                        old_duration=data["old_duration"],
                        new_duration=data["new_duration"],
                        old_memory=data["old_memory"],
                        new_memory=data["new_memory"],
                        differences=data.get("differences"),
                        error=data.get("error"),
                        timestamp=datetime.fromisoformat(data["timestamp"])
                    )
                    self.results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load result from {file_path}: {e}")