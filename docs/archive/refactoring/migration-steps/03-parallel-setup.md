# Step 3: Parallel Development Environment Setup

## Overview

Set up infrastructure for parallel development, allowing old and new implementations to coexist. This enables incremental migration with instant rollback capability.

## Duration

1 week (Week 5)

## Objectives

1. Configure git worktrees for parallel development
2. Implement comprehensive feature flag system
3. Set up comparison testing framework
4. Create A/B testing infrastructure
5. Establish metrics collection for both implementations

## Prerequisites

- ✅ All tests passing (Step 1)
- ✅ Abstraction layer implemented (Step 2)
- ✅ DI container configured

## Detailed Steps

### Day 1: Git Worktree Configuration

#### 1.1 Create Development Branches
```bash
# Ensure we're on main branch with latest changes
cd $MDM_ORIGINAL_ROOT
git checkout main
git pull origin main

# Create feature branch for refactoring
git checkout -b refactor-2025-parallel

# Create worktrees for different aspects
git worktree add ../mdm-refactor-backend refactor-2025-parallel
git worktree add ../mdm-refactor-features refactor-2025-parallel
git worktree add ../mdm-refactor-integration refactor-2025-parallel

# Create comparison testing worktree
git worktree add ../mdm-comparison comparison-testing

# List all worktrees
git worktree list
```

#### 1.2 Set Up Worktree Environment
```bash
# Create setup script for each worktree
cat > setup-worktree.sh << 'EOF'
#!/bin/bash
# Setup script for MDM worktrees

WORKTREE_PATH=$1
if [ -z "$WORKTREE_PATH" ]; then
    echo "Usage: $0 <worktree-path>"
    exit 1
fi

cd "$WORKTREE_PATH"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install uv
uv pip install -e .

# Install additional dev tools
uv pip install pytest-cov pytest-benchmark pytest-xdist memory_profiler

echo "Worktree setup complete at $WORKTREE_PATH"
EOF

chmod +x setup-worktree.sh

# Set up each worktree
./setup-worktree.sh ../mdm-refactor-backend
./setup-worktree.sh ../mdm-refactor-features
./setup-worktree.sh ../mdm-refactor-integration
```

### Day 2: Feature Flag System

#### 2.1 Create Feature Flag Framework
```python
# Create: src/mdm/core/feature_flags.py
from typing import Dict, Any, Optional, Callable
from functools import wraps
import json
import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FeatureFlags:
    """Centralized feature flag management"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".mdm" / "feature_flags.json"
        self._flags: Dict[str, Any] = {}
        self._callbacks: Dict[str, list] = {}
        self._history: list = []
        self.load_flags()
    
    def load_flags(self) -> None:
        """Load flags from configuration file"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                data = json.load(f)
                self._flags = data.get("flags", {})
                self._history = data.get("history", [])
        else:
            # Default flags
            self._flags = {
                "use_new_backend": False,
                "use_new_registrar": False,
                "use_new_features": False,
                "enable_comparison_tests": True,
                "enable_performance_tracking": True,
                "enable_memory_profiling": False,
                "rollout_percentage": {
                    "new_backend": 0,
                    "new_registrar": 0,
                    "new_features": 0
                }
            }
            self.save_flags()
    
    def save_flags(self) -> None:
        """Persist flags to disk"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump({
                "flags": self._flags,
                "history": self._history,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
    
    def get(self, flag_name: str, default: Any = None) -> Any:
        """Get flag value"""
        return self._flags.get(flag_name, default)
    
    def set(self, flag_name: str, value: Any) -> None:
        """Set flag value with history tracking"""
        old_value = self._flags.get(flag_name)
        self._flags[flag_name] = value
        
        # Track change
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "flag": flag_name,
            "old_value": old_value,
            "new_value": value
        })
        
        # Save immediately
        self.save_flags()
        
        # Notify callbacks
        for callback in self._callbacks.get(flag_name, []):
            callback(flag_name, old_value, value)
        
        logger.info(f"Feature flag '{flag_name}' changed from {old_value} to {value}")
    
    def register_callback(self, flag_name: str, callback: Callable) -> None:
        """Register callback for flag changes"""
        if flag_name not in self._callbacks:
            self._callbacks[flag_name] = []
        self._callbacks[flag_name].append(callback)
    
    def is_enabled_for_user(self, flag_name: str, user_id: str) -> bool:
        """Check if feature is enabled for specific user (for gradual rollout)"""
        if self.get(flag_name):
            return True
        
        # Check percentage rollout
        rollout_key = f"rollout_percentage.{flag_name.replace('use_', '')}"
        percentage = self.get(rollout_key, 0)
        
        if percentage == 0:
            return False
        if percentage >= 100:
            return True
        
        # Use consistent hashing for user
        import hashlib
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
        return (user_hash % 100) < percentage


# Global instance
feature_flags = FeatureFlags()


# Decorator for feature-flagged functions
def feature_flag(flag_name: str, fallback: Optional[Callable] = None):
    """Decorator to conditionally execute based on feature flag"""
    def decorator(new_impl: Callable) -> Callable:
        @wraps(new_impl)
        def wrapper(*args, **kwargs):
            if feature_flags.get(flag_name):
                try:
                    return new_impl(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in new implementation: {e}")
                    if fallback and feature_flags.get("auto_fallback", True):
                        logger.warning(f"Falling back to old implementation")
                        return fallback(*args, **kwargs)
                    raise
            elif fallback:
                return fallback(*args, **kwargs)
            else:
                raise NotImplementedError(
                    f"Feature '{flag_name}' is disabled and no fallback provided"
                )
        return wrapper
    return decorator


# CLI for feature flag management
def create_feature_flag_cli():
    """Create CLI commands for feature flag management"""
    import typer
    from rich.console import Console
    from rich.table import Table
    
    app = typer.Typer()
    console = Console()
    
    @app.command()
    def list():
        """List all feature flags"""
        table = Table(title="Feature Flags")
        table.add_column("Flag", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Type", style="yellow")
        
        for flag, value in feature_flags._flags.items():
            table.add_row(flag, str(value), type(value).__name__)
        
        console.print(table)
    
    @app.command()
    def set(flag_name: str, value: str):
        """Set a feature flag value"""
        # Parse value
        if value.lower() in ("true", "false"):
            parsed_value = value.lower() == "true"
        elif value.isdigit():
            parsed_value = int(value)
        else:
            parsed_value = value
        
        feature_flags.set(flag_name, parsed_value)
        console.print(f"[green]Flag '{flag_name}' set to {parsed_value}[/green]")
    
    @app.command()
    def history(flag_name: Optional[str] = None):
        """Show flag change history"""
        history = feature_flags._history
        if flag_name:
            history = [h for h in history if h["flag"] == flag_name]
        
        for entry in history[-10:]:  # Last 10 changes
            console.print(
                f"[dim]{entry['timestamp']}[/dim] "
                f"[cyan]{entry['flag']}[/cyan]: "
                f"{entry['old_value']} → {entry['new_value']}"
            )
    
    return app
```

### Day 3: Comparison Testing Framework

#### 3.1 Create Comparison Test Base
```python
# Create: src/mdm/testing/comparison.py
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import traceback
import json
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from deepdiff import DeepDiff


@dataclass
class ComparisonResult:
    """Result of comparing two implementations"""
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
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def performance_delta(self) -> float:
        """Performance difference as percentage"""
        if self.old_duration == 0:
            return 0
        return ((self.new_duration - self.old_duration) / self.old_duration) * 100
    
    @property
    def memory_delta(self) -> float:
        """Memory difference as percentage"""
        if self.old_memory == 0:
            return 0
        return ((self.new_memory - self.old_memory) / self.old_memory) * 100


class ComparisonTester:
    """Framework for comparing old and new implementations"""
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path.home() / ".mdm" / "comparison_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ComparisonResult] = []
    
    def compare(self, 
                test_name: str,
                old_impl: Callable,
                new_impl: Callable,
                args: tuple = (),
                kwargs: dict = None,
                compare_func: Optional[Callable] = None) -> ComparisonResult:
        """Compare two implementations"""
        kwargs = kwargs or {}
        
        # Run old implementation
        old_result, old_duration, old_memory = self._run_with_metrics(
            old_impl, args, kwargs
        )
        
        # Run new implementation
        new_result, new_duration, new_memory = self._run_with_metrics(
            new_impl, args, kwargs
        )
        
        # Compare results
        if compare_func:
            passed = compare_func(old_result, new_result)
            differences = None
        else:
            differences = self._deep_compare(old_result, new_result)
            passed = not bool(differences)
        
        # Create result
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
        
        return result
    
    def _run_with_metrics(self, func: Callable, args: tuple, 
                          kwargs: dict) -> Tuple[Any, float, float]:
        """Run function and collect metrics"""
        import tracemalloc
        
        # Measure memory
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        # Measure time
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            result = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        
        end_time = time.perf_counter()
        
        # Get memory peak
        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        duration = end_time - start_time
        memory_used = (peak_memory - start_memory) / 1024 / 1024  # MB
        
        return result, duration, memory_used
    
    def _deep_compare(self, old_result: Any, new_result: Any) -> Optional[Dict]:
        """Deep comparison of results"""
        # Handle DataFrame comparison specially
        if isinstance(old_result, pd.DataFrame) and isinstance(new_result, pd.DataFrame):
            try:
                pd.testing.assert_frame_equal(old_result, new_result)
                return None
            except AssertionError as e:
                return {"dataframe_diff": str(e)}
        
        # Use DeepDiff for other types
        diff = DeepDiff(old_result, new_result, ignore_order=True)
        return diff.to_dict() if diff else None
    
    def _save_result(self, result: ComparisonResult) -> None:
        """Save result to disk"""
        filename = f"{result.test_name}_{result.timestamp:%Y%m%d_%H%M%S}.json"
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        data = {
            "test_name": result.test_name,
            "passed": result.passed,
            "timestamp": result.timestamp.isoformat(),
            "performance_delta": result.performance_delta,
            "memory_delta": result.memory_delta,
            "old_duration": result.old_duration,
            "new_duration": result.new_duration,
            "old_memory": result.old_memory,
            "new_memory": result.new_memory,
            "differences": result.differences,
            "error": result.error
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate summary report"""
        if not self.results:
            return "No comparison results available"
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        avg_perf_delta = sum(r.performance_delta for r in self.results) / total
        avg_mem_delta = sum(r.memory_delta for r in self.results) / total
        
        report = f"""
Comparison Test Report
======================
Total Tests: {total}
Passed: {passed} ({passed/total*100:.1f}%)
Failed: {total - passed}

Average Performance Delta: {avg_perf_delta:+.2f}%
Average Memory Delta: {avg_mem_delta:+.2f}%

Failed Tests:
"""
        for result in self.results:
            if not result.passed:
                report += f"\n- {result.test_name}: {result.differences or result.error}"
        
        return report
```

### Day 4: A/B Testing Infrastructure

#### 4.1 Create A/B Testing Router
```python
# Create: src/mdm/core/ab_testing.py
from typing import Any, Callable, Optional, Dict
import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass
import logging

from .feature_flags import feature_flags
from .metrics import metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Configuration for an A/B test"""
    test_name: str
    control_impl: Callable
    treatment_impl: Callable
    traffic_percentage: float = 50.0
    enabled: bool = True
    
    def should_use_treatment(self, identifier: str) -> bool:
        """Determine if treatment should be used for given identifier"""
        if not self.enabled:
            return False
        
        # Use consistent hashing
        hash_val = int(hashlib.md5(f"{self.test_name}:{identifier}".encode()).hexdigest()[:8], 16)
        return (hash_val % 100) < self.traffic_percentage


class ABTestRouter:
    """Routes requests between control and treatment implementations"""
    
    def __init__(self):
        self.tests: Dict[str, ABTestConfig] = {}
        self.active_test: Optional[str] = None
    
    def register_test(self, config: ABTestConfig) -> None:
        """Register an A/B test"""
        self.tests[config.test_name] = config
        logger.info(f"Registered A/B test: {config.test_name}")
    
    @contextmanager
    def test_context(self, test_name: str, identifier: str):
        """Context manager for A/B testing"""
        if test_name not in self.tests:
            yield "control"
            return
        
        config = self.tests[test_name]
        variant = "treatment" if config.should_use_treatment(identifier) else "control"
        
        # Set active test
        old_test = self.active_test
        self.active_test = test_name
        
        # Track metrics
        metrics_collector.increment(f"ab_test.{test_name}.{variant}")
        
        try:
            yield variant
        finally:
            self.active_test = old_test
    
    def route(self, test_name: str, identifier: str, *args, **kwargs) -> Any:
        """Route to appropriate implementation"""
        if test_name not in self.tests:
            raise ValueError(f"Unknown A/B test: {test_name}")
        
        config = self.tests[test_name]
        
        with self.test_context(test_name, identifier) as variant:
            if variant == "treatment":
                logger.debug(f"Routing to treatment for {test_name}")
                return config.treatment_impl(*args, **kwargs)
            else:
                logger.debug(f"Routing to control for {test_name}")
                return config.control_impl(*args, **kwargs)


# Global router
ab_router = ABTestRouter()


# Decorator for A/B tested functions
def ab_test(test_name: str, identifier_param: str = "name"):
    """Decorator for A/B testing"""
    def decorator(treatment_impl: Callable) -> Callable:
        @wraps(treatment_impl)
        def wrapper(*args, **kwargs):
            # Get identifier from parameters
            identifier = kwargs.get(identifier_param, str(args[0] if args else "default"))
            
            # Get control implementation
            control_impl = globals().get(f"{treatment_impl.__name__}_legacy")
            if not control_impl:
                # No legacy version, just use treatment
                return treatment_impl(*args, **kwargs)
            
            # Register test if not already registered
            if test_name not in ab_router.tests:
                ab_router.register_test(ABTestConfig(
                    test_name=test_name,
                    control_impl=control_impl,
                    treatment_impl=treatment_impl,
                    traffic_percentage=feature_flags.get(f"ab_test_{test_name}_percentage", 0)
                ))
            
            # Route to appropriate implementation
            return ab_router.route(test_name, identifier, *args, **kwargs)
        
        return wrapper
    return decorator
```

### Day 5: Metrics Collection System

#### 5.1 Create Unified Metrics Collector
```python
# Create: src/mdm/core/metrics.py
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from pathlib import Path
import threading
from collections import defaultdict
import statistics


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


class MetricsCollector:
    """Collect and aggregate metrics from both implementations"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path.home() / ".mdm" / "metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._metrics: List[Metric] = []
        self._counters: Dict[str, int] = defaultdict(int)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        
        self._lock = threading.Lock()
        self._start_time = time.time()
    
    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter"""
        with self._lock:
            self._counters[name] += value
            self._metrics.append(Metric(name, value, tags=tags or {}))
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value"""
        with self._lock:
            self._gauges[name] = value
            self._metrics.append(Metric(name, value, tags=tags or {}))
    
    def timer(self, name: str):
        """Context manager for timing operations"""
        class Timer:
            def __init__(self, collector, metric_name):
                self.collector = collector
                self.name = metric_name
                self.start = None
            
            def __enter__(self):
                self.start = time.perf_counter()
                return self
            
            def __exit__(self, *args):
                duration = time.perf_counter() - self.start
                self.collector.record_time(self.name, duration)
        
        return Timer(self, name)
    
    def record_time(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing measurement"""
        with self._lock:
            self._timers[name].append(duration)
            self._metrics.append(Metric(f"{name}.duration", duration, tags=tags or {}))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        with self._lock:
            summary = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timers": {}
            }
            
            # Calculate timer statistics
            for name, values in self._timers.items():
                if values:
                    summary["timers"][name] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "stddev": statistics.stdev(values) if len(values) > 1 else 0
                    }
            
            return summary
    
    def export(self, filename: Optional[str] = None):
        """Export metrics to file"""
        if filename is None:
            filename = f"metrics_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        filepath = self.output_dir / filename
        
        with self._lock:
            data = {
                "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
                "export_time": datetime.now().isoformat(),
                "summary": self.get_summary(),
                "metrics": [m.to_dict() for m in self._metrics[-10000:]]  # Last 10k metrics
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def compare_implementations(self, metric_prefix: str = "implementation") -> Dict[str, Any]:
        """Compare metrics between old and new implementations"""
        old_metrics = {k: v for k, v in self._timers.items() if k.startswith(f"{metric_prefix}.old")}
        new_metrics = {k: v for k, v in self._timers.items() if k.startswith(f"{metric_prefix}.new")}
        
        comparison = {}
        
        for old_key, old_values in old_metrics.items():
            # Find corresponding new metric
            base_name = old_key.replace(f"{metric_prefix}.old.", "")
            new_key = f"{metric_prefix}.new.{base_name}"
            
            if new_key in new_metrics:
                new_values = new_metrics[new_key]
                
                old_mean = statistics.mean(old_values) if old_values else 0
                new_mean = statistics.mean(new_values) if new_values else 0
                
                comparison[base_name] = {
                    "old_mean": old_mean,
                    "new_mean": new_mean,
                    "delta": new_mean - old_mean,
                    "delta_percent": ((new_mean - old_mean) / old_mean * 100) if old_mean else 0
                }
        
        return comparison


# Global metrics collector
metrics_collector = MetricsCollector()


# Decorator for automatic metrics collection
def track_metrics(name: str, implementation: str = "unknown"):
    """Decorator to automatically track function metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tags = {"implementation": implementation, "function": func.__name__}
            
            # Track call count
            metrics_collector.increment(f"{name}.calls", tags=tags)
            
            # Track execution time
            with metrics_collector.timer(f"{name}.{implementation}"):
                try:
                    result = func(*args, **kwargs)
                    metrics_collector.increment(f"{name}.success", tags=tags)
                    return result
                except Exception as e:
                    metrics_collector.increment(f"{name}.errors", tags=tags)
                    raise
        
        return wrapper
    return decorator
```

## Validation Checklist

### Environment Setup
- [ ] Git worktrees created and configured
- [ ] Each worktree has its own virtual environment
- [ ] Development tools installed in each worktree

### Feature Flag System
- [ ] Feature flags loading and saving correctly
- [ ] CLI commands working for flag management
- [ ] Callbacks triggering on flag changes
- [ ] Rollout percentages calculating correctly

### Comparison Testing
- [ ] Comparison framework detecting differences accurately
- [ ] Results saving to disk
- [ ] Report generation working
- [ ] DataFrame comparisons handled correctly

### A/B Testing
- [ ] Router directing traffic correctly
- [ ] Consistent hashing working
- [ ] Metrics tracking both variants
- [ ] Context manager preserving state

### Metrics Collection
- [ ] All metric types (counters, gauges, timers) working
- [ ] Thread-safe operations verified
- [ ] Export functionality tested
- [ ] Comparison reports accurate

## Integration Example

```python
# Example: Using all parallel development tools together
from mdm.core.feature_flags import feature_flags, feature_flag
from mdm.core.ab_testing import ab_test
from mdm.core.metrics import track_metrics
from mdm.testing.comparison import ComparisonTester

# Configure feature flags
feature_flags.set("use_new_backend", False)
feature_flags.set("ab_test_registration_percentage", 10)  # 10% to new implementation

# Old implementation
@track_metrics("dataset.register", implementation="old")
def register_dataset_legacy(name: str, path: str) -> Dict[str, Any]:
    # Original implementation
    pass

# New implementation with A/B testing
@ab_test("registration", identifier_param="name")
@feature_flag("use_new_registrar", fallback=register_dataset_legacy)
@track_metrics("dataset.register", implementation="new")
def register_dataset(name: str, path: str) -> Dict[str, Any]:
    # New implementation
    pass

# Comparison testing
def test_implementations():
    tester = ComparisonTester()
    
    result = tester.compare(
        test_name="dataset_registration",
        old_impl=register_dataset_legacy,
        new_impl=register_dataset,
        args=("test_dataset", "/path/to/data"),
        compare_func=lambda old, new: old["status"] == new["status"]
    )
    
    print(f"Comparison passed: {result.passed}")
    print(f"Performance delta: {result.performance_delta:+.2f}%")
```

## Monitoring Dashboard

Create a simple monitoring script:

```bash
#!/bin/bash
# scripts/monitor_parallel.sh

while true; do
    clear
    echo "=== MDM Parallel Development Monitor ==="
    echo "Time: $(date)"
    echo
    
    # Feature flags status
    echo "Feature Flags:"
    python -m mdm.core.feature_flags list
    echo
    
    # Recent comparison results
    echo "Recent Comparisons:"
    ls -t ~/.mdm/comparison_results/*.json | head -5 | while read f; do
        jq -r '"\(.test_name): \(if .passed then "✅" else "❌" end) Perf: \(.performance_delta)%"' "$f"
    done
    echo
    
    # Metrics summary
    echo "Metrics Summary:"
    python -c "
from mdm.core.metrics import metrics_collector
summary = metrics_collector.get_summary()
for counter, value in summary['counters'].items():
    print(f'  {counter}: {value}')
"
    
    sleep 5
done
```

## Troubleshooting

### Issue: Worktree conflicts
```bash
# Clean up broken worktrees
git worktree prune
git worktree list

# Remove specific worktree
git worktree remove ../mdm-refactor-backend
```

### Issue: Feature flag not taking effect
```python
# Debug feature flags
from mdm.core.feature_flags import feature_flags

# Check current value
print(f"Flag value: {feature_flags.get('use_new_backend')}")

# Check history
feature_flags.history('use_new_backend')

# Force reload
feature_flags.load_flags()
```

### Issue: Metrics not being collected
```python
# Check metrics collector state
from mdm.core.metrics import metrics_collector

# View current metrics
print(metrics_collector.get_summary())

# Force export
metrics_collector.export("debug_metrics.json")
```

## Next Steps

With parallel development environment ready, proceed to [04-configuration-migration.md](04-configuration-migration.md) to begin migrating the configuration system.

## Notes

- Keep feature flags conservative initially (start with 0% rollout)
- Monitor metrics continuously during development
- Use comparison tests to validate every change
- Document any deviations from the plan in the migration log