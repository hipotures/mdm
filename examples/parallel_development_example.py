#!/usr/bin/env python3
"""
Example showing how to use parallel development tools.

This demonstrates:
- Feature flags for gradual rollout
- Comparison testing between implementations
- Metrics collection and analysis
- Integration with DI container
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from mdm.core import (
    feature_flags, 
    feature_flag,
    metrics_collector,
    track_metrics,
    container,
    configure_container,
    inject
)
from mdm.testing import ComparisonTester
from mdm.interfaces import IStorageBackend

# Configure system
print("Configuring parallel development environment...")
print("=" * 60)

# 1. Set up feature flags
feature_flags.set('use_new_backend', False)  # Start with old implementation
feature_flags.set('rollout_percentage', {'new_backend': 25})  # 25% gradual rollout
feature_flags.set('enable_comparison_tests', True)
feature_flags.set('enable_performance_tracking', True)

print("Feature flags configured:")
for flag, value in feature_flags.get_all().items():
    print(f"  {flag}: {value}")

# 2. Example implementations with metrics tracking
print("\n\nExample: Dataset Registration")
print("=" * 60)

# Old implementation
@track_metrics("dataset.register", implementation="old")
def register_dataset_old(name: str, path: str) -> dict:
    """Original dataset registration (simulated)."""
    import time
    time.sleep(0.05)  # Simulate some work
    
    return {
        "name": name,
        "path": path,
        "status": "registered",
        "version": "1.0",
        "backend": "sqlite"
    }

# New implementation with feature flag
@feature_flag('use_new_backend', fallback=register_dataset_old)
@track_metrics("dataset.register", implementation="new")
def register_dataset(name: str, path: str) -> dict:
    """New dataset registration (simulated)."""
    import time
    time.sleep(0.02)  # Faster implementation
    
    return {
        "name": name,
        "path": path,
        "status": "registered",
        "version": "2.0",
        "backend": "stateless_sqlite"
    }

# 3. Demonstrate gradual rollout
print("\nTesting gradual rollout (25% to new implementation):")
results = {"old": 0, "new": 0}

for i in range(20):
    dataset_name = f"test_dataset_{i}"
    
    # Check if this user should get new implementation
    if feature_flags.is_enabled_for_user('new_backend', dataset_name):
        result = register_dataset(dataset_name, f"/data/{dataset_name}")
        results["new"] += 1
    else:
        result = register_dataset_old(dataset_name, f"/data/{dataset_name}")
        results["old"] += 1

print(f"  Old implementation: {results['old']} ({results['old']/20*100:.0f}%)")
print(f"  New implementation: {results['new']} ({results['new']/20*100:.0f}%)")

# 4. Run comparison tests
print("\n\nRunning comparison tests:")
print("=" * 60)

tester = ComparisonTester()

# Test with identical inputs
result = tester.compare(
    test_name="registration_basic",
    old_impl=register_dataset_old,
    new_impl=register_dataset,
    args=("comparison_test", "/data/comparison"),
    compare_func=lambda old, new: old["status"] == new["status"]
)

print(f"Basic registration test: {'PASSED' if result.passed else 'FAILED'}")
print(f"  Performance improvement: {-result.performance_delta:.1f}%")
print(f"  Old version: {result.old_result.get('version', 'N/A')}")
print(f"  New version: {result.new_result.get('version', 'N/A')}")

# Test DataFrame operations
def create_dataset_old(rows: int) -> pd.DataFrame:
    """Old way of creating dataset."""
    import time
    time.sleep(0.1)
    return pd.DataFrame({
        'id': range(rows),
        'value': [i * 2 for i in range(rows)],
        'category': ['A' if i % 2 == 0 else 'B' for i in range(rows)]
    })

def create_dataset_new(rows: int) -> pd.DataFrame:
    """New optimized way."""
    # Using numpy for better performance
    import numpy as np
    return pd.DataFrame({
        'id': np.arange(rows),
        'value': np.arange(rows) * 2,
        'category': np.where(np.arange(rows) % 2 == 0, 'A', 'B')
    })

result = tester.compare(
    test_name="dataset_creation",
    old_impl=create_dataset_old,
    new_impl=create_dataset_new,
    args=(1000,)
)

print(f"\nDataset creation test: {'PASSED' if result.passed else 'FAILED'}")
print(f"  Performance improvement: {-result.performance_delta:.1f}%")
print(f"  Memory improvement: {-result.memory_delta:.1f}%")

# 5. View metrics summary
print("\n\nMetrics Summary:")
print("=" * 60)

summary = metrics_collector.get_summary()
print(f"Total metrics collected: {summary['total_metrics']}")
print(f"Uptime: {summary['uptime_seconds']:.1f} seconds")

print("\nFunction call counts:")
for counter, value in summary['counters'].items():
    if 'calls' in counter:
        print(f"  {counter}: {value}")

print("\nPerformance comparison:")
comparison = metrics_collector.compare_implementations("dataset.register")
for operation, stats in comparison.items():
    print(f"  {operation}:")
    print(f"    Old: {stats['old_mean']:.3f}s (p95: {stats['old_p95']:.3f}s)")
    print(f"    New: {stats['new_mean']:.3f}s (p95: {stats['new_p95']:.3f}s)")
    print(f"    Improvement: {-stats['delta_percent']:.1f}%")

# 6. Generate comparison report
print("\n\nGenerating comparison report...")
report_path = tester.results_dir / "example_report.txt"
report = tester.generate_report(report_path)
print(f"Report saved to: {report_path}")

# 7. Export metrics
print("\nExporting metrics...")
metrics_path = metrics_collector.export("example_metrics.json")
print(f"Metrics exported to: {metrics_path}")

# 8. Demonstrate DI integration
print("\n\nDependency Injection Integration:")
print("=" * 60)

# Configure container
from mdm.config import get_config
config = get_config()
configure_container(config.model_dump())

@inject
def process_with_backend(dataset_name: str, backend: IStorageBackend = None):
    """Function that uses injected backend."""
    print(f"  Processing {dataset_name} with {type(backend).__name__}")
    return backend.dataset_exists(dataset_name)

# Test with old backend
feature_flags.set('use_new_backend', False)
configure_container(config.model_dump())
result = process_with_backend("test_di_old")

# Test with new backend
feature_flags.set('use_new_backend', True)
configure_container(config.model_dump())
result = process_with_backend("test_di_new")

print("\n" + "=" * 60)
print("Parallel development example complete!")
print("\nKey takeaways:")
print("- Feature flags control which implementation is used")
print("- Gradual rollout ensures safe migration")
print("- Comparison tests verify functional equivalence")
print("- Metrics track performance improvements")
print("- Everything integrates with DI container")