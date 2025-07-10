"""
Testing utilities for MDM refactoring.

This package provides tools for:
- Comparison testing between old and new implementations
- Performance benchmarking
- A/B testing support
"""

from .comparison import ComparisonTester, ComparisonResult
from .config_comparison import ConfigComparisonTester
from .storage_comparison import StorageComparisonTester, TestResult
from .feature_comparison import FeatureComparisonTester, FeatureTestResult

__all__ = [
    'ComparisonTester',
    'ComparisonResult',
    'ConfigComparisonTester',
    'StorageComparisonTester',
    'TestResult',
    'FeatureComparisonTester',
    'FeatureTestResult',
]