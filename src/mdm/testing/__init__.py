"""
Testing utilities for MDM refactoring.

This package provides tools for:
- Comparison testing between old and new implementations
- Performance benchmarking
- A/B testing support
"""

from .comparison import ComparisonTester, ComparisonResult

__all__ = [
    'ComparisonTester',
    'ComparisonResult',
]