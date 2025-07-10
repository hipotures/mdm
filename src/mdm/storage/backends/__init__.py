"""
Storage backends with full compatibility.

This package contains:
- BackendCompatibilityMixin: Provides backward compatibility for missing methods
- StatelessSQLiteBackend: Stateless SQLite implementation
- StatelessDuckDBBackend: Stateless DuckDB implementation
"""

from .compatibility_mixin import BackendCompatibilityMixin
from .stateless_sqlite import StatelessSQLiteBackend
from .stateless_duckdb import StatelessDuckDBBackend

__all__ = [
    'BackendCompatibilityMixin',
    'StatelessSQLiteBackend', 
    'StatelessDuckDBBackend',
]