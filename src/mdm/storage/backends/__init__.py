"""
Storage backends for MDM.

This package contains:
- StatelessSQLiteBackend: Stateless SQLite implementation
- StatelessDuckDBBackend: Stateless DuckDB implementation
"""

from .stateless_sqlite import StatelessSQLiteBackend
from .stateless_duckdb import StatelessDuckDBBackend

__all__ = [
    'StatelessSQLiteBackend', 
    'StatelessDuckDBBackend',
]