"""
Adapters for existing implementations.

These adapters wrap the existing implementations to provide a consistent
interface during the migration period.
"""

from .storage_adapters import (
    StorageAdapter,
    SQLiteAdapter,
    DuckDBAdapter,
    PostgreSQLAdapter,
)
from .feature_adapters import FeatureGeneratorAdapter
from .dataset_adapters import DatasetRegistrarAdapter, DatasetManagerAdapter

__all__ = [
    'StorageAdapter',
    'SQLiteAdapter',
    'DuckDBAdapter', 
    'PostgreSQLAdapter',
    'FeatureGeneratorAdapter',
    'DatasetRegistrarAdapter',
    'DatasetManagerAdapter',
]