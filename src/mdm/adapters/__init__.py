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
from .storage_manager import (
    get_storage_backend,
    clear_storage_cache,
    get_available_backends,
)
from .feature_adapters import FeatureGeneratorAdapter
from .dataset_adapters import DatasetRegistrarAdapter, DatasetManagerAdapter
from .feature_manager import (
    get_feature_generator,
    clear_feature_cache,
)
from .config_adapters import (
    LegacyConfigAdapter,
    NewConfigAdapter,
    LegacyConfigManagerAdapter,
    NewConfigManagerAdapter,
    get_config_manager,
    get_config,
)

__all__ = [
    'StorageAdapter',
    'SQLiteAdapter',
    'DuckDBAdapter', 
    'PostgreSQLAdapter',
    'get_storage_backend',
    'clear_storage_cache',
    'get_available_backends',
    'FeatureGeneratorAdapter',
    'DatasetRegistrarAdapter',
    'DatasetManagerAdapter',
    'get_feature_generator',
    'clear_feature_cache',
    'LegacyConfigAdapter',
    'NewConfigAdapter',
    'LegacyConfigManagerAdapter',
    'NewConfigManagerAdapter',
    'get_config_manager',
    'get_config',
]