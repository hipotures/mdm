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
from .dataset_manager import (
    get_dataset_registrar,
    get_dataset_manager,
    clear_dataset_cache,
    get_registration_metrics,
)
from .config_adapters import (
    LegacyConfigAdapter,
    NewConfigAdapter,
    LegacyConfigManagerAdapter,
    NewConfigManagerAdapter,
    get_config_manager,
    get_config,
)
from .cli_manager import (
    get_dataset_commands,
    get_batch_commands,
    get_timeseries_commands,
    get_stats_commands,
    get_cli_formatter,
    get_cli_config,
    execute_command,
    clear_cli_cache,
    get_cli_metrics,
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
    'get_dataset_registrar',
    'get_dataset_manager',
    'clear_dataset_cache',
    'get_registration_metrics',
    'LegacyConfigAdapter',
    'NewConfigAdapter',
    'LegacyConfigManagerAdapter',
    'NewConfigManagerAdapter',
    'get_config_manager',
    'get_config',
    'get_dataset_commands',
    'get_batch_commands',
    'get_timeseries_commands',
    'get_stats_commands',
    'get_cli_formatter',
    'get_cli_config',
    'execute_command',
    'clear_cli_cache',
    'get_cli_metrics',
]