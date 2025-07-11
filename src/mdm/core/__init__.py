"""Core functionality for MDM."""

from .di import (
    Container,
    get_container,
    configure_services,
    inject,
    get_service,
    create_scope,
    ServiceNotRegisteredError,
)
from .exceptions import (
    MDMError,
    DatasetError,
    StorageError,
    ConfigError,
)
from .metrics import (
    MetricsCollector,
    metrics_collector,
    track_metrics,
    track_operation,
    track_query,
    track_feature_generation,
    record_dataset_size,
)
from .logging import (
    logger,
    get_logger,
    configure_logging,
)

__all__ = [
    # DI Container
    'Container',
    'get_container',
    'configure_services',
    'inject',
    'get_service',
    'create_scope',
    'ServiceNotRegisteredError',
    # Exceptions
    'MDMError',
    'DatasetError',
    'StorageError',
    'ConfigError',
    # Metrics
    'MetricsCollector',
    'metrics_collector',
    'track_metrics',
    'track_operation',
    'track_query',
    'track_feature_generation',
    'record_dataset_size',
    # Logging
    'logger',
    'get_logger',
    'configure_logging',
]

