"""Core functionality for MDM."""

from .container import (
    DIContainer,
    container,
    configure_container,
    inject,
    get_service,
    has_service,
    ServiceNotFoundError,
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
    'DIContainer',
    'container',
    'configure_container',
    'inject',
    'get_service',
    'has_service',
    'ServiceNotFoundError',
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

