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
from .feature_flags import (
    FeatureFlags,
    feature_flags,
    feature_flag,
    is_new_backend_enabled,
    is_new_registrar_enabled,
    is_new_features_enabled,
    enable_new_backend,
    disable_new_backend,
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
    # Feature Flags
    'FeatureFlags',
    'feature_flags',
    'feature_flag',
    'is_new_backend_enabled',
    'is_new_registrar_enabled',
    'is_new_features_enabled',
    'enable_new_backend',
    'disable_new_backend',
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

