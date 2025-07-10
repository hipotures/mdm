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
    is_new_backend_enabled,
    is_new_registrar_enabled,
    is_new_features_enabled,
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
    'is_new_backend_enabled',
    'is_new_registrar_enabled',
    'is_new_features_enabled',
]

