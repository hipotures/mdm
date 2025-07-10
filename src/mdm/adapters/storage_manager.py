"""Storage backend manager with feature flag support.

This module provides the main entry point for getting storage backends during
the migration period, automatically selecting between legacy and new implementations.
"""
from typing import Any, Dict, Optional
import logging

from mdm.interfaces.storage import IStorageBackend
from mdm.adapters.storage_adapters import SQLiteAdapter, DuckDBAdapter, PostgreSQLAdapter
from mdm.core import feature_flags, metrics_collector

logger = logging.getLogger(__name__)


class StorageBackendManager:
    """Manager for storage backends with caching and feature flag support."""
    
    def __init__(self):
        """Initialize the manager."""
        self._adapters: Dict[str, IStorageBackend] = {}
        self._adapter_classes = {
            "sqlite": SQLiteAdapter,
            "duckdb": DuckDBAdapter,
            "postgresql": PostgreSQLAdapter,
        }
        
    def get_backend(
        self, 
        backend_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> IStorageBackend:
        """Get storage backend based on feature flags.
        
        Args:
            backend_type: Type of backend ('sqlite', 'duckdb', 'postgresql')
            config: Optional backend configuration (not used for legacy adapters)
            
        Returns:
            Storage backend instance implementing IStorageBackend
            
        Raises:
            ValueError: If backend type is not supported
        """
        # Validate backend type
        if backend_type not in self._adapter_classes:
            raise ValueError(
                f"Unsupported backend type: {backend_type}. "
                f"Supported types: {list(self._adapter_classes.keys())}"
            )
        
        # Check feature flag for new storage
        if feature_flags.get("use_new_storage", False):
            logger.info(f"Using new storage backend for {backend_type}")
            metrics_collector.increment(
                "storage.backend_created",
                tags={"type": backend_type, "implementation": "new"}
            )
            
            # Import here to avoid circular imports
            from mdm.core.storage.factory import create_storage_backend
            
            # Create new backend with config
            backend = create_storage_backend(backend_type, config)
            
            # Note: New backends are not cached as they manage their own connections
            return backend
        else:
            logger.debug(f"Using legacy storage backend for {backend_type}")
            metrics_collector.increment(
                "storage.backend_created",
                tags={"type": backend_type, "implementation": "legacy"}
            )
            
            # Return cached adapter if available
            if backend_type in self._adapters:
                return self._adapters[backend_type]
            
            # Create new adapter
            adapter_class = self._adapter_classes[backend_type]
            adapter = adapter_class()
            
            # Cache the adapter
            self._adapters[backend_type] = adapter
            
            return adapter
    
    def clear_cache(self) -> None:
        """Clear all cached backends."""
        logger.info("Clearing storage backend cache")
        
        # Close all adapters
        for backend_type, adapter in self._adapters.items():
            try:
                adapter.close()
                logger.debug(f"Closed {backend_type} adapter")
            except Exception as e:
                logger.warning(f"Error closing {backend_type} adapter: {e}")
        
        # Clear the cache
        self._adapters.clear()
    
    def get_available_backends(self) -> list[str]:
        """Get list of available backend types.
        
        Returns:
            List of backend type identifiers
        """
        return list(self._adapter_classes.keys())
    
    def validate_backend_config(
        self,
        backend_type: str,
        config: Dict[str, Any]
    ) -> bool:
        """Validate configuration for a backend type.
        
        Args:
            backend_type: Type of backend
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        # For now, legacy adapters don't use config parameter
        # This will be implemented when new backends are added
        return backend_type in self._adapter_classes


# Global manager instance
_manager = StorageBackendManager()


def get_storage_backend(
    backend_type: str,
    config: Optional[Dict[str, Any]] = None
) -> IStorageBackend:
    """Get storage backend instance.
    
    This is the main entry point for getting storage backends during migration.
    It automatically selects between legacy and new implementations based on
    feature flags.
    
    Args:
        backend_type: Type of backend to create ('sqlite', 'duckdb', 'postgresql')
        config: Optional backend configuration
        
    Returns:
        Storage backend instance implementing IStorageBackend
        
    Example:
        >>> from mdm.adapters import get_storage_backend
        >>> backend = get_storage_backend("sqlite")
        >>> engine = backend.get_engine("/path/to/database.db")
    """
    return _manager.get_backend(backend_type, config)


def clear_storage_cache() -> None:
    """Clear all cached storage backends.
    
    This should be called when switching feature flags or during cleanup.
    """
    _manager.clear_cache()


def get_available_backends() -> list[str]:
    """Get list of available storage backend types.
    
    Returns:
        List of backend type identifiers
    """
    return _manager.get_available_backends()