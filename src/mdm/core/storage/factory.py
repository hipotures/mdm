"""Factory for creating new storage backend instances.

This module provides the factory pattern for creating storage backends
based on configuration.
"""
from typing import Dict, Any, Optional
import logging

from mdm.interfaces.storage import IStorageBackend
from .sqlite import NewSQLiteBackend
from .duckdb import NewDuckDBBackend
from .postgresql import NewPostgreSQLBackend
from mdm.core.exceptions import StorageError

logger = logging.getLogger(__name__)


# Registry of available backends
BACKEND_REGISTRY = {
    "sqlite": NewSQLiteBackend,
    "duckdb": NewDuckDBBackend,
    "postgresql": NewPostgreSQLBackend,
}


def create_storage_backend(
    backend_type: str,
    config: Optional[Dict[str, Any]] = None
) -> IStorageBackend:
    """Create a new storage backend instance.
    
    Args:
        backend_type: Type of backend to create
        config: Backend-specific configuration
        
    Returns:
        Storage backend instance
        
    Raises:
        StorageError: If backend type is not supported
    """
    if backend_type not in BACKEND_REGISTRY:
        raise StorageError(
            f"Unsupported backend type: {backend_type}. "
            f"Available types: {list(BACKEND_REGISTRY.keys())}"
        )
    
    # Get backend class
    backend_class = BACKEND_REGISTRY[backend_type]
    
    # Create instance with config
    backend = backend_class(config or {})
    
    logger.info(f"Created new {backend_type} storage backend")
    
    return backend


def get_available_backends() -> list[str]:
    """Get list of available backend types.
    
    Returns:
        List of backend type identifiers
    """
    return list(BACKEND_REGISTRY.keys())


def register_backend(backend_type: str, backend_class: type) -> None:
    """Register a custom backend type.
    
    This allows extending the system with custom backends.
    
    Args:
        backend_type: Type identifier for the backend
        backend_class: Backend class that implements IStorageBackend
    """
    if not issubclass(backend_class, IStorageBackend):
        raise ValueError(
            f"Backend class must implement IStorageBackend protocol"
        )
    
    BACKEND_REGISTRY[backend_type] = backend_class
    logger.info(f"Registered custom backend: {backend_type}")