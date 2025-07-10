"""Storage backend factory."""

from typing import Any
import logging

from mdm.core.exceptions import BackendError
from mdm.storage.base import StorageBackend
from mdm.storage.duckdb import DuckDBBackend
from mdm.storage.postgresql import PostgreSQLBackend
from mdm.storage.sqlite import SQLiteBackend

# Import feature flags
from mdm.core.feature_flags import is_new_backend_enabled

logger = logging.getLogger(__name__)


class BackendFactory:
    """Factory for creating storage backend instances."""

    _backends = {
        "sqlite": SQLiteBackend,
        "duckdb": DuckDBBackend,
        "postgresql": PostgreSQLBackend,
    }
    
    _new_backends = None  # Lazy load to avoid circular imports

    @classmethod
    def _get_new_backends(cls):
        """Lazy load new backends to avoid circular imports."""
        if cls._new_backends is None:
            try:
                from mdm.storage.backends.stateless_sqlite import StatelessSQLiteBackend
                from mdm.storage.backends.stateless_duckdb import StatelessDuckDBBackend
                # PostgreSQL not implemented yet in new architecture
                
                cls._new_backends = {
                    "sqlite": StatelessSQLiteBackend,
                    "duckdb": StatelessDuckDBBackend,
                    "postgresql": PostgreSQLBackend,  # Fall back to old for now
                }
                logger.info("New stateless backends loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to load new backends: {e}")
                cls._new_backends = cls._backends  # Fall back to old backends
        
        return cls._new_backends

    @classmethod
    def create(cls, backend_type: str, config: dict[str, Any]) -> StorageBackend:
        """Create storage backend instance.

        Args:
            backend_type: Type of backend ('sqlite', 'duckdb', 'postgresql')
            config: Backend configuration dictionary

        Returns:
            StorageBackend instance

        Raises:
            BackendError: If backend type is not supported
        """
        # Check feature flag
        if is_new_backend_enabled():
            backends = cls._get_new_backends()
            logger.info(f"Using new stateless backend for {backend_type}")
        else:
            backends = cls._backends
            logger.debug(f"Using legacy backend for {backend_type}")
        
        if backend_type not in backends:
            raise BackendError(
                f"Unsupported backend type: {backend_type}. "
                f"Supported backends: {list(backends.keys())}"
            )

        backend_class = backends[backend_type]
        
        # New backends don't take config in constructor
        if is_new_backend_enabled() and backend_type in ["sqlite", "duckdb"]:
            return backend_class()  # type: ignore[abstract]
        else:
            return backend_class(config)  # type: ignore[abstract]

    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """Get list of supported backend types.

        Returns:
            List of backend type identifiers
        """
        return list(cls._backends.keys())

