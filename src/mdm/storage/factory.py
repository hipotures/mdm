"""Storage backend factory."""

from typing import Any
import logging

from mdm.core.exceptions import BackendError
from mdm.storage.base import StorageBackend
from mdm.storage.backends.stateless_sqlite import StatelessSQLiteBackend
from mdm.storage.backends.stateless_duckdb import StatelessDuckDBBackend
from mdm.storage.postgresql import PostgreSQLBackend

logger = logging.getLogger(__name__)


class BackendFactory:
    """Factory for creating storage backend instances."""

    _backends = {
        "sqlite": StatelessSQLiteBackend,
        "duckdb": StatelessDuckDBBackend,
        "postgresql": PostgreSQLBackend,  # TODO: Create stateless version
    }

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
        if backend_type not in cls._backends:
            raise BackendError(
                f"Unsupported backend type: {backend_type}. "
                f"Supported backends: {list(cls._backends.keys())}"
            )

        backend_class = cls._backends[backend_type]
        
        # Stateless backends don't take config in constructor
        if backend_type in ["sqlite", "duckdb"]:
            return backend_class()  # type: ignore[abstract]
        else:
            # PostgreSQL still uses old interface for now
            return backend_class(config)  # type: ignore[abstract]

    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """Get list of supported backend types.

        Returns:
            List of backend type identifiers
        """
        return list(cls._backends.keys())

