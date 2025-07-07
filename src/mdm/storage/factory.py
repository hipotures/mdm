"""Storage backend factory."""

from typing import Any

from mdm.core.exceptions import BackendError
from mdm.storage.base import StorageBackend
from mdm.storage.duckdb import DuckDBBackend
from mdm.storage.postgresql import PostgreSQLBackend
from mdm.storage.sqlite import SQLiteBackend


class BackendFactory:
    """Factory for creating storage backend instances."""

    _backends = {
        "sqlite": SQLiteBackend,
        "duckdb": DuckDBBackend,
        "postgresql": PostgreSQLBackend,
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
        return backend_class(config)  # type: ignore[abstract]

    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """Get list of supported backend types.

        Returns:
            List of backend type identifiers
        """
        return list(cls._backends.keys())

