"""SQLite storage backend implementation."""

import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event

from mdm.core.exceptions import StorageError
from mdm.models.base import Base
from mdm.storage.base import StorageBackend
from mdm.storage.backends.compatibility_mixin import BackendCompatibilityMixin


class SQLiteBackend(BackendCompatibilityMixin, StorageBackend):
    """SQLite storage backend."""

    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return "sqlite"

    def create_engine(self, database_path: str) -> Engine:
        """Create SQLAlchemy engine for SQLite database.

        Args:
            database_path: Path to SQLite database file

        Returns:
            SQLAlchemy Engine instance
        """
        # Expand user path if needed
        db_path = Path(database_path).expanduser()

        # Create connection URL
        url = f"sqlite:///{db_path}"

        # Create engine with SQLite-specific options
        engine = create_engine(
            url,
            echo=self.config.get("sqlalchemy", {}).get("echo", False),
            pool_size=self.config.get("sqlalchemy", {}).get("pool_size", 5),
            max_overflow=self.config.get("sqlalchemy", {}).get("max_overflow", 10),
        )

        # Set SQLite-specific pragmas
        # Check if config has nested sqlite key (from DatasetManager) or flat structure (from registrar)
        if "sqlite" in self.config:
            sqlite_config = self.config.get("sqlite", {})
        else:
            # Config is already SQLite-specific from registrar
            sqlite_config = self.config

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            """Set SQLite pragmas on connection."""
            cursor = dbapi_connection.cursor()

            # Journal mode
            journal_mode = sqlite_config.get("journal_mode", "WAL")
            cursor.execute(f"PRAGMA journal_mode={journal_mode}")

            # Synchronous mode
            synchronous = sqlite_config.get("synchronous", "NORMAL")
            cursor.execute(f"PRAGMA synchronous={synchronous}")

            # Cache size
            cache_size = sqlite_config.get("cache_size", -64000)
            cursor.execute(f"PRAGMA cache_size={cache_size}")

            # Temp store (need to convert string to numeric value)
            temp_store_str = sqlite_config.get("temp_store", "MEMORY")
            temp_store_map = {"DEFAULT": 0, "FILE": 1, "MEMORY": 2}
            temp_store = temp_store_map.get(temp_store_str.upper(), 2)
            cursor.execute(f"PRAGMA temp_store={temp_store}")

            # Memory-mapped I/O
            mmap_size = sqlite_config.get("mmap_size", 268435456)
            cursor.execute(f"PRAGMA mmap_size={mmap_size}")

            cursor.close()

        return engine

    def initialize_database(self, engine: Engine) -> None:
        """Initialize SQLite database with required tables.

        Args:
            engine: SQLAlchemy engine
        """
        # Create all tables defined in Base
        Base.metadata.create_all(engine)

    def get_database_path(self, dataset_name: str, base_path: Path) -> str:
        """Get SQLite database file path for dataset.

        Args:
            dataset_name: Name of the dataset
            base_path: Base path for datasets

        Returns:
            Full path to SQLite database file
        """
        dataset_dir = base_path / dataset_name.lower()
        return str(dataset_dir / "dataset.sqlite")

    def database_exists(self, database_path: str) -> bool:
        """Check if SQLite database file exists.

        Args:
            database_path: Path to SQLite database file

        Returns:
            True if database exists
        """
        db_path = Path(database_path).expanduser()
        return db_path.exists() and db_path.is_file()

    def create_database(self, database_path: str) -> None:
        """Create a new SQLite database.

        Args:
            database_path: Path to SQLite database file
        """
        db_path = Path(database_path).expanduser()

        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty database
        try:
            conn = sqlite3.connect(str(db_path))
            conn.close()
        except Exception as e:
            raise StorageError(f"Failed to create SQLite database: {e}") from e

    def drop_database(self, database_path: str) -> None:
        """Drop (delete) SQLite database file.

        Args:
            database_path: Path to SQLite database file
        """
        db_path = Path(database_path).expanduser()

        if db_path.exists():
            try:
                # Close any existing connections
                self.close_connections()
                # Delete the file
                db_path.unlink()
            except Exception as e:
                raise StorageError(f"Failed to drop SQLite database: {e}") from e

