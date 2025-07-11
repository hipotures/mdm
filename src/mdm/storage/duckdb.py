"""DuckDB storage backend implementation."""

from pathlib import Path

from sqlalchemy import Engine, create_engine, text

from mdm.core.exceptions import StorageError
from mdm.models.base import Base
from mdm.storage.base import StorageBackend
from mdm.storage.backends.compatibility_mixin import BackendCompatibilityMixin


class DuckDBBackend(BackendCompatibilityMixin, StorageBackend):
    """DuckDB storage backend."""

    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return "duckdb"

    def create_engine(self, database_path: str) -> Engine:
        """Create SQLAlchemy engine for DuckDB database.

        Args:
            database_path: Path to DuckDB database file

        Returns:
            SQLAlchemy Engine instance
        """
        # Expand user path if needed
        db_path = Path(database_path).expanduser()

        # Create connection URL
        url = f"duckdb:///{db_path}"

        # Get DuckDB-specific configuration
        duckdb_config = self.config.get("duckdb", {})

        # Create connect args for DuckDB
        connect_args = {}

        # Memory limit
        if "memory_limit" in duckdb_config:
            connect_args["memory_limit"] = duckdb_config["memory_limit"]

        # Number of threads
        if "threads" in duckdb_config:
            connect_args["threads"] = duckdb_config["threads"]

        # Temp directory
        if "temp_directory" in duckdb_config:
            connect_args["temp_directory"] = duckdb_config["temp_directory"]

        # Access mode
        if "access_mode" in duckdb_config:
            read_only = duckdb_config["access_mode"] == "READ_ONLY"
            connect_args["read_only"] = read_only

        # Create engine with DuckDB-specific options
        return create_engine(
            url,
            echo=self.config.get("sqlalchemy", {}).get("echo", False),
            pool_size=self.config.get("sqlalchemy", {}).get("pool_size", 5),
            max_overflow=self.config.get("sqlalchemy", {}).get("max_overflow", 10),
            connect_args=connect_args,
        )

    def initialize_database(self, engine: Engine) -> None:
        """Initialize DuckDB database with required tables.

        Args:
            engine: SQLAlchemy engine
        """
        # Create all tables defined in Base
        Base.metadata.create_all(engine)

        # Install useful DuckDB extensions
        with engine.connect() as conn:
            # Install parquet extension for data import/export
            conn.execute(text("INSTALL parquet"))
            conn.execute(text("LOAD parquet"))
            conn.commit()

    def get_database_path(self, dataset_name: str, base_path: Path) -> str:
        """Get DuckDB database file path for dataset.

        Args:
            dataset_name: Name of the dataset
            base_path: Base path for datasets

        Returns:
            Full path to DuckDB database file
        """
        dataset_dir = base_path / dataset_name.lower()
        return str(dataset_dir / "dataset.duckdb")

    def database_exists(self, database_path: str) -> bool:
        """Check if DuckDB database file exists.

        Args:
            database_path: Path to DuckDB database file

        Returns:
            True if database exists
        """
        db_path = Path(database_path).expanduser()
        return db_path.exists() and db_path.is_file()

    def create_database(self, database_path: str) -> None:
        """Create a new DuckDB database.

        Args:
            database_path: Path to DuckDB database file
        """
        db_path = Path(database_path).expanduser()

        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty database by connecting and immediately closing
        try:
            import duckdb

            conn = duckdb.connect(str(db_path))
            conn.close()
        except Exception as e:
            raise StorageError(f"Failed to create DuckDB database: {e}") from e

    def drop_database(self, database_path: str) -> None:
        """Drop (delete) DuckDB database file.

        Args:
            database_path: Path to DuckDB database file
        """
        db_path = Path(database_path).expanduser()

        if db_path.exists():
            try:
                # Close any existing connections
                self.close_connections()
                # Delete the file
                db_path.unlink()
                # Also delete any WAL files
                wal_path = db_path.with_suffix(".duckdb.wal")
                if wal_path.exists():
                    wal_path.unlink()
            except Exception as e:
                raise StorageError(f"Failed to drop DuckDB database: {e}") from e

