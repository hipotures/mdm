"""PostgreSQL storage backend implementation."""

from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus, urlparse

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import Engine, create_engine, text

from mdm.core.exceptions import StorageError
from mdm.models.base import Base
from mdm.storage.base import StorageBackend


class PostgreSQLBackend(StorageBackend):
    """PostgreSQL storage backend."""

    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return "postgresql"

    def _build_connection_string(
        self, database_name: Optional[str] = None, admin: bool = False
    ) -> str:
        """Build PostgreSQL connection string.

        Args:
            database_name: Optional database name
            admin: If True, connect to postgres database for admin operations

        Returns:
            PostgreSQL connection string
        """
        pg_config = self.config.get("postgresql", {})

        # Get connection parameters
        host = pg_config.get("host", "localhost")
        port = pg_config.get("port", 5432)
        user = pg_config.get("user")
        password = pg_config.get("password")

        if not user:
            raise StorageError("PostgreSQL user not configured")

        # Build base URL
        if password:
            password_encoded = quote_plus(password)
            base_url = f"postgresql://{user}:{password_encoded}@{host}:{port}"
        else:
            base_url = f"postgresql://{user}@{host}:{port}"

        # Add database name
        if admin:
            db_url = f"{base_url}/postgres"
        elif database_name:
            db_url = f"{base_url}/{database_name}"
        else:
            db_url = base_url

        # Add SSL parameters if configured
        sslmode = pg_config.get("sslmode", "prefer")
        params = [f"sslmode={sslmode}"]

        if "sslcert" in pg_config:
            params.append(f"sslcert={pg_config['sslcert']}")
        if "sslkey" in pg_config:
            params.append(f"sslkey={pg_config['sslkey']}")

        if params:
            db_url += "?" + "&".join(params)

        return db_url

    def create_engine(self, database_path: str) -> Engine:
        """Create SQLAlchemy engine for PostgreSQL database.

        Args:
            database_path: PostgreSQL connection string or database name

        Returns:
            SQLAlchemy Engine instance
        """
        # If database_path looks like a connection string, use it directly
        if database_path.startswith("postgresql://"):
            url = database_path
        else:
            # Otherwise, treat it as a database name
            url = self._build_connection_string(database_path)

        pg_config = self.config.get("postgresql", {})

        # Create engine with PostgreSQL-specific options
        return create_engine(
            url,
            echo=self.config.get("sqlalchemy", {}).get("echo", False),
            pool_size=pg_config.get("pool_size", 10),
            max_overflow=self.config.get("sqlalchemy", {}).get("max_overflow", 20),
            pool_timeout=self.config.get("sqlalchemy", {}).get("pool_timeout", 30),
            pool_recycle=self.config.get("sqlalchemy", {}).get("pool_recycle", 3600),
        )

    def initialize_database(self, engine: Engine) -> None:
        """Initialize PostgreSQL database with required tables.

        Args:
            engine: SQLAlchemy engine
        """
        # Create all tables defined in Base
        Base.metadata.create_all(engine)

        # Create useful extensions
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
            conn.commit()

    def get_database_path(self, dataset_name: str, base_path: Path) -> str:
        """Get PostgreSQL database name for dataset.

        Args:
            dataset_name: Name of the dataset
            base_path: Base path for datasets (not used for PostgreSQL)

        Returns:
            PostgreSQL database name
        """
        pg_config = self.config.get("postgresql", {})
        prefix = pg_config.get("database_prefix", "mdm_")
        return f"{prefix}{dataset_name.lower()}"

    def database_exists(self, database_path: str) -> bool:
        """Check if PostgreSQL database exists.

        Args:
            database_path: Database name or connection string

        Returns:
            True if database exists
        """
        # Extract database name if full connection string provided
        if database_path.startswith("postgresql://"):
            parsed = urlparse(database_path)
            db_name = parsed.path.lstrip("/")
        else:
            db_name = database_path

        try:
            # Connect to postgres database to check
            admin_url = self._build_connection_string(admin=True)
            engine = create_engine(admin_url)

            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                    {"dbname": db_name},
                )
                exists = result.scalar() is not None

            engine.dispose()
            return exists

        except Exception as e:
            raise StorageError(f"Failed to check database existence: {e}") from e

    def create_database(self, database_path: str) -> None:
        """Create a new PostgreSQL database.

        Args:
            database_path: Database name
        """
        # Extract database name if needed
        if database_path.startswith("postgresql://"):
            parsed = urlparse(database_path)
            db_name = parsed.path.lstrip("/")
        else:
            db_name = database_path

        try:
            # Connect to postgres database to create new database
            pg_config = self.config.get("postgresql", {})
            conn_params = {
                "host": pg_config.get("host", "localhost"),
                "port": pg_config.get("port", 5432),
                "user": pg_config.get("user"),
                "password": pg_config.get("password"),
                "database": "postgres",
            }

            # Remove None values
            conn_params = {k: v for k, v in conn_params.items() if v is not None}

            # Create database using psycopg2 (SQLAlchemy doesn't support CREATE DATABASE well)
            conn = psycopg2.connect(**conn_params)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Create database with UTF8 encoding
            cursor.execute(
                f'CREATE DATABASE "{db_name}" ENCODING \'UTF8\' LC_COLLATE=\'en_US.UTF-8\' LC_CTYPE=\'en_US.UTF-8\''
            )

            cursor.close()
            conn.close()

        except psycopg2.errors.DuplicateDatabase:
            # Database already exists, that's fine
            pass
        except Exception as e:
            raise StorageError(f"Failed to create PostgreSQL database: {e}") from e

    def drop_database(self, database_path: str) -> None:
        """Drop PostgreSQL database.

        Args:
            database_path: Database name or connection string
        """
        # Extract database name if needed
        if database_path.startswith("postgresql://"):
            parsed = urlparse(database_path)
            db_name = parsed.path.lstrip("/")
        else:
            db_name = database_path

        try:
            # Close any existing connections
            self.close_connections()

            # Connect to postgres database to drop target database
            pg_config = self.config.get("postgresql", {})
            conn_params = {
                "host": pg_config.get("host", "localhost"),
                "port": pg_config.get("port", 5432),
                "user": pg_config.get("user"),
                "password": pg_config.get("password"),
                "database": "postgres",
            }

            # Remove None values
            conn_params = {k: v for k, v in conn_params.items() if v is not None}

            conn = psycopg2.connect(**conn_params)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Terminate existing connections to the database
            cursor.execute(
                f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{db_name}'
                AND pid <> pg_backend_pid()
                """
            )

            # Drop the database
            cursor.execute(f'DROP DATABASE IF EXISTS "{db_name}"')

            cursor.close()
            conn.close()

        except Exception as e:
            raise StorageError(f"Failed to drop PostgreSQL database: {e}") from e

