"""Base storage backend interface."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import Engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from mdm.core.exceptions import StorageError


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    def __init__(self, config: dict[str, Any]):
        """Initialize storage backend.

        Args:
            config: Backend-specific configuration
        """
        self.config = config
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Get backend type identifier."""
        pass

    @abstractmethod
    def create_engine(self, database_path: str) -> Engine:
        """Create SQLAlchemy engine for the database.

        Args:
            database_path: Path or connection string to database

        Returns:
            SQLAlchemy Engine instance
        """
        pass

    @abstractmethod
    def initialize_database(self, engine: Engine) -> None:
        """Initialize database with required tables.

        Args:
            engine: SQLAlchemy engine
        """
        pass

    @abstractmethod
    def get_database_path(self, dataset_name: str, base_path: Path) -> str:
        """Get database path or connection string for dataset.

        Args:
            dataset_name: Name of the dataset
            base_path: Base path for datasets

        Returns:
            Database path or connection string
        """
        pass

    @abstractmethod
    def database_exists(self, database_path: str) -> bool:
        """Check if database exists.

        Args:
            database_path: Path or connection string to database

        Returns:
            True if database exists
        """
        pass

    @abstractmethod
    def create_database(self, database_path: str) -> None:
        """Create a new database.

        Args:
            database_path: Path or connection string to database
        """
        pass

    @abstractmethod
    def drop_database(self, database_path: str) -> None:
        """Drop an existing database.

        Args:
            database_path: Path or connection string to database
        """
        pass

    def get_engine(self, database_path: str) -> Engine:
        """Get or create engine for database.

        Args:
            database_path: Path or connection string to database

        Returns:
            SQLAlchemy Engine instance
        """
        if self._engine is None:
            self._engine = self.create_engine(database_path)
            self._session_factory = sessionmaker(bind=self._engine)
        return self._engine

    @contextmanager
    def session(self, database_path: str) -> Generator[Session, None, None]:
        """Create a database session context manager.

        Args:
            database_path: Path or connection string to database

        Yields:
            SQLAlchemy Session instance
        """
        engine = self.get_engine(database_path)
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=engine)

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def table_exists(self, engine: Engine, table_name: str) -> bool:
        """Check if table exists in database.

        Args:
            engine: SQLAlchemy engine
            table_name: Name of the table

        Returns:
            True if table exists
        """
        inspector = inspect(engine)
        return table_name in inspector.get_table_names()

    def get_table_names(self, engine: Engine) -> list[str]:
        """Get list of table names in database.

        Args:
            engine: SQLAlchemy engine

        Returns:
            List of table names
        """
        inspector = inspect(engine)
        return inspector.get_table_names()

    def create_table_from_dataframe(
        self, df: pd.DataFrame, table_name: str, engine: Engine, if_exists: str = "fail"
    ) -> None:
        """Create table from pandas DataFrame.

        Args:
            df: Pandas DataFrame
            table_name: Name of the table to create
            engine: SQLAlchemy engine
            if_exists: What to do if table exists ('fail', 'replace', 'append')
        """
        try:
            df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        except Exception as e:
            raise StorageError(f"Failed to create table {table_name}: {e}") from e

    def read_table_to_dataframe(
        self, table_name: str, engine: Engine, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Read table into pandas DataFrame.

        Args:
            table_name: Name of the table
            engine: SQLAlchemy engine
            limit: Optional row limit

        Returns:
            Pandas DataFrame
        """
        try:
            query = f"SELECT * FROM {table_name}"
            if limit:
                query += f" LIMIT {limit}"
            return pd.read_sql_query(query, engine)
        except Exception as e:
            raise StorageError(f"Failed to read table {table_name}: {e}") from e

    def get_table_info(self, table_name: str, engine: Engine) -> dict[str, Any]:
        """Get table information.

        Args:
            table_name: Name of the table
            engine: SQLAlchemy engine

        Returns:
            Dictionary with table information
        """
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)

        # Get row count
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()

        return {
            "name": table_name,
            "columns": columns,
            "row_count": row_count,
            "column_count": len(columns),
        }

    def execute_query(self, query: str, engine: Engine) -> Any:
        """Execute arbitrary SQL query.

        Args:
            query: SQL query string
            engine: SQLAlchemy engine

        Returns:
            Query result
        """
        try:
            with engine.connect() as conn:
                return conn.execute(text(query))
        except Exception as e:
            raise StorageError(f"Failed to execute query: {e}") from e

    def close_connections(self) -> None:
        """Close all database connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None

