"""Base storage backend interface."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Dict

import pandas as pd
from sqlalchemy import Engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from mdm.core.exceptions import StorageError
from mdm.performance import (
    QueryOptimizer,
    ConnectionPool,
    PoolConfig,
    CacheManager,
    BatchOptimizer,
    get_monitor
)


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
        
        # Performance optimization components
        self._query_optimizer: Optional[QueryOptimizer] = None
        self._connection_pool: Optional[ConnectionPool] = None
        self._cache_manager: Optional[CacheManager] = None
        self._batch_optimizer: Optional[BatchOptimizer] = None
        self._monitor = get_monitor()
        
        # Initialize performance optimizations if enabled
        if config.get('enable_performance_optimizations', True):
            self._initialize_performance_optimizations()

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

    def _initialize_performance_optimizations(self) -> None:
        """Initialize performance optimization components."""
        # Query optimizer
        self._query_optimizer = QueryOptimizer(
            cache_query_plans=self.config.get('cache_query_plans', True)
        )
        
        # Cache manager
        cache_config = self.config.get('cache', {})
        self._cache_manager = CacheManager(
            max_size_mb=cache_config.get('max_size_mb', 100),
            default_ttl=cache_config.get('default_ttl', 300)
        )
        
        # Batch optimizer
        batch_config = self.config.get('batch', {})
        from mdm.performance import BatchConfig
        self._batch_optimizer = BatchOptimizer(
            BatchConfig(
                batch_size=batch_config.get('batch_size', 10000),
                max_workers=batch_config.get('max_workers', 4),
                enable_parallel=batch_config.get('enable_parallel', True)
            )
        )
    
    def get_engine(self, database_path: str) -> Engine:
        """Get or create engine for database.

        Args:
            database_path: Path or connection string to database

        Returns:
            SQLAlchemy Engine instance
        """
        if self._engine is None:
            # Use connection pool if available
            if self._connection_pool is None and self.config.get('use_connection_pool', True):
                pool_config = self.config.get('pool', {})
                # For SQLite and DuckDB, convert file path to proper URL
                # For PostgreSQL, let the backend build the connection string
                connection_string = database_path
                if self.backend_type == "sqlite" and not database_path.startswith("sqlite:"):
                    connection_string = f"sqlite:///{database_path}"
                elif self.backend_type == "duckdb" and not database_path.startswith("duckdb:"):
                    connection_string = f"duckdb:///{database_path}"
                elif self.backend_type == "postgresql" and not database_path.startswith("postgresql:"):
                    # PostgreSQL backend needs to build the connection string
                    from mdm.storage.postgresql import PostgreSQLBackend
                    pg_backend = self if isinstance(self, PostgreSQLBackend) else None
                    if pg_backend:
                        connection_string = pg_backend._build_connection_string(database_path)
                
                self._connection_pool = ConnectionPool(
                    connection_string,
                    PoolConfig(
                        pool_size=pool_config.get('pool_size', 5),
                        max_overflow=pool_config.get('max_overflow', 10),
                        timeout=pool_config.get('timeout', 30.0)
                    )
                )
                self._engine = self._connection_pool._engine
            else:
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
            # Use batch optimizer for large dataframes
            if self._batch_optimizer and len(df) > 10000:
                with self._monitor.track_operation("table_creation", table=table_name):
                    # Insert in batches
                    df.to_sql(table_name, engine, if_exists=if_exists, index=False, 
                            chunksize=self._batch_optimizer.config.batch_size)
            else:
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
            
            # Check cache first
            cache_key = f"table:{table_name}:{limit}"
            if self._cache_manager:
                cached_df = self._cache_manager.get(cache_key)
                if cached_df is not None:
                    self._monitor.track_cache("table_read", hit=True)
                    return cached_df
                else:
                    self._monitor.track_cache("table_read", hit=False)
            
            # Optimize query if optimizer available
            if self._query_optimizer:
                query, plan = self._query_optimizer.optimize_query(query, engine)
            
            with self._monitor.track_operation("table_read", table=table_name):
                df = pd.read_sql_query(query, engine)
            
            # Cache result
            if self._cache_manager and len(df) < 100000:  # Don't cache very large results
                self._cache_manager.set(cache_key, df, ttl=300)
            
            return df
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
            query: SQL query to execute
            engine: SQLAlchemy engine
            
        Returns:
            Query result
        """
        with self._monitor.track_operation("query_execution"):
            # Optimize query if optimizer available
            if self._query_optimizer:
                query, plan = self._query_optimizer.optimize_query(query, engine)
                if plan.execution_time:
                    self._monitor.track_query(
                        plan.query_type.value,
                        plan.execution_time,
                        plan.estimated_rows
                    )
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return result.fetchall()

    def query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame.

        Args:
            query: SQL query string

        Returns:
            Query results as pandas DataFrame
        """
        if self._engine is None:
            raise StorageError("No engine available. Call get_engine() first.")
        
        try:
            return pd.read_sql_query(query, self._engine)
        except Exception as e:
            raise StorageError(f"Failed to execute query: {e}") from e

    def close_connections(self) -> None:
        """Close all database connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None


