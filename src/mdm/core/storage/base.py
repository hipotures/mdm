"""Base class for new storage backend implementations.

This module provides the base implementation for all new storage backends,
implementing the IStorageBackend protocol with improved design patterns.
"""
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from contextlib import contextmanager
from abc import ABC, abstractmethod
import logging
import pandas as pd
from sqlalchemy import Engine, create_engine, inspect, text, MetaData, Table
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from mdm.interfaces.storage import IStorageBackend
from mdm.core.exceptions import StorageError
from mdm.core import metrics_collector

logger = logging.getLogger(__name__)


class NewStorageBackend(ABC, IStorageBackend):
    """Base class for new storage backend implementations.
    
    This class provides common functionality for all storage backends while
    implementing the IStorageBackend protocol. Key improvements over legacy:
    
    - Better separation of concerns
    - Cleaner engine management
    - Improved error handling
    - Built-in metrics collection
    - Thread-safe operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize storage backend.
        
        Args:
            config: Backend-specific configuration
        """
        self.config = config
        self._engines: Dict[str, Engine] = {}
        self._metadata_cache: Dict[str, MetaData] = {}
        self._session_factory = sessionmaker()
        
    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Get backend type identifier."""
        pass
    
    @abstractmethod
    def _create_connection_string(self, database_path: str) -> str:
        """Create backend-specific connection string.
        
        Args:
            database_path: Path or identifier for database
            
        Returns:
            Connection string for SQLAlchemy
        """
        pass
    
    @abstractmethod
    def _initialize_backend_specific(self, engine: Engine) -> None:
        """Perform backend-specific initialization.
        
        Args:
            engine: SQLAlchemy engine
        """
        pass
    
    def get_engine(self, database_path: str) -> Engine:
        """Get SQLAlchemy engine for database."""
        with metrics_collector.timer(
            "storage.get_engine", 
            tags={"backend": self.backend_type, "implementation": "new"}
        ):
            # Return cached engine if available
            if database_path in self._engines:
                return self._engines[database_path]
            
            # Create new engine
            conn_string = self._create_connection_string(database_path)
            
            # Common engine arguments
            engine_args = {
                "poolclass": NullPool,  # Disable connection pooling by default
                "echo": self.config.get("echo", False),
            }
            
            # Add backend-specific arguments
            engine_args.update(self._get_engine_args())
            
            try:
                engine = create_engine(conn_string, **engine_args)
                
                # Test connection
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                # Cache engine
                self._engines[database_path] = engine
                
                # Perform backend-specific initialization
                self._initialize_backend_specific(engine)
                
                logger.debug(f"Created engine for {database_path}")
                return engine
                
            except Exception as e:
                logger.error(f"Failed to create engine for {database_path}: {e}")
                raise StorageError(f"Failed to create engine: {e}")
    
    def _get_engine_args(self) -> Dict[str, Any]:
        """Get backend-specific engine arguments.
        
        Subclasses can override this to provide custom arguments.
        """
        return {}
    
    def create_table_from_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str,
        engine: Engine,
        if_exists: str = "fail"
    ) -> None:
        """Create table from pandas DataFrame."""
        with metrics_collector.timer(
            "storage.create_table",
            tags={"backend": self.backend_type, "implementation": "new"}
        ):
            try:
                # Use pandas to_sql with proper error handling
                df.to_sql(
                    table_name, 
                    engine, 
                    if_exists=if_exists,
                    index=False,
                    method="multi"  # Use multi-row inserts for better performance
                )
                
                # Clear metadata cache for this database
                db_key = str(engine.url)
                if db_key in self._metadata_cache:
                    del self._metadata_cache[db_key]
                    
                logger.debug(f"Created table {table_name} from DataFrame")
                
            except Exception as e:
                logger.error(f"Failed to create table {table_name}: {e}")
                raise StorageError(f"Failed to create table: {e}")
    
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame.
        
        Note: This assumes a default engine is available (legacy compatibility).
        New code should use engine-specific methods.
        """
        # For backward compatibility, use the first available engine
        if not self._engines:
            raise StorageError("No engine available for query execution")
        
        engine = next(iter(self._engines.values()))
        
        with metrics_collector.timer(
            "storage.query",
            tags={"backend": self.backend_type, "implementation": "new"}
        ):
            try:
                return pd.read_sql_query(query, engine, params=params)
            except Exception as e:
                logger.error(f"Query failed: {e}")
                raise StorageError(f"Query failed: {e}")
    
    def read_table_to_dataframe(
        self, 
        table_name: str, 
        engine: Engine,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Read entire table to DataFrame."""
        with metrics_collector.timer(
            "storage.read_table",
            tags={"backend": self.backend_type, "implementation": "new"}
        ):
            try:
                query = f"SELECT * FROM {table_name}"
                if limit:
                    query += f" LIMIT {limit}"
                
                return pd.read_sql_query(query, engine)
                
            except Exception as e:
                logger.error(f"Failed to read table {table_name}: {e}")
                raise StorageError(f"Failed to read table: {e}")
    
    def close_connections(self) -> None:
        """Close all database connections."""
        with metrics_collector.timer(
            "storage.close",
            tags={"backend": self.backend_type, "implementation": "new"}
        ):
            for path, engine in self._engines.items():
                try:
                    engine.dispose()
                    logger.debug(f"Closed engine for {path}")
                except Exception as e:
                    logger.warning(f"Error closing engine for {path}: {e}")
            
            self._engines.clear()
            self._metadata_cache.clear()
    
    def read_table(
        self, 
        table_name: str, 
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        engine: Optional[Engine] = None
    ) -> pd.DataFrame:
        """Read table with optional filtering."""
        with metrics_collector.timer(
            "storage.read_filtered",
            tags={"backend": self.backend_type, "implementation": "new"}
        ):
            # Use provided engine or get default
            if engine is None:
                if not self._engines:
                    raise StorageError("No engine available")
                engine = next(iter(self._engines.values()))
            
            try:
                # Build query
                if columns:
                    column_str = ", ".join(columns)
                else:
                    column_str = "*"
                
                query = f"SELECT {column_str} FROM {table_name}"
                
                if where:
                    query += f" WHERE {where}"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                return pd.read_sql_query(query, engine)
                
            except Exception as e:
                logger.error(f"Failed to read table {table_name}: {e}")
                raise StorageError(f"Failed to read table: {e}")
    
    def write_table(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        if_exists: str = "replace",
        engine: Optional[Engine] = None
    ) -> None:
        """Write DataFrame to table."""
        with metrics_collector.timer(
            "storage.write_table",
            tags={"backend": self.backend_type, "implementation": "new"}
        ):
            # Use provided engine or get default
            if engine is None:
                if not self._engines:
                    raise StorageError("No engine available")
                engine = next(iter(self._engines.values()))
            
            try:
                df.to_sql(
                    table_name,
                    engine,
                    if_exists=if_exists,
                    index=False,
                    method="multi"
                )
                
                # Clear metadata cache
                db_key = str(engine.url)
                if db_key in self._metadata_cache:
                    del self._metadata_cache[db_key]
                    
            except Exception as e:
                logger.error(f"Failed to write table {table_name}: {e}")
                raise StorageError(f"Failed to write table: {e}")
    
    def get_table_info(
        self, 
        table_name: str, 
        engine: Engine
    ) -> Dict[str, Any]:
        """Get table metadata and statistics."""
        with metrics_collector.timer(
            "storage.table_info",
            tags={"backend": self.backend_type, "implementation": "new"}
        ):
            try:
                inspector = inspect(engine)
                
                # Get columns
                columns = inspector.get_columns(table_name)
                
                # Get row count
                result = engine.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = result.scalar()
                
                # Get indexes
                indexes = inspector.get_indexes(table_name)
                
                # Get primary keys
                pk = inspector.get_pk_constraint(table_name)
                
                return {
                    "columns": columns,
                    "row_count": row_count,
                    "indexes": indexes,
                    "primary_key": pk,
                    "table_name": table_name
                }
                
            except Exception as e:
                logger.error(f"Failed to get table info for {table_name}: {e}")
                raise StorageError(f"Failed to get table info: {e}")
    
    def execute_query(
        self, 
        query: str, 
        engine: Engine
    ) -> Any:
        """Execute query without returning DataFrame."""
        with metrics_collector.timer(
            "storage.execute",
            tags={"backend": self.backend_type, "implementation": "new"}
        ):
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    conn.commit()
                    return result
                    
            except Exception as e:
                logger.error(f"Failed to execute query: {e}")
                raise StorageError(f"Failed to execute query: {e}")
    
    def get_connection(self) -> Any:
        """Get raw database connection."""
        # Return connection from first available engine
        if not self._engines:
            raise StorageError("No engine available")
        
        engine = next(iter(self._engines.values()))
        return engine.raw_connection()
    
    def get_columns(
        self, 
        table_name: str,
        engine: Optional[Engine] = None
    ) -> List[str]:
        """Get column names for table."""
        # Use provided engine or get default
        if engine is None:
            if not self._engines:
                raise StorageError("No engine available")
            engine = next(iter(self._engines.values()))
        
        try:
            inspector = inspect(engine)
            columns = inspector.get_columns(table_name)
            return [col["name"] for col in columns]
            
        except Exception as e:
            logger.error(f"Failed to get columns for {table_name}: {e}")
            raise StorageError(f"Failed to get columns: {e}")
    
    def analyze_column(
        self, 
        table_name: str, 
        column_name: str,
        engine: Optional[Engine] = None
    ) -> Dict[str, Any]:
        """Get column statistics."""
        # Use provided engine or get default
        if engine is None:
            if not self._engines:
                raise StorageError("No engine available")
            engine = next(iter(self._engines.values()))
        
        try:
            # Get basic stats using SQL
            stats_query = f"""
            SELECT 
                COUNT(*) as count,
                COUNT(DISTINCT {column_name}) as unique_count,
                MIN({column_name}) as min_value,
                MAX({column_name}) as max_value,
                COUNT(*) - COUNT({column_name}) as null_count
            FROM {table_name}
            """
            
            result = engine.execute(text(stats_query))
            row = result.fetchone()
            
            return {
                "count": row[0],
                "unique_count": row[1],
                "min_value": row[2],
                "max_value": row[3],
                "null_count": row[4],
                "null_percentage": (row[4] / row[0] * 100) if row[0] > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze column {column_name}: {e}")
            raise StorageError(f"Failed to analyze column: {e}")
    
    @abstractmethod
    def database_exists(self, database_path: str) -> bool:
        """Check if database exists."""
        pass
    
    @abstractmethod
    def create_database(self, database_path: str) -> None:
        """Create empty database."""
        pass
    
    # Dataset-oriented methods
    def create_dataset(self, dataset_name: str, config: Dict[str, Any]) -> None:
        """Create a new dataset with given configuration."""
        base_path = Path(config.get("base_path", Path.home() / ".mdm" / "datasets"))
        database_path = self._get_dataset_path(dataset_name, base_path)
        
        # Create database if needed
        if not self.database_exists(database_path):
            self.create_database(database_path)
        
        # Get engine and initialize
        engine = self.get_engine(database_path)
        
        # Create metadata tables
        self._create_metadata_tables(engine)
        
        # Store initial metadata
        metadata = {
            "dataset_name": dataset_name,
            "backend_type": self.backend_type,
            "created_at": pd.Timestamp.now().isoformat(),
            **config.get("metadata", {})
        }
        self.update_metadata(dataset_name, metadata)
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset exists."""
        base_path = Path.home() / ".mdm" / "datasets"
        database_path = self._get_dataset_path(dataset_name, base_path)
        return self.database_exists(database_path)
    
    def drop_dataset(self, dataset_name: str) -> None:
        """Remove dataset and all associated data."""
        base_path = Path.home() / ".mdm" / "datasets"
        database_path = self._get_dataset_path(dataset_name, base_path)
        
        # Close connections
        if database_path in self._engines:
            self._engines[database_path].dispose()
            del self._engines[database_path]
        
        # Drop database
        self.drop_database(database_path)
    
    def load_data(self, dataset_name: str, table_name: str = "data") -> pd.DataFrame:
        """Load data from dataset table."""
        engine = self._get_dataset_engine(dataset_name)
        return self.read_table_to_dataframe(table_name, engine)
    
    def save_data(
        self, 
        dataset_name: str, 
        data: pd.DataFrame,
        table_name: str = "data", 
        if_exists: str = "replace"
    ) -> None:
        """Save DataFrame to dataset table."""
        engine = self._get_dataset_engine(dataset_name)
        self.create_table_from_dataframe(data, table_name, engine, if_exists)
    
    def get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset metadata."""
        engine = self._get_dataset_engine(dataset_name)
        
        try:
            df = self.read_table_to_dataframe("_metadata", engine)
            metadata = {}
            for _, row in df.iterrows():
                metadata[row["key"]] = row["value"]
            return metadata
        except:
            return {}
    
    def update_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        """Update dataset metadata."""
        engine = self._get_dataset_engine(dataset_name)
        
        # Convert to DataFrame
        rows = [{"key": k, "value": str(v)} for k, v in metadata.items()]
        df = pd.DataFrame(rows)
        
        self.create_table_from_dataframe(df, "_metadata", engine, "replace")
    
    def close(self) -> None:
        """Close any open resources."""
        self.close_connections()
    
    # Helper methods
    def _get_dataset_path(self, dataset_name: str, base_path: Path) -> str:
        """Get path for dataset database."""
        dataset_dir = base_path / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return str(dataset_dir / f"{dataset_name}.{self._get_file_extension()}")
    
    def _get_dataset_engine(self, dataset_name: str) -> Engine:
        """Get or create engine for dataset."""
        base_path = Path.home() / ".mdm" / "datasets"
        database_path = self._get_dataset_path(dataset_name, base_path)
        return self.get_engine(database_path)
    
    def _create_metadata_tables(self, engine: Engine) -> None:
        """Create standard metadata tables."""
        # Create metadata table
        metadata_df = pd.DataFrame(columns=["key", "value"])
        self.create_table_from_dataframe(metadata_df, "_metadata", engine, "replace")
        
        # Create stats table
        stats_df = pd.DataFrame(columns=["table_name", "column_name", "stat_name", "stat_value"])
        self.create_table_from_dataframe(stats_df, "_stats", engine, "replace")
    
    @abstractmethod
    def _get_file_extension(self) -> str:
        """Get file extension for database files."""
        pass
    
    @abstractmethod
    def drop_database(self, database_path: str) -> None:
        """Drop an existing database."""
        pass
    
    def get_table_names(self, engine: Engine) -> List[str]:
        """Get list of table names in database."""
        inspector = inspect(engine)
        return inspector.get_table_names()