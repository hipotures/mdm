"""
Storage backend adapters for existing implementations.

These adapters wrap the legacy backends to provide the IStorageBackend interface
while maintaining full backward compatibility.
"""
from typing import Any, Dict, Optional, List
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
import logging
from pathlib import Path

from ..interfaces.storage import IStorageBackend
from ..storage.sqlite import SQLiteBackend
from ..storage.duckdb import DuckDBBackend
from ..storage.postgresql import PostgreSQLBackend
from ..config import get_config

logger = logging.getLogger(__name__)


class StorageAdapter:
    """Base adapter with common functionality and metrics tracking."""
    
    def __init__(self, backend: Any):
        self._backend = backend
        self._call_count = {}  # For metrics
        self._config = get_config()
    
    def _track_call(self, method: str) -> None:
        """Track method calls for metrics."""
        self._call_count[method] = self._call_count.get(method, 0) + 1
        logger.debug(f"Method called: {method} (count: {self._call_count[method]})")
    
    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return self._backend.backend_type
    
    def get_metrics(self) -> Dict[str, int]:
        """Get call metrics."""
        return self._call_count.copy()


class SQLiteAdapter(StorageAdapter, IStorageBackend):
    """Adapter for SQLite backend implementing full IStorageBackend interface."""
    
    def __init__(self):
        backend = SQLiteBackend(get_config().database.sqlite)
        super().__init__(backend)
        logger.info("Initialized SQLiteAdapter")
    
    # ==========================================
    # High Usage Methods (from API analysis)
    # ==========================================
    
    def get_engine(self, database_path: str) -> Engine:
        """Get SQLAlchemy engine for database."""
        self._track_call("get_engine")
        return self._backend.get_engine(database_path)
    
    def create_table_from_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str,
        engine: Engine,
        if_exists: str = "fail"
    ) -> None:
        """Create table from pandas DataFrame."""
        self._track_call("create_table_from_dataframe")
        return self._backend.create_table_from_dataframe(df, table_name, engine, if_exists)
    
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        self._track_call("query")
        return self._backend.query(query, params)
    
    def read_table_to_dataframe(
        self, 
        table_name: str, 
        engine: Engine,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Read entire table to DataFrame."""
        self._track_call("read_table_to_dataframe")
        return self._backend.read_table_to_dataframe(table_name, engine, limit)
    
    def close_connections(self) -> None:
        """Close all database connections."""
        self._track_call("close_connections")
        return self._backend.close_connections()
    
    def read_table(
        self, 
        table_name: str, 
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        engine: Optional[Engine] = None
    ) -> pd.DataFrame:
        """Read table with optional filtering."""
        self._track_call("read_table")
        return self._backend.read_table(table_name, columns, where, limit, engine)
    
    # ==========================================
    # Medium Usage Methods
    # ==========================================
    
    def write_table(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        if_exists: str = "replace",
        engine: Optional[Engine] = None
    ) -> None:
        """Write DataFrame to table."""
        self._track_call("write_table")
        return self._backend.write_table(table_name, df, if_exists, engine)
    
    def get_table_info(
        self, 
        table_name: str, 
        engine: Engine
    ) -> Dict[str, Any]:
        """Get table metadata and statistics."""
        self._track_call("get_table_info")
        return self._backend.get_table_info(table_name, engine)
    
    # ==========================================
    # Low Usage Methods
    # ==========================================
    
    def execute_query(
        self, 
        query: str, 
        engine: Engine
    ) -> Any:
        """Execute query without returning DataFrame."""
        self._track_call("execute_query")
        return self._backend.execute_query(query, engine)
    
    def get_connection(self) -> Any:
        """Get raw database connection."""
        self._track_call("get_connection")
        return self._backend.get_connection()
    
    def get_columns(
        self, 
        table_name: str,
        engine: Optional[Engine] = None
    ) -> List[str]:
        """Get column names for table."""
        self._track_call("get_columns")
        return self._backend.get_columns(table_name, engine)
    
    def analyze_column(
        self, 
        table_name: str, 
        column_name: str,
        engine: Optional[Engine] = None
    ) -> Dict[str, Any]:
        """Get column statistics."""
        self._track_call("analyze_column")
        return self._backend.analyze_column(table_name, column_name, engine)
    
    def database_exists(self, database_path: str) -> bool:
        """Check if database exists."""
        self._track_call("database_exists")
        return self._backend.database_exists(database_path)
    
    def create_database(self, database_path: str) -> None:
        """Create empty database."""
        self._track_call("create_database")
        return self._backend.create_database(database_path)
    
    # ==========================================
    # New architecture methods
    # ==========================================
    
    def create_dataset(self, dataset_name: str, config: Dict[str, Any]) -> None:
        """Create a new dataset with given configuration."""
        self._track_call("create_dataset")
        return self._backend.create_dataset(dataset_name, config)
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset exists."""
        self._track_call("dataset_exists")
        return self._backend.dataset_exists(dataset_name)
    
    def drop_dataset(self, dataset_name: str) -> None:
        """Remove dataset and all associated data."""
        self._track_call("drop_dataset")
        return self._backend.drop_dataset(dataset_name)
    
    def load_data(self, dataset_name: str, table_name: str = "data") -> pd.DataFrame:
        """Load data from dataset table."""
        self._track_call("load_data")
        return self._backend.load_data(dataset_name, table_name)
    
    def save_data(
        self, 
        dataset_name: str, 
        data: pd.DataFrame,
        table_name: str = "data", 
        if_exists: str = "replace"
    ) -> None:
        """Save DataFrame to dataset table."""
        self._track_call("save_data")
        return self._backend.save_data(dataset_name, data, table_name, if_exists)
    
    def get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset metadata."""
        self._track_call("get_metadata")
        return self._backend.get_metadata(dataset_name)
    
    def update_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        """Update dataset metadata."""
        self._track_call("update_metadata")
        return self._backend.update_metadata(dataset_name, metadata)
    
    def close(self) -> None:
        """Close any open resources."""
        self._track_call("close")
        # SQLite backend doesn't have explicit close in legacy version
        if hasattr(self._backend, '_engine') and self._backend._engine:
            self._backend._engine.dispose()


class DuckDBAdapter(StorageAdapter, IStorageBackend):
    """Adapter for DuckDB backend implementing full IStorageBackend interface."""
    
    def __init__(self):
        backend = DuckDBBackend(get_config().database.duckdb)
        super().__init__(backend)
        logger.info("Initialized DuckDBAdapter")
    
    # Implementation identical to SQLiteAdapter but wrapping DuckDB backend
    # For brevity, showing key differences only
    
    def get_engine(self, database_path: str) -> Engine:
        """Get SQLAlchemy engine for database."""
        self._track_call("get_engine")
        return self._backend.get_engine(database_path)
    
    def create_table_from_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str,
        engine: Engine,
        if_exists: str = "fail"
    ) -> None:
        """Create table from pandas DataFrame."""
        self._track_call("create_table_from_dataframe")
        return self._backend.create_table_from_dataframe(df, table_name, engine, if_exists)
    
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        self._track_call("query")
        return self._backend.query(query, params)
    
    def read_table_to_dataframe(
        self, 
        table_name: str, 
        engine: Engine,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Read entire table to DataFrame."""
        self._track_call("read_table_to_dataframe")
        return self._backend.read_table_to_dataframe(table_name, engine, limit)
    
    def close_connections(self) -> None:
        """Close all database connections."""
        self._track_call("close_connections")
        return self._backend.close_connections()
    
    def read_table(
        self, 
        table_name: str, 
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        engine: Optional[Engine] = None
    ) -> pd.DataFrame:
        """Read table with optional filtering."""
        self._track_call("read_table")
        return self._backend.read_table(table_name, columns, where, limit, engine)
    
    def write_table(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        if_exists: str = "replace",
        engine: Optional[Engine] = None
    ) -> None:
        """Write DataFrame to table."""
        self._track_call("write_table")
        return self._backend.write_table(table_name, df, if_exists, engine)
    
    def get_table_info(
        self, 
        table_name: str, 
        engine: Engine
    ) -> Dict[str, Any]:
        """Get table metadata and statistics."""
        self._track_call("get_table_info")
        return self._backend.get_table_info(table_name, engine)
    
    def execute_query(
        self, 
        query: str, 
        engine: Engine
    ) -> Any:
        """Execute query without returning DataFrame."""
        self._track_call("execute_query")
        return self._backend.execute_query(query, engine)
    
    def get_connection(self) -> Any:
        """Get raw database connection."""
        self._track_call("get_connection")
        return self._backend.get_connection()
    
    def get_columns(
        self, 
        table_name: str,
        engine: Optional[Engine] = None
    ) -> List[str]:
        """Get column names for table."""
        self._track_call("get_columns")
        return self._backend.get_columns(table_name, engine)
    
    def analyze_column(
        self, 
        table_name: str, 
        column_name: str,
        engine: Optional[Engine] = None
    ) -> Dict[str, Any]:
        """Get column statistics."""
        self._track_call("analyze_column")
        return self._backend.analyze_column(table_name, column_name, engine)
    
    def database_exists(self, database_path: str) -> bool:
        """Check if database exists."""
        self._track_call("database_exists")
        return self._backend.database_exists(database_path)
    
    def create_database(self, database_path: str) -> None:
        """Create empty database."""
        self._track_call("create_database")
        return self._backend.create_database(database_path)
    
    def create_dataset(self, dataset_name: str, config: Dict[str, Any]) -> None:
        """Create a new dataset with given configuration."""
        self._track_call("create_dataset")
        return self._backend.create_dataset(dataset_name, config)
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset exists."""
        self._track_call("dataset_exists")
        return self._backend.dataset_exists(dataset_name)
    
    def drop_dataset(self, dataset_name: str) -> None:
        """Remove dataset and all associated data."""
        self._track_call("drop_dataset")
        return self._backend.drop_dataset(dataset_name)
    
    def load_data(self, dataset_name: str, table_name: str = "data") -> pd.DataFrame:
        """Load data from dataset table."""
        self._track_call("load_data")
        return self._backend.load_data(dataset_name, table_name)
    
    def save_data(
        self, 
        dataset_name: str, 
        data: pd.DataFrame,
        table_name: str = "data", 
        if_exists: str = "replace"
    ) -> None:
        """Save DataFrame to dataset table."""
        self._track_call("save_data")
        return self._backend.save_data(dataset_name, data, table_name, if_exists)
    
    def get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset metadata."""
        self._track_call("get_metadata")
        return self._backend.get_metadata(dataset_name)
    
    def update_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        """Update dataset metadata."""
        self._track_call("update_metadata")
        return self._backend.update_metadata(dataset_name, metadata)
    
    def close(self) -> None:
        """Close any open resources."""
        self._track_call("close")
        if hasattr(self._backend, '_conn') and self._backend._conn:
            self._backend._conn.close()


class PostgreSQLAdapter(StorageAdapter, IStorageBackend):
    """Adapter for PostgreSQL backend implementing full IStorageBackend interface."""
    
    def __init__(self):
        backend = PostgreSQLBackend(get_config().database.postgresql)
        super().__init__(backend)
        logger.info("Initialized PostgreSQLAdapter")
    
    # Implementation identical to SQLiteAdapter but wrapping PostgreSQL backend
    # PostgreSQL has some unique features but adapter interface remains same
    
    def get_engine(self, database_path: str) -> Engine:
        """Get SQLAlchemy engine for database."""
        self._track_call("get_engine")
        return self._backend.get_engine(database_path)
    
    def create_table_from_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str,
        engine: Engine,
        if_exists: str = "fail"
    ) -> None:
        """Create table from pandas DataFrame."""
        self._track_call("create_table_from_dataframe")
        return self._backend.create_table_from_dataframe(df, table_name, engine, if_exists)
    
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        self._track_call("query")
        return self._backend.query(query, params)
    
    def read_table_to_dataframe(
        self, 
        table_name: str, 
        engine: Engine,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Read entire table to DataFrame."""
        self._track_call("read_table_to_dataframe")
        return self._backend.read_table_to_dataframe(table_name, engine, limit)
    
    def close_connections(self) -> None:
        """Close all database connections."""
        self._track_call("close_connections")
        return self._backend.close_connections()
    
    def read_table(
        self, 
        table_name: str, 
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        engine: Optional[Engine] = None
    ) -> pd.DataFrame:
        """Read table with optional filtering."""
        self._track_call("read_table")
        return self._backend.read_table(table_name, columns, where, limit, engine)
    
    def write_table(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        if_exists: str = "replace",
        engine: Optional[Engine] = None
    ) -> None:
        """Write DataFrame to table."""
        self._track_call("write_table")
        return self._backend.write_table(table_name, df, if_exists, engine)
    
    def get_table_info(
        self, 
        table_name: str, 
        engine: Engine
    ) -> Dict[str, Any]:
        """Get table metadata and statistics."""
        self._track_call("get_table_info")
        return self._backend.get_table_info(table_name, engine)
    
    def execute_query(
        self, 
        query: str, 
        engine: Engine
    ) -> Any:
        """Execute query without returning DataFrame."""
        self._track_call("execute_query")
        return self._backend.execute_query(query, engine)
    
    def get_connection(self) -> Any:
        """Get raw database connection."""
        self._track_call("get_connection")
        return self._backend.get_connection()
    
    def get_columns(
        self, 
        table_name: str,
        engine: Optional[Engine] = None
    ) -> List[str]:
        """Get column names for table."""
        self._track_call("get_columns")
        return self._backend.get_columns(table_name, engine)
    
    def analyze_column(
        self, 
        table_name: str, 
        column_name: str,
        engine: Optional[Engine] = None
    ) -> Dict[str, Any]:
        """Get column statistics."""
        self._track_call("analyze_column")
        return self._backend.analyze_column(table_name, column_name, engine)
    
    def database_exists(self, database_path: str) -> bool:
        """Check if database exists."""
        self._track_call("database_exists")
        return self._backend.database_exists(database_path)
    
    def create_database(self, database_path: str) -> None:
        """Create empty database."""
        self._track_call("create_database")
        return self._backend.create_database(database_path)
    
    def create_dataset(self, dataset_name: str, config: Dict[str, Any]) -> None:
        """Create a new dataset with given configuration."""
        self._track_call("create_dataset")
        return self._backend.create_dataset(dataset_name, config)
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset exists."""
        self._track_call("dataset_exists")
        return self._backend.dataset_exists(dataset_name)
    
    def drop_dataset(self, dataset_name: str) -> None:
        """Remove dataset and all associated data."""
        self._track_call("drop_dataset")
        return self._backend.drop_dataset(dataset_name)
    
    def load_data(self, dataset_name: str, table_name: str = "data") -> pd.DataFrame:
        """Load data from dataset table."""
        self._track_call("load_data")
        return self._backend.load_data(dataset_name, table_name)
    
    def save_data(
        self, 
        dataset_name: str, 
        data: pd.DataFrame,
        table_name: str = "data", 
        if_exists: str = "replace"
    ) -> None:
        """Save DataFrame to dataset table."""
        self._track_call("save_data")
        return self._backend.save_data(dataset_name, data, table_name, if_exists)
    
    def get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset metadata."""
        self._track_call("get_metadata")
        return self._backend.get_metadata(dataset_name)
    
    def update_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        """Update dataset metadata."""
        self._track_call("update_metadata")
        return self._backend.update_metadata(dataset_name, metadata)
    
    def close(self) -> None:
        """Close any open resources."""
        self._track_call("close")
        if hasattr(self._backend, '_engine') and self._backend._engine:
            self._backend._engine.dispose()