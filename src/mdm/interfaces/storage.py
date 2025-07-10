"""
Storage backend interface definition.

This interface is based on ACTUAL API usage analysis, not idealistic design.
It includes ALL methods found to be in use across the codebase.
"""
from typing import Protocol, Any, Dict, List, Optional, runtime_checkable, Generator
from pathlib import Path
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
import pandas as pd


@runtime_checkable
class IStorageBackend(Protocol):
    """
    Storage backend interface based on actual usage analysis.
    
    This interface includes ALL 14 methods identified in the API analysis:
    - 11 methods heavily used but missing from initial design
    - 3 methods that were included in initial design
    
    Total method calls found: 62 across the codebase
    """
    
    @property
    def backend_type(self) -> str:
        """Get backend type identifier ('sqlite', 'duckdb', 'postgresql')."""
        ...
    
    # ==========================================
    # High Usage Methods (>5 calls)
    # ==========================================
    
    def get_engine(self, database_path: str) -> Engine:
        """
        Get SQLAlchemy engine for database.
        
        Usage: 11 calls - Most frequently used method
        Critical for: Database connections
        
        Args:
            database_path: Path to database file
            
        Returns:
            SQLAlchemy Engine instance
        """
        ...
    
    def create_table_from_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str,
        engine: Engine,
        if_exists: str = "fail"
    ) -> None:
        """
        Create table from pandas DataFrame.
        
        Usage: 10 calls - Essential for data loading
        Critical for: Dataset registration, chunked loading
        
        Args:
            df: DataFrame to save
            table_name: Name of the table to create
            engine: SQLAlchemy engine instance
            if_exists: What to do if table exists ('fail', 'replace', 'append')
        """
        ...
    
    def query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Usage: 9 calls - Primary query method
        Critical for: Data retrieval, statistics
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as pandas DataFrame
            
        Note:
            Assumes engine is already available (singleton pattern)
        """
        ...
    
    def read_table_to_dataframe(
        self, 
        table_name: str, 
        engine: Engine,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read entire table to DataFrame.
        
        Usage: 7 calls - Common data loading pattern
        Critical for: Loading datasets, exports
        
        Args:
            table_name: Name of the table to read
            engine: SQLAlchemy engine instance
            limit: Optional row limit
            
        Returns:
            Table contents as DataFrame
        """
        ...
    
    def close_connections(self) -> None:
        """
        Close all database connections.
        
        Usage: 7 calls - Resource cleanup
        Critical for: Preventing connection leaks
        
        Note:
            Must be idempotent (safe to call multiple times)
        """
        ...
    
    def read_table(
        self, 
        table_name: str, 
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        engine: Optional[Engine] = None
    ) -> pd.DataFrame:
        """
        Read table with optional filtering.
        
        Usage: 7 calls - Feature engineering dependency
        Critical for: Selective data loading
        
        Args:
            table_name: Name of the table
            columns: Optional columns to select
            where: Optional WHERE clause
            limit: Optional row limit
            engine: Optional engine (uses cached if not provided)
            
        Returns:
            Filtered table data
        """
        ...
    
    # ==========================================
    # Medium Usage Methods (2-5 calls)
    # ==========================================
    
    def write_table(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        if_exists: str = "replace",
        engine: Optional[Engine] = None
    ) -> None:
        """
        Write DataFrame to table.
        
        Usage: 3 calls - Feature output
        Critical for: Saving processed features
        
        Args:
            table_name: Target table name
            df: DataFrame to write
            if_exists: What to do if table exists
            engine: Optional engine
        """
        ...
    
    def get_table_info(
        self, 
        table_name: str, 
        engine: Engine
    ) -> Dict[str, Any]:
        """
        Get table metadata and statistics.
        
        Usage: 2 calls - Metadata operations
        Critical for: Schema inspection
        
        Args:
            table_name: Name of the table
            engine: SQLAlchemy engine
            
        Returns:
            Dictionary with table information
        """
        ...
    
    # ==========================================
    # Low Usage Methods (1 call each)
    # ==========================================
    
    def execute_query(
        self, 
        query: str, 
        engine: Engine
    ) -> Any:
        """
        Execute query without returning DataFrame.
        
        Usage: 1 call
        Critical for: DDL operations
        
        Args:
            query: SQL query string
            engine: SQLAlchemy engine
            
        Returns:
            Query result (backend-specific)
        """
        ...
    
    def get_connection(self) -> Any:
        """
        Get raw database connection.
        
        Usage: 1 call
        Critical for: Low-level operations
        
        Returns:
            Backend-specific connection object
        """
        ...
    
    def get_columns(
        self, 
        table_name: str,
        engine: Optional[Engine] = None
    ) -> List[str]:
        """
        Get column names for table.
        
        Usage: 1 call
        Critical for: Schema operations
        
        Args:
            table_name: Name of the table
            engine: Optional engine
            
        Returns:
            List of column names
        """
        ...
    
    def analyze_column(
        self, 
        table_name: str, 
        column_name: str,
        engine: Optional[Engine] = None
    ) -> Dict[str, Any]:
        """
        Get column statistics.
        
        Usage: 1 call
        Critical for: Data profiling
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            engine: Optional engine
            
        Returns:
            Dictionary with column statistics
        """
        ...
    
    def database_exists(self, database_path: str) -> bool:
        """
        Check if database exists.
        
        Usage: 1 call (but critical for registration)
        Critical for: Dataset registration flow
        
        Args:
            database_path: Path to database file
            
        Returns:
            True if database exists
        """
        ...
    
    def create_database(self, database_path: str) -> None:
        """
        Create empty database.
        
        Usage: 1 call (but critical for registration)
        Critical for: Dataset initialization
        
        Args:
            database_path: Path to database file
        """
        ...
    
    # ==========================================
    # Additional methods for new architecture
    # ==========================================
    
    def create_dataset(self, dataset_name: str, config: Dict[str, Any]) -> None:
        """Create a new dataset with given configuration."""
        ...
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset exists."""
        ...
    
    def drop_dataset(self, dataset_name: str) -> None:
        """Remove dataset and all associated data."""
        ...
    
    def load_data(self, dataset_name: str, table_name: str = "data") -> pd.DataFrame:
        """Load data from dataset table."""
        ...
    
    def save_data(
        self, 
        dataset_name: str, 
        data: pd.DataFrame,
        table_name: str = "data", 
        if_exists: str = "replace"
    ) -> None:
        """Save DataFrame to dataset table."""
        ...
    
    def get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset metadata."""
        ...
    
    def update_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        """Update dataset metadata."""
        ...
    
    def close(self) -> None:
        """Close any open resources (new pattern)."""
        ...