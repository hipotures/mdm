"""
Compatibility mixin to add missing methods to stateless backends.

This is a TEMPORARY solution until the codebase is updated to use new patterns.
All methods log deprecation warnings to encourage migration.
"""
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
from sqlalchemy import text, inspect, Engine
import logging
from contextlib import contextmanager

from ...core.exceptions import StorageError

logger = logging.getLogger(__name__)


class BackendCompatibilityMixin:
    """
    Mixin that adds backward compatibility methods to stateless backends.
    
    This mixin provides methods that exist in the old backends but are
    missing from the new stateless implementations. Each method logs a
    deprecation warning to encourage migration to new patterns.
    
    IMPORTANT: This is a temporary solution. These methods should be
    removed once all code is migrated to use the new API.
    """
    
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame (compatibility method).
        
        Args:
            query: SQL query string
            params: Optional parameters for parameterized queries
            
        Returns:
            Query results as pandas DataFrame
            
        Raises:
            StorageError: If query execution fails
            
        Note:
            This method is DEPRECATED. New code should use:
            - with backend.get_engine_context(dataset) as engine:
                  df = pd.read_sql_query(query, engine)
        """
        logger.debug(f"Using compatibility method 'query' with query: {query[:50]}...")
        
        if hasattr(self, '_engine') and self._engine is not None:
            # Singleton pattern compatibility - use cached engine
            try:
                return pd.read_sql_query(query, self._engine, params=params)
            except Exception as e:
                raise StorageError(f"Failed to execute query: {e}") from e
        else:
            raise StorageError(
                "No engine available. Call get_engine() first or use "
                "get_engine_context() for proper connection management."
            )
    
    def create_table_from_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str,
        engine: Engine,
        if_exists: str = "fail"
    ) -> None:
        """
        Create table from pandas DataFrame (compatibility method).
        
        Args:
            df: Pandas DataFrame to save
            table_name: Name of the table to create
            engine: SQLAlchemy engine instance
            if_exists: Behavior if table exists ('fail', 'replace', 'append')
            
        Raises:
            StorageError: If table creation fails
            
        Note:
            This method is DEPRECATED. New code should use:
            - backend.save_data(dataset_name, df, table_name, if_exists)
        """
        logger.debug(
            f"Using compatibility method 'create_table_from_dataframe' "
            f"for table '{table_name}' with {len(df)} rows"
        )
        
        try:
            # Use pandas to_sql with the provided engine
            df.to_sql(
                table_name, 
                engine, 
                if_exists=if_exists, 
                index=False,
                method='multi',  # Use multi-row insert for better performance
                chunksize=10000  # Match the configured batch size
            )
        except Exception as e:
            raise StorageError(
                f"Failed to create table '{table_name}' from DataFrame: {e}"
            ) from e
    
    def close_connections(self) -> None:
        """
        Close all database connections (compatibility method).
        
        Note:
            This method is DEPRECATED. New code should use:
            - backend.close()
            - Or better: use context managers for automatic cleanup
        """
        logger.debug("Using compatibility method 'close_connections'")
        
        # Close connection pool if exists
        if hasattr(self, 'pool') and self.pool is not None:
            try:
                self.pool.close_all()
            except Exception as e:
                logger.warning(f"Error closing connection pool: {e}")
        
        # Clear singleton engine for compatibility
        if hasattr(self, '_engine'):
            if self._engine is not None:
                try:
                    self._engine.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing engine: {e}")
            self._engine = None
            
        # Clear session factory if exists
        if hasattr(self, '_session_factory'):
            self._session_factory = None
        
        # Call new close method if exists
        if hasattr(self, 'close') and callable(self.close):
            try:
                self.close()
            except Exception as e:
                logger.warning(f"Error calling close(): {e}")
    
    def read_table_to_dataframe(
        self, 
        table_name: str, 
        engine: Engine,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read entire table to DataFrame (compatibility method).
        
        Args:
            table_name: Name of the table to read
            engine: SQLAlchemy engine instance
            limit: Optional row limit for large tables
            
        Returns:
            Table contents as pandas DataFrame
            
        Raises:
            StorageError: If table doesn't exist or read fails
            
        Note:
            This method is DEPRECATED. New code should use:
            - backend.load_data(dataset_name, table_name)
        """
        logger.debug(
            f"Using compatibility method 'read_table_to_dataframe' "
            f"for table '{table_name}'"
        )
        
        try:
            query = f"SELECT * FROM {table_name}"
            if limit:
                # Handle different SQL dialects
                if 'postgresql' in str(engine.url):
                    query += f" LIMIT {limit}"
                elif 'sqlite' in str(engine.url) or 'duckdb' in str(engine.url):
                    query += f" LIMIT {limit}"
                else:
                    # Generic SQL
                    query += f" LIMIT {limit}"
                    
            return pd.read_sql_query(query, engine)
        except Exception as e:
            raise StorageError(f"Failed to read table '{table_name}': {e}") from e
    
    def read_table(
        self, 
        table_name: str, 
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        engine: Optional[Engine] = None
    ) -> pd.DataFrame:
        """
        Read table with optional filtering (compatibility method).
        
        Args:
            table_name: Name of the table
            columns: Optional list of columns to select
            where: Optional WHERE clause (without 'WHERE' keyword)
            limit: Optional row limit
            engine: Optional engine (uses cached if not provided)
            
        Returns:
            Filtered table data as DataFrame
            
        Note:
            This method is DEPRECATED. New code should build queries
            explicitly and use pd.read_sql_query directly.
        """
        logger.debug(f"Using compatibility method 'read_table' for table '{table_name}'")
        
        # Build query
        if columns:
            cols = ", ".join(columns)
        else:
            cols = "*"
            
        query = f"SELECT {cols} FROM {table_name}"
        
        if where:
            query += f" WHERE {where}"
            
        if limit:
            query += f" LIMIT {limit}"
        
        # Use provided engine or cached one
        if engine is None:
            if hasattr(self, '_engine') and self._engine is not None:
                engine = self._engine
            else:
                raise StorageError("No engine provided and no cached engine available")
                
        try:
            return pd.read_sql_query(query, engine)
        except Exception as e:
            raise StorageError(f"Failed to read table '{table_name}': {e}") from e
    
    def write_table(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        if_exists: str = "replace",
        engine: Optional[Engine] = None
    ) -> None:
        """
        Write DataFrame to table (compatibility method).
        
        Args:
            table_name: Target table name
            df: DataFrame to write
            if_exists: Behavior if table exists
            engine: Optional engine (uses cached if not provided)
            
        Note:
            This method is DEPRECATED. It's redundant with
            create_table_from_dataframe.
        """
        logger.debug(f"Using compatibility method 'write_table' for table '{table_name}'")
        
        # Use provided engine or cached one
        if engine is None:
            if hasattr(self, '_engine') and self._engine is not None:
                engine = self._engine
            else:
                raise StorageError("No engine provided and no cached engine available")
                
        self.create_table_from_dataframe(df, table_name, engine, if_exists)
    
    def get_table_info(
        self, 
        table_name: str, 
        engine: Engine
    ) -> Dict[str, Any]:
        """
        Get table schema information (compatibility method).
        
        Args:
            table_name: Name of the table
            engine: SQLAlchemy engine
            
        Returns:
            Dictionary containing:
            - name: Table name
            - columns: List of column info dicts
            - row_count: Number of rows
            - column_count: Number of columns
            
        Note:
            This method is DEPRECATED. Use SQLAlchemy inspector directly.
        """
        logger.debug(f"Using compatibility method 'get_table_info' for table '{table_name}'")
        
        try:
            # Get schema information
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
        except Exception as e:
            raise StorageError(f"Failed to get info for table '{table_name}': {e}") from e
    
    def execute_query(
        self, 
        query: str, 
        engine: Engine
    ) -> Any:
        """
        Execute query without returning DataFrame (compatibility method).
        
        Args:
            query: SQL query string
            engine: SQLAlchemy engine
            
        Returns:
            Query result (backend-specific)
            
        Note:
            This method is DEPRECATED. Use engine.execute() directly.
        """
        logger.debug(f"Using compatibility method 'execute_query'")
        
        try:
            with engine.begin() as conn:
                return conn.execute(text(query))
        except Exception as e:
            raise StorageError(f"Failed to execute query: {e}") from e
    
    def get_connection(self) -> Any:
        """
        Get raw database connection (compatibility method).
        
        Returns:
            Backend-specific connection object
            
        Note:
            This method is DEPRECATED. Use get_engine_context() instead.
        """
        logger.debug("Using compatibility method 'get_connection'")
        
        if hasattr(self, '_engine') and self._engine is not None:
            return self._engine.raw_connection()
        else:
            raise StorageError("No engine available. Call get_engine() first.")
    
    def get_columns(
        self, 
        table_name: str,
        engine: Optional[Engine] = None
    ) -> List[str]:
        """
        Get column names for table (compatibility method).
        
        Args:
            table_name: Name of the table
            engine: Optional engine (uses cached if not provided)
            
        Returns:
            List of column names
            
        Note:
            This method is DEPRECATED. Use get_table_info() or
            SQLAlchemy inspector directly.
        """
        logger.debug(f"Using compatibility method 'get_columns' for table '{table_name}'")
        
        # Use provided engine or cached one
        if engine is None:
            if hasattr(self, '_engine') and self._engine is not None:
                engine = self._engine
            else:
                raise StorageError("No engine provided and no cached engine available")
                
        try:
            info = self.get_table_info(table_name, engine)
            return [col['name'] for col in info['columns']]
        except Exception as e:
            raise StorageError(f"Failed to get columns for table '{table_name}': {e}") from e
    
    def analyze_column(
        self, 
        table_name: str, 
        column_name: str,
        engine: Optional[Engine] = None
    ) -> Dict[str, Any]:
        """
        Analyze column statistics (compatibility method).
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            engine: Optional engine (uses cached if not provided)
            
        Returns:
            Dictionary with statistics:
            - count: Non-null count
            - unique: Unique value count  
            - min: Minimum value
            - max: Maximum value
            - mean: Average (for numeric)
            - std: Std deviation (for numeric)
            
        Note:
            This method is DEPRECATED. Use dedicated statistics functions.
        """
        logger.debug(
            f"Using compatibility method 'analyze_column' "
            f"for {table_name}.{column_name}"
        )
        
        # Use provided engine or cached one
        if engine is None:
            if hasattr(self, '_engine') and self._engine is not None:
                engine = self._engine
            else:
                raise StorageError("No engine provided and no cached engine available")
        
        try:
            # Basic statistics query
            query = f"""
            SELECT 
                COUNT({column_name}) as count,
                COUNT(DISTINCT {column_name}) as unique_count,
                MIN({column_name}) as min_value,
                MAX({column_name}) as max_value
            FROM {table_name}
            """
            
            stats_df = pd.read_sql_query(query, engine)
            stats = stats_df.iloc[0].to_dict()
            
            # Try to get numeric statistics
            try:
                numeric_query = f"""
                SELECT 
                    AVG(CAST({column_name} AS FLOAT)) as mean,
                    STDDEV(CAST({column_name} AS FLOAT)) as std
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                """
                numeric_df = pd.read_sql_query(numeric_query, engine)
                stats.update(numeric_df.iloc[0].to_dict())
            except:
                # Not a numeric column or dialect doesn't support STDDEV
                stats['mean'] = None
                stats['std'] = None
                
            return stats
            
        except Exception as e:
            raise StorageError(
                f"Failed to analyze column '{column_name}' in table '{table_name}': {e}"
            ) from e
            
    # Helper method for debugging
    def _log_compatibility_usage(self):
        """Log current usage of compatibility methods for migration tracking."""
        methods = [
            'query', 'create_table_from_dataframe', 'close_connections',
            'read_table_to_dataframe', 'read_table', 'write_table',
            'get_table_info', 'execute_query', 'get_connection',
            'get_columns', 'analyze_column'
        ]
        
        logger.info("Compatibility method usage tracking:")
        for method in methods:
            if hasattr(self, f'_{method}_count'):
                count = getattr(self, f'_{method}_count', 0)
                if count > 0:
                    logger.info(f"  {method}: {count} calls")