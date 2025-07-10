"""New DuckDB storage backend implementation.

This module provides the refactored DuckDB backend with improved design
and better separation of concerns.
"""
from typing import Any, Dict
from pathlib import Path
import logging
import os

from sqlalchemy import Engine, text
import duckdb

from .base import NewStorageBackend
from mdm.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class NewDuckDBBackend(NewStorageBackend):
    """New DuckDB storage backend implementation.
    
    Key improvements over legacy:
    - Better connection management
    - Proper configuration handling
    - Improved error handling
    - Better memory management
    """
    
    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return "duckdb"
    
    def _create_connection_string(self, database_path: str) -> str:
        """Create DuckDB connection string.
        
        Args:
            database_path: Path to DuckDB database file
            
        Returns:
            DuckDB connection string
        """
        # Ensure absolute path
        db_path = Path(database_path).absolute()
        
        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        return f"duckdb:///{db_path}"
    
    def _get_engine_args(self) -> Dict[str, Any]:
        """Get DuckDB-specific engine arguments."""
        # DuckDB configuration
        config = {
            "memory_limit": self.config.get("memory_limit", "8GB"),
            "threads": self.config.get("threads", 4),
            "temp_directory": self.config.get("temp_directory", "/tmp"),
            "access_mode": self.config.get("access_mode", "READ_WRITE"),
        }
        
        return {
            "connect_args": {
                "config": config,
                "read_only": config["access_mode"] == "READ_ONLY"
            }
        }
    
    def _initialize_backend_specific(self, engine: Engine) -> None:
        """Perform DuckDB-specific initialization.
        
        Sets DuckDB-specific settings and loads extensions if needed.
        """
        with engine.connect() as conn:
            # Set additional runtime settings
            settings = {
                "enable_progress_bar": False,
                "enable_profiling": self.config.get("enable_profiling", False),
                "explain_output": self.config.get("explain_output", False),
            }
            
            for setting, value in settings.items():
                try:
                    if isinstance(value, bool):
                        value_str = "true" if value else "false"
                    else:
                        value_str = str(value)
                    
                    conn.execute(text(f"SET {setting} = {value_str}"))
                    logger.debug(f"Set DuckDB setting {setting} = {value_str}")
                except Exception as e:
                    logger.warning(f"Failed to set DuckDB setting {setting}: {e}")
            
            # Load commonly used extensions
            extensions = self.config.get("extensions", ["parquet", "json"])
            for ext in extensions:
                try:
                    conn.execute(text(f"INSTALL {ext}"))
                    conn.execute(text(f"LOAD {ext}"))
                    logger.debug(f"Loaded DuckDB extension: {ext}")
                except Exception as e:
                    logger.warning(f"Failed to load extension {ext}: {e}")
    
    def database_exists(self, database_path: str) -> bool:
        """Check if database exists.
        
        Args:
            database_path: Path to DuckDB database file
            
        Returns:
            True if database file exists
        """
        return Path(database_path).exists()
    
    def create_database(self, database_path: str) -> None:
        """Create empty database.
        
        For DuckDB, this creates an empty database file.
        
        Args:
            database_path: Path to DuckDB database file
        """
        db_path = Path(database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create empty database by connecting and immediately closing
        try:
            conn = duckdb.connect(str(db_path))
            conn.close()
            logger.info(f"Created DuckDB database at {database_path}")
        except Exception as e:
            raise StorageError(f"Failed to create database: {e}")
    
    def drop_database(self, database_path: str) -> None:
        """Drop an existing database.
        
        For DuckDB, this removes the database file.
        
        Args:
            database_path: Path to DuckDB database file
        """
        db_path = Path(database_path)
        
        # Close any existing connections first
        if str(db_path) in self._engines:
            self._engines[str(db_path)].dispose()
            del self._engines[str(db_path)]
        
        # Remove database file
        if db_path.exists():
            try:
                os.remove(db_path)
                logger.info(f"Dropped DuckDB database at {database_path}")
            except Exception as e:
                raise StorageError(f"Failed to drop database: {e}")
        
        # Also remove WAL file if it exists
        wal_file = Path(str(db_path) + ".wal")
        if wal_file.exists():
            try:
                os.remove(wal_file)
            except Exception as e:
                logger.warning(f"Failed to remove WAL file: {e}")
    
    def _get_file_extension(self) -> str:
        """Get file extension for database files."""
        return "duckdb"
    
    def optimize_database(self, engine: Engine) -> None:
        """Run DuckDB-specific optimizations.
        
        Args:
            engine: SQLAlchemy engine
        """
        with engine.connect() as conn:
            # Run checkpoint to flush WAL
            conn.execute(text("CHECKPOINT"))
            
            # Force garbage collection
            conn.execute(text("PRAGMA force_checkpoint"))
            
            # Update table statistics
            tables = self.get_table_names(engine)
            for table in tables:
                if not table.startswith("_"):  # Skip metadata tables
                    try:
                        conn.execute(text(f"ANALYZE {table}"))
                    except Exception as e:
                        logger.warning(f"Failed to analyze table {table}: {e}")
            
            logger.info("Optimized DuckDB database")
    
    def get_database_size(self, database_path: str) -> int:
        """Get database file size in bytes.
        
        Args:
            database_path: Path to DuckDB database file
            
        Returns:
            Size in bytes
        """
        db_path = Path(database_path)
        if not db_path.exists():
            return 0
        
        # Get main database size
        size = db_path.stat().st_size
        
        # Add WAL file size if it exists
        wal_path = Path(str(db_path) + ".wal")
        if wal_path.exists():
            size += wal_path.stat().st_size
        
        return size
    
    def export_to_parquet(self, table_name: str, output_path: str, engine: Engine) -> None:
        """Export table to Parquet file.
        
        DuckDB has excellent Parquet support.
        
        Args:
            table_name: Name of table to export
            output_path: Path to output Parquet file
            engine: SQLAlchemy engine
        """
        with engine.connect() as conn:
            conn.execute(text(f"COPY {table_name} TO '{output_path}' (FORMAT PARQUET)"))
            logger.info(f"Exported {table_name} to {output_path}")
    
    def import_from_parquet(self, parquet_path: str, table_name: str, engine: Engine) -> None:
        """Import data from Parquet file.
        
        Args:
            parquet_path: Path to Parquet file
            table_name: Target table name
            engine: SQLAlchemy engine
        """
        with engine.connect() as conn:
            conn.execute(text(f"CREATE TABLE {table_name} AS SELECT * FROM '{parquet_path}'"))
            logger.info(f"Imported {parquet_path} to {table_name}")