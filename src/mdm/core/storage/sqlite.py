"""New SQLite storage backend implementation.

This module provides the refactored SQLite backend with improved design
and better separation of concerns.
"""
from typing import Any, Dict
from pathlib import Path
import logging
import os

from sqlalchemy import Engine, text

from .base import NewStorageBackend
from mdm.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class NewSQLiteBackend(NewStorageBackend):
    """New SQLite storage backend implementation.
    
    Key improvements over legacy:
    - Better connection management
    - Proper PRAGMA handling
    - Improved error handling
    - Thread-safe operations
    """
    
    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return "sqlite"
    
    def _create_connection_string(self, database_path: str) -> str:
        """Create SQLite connection string.
        
        Args:
            database_path: Path to SQLite database file
            
        Returns:
            SQLite connection string
        """
        # Ensure absolute path
        db_path = Path(database_path).absolute()
        
        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        return f"sqlite:///{db_path}"
    
    def _get_engine_args(self) -> Dict[str, Any]:
        """Get SQLite-specific engine arguments."""
        return {
            "connect_args": {
                "check_same_thread": False,  # Allow multi-threaded access
                "timeout": self.config.get("timeout", 30),
            }
        }
    
    def _initialize_backend_specific(self, engine: Engine) -> None:
        """Perform SQLite-specific initialization.
        
        Sets PRAGMAs and other SQLite-specific settings.
        """
        pragma_settings = {
            "journal_mode": self.config.get("journal_mode", "WAL"),
            "synchronous": self.config.get("synchronous", "NORMAL"),
            "cache_size": self.config.get("cache_size", -64000),
            "temp_store": self.config.get("temp_store", "MEMORY"),
            "mmap_size": self.config.get("mmap_size", 268435456),
        }
        
        with engine.connect() as conn:
            for pragma, value in pragma_settings.items():
                try:
                    if isinstance(value, str):
                        conn.execute(text(f"PRAGMA {pragma} = '{value}'"))
                    else:
                        conn.execute(text(f"PRAGMA {pragma} = {value}"))
                    
                    # Verify setting
                    result = conn.execute(text(f"PRAGMA {pragma}"))
                    actual_value = result.scalar()
                    logger.debug(f"Set SQLite PRAGMA {pragma} = {actual_value}")
                    
                except Exception as e:
                    logger.warning(f"Failed to set PRAGMA {pragma}: {e}")
    
    def database_exists(self, database_path: str) -> bool:
        """Check if database exists.
        
        Args:
            database_path: Path to SQLite database file
            
        Returns:
            True if database file exists
        """
        return Path(database_path).exists()
    
    def create_database(self, database_path: str) -> None:
        """Create empty database.
        
        For SQLite, this just ensures the directory exists.
        The database file will be created on first connection.
        
        Args:
            database_path: Path to SQLite database file
        """
        db_path = Path(database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Touch the file to create it
        db_path.touch(exist_ok=True)
        
        logger.info(f"Created SQLite database at {database_path}")
    
    def drop_database(self, database_path: str) -> None:
        """Drop an existing database.
        
        For SQLite, this removes the database file.
        
        Args:
            database_path: Path to SQLite database file
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
                logger.info(f"Dropped SQLite database at {database_path}")
            except Exception as e:
                raise StorageError(f"Failed to drop database: {e}")
        
        # Also remove WAL and SHM files if they exist
        for suffix in ["-wal", "-shm"]:
            aux_file = Path(str(db_path) + suffix)
            if aux_file.exists():
                try:
                    os.remove(aux_file)
                except Exception as e:
                    logger.warning(f"Failed to remove {aux_file}: {e}")
    
    def _get_file_extension(self) -> str:
        """Get file extension for database files."""
        return "db"
    
    def optimize_database(self, engine: Engine) -> None:
        """Run SQLite-specific optimizations.
        
        Args:
            engine: SQLAlchemy engine
        """
        with engine.connect() as conn:
            # Run VACUUM to reclaim space
            conn.execute(text("VACUUM"))
            
            # Run ANALYZE to update statistics
            conn.execute(text("ANALYZE"))
            
            # Optimize query planner
            conn.execute(text("PRAGMA optimize"))
            
            logger.info("Optimized SQLite database")
    
    def get_database_size(self, database_path: str) -> int:
        """Get database file size in bytes.
        
        Args:
            database_path: Path to SQLite database file
            
        Returns:
            Size in bytes
        """
        db_path = Path(database_path)
        if not db_path.exists():
            return 0
        
        # Get main database size
        size = db_path.stat().st_size
        
        # Add WAL file size if it exists
        wal_path = Path(str(db_path) + "-wal")
        if wal_path.exists():
            size += wal_path.stat().st_size
        
        return size