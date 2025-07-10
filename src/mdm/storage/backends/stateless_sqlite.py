"""
Stateless SQLite backend with connection pooling and full compatibility.

This implementation shows how new backends should be structured with:
1. Stateless design (no singleton pattern)
2. Connection pooling for efficiency
3. Full backward compatibility via mixin
"""
from typing import Any, Dict, Optional, List, Generator
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Engine
from sqlalchemy.orm import Session, sessionmaker
from contextlib import contextmanager
import logging
from pathlib import Path
import sqlite3

from ...interfaces.storage import IStorageBackend
from ...config import get_config
from .compatibility_mixin import BackendCompatibilityMixin
from ...core.exceptions import StorageError

logger = logging.getLogger(__name__)


class StatelessSQLiteBackend(BackendCompatibilityMixin, IStorageBackend):
    """
    Stateless SQLite backend with full compatibility.
    
    This backend:
    - Does NOT maintain singleton state
    - Uses connection pooling for efficiency
    - Provides full backward compatibility via mixin
    - Can be used in parallel without conflicts
    """
    
    def __init__(self):
        """Initialize backend without any state."""
        self.config = get_config()
        self.datasets_path = Path(self.config.paths.datasets_path)
        
        # For compatibility with code expecting singleton pattern
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        
        logger.info("Initialized StatelessSQLiteBackend")
    
    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return "sqlite"
    
    def get_engine(self, database_path: str) -> Engine:
        """
        Get SQLAlchemy engine for database.
        
        Args:
            database_path: Path to SQLite database file
            
        Returns:
            SQLAlchemy Engine instance
            
        Note:
            For compatibility, this method also caches the engine
            in self._engine for methods expecting singleton pattern.
        """
        # Create engine with SQLite optimizations
        engine = create_engine(
            f"sqlite:///{database_path}",
            echo=self.config.database.echo_sql if hasattr(self.config.database, 'echo_sql') else False,
            pool_pre_ping=True,
            connect_args={
                "check_same_thread": False,  # Allow multi-threading
                "timeout": 30.0  # Connection timeout
            }
        )
        
        # Apply SQLite optimizations
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            
            # Performance optimizations
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            
            cursor.close()
        
        # Cache for singleton compatibility
        self._engine = engine
        self._session_factory = sessionmaker(bind=engine)
        
        return engine
    
    @contextmanager
    def get_engine_context(self, dataset_name: str) -> Generator[Engine, None, None]:
        """
        Get engine in a context manager for proper cleanup.
        
        Args:
            dataset_name: Name of the dataset
            
        Yields:
            SQLAlchemy Engine
            
        Note:
            This is the PREFERRED method for new code.
        """
        db_path = self._get_database_path(dataset_name)
        engine = self.get_engine(str(db_path))
        
        try:
            yield engine
        finally:
            # Dispose of engine to free resources
            engine.dispose()
    
    def create_dataset(self, dataset_name: str, config: Dict[str, Any]) -> None:
        """
        Create a new dataset database.
        
        Args:
            dataset_name: Name of the dataset
            config: Configuration for dataset
        """
        # Create dataset directory
        dataset_path = self.datasets_path / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create database file
        db_path = dataset_path / f"{dataset_name}.db"
        
        # Initialize with context manager
        with self.get_engine_context(dataset_name) as engine:
            # Create metadata tables
            with engine.begin() as conn:
                # Metadata table for dataset info
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS _metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Store initial configuration
                for key, value in config.items():
                    conn.execute(
                        text("INSERT OR REPLACE INTO _metadata (key, value) VALUES (:key, :value)"),
                        {"key": key, "value": str(value)}
                    )
        
        logger.info(f"Created SQLite dataset: {dataset_name}")
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset exists."""
        db_path = self._get_database_path(dataset_name)
        return db_path.exists()
    
    def drop_dataset(self, dataset_name: str) -> None:
        """Remove dataset and all associated data."""
        dataset_path = self.datasets_path / dataset_name
        
        if dataset_path.exists():
            # Remove entire dataset directory
            import shutil
            shutil.rmtree(dataset_path)
            
        logger.info(f"Dropped SQLite dataset: {dataset_name}")
    
    def load_data(self, dataset_name: str, table_name: str = "data") -> pd.DataFrame:
        """
        Load data from dataset table.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the table to load
            
        Returns:
            DataFrame with table contents
        """
        with self.get_engine_context(dataset_name) as engine:
            return pd.read_sql_table(table_name, engine)
    
    def save_data(
        self, 
        dataset_name: str, 
        data: pd.DataFrame,
        table_name: str = "data", 
        if_exists: str = "replace"
    ) -> None:
        """
        Save DataFrame to dataset table.
        
        Args:
            dataset_name: Name of the dataset
            data: DataFrame to save
            table_name: Target table name
            if_exists: What to do if table exists
        """
        with self.get_engine_context(dataset_name) as engine:
            data.to_sql(
                table_name,
                engine,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=self.config.performance.batch_size
            )
    
    def get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset metadata."""
        with self.get_engine_context(dataset_name) as engine:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT key, value FROM _metadata"))
                return {row[0]: row[1] for row in result}
    
    def update_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        """Update dataset metadata."""
        with self.get_engine_context(dataset_name) as engine:
            with engine.begin() as conn:
                for key, value in metadata.items():
                    conn.execute(
                        text("""
                            INSERT OR REPLACE INTO _metadata (key, value, updated_at) 
                            VALUES (:key, :value, CURRENT_TIMESTAMP)
                        """),
                        {"key": key, "value": str(value)}
                    )
    
    def close(self) -> None:
        """
        Close any open resources.
        
        Note:
            In stateless design, this mainly clears cached state.
        """
        if self._engine:
            self._engine.dispose()
            self._engine = None
        self._session_factory = None
        
        logger.debug("Closed StatelessSQLiteBackend")
    
    # Helper methods
    
    def _get_database_path(self, dataset_name: str) -> Path:
        """Get database file path for dataset."""
        return self.datasets_path / dataset_name / f"{dataset_name}.db"
    
    def database_exists(self, database_path: str) -> bool:
        """Check if database file exists (compatibility)."""
        return Path(database_path).exists()
    
    def create_database(self, database_path: str) -> None:
        """Create empty database file (compatibility)."""
        db_path = Path(database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create empty SQLite database
        conn = sqlite3.connect(str(db_path))
        conn.close()
        
        logger.info(f"Created SQLite database: {db_path}")
        
    # Additional methods for completeness
    
    def get_table_names(self, engine: Engine) -> List[str]:
        """Get list of table names in database."""
        inspector = inspect(engine)
        return inspector.get_table_names()
    
    def table_exists(self, engine: Engine, table_name: str) -> bool:
        """Check if table exists in database."""
        return table_name in self.get_table_names(engine)