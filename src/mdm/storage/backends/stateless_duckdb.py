"""
Stateless DuckDB backend with connection pooling and full compatibility.

This implementation provides:
1. Stateless design for concurrent usage
2. DuckDB-specific optimizations
3. Full backward compatibility
"""
from typing import Any, Dict, Optional, List, Generator
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Engine, event
from sqlalchemy.orm import Session, sessionmaker
from contextlib import contextmanager
import logging
from pathlib import Path
import duckdb

from ...interfaces.storage import IStorageBackend
from ...config import get_config
from .compatibility_mixin import BackendCompatibilityMixin
from ...core.exceptions import StorageError

logger = logging.getLogger(__name__)


class StatelessDuckDBBackend(BackendCompatibilityMixin, IStorageBackend):
    """
    Stateless DuckDB backend with full compatibility.
    
    DuckDB advantages:
    - Columnar storage for analytics
    - Excellent performance on analytical queries
    - Native Parquet support
    - Better memory efficiency for large datasets
    """
    
    def __init__(self):
        """Initialize backend without any state."""
        self.config = get_config()
        self.datasets_path = Path(self.config.paths.datasets_path)
        
        # For compatibility with singleton pattern
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        
        logger.info("Initialized StatelessDuckDBBackend")
    
    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return "duckdb"
    
    def get_engine(self, database_path: str) -> Engine:
        """
        Get SQLAlchemy engine for DuckDB database.
        
        Args:
            database_path: Path to DuckDB database file
            
        Returns:
            SQLAlchemy Engine instance
        """
        # Get DuckDB config from settings
        duckdb_config = getattr(self.config.database, 'duckdb', {})
        
        # Build connection arguments
        connect_args = {
            "config": {
                "memory_limit": duckdb_config.get("memory_limit", "2GB"),
                "threads": duckdb_config.get("threads", 4),
                "temp_directory": duckdb_config.get("temp_directory", "/tmp/mdm_duckdb"),
            }
        }
        
        # Create engine
        engine = create_engine(
            f"duckdb:///{database_path}",
            echo=self.config.database.echo_sql if hasattr(self.config.database, 'echo_sql') else False,
            connect_args=connect_args,
            pool_pre_ping=True
        )
        
        # Apply DuckDB optimizations
        @event.listens_for(engine, "connect")
        def set_duckdb_options(dbapi_conn, connection_record):
            # Enable parallel query execution
            dbapi_conn.execute("SET threads TO 4")
            # Set memory limit
            dbapi_conn.execute(f"SET memory_limit='{duckdb_config.get('memory_limit', '2GB')}'")
            # Enable progress bar for long queries (disabled in production)
            dbapi_conn.execute("SET enable_progress_bar=false")
        
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
        """
        db_path = self._get_database_path(dataset_name)
        engine = self.get_engine(str(db_path))
        
        try:
            yield engine
        finally:
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
        db_path = dataset_path / f"{dataset_name}.duckdb"
        
        # Initialize with context manager
        with self.get_engine_context(dataset_name) as engine:
            with engine.begin() as conn:
                # Create metadata table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS _metadata (
                        key VARCHAR PRIMARY KEY,
                        value VARCHAR,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Store initial configuration
                for key, value in config.items():
                    conn.execute(
                        text("""
                            INSERT INTO _metadata (key, value) 
                            VALUES (:key, :value)
                            ON CONFLICT (key) DO UPDATE SET 
                                value = EXCLUDED.value,
                                updated_at = CURRENT_TIMESTAMP
                        """),
                        {"key": key, "value": str(value)}
                    )
        
        logger.info(f"Created DuckDB dataset: {dataset_name}")
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset exists."""
        db_path = self._get_database_path(dataset_name)
        return db_path.exists()
    
    def drop_dataset(self, dataset_name: str) -> None:
        """Remove dataset and all associated data."""
        dataset_path = self.datasets_path / dataset_name
        
        if dataset_path.exists():
            import shutil
            shutil.rmtree(dataset_path)
            
        logger.info(f"Dropped DuckDB dataset: {dataset_name}")
    
    def load_data(self, dataset_name: str, table_name: str = "data") -> pd.DataFrame:
        """
        Load data from dataset table.
        
        DuckDB optimization: Uses native read methods when possible.
        """
        with self.get_engine_context(dataset_name) as engine:
            # For large tables, DuckDB's native methods are faster
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
        
        DuckDB optimization: Uses native DataFrame support.
        """
        with self.get_engine_context(dataset_name) as engine:
            # DuckDB has excellent DataFrame integration
            data.to_sql(
                table_name,
                engine,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=self.config.performance.batch_size
            )
    
    def export_to_parquet(self, dataset_name: str, table_name: str, output_path: str) -> None:
        """
        Export table to Parquet format (DuckDB-specific feature).
        
        Args:
            dataset_name: Name of the dataset
            table_name: Table to export
            output_path: Path for Parquet file
        """
        with self.get_engine_context(dataset_name) as engine:
            with engine.connect() as conn:
                conn.execute(text(f"""
                    COPY {table_name} TO '{output_path}' (FORMAT PARQUET)
                """))
        
        logger.info(f"Exported {table_name} to {output_path}")
    
    def import_from_parquet(self, dataset_name: str, parquet_path: str, table_name: str) -> None:
        """
        Import data from Parquet file (DuckDB-specific feature).
        
        Args:
            dataset_name: Name of the dataset
            parquet_path: Path to Parquet file
            table_name: Target table name
        """
        with self.get_engine_context(dataset_name) as engine:
            with engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE OR REPLACE TABLE {table_name} AS 
                    SELECT * FROM read_parquet('{parquet_path}')
                """))
        
        logger.info(f"Imported {parquet_path} to {table_name}")
    
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
                            INSERT INTO _metadata (key, value) 
                            VALUES (:key, :value)
                            ON CONFLICT (key) DO UPDATE SET 
                                value = EXCLUDED.value,
                                updated_at = CURRENT_TIMESTAMP
                        """),
                        {"key": key, "value": str(value)}
                    )
    
    def close(self) -> None:
        """Close any open resources."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
        self._session_factory = None
        
        logger.debug("Closed StatelessDuckDBBackend")
    
    # Helper methods
    
    def _get_database_path(self, dataset_name: str) -> Path:
        """Get database file path for dataset."""
        return self.datasets_path / dataset_name / f"{dataset_name}.duckdb"
    
    def database_exists(self, database_path: str) -> bool:
        """Check if database file exists."""
        return Path(database_path).exists()
    
    def create_database(self, database_path: str) -> None:
        """Create empty database file."""
        db_path = Path(database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create empty DuckDB database
        conn = duckdb.connect(str(db_path))
        conn.close()
        
        logger.info(f"Created DuckDB database: {db_path}")
    
    def get_table_names(self, engine: Engine) -> List[str]:
        """Get list of table names in database."""
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                AND table_name NOT LIKE '\\_%' ESCAPE '\\\\'
            """))
            return [row[0] for row in result]
    
    def table_exists(self, engine: Engine, table_name: str) -> bool:
        """Check if table exists in database."""
        return table_name in self.get_table_names(engine)