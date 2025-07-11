"""Database implementation of feature loader."""

from typing import Dict, Optional, List
import pandas as pd
from sqlalchemy import Engine

from loguru import logger

from mdm.storage.factory import BackendFactory
from mdm.features.domain.interfaces import IFeatureLoader


class DatabaseFeatureLoader(IFeatureLoader):
    """Loads data from database for feature generation."""
    
    def __init__(self, backend_factory: Optional[BackendFactory] = None):
        """Initialize the database feature loader.
        
        Args:
            backend_factory: Factory for creating storage backends
        """
        self.backend_factory = backend_factory or BackendFactory
        self._engines: Dict[str, Engine] = {}
        self._backends: Dict[str, any] = {}
    
    def load_data(self, dataset_name: str, table_name: str, 
                 batch_size: Optional[int] = None,
                 offset: int = 0,
                 columns: Optional[List[str]] = None,
                 datetime_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load data for feature generation.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the table to load
            batch_size: Optional batch size for loading
            offset: Offset for batch loading
            columns: Specific columns to load (None for all)
            datetime_columns: Columns to parse as datetime
            
        Returns:
            DataFrame with loaded data
        """
        try:
            engine = self._get_engine(dataset_name)
            
            # Build query
            if columns:
                columns_str = ", ".join(columns)
                query = f"SELECT {columns_str} FROM {table_name}"
            else:
                query = f"SELECT * FROM {table_name}"
            
            if batch_size:
                query += f" LIMIT {batch_size} OFFSET {offset}"
            
            logger.debug(f"Loading data with query: {query}")
            
            # Load data with datetime parsing if needed
            if datetime_columns:
                df = pd.read_sql_query(query, engine, parse_dates=datetime_columns)
            else:
                df = pd.read_sql_query(query, engine)
            
            logger.debug(f"Loaded {len(df)} rows from {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from {table_name}: {e}")
            raise
    
    def get_row_count(self, dataset_name: str, table_name: str) -> int:
        """Get the number of rows in a table.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the table
            
        Returns:
            Number of rows
        """
        try:
            backend = self._get_backend(dataset_name)
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            result = backend.query(query)
            
            if result is not None and not result.empty:
                return int(result.iloc[0]['count'])
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get row count for {table_name}: {e}")
            return 0
    
    def table_exists(self, dataset_name: str, table_name: str) -> bool:
        """Check if a table exists.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the table
            
        Returns:
            True if table exists
        """
        try:
            engine = self._get_engine(dataset_name)
            
            # Use SQLAlchemy inspection
            from sqlalchemy import inspect
            inspector = inspect(engine)
            
            return table_name in inspector.get_table_names()
            
        except Exception as e:
            logger.error(f"Failed to check if table {table_name} exists: {e}")
            return False
    
    def _get_engine(self, dataset_name: str) -> Engine:
        """Get or create database engine for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            SQLAlchemy engine
        """
        if dataset_name not in self._engines:
            backend = self._get_backend(dataset_name)
            
            # Get database path from dataset info
            dataset_info = self._load_dataset_info(dataset_name)
            
            if 'path' in dataset_info.get('database', {}):
                db_path = dataset_info['database']['path']
            else:
                # Construct path
                from mdm.config import get_config_manager
                config_manager = get_config_manager()
                datasets_dir = config_manager.base_path / config_manager.config.paths.datasets_path
                dataset_dir = datasets_dir / dataset_name
                
                backend_type = dataset_info.get('database', {}).get('backend', 'duckdb')
                if backend_type == 'duckdb':
                    db_path = str(dataset_dir / f'{dataset_name}.duckdb')
                elif backend_type == 'sqlite':
                    db_path = str(dataset_dir / f'{dataset_name}.sqlite')
                else:
                    raise ValueError(f"Unsupported backend type: {backend_type}")
            
            self._engines[dataset_name] = backend.get_engine(db_path)
        
        return self._engines[dataset_name]
    
    def _get_backend(self, dataset_name: str):
        """Get or create backend for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Storage backend
        """
        if dataset_name not in self._backends:
            dataset_info = self._load_dataset_info(dataset_name)
            
            # Get backend configuration
            backend_type = dataset_info.get('database', {}).get('backend', 'duckdb')
            backend_config = dataset_info.get('database', {}).copy()
            
            # Create backend
            self._backends[dataset_name] = self.backend_factory.create(
                backend_type, backend_config
            )
        
        return self._backends[dataset_name]
    
    def _load_dataset_info(self, dataset_name: str) -> Dict:
        """Load dataset information from YAML.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset information dictionary
        """
        from mdm.config import get_config_manager
        config_manager = get_config_manager()
        base_path = config_manager.base_path
        config = config_manager.config
        
        dataset_registry_dir = base_path / config.paths.configs_path
        yaml_file = dataset_registry_dir / f"{dataset_name}.yaml"
        
        if not yaml_file.exists():
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        import yaml
        with open(yaml_file) as f:
            return yaml.safe_load(f)
    
    def __del__(self):
        """Clean up database connections."""
        for engine in self._engines.values():
            try:
                engine.dispose()
            except:
                pass
        
        for backend in self._backends.values():
            try:
                if hasattr(backend, 'close_connections'):
                    backend.close_connections()
            except:
                pass