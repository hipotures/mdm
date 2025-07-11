"""Database implementation of feature writer."""

from typing import Dict, Optional
import pandas as pd
from sqlalchemy import Engine

from loguru import logger

from mdm.storage.factory import BackendFactory
from mdm.features.domain.interfaces import IFeatureWriter


class DatabaseFeatureWriter(IFeatureWriter):
    """Writes generated features to database storage."""
    
    def __init__(self, backend_factory: Optional[BackendFactory] = None):
        """Initialize the database feature writer.
        
        Args:
            backend_factory: Factory for creating storage backends
        """
        self.backend_factory = backend_factory or BackendFactory
        self._engines: Dict[str, Engine] = {}
    
    def write_features(self, dataset_name: str, table_name: str, 
                      features: pd.DataFrame, mode: str = "replace") -> None:
        """Write generated features to storage.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the feature table
            features: DataFrame containing features
            mode: Write mode ('replace' or 'append')
        """
        try:
            engine = self._get_engine(dataset_name)
            
            logger.info(f"Writing {len(features.columns)} features to {table_name} "
                       f"({len(features)} rows, mode={mode})")
            
            # Write to database
            features.to_sql(
                table_name,
                engine,
                if_exists=mode,
                index=False
            )
            
            logger.info(f"Successfully wrote features to {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to write features to {table_name}: {e}")
            raise
    
    def write_features_batch(self, dataset_name: str, table_name: str,
                           features: pd.DataFrame, batch_number: int,
                           is_first_batch: bool = False) -> None:
        """Write features in batches for large datasets.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the feature table
            features: DataFrame containing features for this batch
            batch_number: Current batch number
            is_first_batch: Whether this is the first batch (for table creation)
        """
        try:
            engine = self._get_engine(dataset_name)
            
            # Determine write mode
            if_exists = "replace" if is_first_batch else "append"
            
            logger.debug(f"Writing batch {batch_number} to {table_name} "
                        f"({len(features)} rows, if_exists={if_exists})")
            
            # Write batch to database
            features.to_sql(
                table_name,
                engine,
                if_exists=if_exists,
                index=False
            )
            
            logger.debug(f"Successfully wrote batch {batch_number}")
            
        except Exception as e:
            logger.error(f"Failed to write batch {batch_number} to {table_name}: {e}")
            raise
    
    def _get_engine(self, dataset_name: str) -> Engine:
        """Get or create database engine for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            SQLAlchemy engine
        """
        if dataset_name not in self._engines:
            # Load dataset configuration
            from mdm.config import get_config_manager
            config_manager = get_config_manager()
            base_path = config_manager.base_path
            config = config_manager.config
            
            # Get dataset info
            dataset_registry_dir = base_path / config.paths.configs_path
            yaml_file = dataset_registry_dir / f"{dataset_name}.yaml"
            
            if not yaml_file.exists():
                raise ValueError(f"Dataset '{dataset_name}' not found")
            
            import yaml
            with open(yaml_file) as f:
                dataset_info = yaml.safe_load(f)
            
            # Get backend configuration
            backend_type = dataset_info.get('database', {}).get('backend', 'duckdb')
            backend_config = dataset_info.get('database', {}).copy()
            
            # Create backend
            backend = self.backend_factory.create(backend_type, backend_config)
            
            # Get database path
            if 'path' in dataset_info.get('database', {}):
                db_path = dataset_info['database']['path']
            else:
                datasets_dir = base_path / config.paths.datasets_path
                dataset_dir = datasets_dir / dataset_name
                
                if backend_type == 'duckdb':
                    db_path = str(dataset_dir / f'{dataset_name}.duckdb')
                elif backend_type == 'sqlite':
                    db_path = str(dataset_dir / f'{dataset_name}.sqlite')
                else:
                    raise ValueError(f"Unsupported backend type: {backend_type}")
            
            # Get engine
            self._engines[dataset_name] = backend.get_engine(db_path)
        
        return self._engines[dataset_name]
    
    def __del__(self):
        """Clean up database connections."""
        for engine in self._engines.values():
            try:
                engine.dispose()
            except:
                pass