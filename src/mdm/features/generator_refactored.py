"""Refactored feature generator using DDD architecture.

This module provides a facade that maintains backward compatibility
while using the new domain-driven design architecture internally.
"""

import time
from typing import Optional, Any, Dict, List
import pandas as pd
from loguru import logger
from sqlalchemy import Engine

from mdm.config import get_config_manager
from mdm.models.enums import ColumnType

# Import from new DDD architecture
from .domain import (
    FeatureService,
    FeatureGenerationConfig,
    BatchProcessingConfig
)
from .application import (
    FeatureGenerationUseCase,
    FeatureGenerationRequest,
    FeatureGenerationMode,
    FeaturePipelineBuilder
)
from .infrastructure import (
    DatabaseFeatureLoader,
    DatabaseFeatureWriter,
    FileFeatureLoader,
    FileSystemFeatureRepository,
    BatchProgressTracker,
    NoOpProgressTracker
)


class FeatureGenerator:
    """Refactored feature generator using DDD architecture.
    
    This class maintains the same public interface as the original
    FeatureGenerator but delegates to the new DDD components internally.
    """
    
    def __init__(self):
        """Initialize feature generator with DDD components."""
        config_manager = get_config_manager()
        self.config = config_manager.config
        self.base_path = config_manager.base_path
        
        # Initialize infrastructure components
        self._feature_loader = DatabaseFeatureLoader()
        self._feature_writer = DatabaseFeatureWriter()
        self._custom_loader = FileFeatureLoader()
        self._feature_repository = FileSystemFeatureRepository()
        
        # Initialize domain service
        self._feature_service = FeatureService()
        
        # Initialize application service
        self._use_case = FeatureGenerationUseCase(
            feature_service=self._feature_service,
            feature_loader=self._feature_loader,
            feature_writer=self._feature_writer,
            custom_loader=self._custom_loader,
            feature_repository=self._feature_repository
        )
        
        # Build pipeline for more complex workflows
        self._pipeline = (
            FeaturePipelineBuilder()
            .with_validation()
            .with_data_loading(self._feature_loader)
            .with_feature_generation(self._use_case)
            .with_metadata_persistence(self._feature_repository)
            .build()
        )
    
    def generate_features(self,
                         df: pd.DataFrame,
                         dataset_name: str,
                         column_types: Dict[str, ColumnType],
                         target_column: Optional[str] = None,
                         id_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate features for a DataFrame (backward compatible interface).
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset
            column_types: Mapping of column names to types
            target_column: Target column name (if any)
            id_columns: ID column names
            
        Returns:
            DataFrame with original and generated features
        """
        logger.info(f"Generating features for {dataset_name} ({len(df)} rows)")
        
        # Create configuration
        config = FeatureGenerationConfig(
            dataset_name=dataset_name,
            target_column=target_column,
            id_columns=id_columns,
            column_types=column_types
        )
        
        # Load custom features if available
        custom_features = self._custom_loader.load(dataset_name)
        
        # Generate features using domain service
        feature_set = self._feature_service.generate_features(
            df, config, custom_features
        )
        
        logger.info(
            f"Feature generation complete: {feature_set.feature_count} features created "
            f"({feature_set.discarded_count} discarded) in {feature_set.generation_time:.2f}s"
        )
        
        return feature_set.features
    
    def generate_feature_tables(self,
                               engine: Engine,
                               dataset_name: str,
                               source_tables: Dict[str, str],
                               column_types: Dict[str, ColumnType],
                               target_column: Optional[str] = None,
                               id_columns: Optional[List[str]] = None,
                               progress: Optional[Any] = None,
                               datetime_columns: Optional[List[str]] = None) -> Dict[str, str]:
        """Generate feature tables for all source tables (backward compatible).
        
        This method now uses the new DDD architecture internally while
        maintaining the same interface for backward compatibility.
        
        Args:
            engine: SQLAlchemy engine (not used directly anymore)
            dataset_name: Name of the dataset
            source_tables: Mapping of table type to table name
            column_types: Mapping of column names to types
            target_column: Target column name
            id_columns: ID column names
            progress: Progress object (optional)
            datetime_columns: Datetime column names
            
        Returns:
            Mapping of feature table type to table name
        """
        # Get batch size from config
        batch_size = self.config.performance.batch_size
        
        # Create feature generation request
        request = FeatureGenerationRequest(
            dataset_name=dataset_name,
            table_names=source_tables,
            column_types=column_types,
            target_column=target_column,
            id_columns=id_columns,
            datetime_columns=datetime_columns,
            mode=FeatureGenerationMode.FULL,
            batch_size=batch_size,
            enable_progress=(progress is not None)
        )
        
        # Execute feature generation
        response = self._use_case.execute(request)
        
        if response.success:
            logger.info(
                f"Created {len(response.feature_tables)} feature tables with "
                f"{response.total_features_generated} total features in "
                f"{response.generation_time:.2f}s"
            )
            return response.feature_tables
        else:
            logger.error(f"Feature generation failed: {response.error_message}")
            raise RuntimeError(f"Feature generation failed: {response.error_message}")
    
    def _load_custom_features(self, dataset_name: str) -> Optional[Any]:
        """Load custom features for a dataset (backward compatible).
        
        This method is maintained for backward compatibility but
        delegates to the new infrastructure component.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Custom feature instance or None
        """
        return self._custom_loader.load(dataset_name)