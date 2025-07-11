"""Domain service for feature generation logic."""

import time
from typing import Dict, List, Optional
import pandas as pd

from loguru import logger

from mdm.features.registry import feature_registry
from mdm.features.custom.base import BaseDomainFeatures
from mdm.models.enums import ColumnType

from .value_objects import (
    FeatureSet, 
    FeatureMetadata, 
    FeatureGenerationConfig
)


class FeatureService:
    """Domain service containing pure feature generation logic.
    
    This service contains only business logic for feature generation,
    without any I/O operations or infrastructure concerns.
    """
    
    def __init__(self):
        """Initialize feature service."""
        self.registry = feature_registry
    
    def generate_features(self, 
                         df: pd.DataFrame,
                         config: FeatureGenerationConfig,
                         custom_features: Optional[BaseDomainFeatures] = None) -> FeatureSet:
        """Generate features for a DataFrame using pure domain logic.
        
        Args:
            df: Input DataFrame
            config: Feature generation configuration
            custom_features: Optional custom feature generator
            
        Returns:
            FeatureSet containing generated features and metadata
        """
        logger.info(f"Generating features for {config.dataset_name} ({len(df)} rows)")
        start_time = time.time()
        
        # Start with original columns
        feature_df = df.copy()
        metadata: Dict[str, FeatureMetadata] = {}
        discarded_count = 0
        
        # Generate features based on column types
        for column, col_type in config.column_types.items():
            if not config.should_process_column(column):
                continue
            
            # Get transformers for this column type
            transformers = self.registry.get_transformers(col_type)
            
            for transformer in transformers:
                logger.debug(
                    f"[{transformer.__class__.__name__}] Processing column '{column}'"
                )
                
                # Generate features using transformer
                features = transformer.generate_features(df, [column])
                
                # Add features and metadata
                for feature_name, feature_values in features.items():
                    if feature_name not in feature_df.columns:
                        feature_df[feature_name] = feature_values
                        
                        # Create metadata for this feature
                        metadata[feature_name] = FeatureMetadata(
                            name=feature_name,
                            source_columns=[column],
                            transformer_name=transformer.__class__.__name__,
                            column_type=col_type
                        )
        
        # Apply custom features if available
        if custom_features:
            logger.info(f"[CustomFeatures] Applying {config.dataset_name}-specific features")
            custom_feature_dict = custom_features.generate_all_features(df)
            
            for feature_name, feature_values in custom_feature_dict.items():
                if feature_name not in feature_df.columns:
                    feature_df[feature_name] = feature_values
                    
                    # Create metadata for custom feature
                    metadata[feature_name] = FeatureMetadata(
                        name=feature_name,
                        source_columns=list(df.columns),  # Custom features may use any columns
                        transformer_name="CustomFeatures",
                        column_type=ColumnType.NUMERIC  # Default type for custom features
                    )
        
        generation_time = time.time() - start_time
        
        return FeatureSet(
            features=feature_df,
            metadata=metadata,
            generation_time=generation_time,
            feature_count=len(metadata),
            discarded_count=discarded_count
        )
    
    def generate_features_for_columns(self,
                                    df: pd.DataFrame,
                                    columns: List[str],
                                    column_types: Dict[str, ColumnType]) -> FeatureSet:
        """Generate features for specific columns only.
        
        Args:
            df: Input DataFrame
            columns: List of columns to process
            column_types: Mapping of column names to types
            
        Returns:
            FeatureSet containing generated features
        """
        start_time = time.time()
        feature_df = pd.DataFrame(index=df.index)
        metadata: Dict[str, FeatureMetadata] = {}
        
        for column in columns:
            if column not in column_types:
                logger.warning(f"Column {column} not found in column types")
                continue
            
            col_type = column_types[column]
            transformers = self.registry.get_transformers(col_type)
            
            for transformer in transformers:
                features = transformer.generate_features(df, [column])
                
                for feature_name, feature_values in features.items():
                    if feature_name not in feature_df.columns:
                        feature_df[feature_name] = feature_values
                        metadata[feature_name] = FeatureMetadata(
                            name=feature_name,
                            source_columns=[column],
                            transformer_name=transformer.__class__.__name__,
                            column_type=col_type
                        )
        
        generation_time = time.time() - start_time
        
        return FeatureSet(
            features=feature_df,
            metadata=metadata,
            generation_time=generation_time,
            feature_count=len(metadata),
            discarded_count=0
        )
    
    def validate_features(self, feature_set: FeatureSet) -> bool:
        """Validate generated features.
        
        Args:
            feature_set: FeatureSet to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check if features DataFrame is empty
        if feature_set.features.empty:
            logger.error("Feature DataFrame is empty")
            return False
        
        # Check if metadata matches features
        feature_columns = set(feature_set.features.columns)
        metadata_columns = set(feature_set.metadata.keys())
        
        # Original columns won't have metadata
        generated_features = feature_columns - metadata_columns
        if generated_features and not any(col.startswith(tuple(metadata_columns)) 
                                         for col in generated_features):
            logger.warning(f"Features without metadata: {generated_features}")
        
        # Check for invalid values
        for col in feature_set.features.columns:
            if feature_set.features[col].isna().all():
                logger.warning(f"Feature {col} contains only NaN values")
        
        return True
    
    def merge_feature_sets(self, *feature_sets: FeatureSet) -> FeatureSet:
        """Merge multiple feature sets into one.
        
        Args:
            *feature_sets: Variable number of FeatureSets to merge
            
        Returns:
            Merged FeatureSet
        """
        if not feature_sets:
            raise ValueError("No feature sets provided to merge")
        
        if len(feature_sets) == 1:
            return feature_sets[0]
        
        # Start with the first set
        merged = feature_sets[0]
        
        # Merge remaining sets
        for feature_set in feature_sets[1:]:
            merged = merged.merge(feature_set)
        
        return merged