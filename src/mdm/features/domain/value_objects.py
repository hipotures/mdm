"""Value objects for feature engineering domain."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
import pandas as pd

from mdm.models.enums import ColumnType


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata for a generated feature."""
    name: str
    source_columns: List[str]
    transformer_name: str
    column_type: ColumnType
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate metadata."""
        if not self.name:
            raise ValueError("Feature name cannot be empty")
        if not self.source_columns:
            raise ValueError("Source columns cannot be empty")


@dataclass(frozen=True)
class FeatureGenerationConfig:
    """Configuration for feature generation."""
    dataset_name: str
    target_column: Optional[str] = None
    id_columns: Optional[List[str]] = None
    exclude_columns: Optional[Set[str]] = None
    column_types: Dict[str, ColumnType] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize exclude columns set."""
        if self.exclude_columns is None:
            exclude_cols = set()
            if self.id_columns:
                exclude_cols.update(self.id_columns)
            if self.target_column:
                exclude_cols.add(self.target_column)
            object.__setattr__(self, 'exclude_columns', exclude_cols)
    
    def should_process_column(self, column: str) -> bool:
        """Check if a column should be processed for feature generation."""
        return column not in self.exclude_columns


@dataclass
class FeatureSet:
    """Collection of generated features with metadata."""
    features: pd.DataFrame
    metadata: Dict[str, FeatureMetadata]
    generation_time: float
    feature_count: int = 0
    discarded_count: int = 0
    
    def __post_init__(self):
        """Calculate feature count if not provided."""
        if self.feature_count == 0:
            self.feature_count = len(self.metadata)
    
    def merge(self, other: 'FeatureSet') -> 'FeatureSet':
        """Merge with another feature set.
        
        Args:
            other: Another FeatureSet to merge
            
        Returns:
            New FeatureSet with merged features
        """
        # Merge DataFrames
        merged_features = pd.concat([self.features, other.features], axis=1)
        
        # Merge metadata
        merged_metadata = {**self.metadata, **other.metadata}
        
        # Calculate combined metrics
        total_time = self.generation_time + other.generation_time
        total_features = self.feature_count + other.feature_count
        total_discarded = self.discarded_count + other.discarded_count
        
        return FeatureSet(
            features=merged_features,
            metadata=merged_metadata,
            generation_time=total_time,
            feature_count=total_features,
            discarded_count=total_discarded
        )
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return list(self.metadata.keys())
    
    def get_features_by_source(self, source_column: str) -> List[str]:
        """Get features derived from a specific source column."""
        features = []
        for name, meta in self.metadata.items():
            if source_column in meta.source_columns:
                features.append(name)
        return features


@dataclass(frozen=True)
class BatchProcessingConfig:
    """Configuration for batch feature processing."""
    batch_size: int = 10000
    enable_progress: bool = True
    memory_limit_mb: int = 100
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("Memory limit must be positive")