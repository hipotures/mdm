"""Data Transfer Objects for feature generation application layer."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from mdm.models.enums import ColumnType


class FeatureGenerationMode(Enum):
    """Mode for feature generation."""
    FULL = "full"
    INCREMENTAL = "incremental"
    SELECTED_COLUMNS = "selected_columns"


@dataclass
class FeatureGenerationRequest:
    """Request for feature generation."""
    dataset_name: str
    table_names: Dict[str, str]  # table_type -> table_name mapping
    column_types: Dict[str, ColumnType]
    target_column: Optional[str] = None
    id_columns: Optional[List[str]] = None
    datetime_columns: Optional[List[str]] = None
    mode: FeatureGenerationMode = FeatureGenerationMode.FULL
    selected_columns: Optional[List[str]] = None  # For SELECTED_COLUMNS mode
    batch_size: int = 10000
    enable_custom_features: bool = True
    enable_progress: bool = True
    
    def __post_init__(self):
        """Validate request."""
        if self.mode == FeatureGenerationMode.SELECTED_COLUMNS and not self.selected_columns:
            raise ValueError("selected_columns must be provided for SELECTED_COLUMNS mode")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class FeatureGenerationResponse:
    """Response from feature generation."""
    dataset_name: str
    feature_tables: Dict[str, str]  # table_type -> feature_table_name mapping
    total_features_generated: int
    features_per_table: Dict[str, int]
    generation_time: float
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


@dataclass
class BatchProcessingRequest:
    """Request for batch feature processing."""
    dataset_name: str
    source_table: str
    target_table: str
    column_types: Dict[str, ColumnType]
    batch_size: int = 10000
    offset: int = 0
    limit: Optional[int] = None
    
    def get_query(self) -> str:
        """Get SQL query for this batch."""
        query = f"SELECT * FROM {self.source_table}"
        if self.limit:
            query += f" LIMIT {self.batch_size} OFFSET {self.offset}"
        return query


@dataclass
class BatchProcessingResponse:
    """Response from batch processing."""
    batch_number: int
    rows_processed: int
    features_generated: int
    processing_time: float
    has_more: bool = False
    next_offset: Optional[int] = None


@dataclass
class FeatureMetadataDto:
    """DTO for feature metadata."""
    feature_name: str
    source_columns: List[str]
    transformer_name: str
    column_type: str
    statistics: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_domain(cls, metadata: Any) -> 'FeatureMetadataDto':
        """Create from domain metadata."""
        return cls(
            feature_name=metadata.name,
            source_columns=metadata.source_columns,
            transformer_name=metadata.transformer_name,
            column_type=metadata.column_type.value if hasattr(metadata.column_type, 'value') else str(metadata.column_type)
        )


@dataclass  
class FeatureValidationRequest:
    """Request for feature validation."""
    dataset_name: str
    feature_table: str
    validation_rules: Optional[Dict[str, Any]] = None
    sample_size: int = 1000


@dataclass
class FeatureValidationResponse:
    """Response from feature validation."""
    is_valid: bool
    total_features: int
    invalid_features: List[str] = field(default_factory=list)
    validation_errors: Dict[str, List[str]] = field(default_factory=dict)
    statistics: Optional[Dict[str, Any]] = None