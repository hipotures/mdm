"""Dataset-related models."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from mdm.models.enums import ColumnType, FileType, ProblemType


class DatasetInfo(BaseModel):
    """Dataset information model."""

    name: str = Field(description="Dataset name (lowercase, no spaces)")
    display_name: Optional[str] = Field(default=None, description="Display name")
    description: Optional[str] = Field(default=None, description="Dataset description")
    database: dict = Field(description="Database connection info")
    tables: dict[str, str] = Field(default_factory=dict, description="Table mappings")
    problem_type: Optional[
        Literal["binary_classification", "multiclass_classification", "regression", "time_series", "clustering"]
    ] = Field(default=None, description="ML problem type")
    target_column: Optional[str] = Field(default=None, description="Target column name")
    id_columns: list[str] = Field(default_factory=list, description="ID columns")
    time_column: Optional[str] = Field(default=None, description="Time column for time series")
    group_column: Optional[str] = Field(default=None, description="Group column for grouped time series")
    feature_tables: dict[str, str] = Field(default_factory=dict, description="Feature table mappings")
    registered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Registration timestamp"
    )
    last_updated_at: Optional[datetime] = Field(
        default=None, description="Last update timestamp"
    )
    tags: list[str] = Field(default_factory=list, description="Dataset tags")
    source: Optional[str] = Field(default=None, description="Data source")
    version: str = Field(default="1.0.0", description="Dataset version")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate dataset name."""
        if not v:
            raise ValueError("Dataset name cannot be empty")
        # Normalize to lowercase
        v = v.lower()
        # Check for valid characters
        if not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError("Dataset name can only contain alphanumeric, underscore, and dash")
        return v

    @property
    def has_train_test_split(self) -> bool:
        """Check if dataset has train/test split."""
        return "train" in self.tables and "test" in self.tables

    @property
    def has_validation(self) -> bool:
        """Check if dataset has validation set."""
        return "validation" in self.tables

    def get_database_path(self) -> Optional[Path]:
        """Get database file path if applicable."""
        if "path" in self.database:
            return Path(self.database["path"]).expanduser()
        return None

    def get_connection_string(self) -> Optional[str]:
        """Get database connection string if applicable."""
        return self.database.get("connection_string")


class DatasetStatistics(BaseModel):
    """Dataset statistics model."""

    row_count: int = Field(description="Total number of rows")
    column_count: int = Field(description="Total number of columns")
    memory_usage_mb: float = Field(description="Memory usage in MB")
    missing_values: dict[str, int] = Field(
        default_factory=dict, description="Missing values per column"
    )
    column_types: dict[str, str] = Field(
        default_factory=dict, description="Data types per column"
    )
    numeric_columns: list[str] = Field(
        default_factory=list, description="List of numeric columns"
    )
    categorical_columns: list[str] = Field(
        default_factory=list, description="List of categorical columns"
    )
    datetime_columns: list[str] = Field(
        default_factory=list, description="List of datetime columns"
    )
    text_columns: list[str] = Field(default_factory=list, description="List of text columns")
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When statistics were computed"
    )


class FileInfo(BaseModel):
    """Information about a data file."""

    path: Path = Field(description="File path")
    name: str = Field(description="File name")
    size_bytes: int = Field(description="File size in bytes")
    file_type: FileType = Field(description="Type of the file (train, test, etc.)")
    format: str = Field(description="File format (csv, parquet, etc.)")
    encoding: str = Field(default="utf-8", description="File encoding")
    row_count: Optional[int] = Field(default=None, description="Number of rows")
    column_count: Optional[int] = Field(default=None, description="Number of columns")
    created_at: Optional[datetime] = Field(default=None, description="File creation time")
    modified_at: Optional[datetime] = Field(default=None, description="File modification time")
    checksum: Optional[str] = Field(default=None, description="File checksum")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Validate file path."""
        if isinstance(v, str):
            v = Path(v)
        return v.expanduser().resolve()


class ColumnInfo(BaseModel):
    """Column metadata."""

    name: str = Field(description="Column name")
    dtype: str = Field(description="Data type as string")
    column_type: ColumnType = Field(description="Semantic column type")
    nullable: bool = Field(default=True, description="Whether column can contain nulls")
    unique: bool = Field(default=False, description="Whether values are unique")
    missing_count: int = Field(default=0, description="Number of missing values")
    missing_ratio: float = Field(default=0.0, description="Ratio of missing values")
    cardinality: Optional[int] = Field(default=None, description="Number of unique values")
    min_value: Optional[Any] = Field(default=None, description="Minimum value")
    max_value: Optional[Any] = Field(default=None, description="Maximum value")
    mean_value: Optional[float] = Field(default=None, description="Mean value (numeric)")
    std_value: Optional[float] = Field(default=None, description="Standard deviation (numeric)")
    sample_values: list[Any] = Field(default_factory=list, description="Sample values")
    description: Optional[str] = Field(default=None, description="Column description")


class DatasetInfoExtended(BaseModel):
    """Extended dataset information including files and columns."""

    name: str = Field(description="Dataset name")
    display_name: Optional[str] = Field(default=None, description="Display name")
    description: Optional[str] = Field(default=None, description="Dataset description")
    problem_type: Optional[ProblemType] = Field(default=None, description="ML problem type")
    target_column: Optional[str] = Field(default=None, description="Target column name")
    id_columns: list[str] = Field(default_factory=list, description="ID columns")
    datetime_columns: list[str] = Field(default_factory=list, description="DateTime columns")
    feature_columns: list[str] = Field(default_factory=list, description="Feature columns")
    files: dict[str, FileInfo] = Field(default_factory=dict, description="Dataset files")
    columns: dict[str, ColumnInfo] = Field(default_factory=dict, description="Column information")
    row_count: Optional[int] = Field(default=None, description="Total row count")
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage in MB")
    source: Optional[str] = Field(default=None, description="Data source")
    version: str = Field(default="1.0.0", description="Dataset version")
    tags: list[str] = Field(default_factory=list, description="Dataset tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")


class RegistrationParams(BaseModel):
    """Parameters for dataset registration."""

    name: str = Field(description="Dataset name")
    display_name: Optional[str] = Field(default=None, description="Display name")
    description: Optional[str] = Field(default=None, description="Dataset description")
    data_path: Optional[Path] = Field(default=None, description="Path to data files")
    train_file: Optional[str] = Field(default=None, description="Training file name")
    test_file: Optional[str] = Field(default=None, description="Test file name")
    validation_file: Optional[str] = Field(default=None, description="Validation file name")
    submission_file: Optional[str] = Field(default=None, description="Submission file name")
    target_column: Optional[str] = Field(default=None, description="Target column name")
    id_columns: Optional[list[str]] = Field(default=None, description="ID columns")
    datetime_columns: Optional[list[str]] = Field(default=None, description="DateTime columns")
    exclude_columns: Optional[list[str]] = Field(default=None, description="Columns to exclude")
    problem_type: Optional[ProblemType] = Field(default=None, description="ML problem type")
    auto_detect: bool = Field(default=True, description="Auto-detect dataset properties")
    validate_data: bool = Field(default=True, description="Validate data during registration")
    create_database: bool = Field(default=True, description="Create database for dataset")
    database_backend: Optional[Literal["sqlite", "duckdb", "postgresql"]] = Field(
        default=None, description="Database backend to use"
    )
    force: bool = Field(default=False, description="Force overwrite if exists")
    tags: Optional[list[str]] = Field(default=None, description="Dataset tags")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Additional metadata")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate dataset name."""
        if not v:
            raise ValueError("Dataset name cannot be empty")
        v = v.lower()
        if not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError("Dataset name can only contain alphanumeric, underscore, and dash")
        return v

    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate data path."""
        if v is not None:
            if isinstance(v, str):
                v = Path(v)
            v = v.expanduser().resolve()
            if not v.exists():
                raise ValueError(f"Data path does not exist: {v}")
            if not v.is_dir():
                raise ValueError(f"Data path must be a directory: {v}")
        return v


class RegistrationResult(BaseModel):
    """Result of dataset registration process."""

    success: bool = Field(description="Whether registration was successful")
    dataset_name: str = Field(description="Registered dataset name")
    dataset_info: Optional[DatasetInfoExtended] = Field(
        default=None, description="Dataset information"
    )
    database_path: Optional[Path] = Field(default=None, description="Database file path")
    files_processed: list[str] = Field(default_factory=list, description="Files processed")
    tables_created: list[str] = Field(default_factory=list, description="Database tables created")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")
    errors: list[str] = Field(default_factory=list, description="Error messages")
    duration_seconds: Optional[float] = Field(default=None, description="Registration duration")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

