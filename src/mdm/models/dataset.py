"""Dataset-related models."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class DatasetInfo(BaseModel):
    """Dataset information model."""

    name: str = Field(description="Dataset name (lowercase, no spaces)")
    display_name: Optional[str] = Field(default=None, description="Display name")
    description: Optional[str] = Field(default=None, description="Dataset description")
    database: dict = Field(description="Database connection info")
    tables: dict[str, str] = Field(default_factory=dict, description="Table mappings")
    problem_type: Optional[
        Literal["binary_classification", "multiclass_classification", "regression"]
    ] = Field(default=None, description="ML problem type")
    target_column: Optional[str] = Field(default=None, description="Target column name")
    id_columns: list[str] = Field(default_factory=list, description="ID columns")
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

