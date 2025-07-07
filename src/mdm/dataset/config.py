"""Dataset configuration handling."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from mdm.core.exceptions import DatasetError
from mdm.models.dataset import DatasetInfo


class DatasetConfig(BaseModel):
    """Dataset configuration management."""

    name: str = Field(description="Dataset name")
    display_name: Optional[str] = Field(default=None, description="Display name")
    description: Optional[str] = Field(default=None, description="Dataset description")
    database: Dict[str, Any] = Field(description="Database configuration")
    tables: Dict[str, str] = Field(default_factory=dict, description="Table mappings")
    problem_type: Optional[str] = Field(default=None, description="ML problem type")
    target_column: Optional[str] = Field(default=None, description="Target column name")
    id_columns: list[str] = Field(default_factory=list, description="ID columns")
    source: Optional[str] = Field(default=None, description="Data source")
    version: str = Field(default="1.0.0", description="Dataset version")
    tags: list[str] = Field(default_factory=list, description="Dataset tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

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

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetConfig":
        """Load configuration from YAML file."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise DatasetError(f"Configuration file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            return cls(**data)
        except yaml.YAMLError as e:
            raise DatasetError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to load configuration: {e}")

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = self.model_dump(exclude_none=True)
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise DatasetError(f"Failed to save configuration: {e}")

    def to_dataset_info(self) -> DatasetInfo:
        """Convert to DatasetInfo model."""
        return DatasetInfo(
            name=self.name,
            display_name=self.display_name,
            description=self.description,
            database=self.database,
            tables=self.tables,
            problem_type=self.problem_type,
            target_column=self.target_column,
            id_columns=self.id_columns,
            source=self.source,
            version=self.version,
            tags=self.tags,
        )

    @classmethod
    def from_dataset_info(cls, info: DatasetInfo) -> "DatasetConfig":
        """Create from DatasetInfo model."""
        return cls(
            name=info.name,
            display_name=info.display_name,
            description=info.description,
            database=info.database,
            tables=info.tables,
            problem_type=info.problem_type,
            target_column=info.target_column,
            id_columns=info.id_columns,
            source=info.source,
            version=info.version,
            tags=info.tags,
        )


def create_dataset_pointer(dataset_name: str, dataset_path: Path) -> Path:
    """Create a YAML pointer in ~/.mdm/config/datasets/.
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the actual dataset directory
    
    Returns:
        Path to the created pointer file
    """
    from mdm.core.config import get_config

    config = get_config()
    pointer_dir = config.dataset_registry_dir
    pointer_dir.mkdir(parents=True, exist_ok=True)

    # Normalize dataset name
    dataset_name = dataset_name.lower()
    pointer_path = pointer_dir / f"{dataset_name}.yaml"

    # Create pointer content
    pointer_data = {
        "path": str(dataset_path.expanduser().resolve())
    }

    try:
        with open(pointer_path, "w") as f:
            yaml.dump(pointer_data, f, default_flow_style=False)
        return pointer_path
    except Exception as e:
        raise DatasetError(f"Failed to create dataset pointer: {e}")


def load_dataset_config(dataset_name: str) -> DatasetConfig:
    """Load dataset configuration from registry.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        DatasetConfig instance
    """
    from mdm.core.config import get_config
    from mdm.dataset.utils import normalize_dataset_name

    config = get_config()
    dataset_name = normalize_dataset_name(dataset_name)

    # Check if pointer exists
    pointer_path = config.dataset_registry_dir / f"{dataset_name}.yaml"
    if not pointer_path.exists():
        raise DatasetError(f"Dataset '{dataset_name}' not found in registry")

    # Load pointer to get dataset path
    try:
        with open(pointer_path) as f:
            pointer_data = yaml.safe_load(f)
        dataset_path = Path(pointer_data["path"])
    except Exception as e:
        raise DatasetError(f"Failed to load dataset pointer: {e}")

    # Load actual dataset config
    config_path = dataset_path / "dataset.yaml"
    if not config_path.exists():
        raise DatasetError(f"Dataset configuration not found: {config_path}")

    return DatasetConfig.from_yaml(config_path)


def save_dataset_config(dataset_name: str, config: DatasetConfig) -> None:
    """Save dataset configuration.
    
    Args:
        dataset_name: Name of the dataset
        config: DatasetConfig instance to save
    """
    from mdm.dataset.utils import get_dataset_path, normalize_dataset_name

    dataset_name = normalize_dataset_name(dataset_name)
    dataset_path = get_dataset_path(dataset_name)

    if not dataset_path.exists():
        raise DatasetError(f"Dataset directory not found: {dataset_path}")

    config_path = dataset_path / "dataset.yaml"
    config.to_yaml(config_path)


def validate_dataset_config(config: DatasetConfig) -> None:
    """Validate dataset configuration.
    
    Args:
        config: DatasetConfig instance to validate
    
    Raises:
        DatasetError: If configuration is invalid
    """
    # Check required fields
    if not config.name:
        raise DatasetError("Dataset name is required")

    if not config.database:
        raise DatasetError("Database configuration is required")

    # Validate database configuration
    if "type" not in config.database:
        raise DatasetError("Database type is required")

    db_type = config.database["type"]
    if db_type not in ["sqlite", "duckdb", "postgresql"]:
        raise DatasetError(f"Unsupported database type: {db_type}")

    # For file-based databases, check path
    if db_type in ["sqlite", "duckdb"]:
        if "path" not in config.database:
            raise DatasetError(f"Database path is required for {db_type}")

    # For PostgreSQL, check connection string
    if db_type == "postgresql":
        if "connection_string" not in config.database:
            raise DatasetError("Connection string is required for PostgreSQL")

    # Validate problem type if specified
    if config.problem_type:
        valid_types = [
            "binary_classification",
            "multiclass_classification",
            "regression",
            "clustering",
            "time_series"
        ]
        if config.problem_type not in valid_types:
            raise DatasetError(f"Invalid problem type: {config.problem_type}")

    # Validate table mappings
    if config.tables:
        valid_table_types = ["train", "test", "validation", "submission"]
        for table_type, table_name in config.tables.items():
            if table_type not in valid_table_types:
                raise DatasetError(f"Invalid table type: {table_type}")
            if not table_name:
                raise DatasetError(f"Table name cannot be empty for {table_type}")
