"""Configuration models for MDM."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PathsConfig(BaseModel):
    """Paths configuration."""

    datasets_path: str = Field(default="datasets/", description="Path to datasets")
    configs_path: str = Field(default="config/datasets/", description="Path to dataset configs")
    logs_path: str = Field(default="logs/", description="Path to logs")
    custom_features_path: str = Field(
        default="config/custom_features/", description="Path to custom features"
    )


class SQLAlchemyConfig(BaseModel):
    """SQLAlchemy configuration."""

    echo: bool = Field(default=False, description="Echo SQL statements")
    pool_size: int = Field(default=5, ge=1, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, description="Max overflow connections")
    pool_timeout: int = Field(default=30, ge=1, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, ge=-1, description="Connection recycle time")


class SQLiteConfig(BaseModel):
    """SQLite-specific configuration."""

    journal_mode: Literal["DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"] = Field(
        default="WAL", description="Journal mode"
    )
    synchronous: Literal["OFF", "NORMAL", "FULL", "EXTRA"] = Field(
        default="NORMAL", description="Synchronous mode"
    )
    cache_size: int = Field(default=-64000, description="Cache size (negative=KB, positive=pages)")
    temp_store: Literal["DEFAULT", "FILE", "MEMORY"] = Field(
        default="MEMORY", description="Temporary storage location"
    )
    mmap_size: int = Field(default=268435456, ge=0, description="Memory-mapped I/O size")


class DuckDBConfig(BaseModel):
    """DuckDB-specific configuration."""

    memory_limit: str = Field(default="8GB", description="Memory limit")
    threads: int = Field(default=4, ge=1, description="Number of threads")
    temp_directory: str = Field(default="/tmp", description="Temporary directory")
    access_mode: Literal["READ_WRITE", "READ_ONLY"] = Field(
        default="READ_WRITE", description="Access mode"
    )


class PostgreSQLConfig(BaseModel):
    """PostgreSQL-specific configuration."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    user: Optional[str] = Field(default=None, description="Database user")
    password: Optional[str] = Field(default=None, description="Database password")
    database_prefix: str = Field(default="mdm_", description="Database name prefix")
    sslmode: Literal[
        "disable", "allow", "prefer", "require", "verify-ca", "verify-full"
    ] = Field(default="prefer", description="SSL mode")
    pool_size: int = Field(default=10, ge=1, description="Connection pool size")


class DatabaseConfig(BaseModel):
    """Database configuration."""

    default_backend: Literal["sqlite", "duckdb", "postgresql"] = Field(
        default="sqlite", description="Default database backend"
    )
    connection_timeout: int = Field(default=30, ge=1, description="Connection timeout")
    sqlalchemy: SQLAlchemyConfig = Field(default_factory=SQLAlchemyConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    duckdb: DuckDBConfig = Field(default_factory=DuckDBConfig)
    postgresql: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)


class PerformanceConfig(BaseModel):
    """Performance configuration."""

    batch_size: int = Field(default=10000, ge=100, description="Batch processing size")
    max_concurrent_operations: int = Field(default=5, ge=1, description="Max concurrent operations")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    file: str = Field(default="mdm.log", description="Log file name")
    max_bytes: int = Field(default=10485760, ge=1, description="Max log file size")
    backup_count: int = Field(default=5, ge=0, description="Number of backup files")
    format: Literal["json", "console"] = Field(default="json", description="Log format")


class TemporalConfig(BaseModel):
    """Temporal feature configuration."""

    enabled: bool = Field(default=True, description="Enable temporal features")
    include_cyclical: bool = Field(default=True, description="Include cyclical features")
    include_lag: bool = Field(default=False, description="Include lag features")


class CategoricalConfig(BaseModel):
    """Categorical feature configuration."""

    enabled: bool = Field(default=True, description="Enable categorical features")
    max_cardinality: int = Field(default=50, ge=2, description="Max cardinality")
    min_frequency: float = Field(default=0.01, ge=0.0, le=1.0, description="Min frequency")


class StatisticalConfig(BaseModel):
    """Statistical feature configuration."""

    enabled: bool = Field(default=True, description="Enable statistical features")
    include_log: bool = Field(default=True, description="Include log transform")
    include_zscore: bool = Field(default=True, description="Include z-score")
    outlier_threshold: float = Field(default=3.0, ge=1.0, description="Outlier threshold")


class TextConfig(BaseModel):
    """Text feature configuration."""

    enabled: bool = Field(default=True, description="Enable text features")
    min_text_length: int = Field(default=20, ge=1, description="Min text length")


class BinningConfig(BaseModel):
    """Binning feature configuration."""

    enabled: bool = Field(default=True, description="Enable binning features")
    n_bins: list[int] = Field(default=[5, 10], description="Number of bins")

    @field_validator("n_bins")
    @classmethod
    def validate_n_bins(cls, v: list[int]) -> list[int]:
        """Validate n_bins."""
        if not v:
            raise ValueError("n_bins must not be empty")
        if any(b < 2 for b in v):
            raise ValueError("All bin counts must be >= 2")
        return v


class GenericFeaturesConfig(BaseModel):
    """Generic feature configuration."""

    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    categorical: CategoricalConfig = Field(default_factory=CategoricalConfig)
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    binning: BinningConfig = Field(default_factory=BinningConfig)


class TypeDetectionConfig(BaseModel):
    """Type detection configuration."""
    
    categorical_threshold: int = Field(
        default=20, 
        ge=2, 
        description="Max unique values to consider as categorical"
    )
    text_min_avg_length: int = Field(
        default=50,
        ge=1,
        description="Min average length to consider as text"
    )
    detect_datetime: bool = Field(
        default=True,
        description="Automatically detect datetime columns"
    )


class FeatureEngineeringConfig(BaseModel):
    """Feature engineering configuration."""

    enabled: bool = Field(default=True, description="Enable feature engineering")
    generic: GenericFeaturesConfig = Field(default_factory=GenericFeaturesConfig)
    type_detection: TypeDetectionConfig = Field(default_factory=TypeDetectionConfig)


class ExportConfig(BaseModel):
    """Export configuration."""

    default_format: Literal["csv", "parquet", "json"] = Field(
        default="csv", description="Default export format"
    )
    compression: Literal["none", "zip", "gzip", "snappy", "lz4"] = Field(
        default="zip", description="Compression type"
    )
    include_metadata: bool = Field(default=True, description="Include metadata in export")


class ValidationConfig(BaseModel):
    """Validation configuration."""

    check_duplicates: bool = Field(default=True, description="Check for duplicates")
    drop_duplicates: bool = Field(default=True, description="Drop duplicates")
    signal_detection: bool = Field(default=False, description="Check signal ratio")
    max_missing_ratio: float = Field(
        default=0.99, ge=0.0, le=1.0, description="Max missing data ratio"
    )
    min_rows: int = Field(default=10, ge=1, description="Minimum rows required")


class ValidationSettings(BaseModel):
    """Validation settings."""

    before_features: ValidationConfig = Field(
        default_factory=lambda: ValidationConfig(
            check_duplicates=True,
            drop_duplicates=True,
            signal_detection=False,
            max_missing_ratio=0.99,
            min_rows=10,
        )
    )
    after_features: ValidationConfig = Field(
        default_factory=lambda: ValidationConfig(
            check_duplicates=False,
            drop_duplicates=False,
            signal_detection=True,
            max_missing_ratio=0.99,
            min_rows=10,
        )
    )


class CLIConfig(BaseModel):
    """CLI configuration."""

    default_output_format: Literal["rich", "json", "csv", "table"] = Field(
        default="rich", description="Default output format"
    )
    page_size: int = Field(default=20, ge=1, description="Page size for listings")
    confirm_destructive: bool = Field(default=True, description="Confirm destructive operations")
    show_progress: bool = Field(default=True, description="Show progress bars")


class DevelopmentConfig(BaseModel):
    """Development configuration."""

    debug_mode: bool = Field(default=False, description="Debug mode")
    profile_operations: bool = Field(default=False, description="Profile operations")
    explain_queries: bool = Field(default=False, description="Explain SQL queries")


class MDMConfig(BaseModel):
    """Main MDM configuration."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig
    )
    export: ExportConfig = Field(default_factory=ExportConfig)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    cli: CLIConfig = Field(default_factory=CLIConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)

    def get_full_path(self, path_key: str, base_path: Path) -> Path:
        """Get full path from relative path."""
        relative_path: str = getattr(self.paths, path_key)
        return base_path / relative_path

