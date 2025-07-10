"""New configuration system for MDM.

This module provides a complete configuration system using Pydantic Settings
with full feature parity to the legacy system but simplified structure.
"""
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import os
import yaml
import logging

from pydantic import Field, validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    default_backend: Literal["sqlite", "duckdb", "postgresql"] = Field(
        default="sqlite",
        description="Default database backend"
    )
    connection_timeout: int = Field(default=30, ge=1)
    
    # SQLite settings
    sqlite_journal_mode: str = Field(default="WAL")
    sqlite_synchronous: str = Field(default="NORMAL")
    sqlite_cache_size: int = Field(default=-64000)
    sqlite_temp_store: str = Field(default="MEMORY")
    sqlite_mmap_size: int = Field(default=268435456)
    
    # DuckDB settings
    duckdb_memory_limit: str = Field(default="8GB")
    duckdb_threads: int = Field(default=4)
    duckdb_temp_directory: str = Field(default="/tmp")
    duckdb_access_mode: str = Field(default="READ_WRITE")
    
    # PostgreSQL settings
    postgresql_host: str = Field(default="localhost")
    postgresql_port: int = Field(default=5432)
    postgresql_user: Optional[str] = Field(default=None)
    postgresql_password: Optional[str] = Field(default=None)
    postgresql_database_prefix: str = Field(default="mdm_")
    postgresql_sslmode: str = Field(default="prefer")
    postgresql_pool_size: int = Field(default=10)
    
    # SQLAlchemy settings
    sqlalchemy_echo: bool = Field(default=False)
    sqlalchemy_pool_size: int = Field(default=5)
    sqlalchemy_max_overflow: int = Field(default=10)
    sqlalchemy_pool_timeout: int = Field(default=30)
    sqlalchemy_pool_recycle: int = Field(default=3600)
    
    class Config:
        env_prefix = "MDM_DATABASE_"


class PerformanceSettings(BaseSettings):
    """Performance configuration."""
    
    batch_size: int = Field(default=10000, ge=100)
    max_concurrent_operations: int = Field(default=5, ge=1)
    chunk_size: int = Field(default=10000, ge=100)
    max_workers: int = Field(default=4, ge=1)
    
    class Config:
        env_prefix = "MDM_PERFORMANCE_"


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    file: Optional[str] = Field(default="mdm.log")
    max_bytes: int = Field(default=10485760, ge=1)
    backup_count: int = Field(default=5, ge=0)
    format: Literal["json", "console"] = Field(default="json")
    
    class Config:
        env_prefix = "MDM_LOGGING_"


class FeatureEngineeringSettings(BaseSettings):
    """Feature engineering configuration."""
    
    enabled: bool = Field(default=True)
    
    # Type detection
    categorical_threshold: int = Field(default=20, ge=2)
    text_min_avg_length: int = Field(default=50, ge=1)
    detect_datetime: bool = Field(default=True)
    
    # Temporal features
    temporal_enabled: bool = Field(default=True)
    temporal_include_cyclical: bool = Field(default=True)
    temporal_include_lag: bool = Field(default=False)
    
    # Categorical features
    categorical_enabled: bool = Field(default=True)
    categorical_max_cardinality: int = Field(default=50, ge=2)
    categorical_min_frequency: float = Field(default=0.01, ge=0.0, le=1.0)
    
    # Statistical features
    statistical_enabled: bool = Field(default=True)
    statistical_include_log: bool = Field(default=True)
    statistical_include_zscore: bool = Field(default=True)
    statistical_outlier_threshold: float = Field(default=3.0, ge=1.0)
    
    # Text features
    text_enabled: bool = Field(default=True)
    text_min_text_length: int = Field(default=20, ge=1)
    
    # Binning features
    binning_enabled: bool = Field(default=True)
    binning_n_bins: str = Field(default="5,10")  # Comma-separated
    
    @validator("binning_n_bins")
    def parse_binning_bins(cls, v):
        """Parse comma-separated bins."""
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return v
    
    class Config:
        env_prefix = "MDM_FEATURE_ENGINEERING_"


class ExportSettings(BaseSettings):
    """Export configuration."""
    
    default_format: Literal["csv", "parquet", "json"] = Field(default="csv")
    compression: Literal["none", "zip", "gzip", "snappy", "lz4"] = Field(
        default="zip"
    )
    include_metadata: bool = Field(default=True)
    
    class Config:
        env_prefix = "MDM_EXPORT_"


class ValidationSettings(BaseSettings):
    """Validation configuration."""
    
    # Before features
    before_check_duplicates: bool = Field(default=True)
    before_drop_duplicates: bool = Field(default=True)
    before_signal_detection: bool = Field(default=False)
    before_max_missing_ratio: float = Field(default=0.99, ge=0.0, le=1.0)
    before_min_rows: int = Field(default=10, ge=1)
    
    # After features
    after_check_duplicates: bool = Field(default=False)
    after_drop_duplicates: bool = Field(default=False) 
    after_signal_detection: bool = Field(default=True)
    after_max_missing_ratio: float = Field(default=0.99, ge=0.0, le=1.0)
    after_min_rows: int = Field(default=10, ge=1)
    
    class Config:
        env_prefix = "MDM_VALIDATION_"


class CLISettings(BaseSettings):
    """CLI configuration."""
    
    default_output_format: Literal["rich", "json", "csv", "table"] = Field(
        default="rich"
    )
    page_size: int = Field(default=20, ge=1)
    confirm_destructive: bool = Field(default=True)
    show_progress: bool = Field(default=True)
    
    class Config:
        env_prefix = "MDM_CLI_"


class DevelopmentSettings(BaseSettings):
    """Development configuration."""
    
    debug_mode: bool = Field(default=False)
    profile_operations: bool = Field(default=False)
    explain_queries: bool = Field(default=False)
    
    class Config:
        env_prefix = "MDM_DEVELOPMENT_"


class PathSettings(BaseSettings):
    """Path configuration."""
    
    datasets_path: str = Field(default="datasets/")
    configs_path: str = Field(default="config/datasets/")
    logs_path: str = Field(default="logs/")
    custom_features_path: str = Field(default="config/custom_features/")
    cache_path: str = Field(default="cache/")
    
    class Config:
        env_prefix = "MDM_PATHS_"


class NewMDMConfig(BaseSettings):
    """New unified MDM configuration.
    
    This configuration system provides:
    - Simplified structure compared to legacy
    - Full environment variable support
    - YAML file support
    - Backward compatibility through adapters
    """
    
    # Base configuration
    home_dir: Path = Field(
        default_factory=lambda: Path(os.environ.get("MDM_HOME_DIR", str(Path.home() / ".mdm"))),
        description="MDM home directory"
    )
    data_dir: Optional[Path] = Field(default=None)
    
    # Feature flags
    enable_auto_detect: bool = Field(default=True)
    enable_validation: bool = Field(default=True)
    
    # Sub-configurations
    paths: PathSettings = Field(default_factory=PathSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    feature_engineering: FeatureEngineeringSettings = Field(
        default_factory=FeatureEngineeringSettings
    )
    export: ExportSettings = Field(default_factory=ExportSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    cli: CLISettings = Field(default_factory=CLISettings)
    development: DevelopmentSettings = Field(default_factory=DevelopmentSettings)
    
    class Config:
        env_prefix = "MDM_"
        env_file = ".env"
        env_nested_delimiter = "__"  # Allows MDM_DATABASE__DEFAULT_BACKEND
        
    @classmethod
    def from_yaml(cls, path: Path) -> "NewMDMConfig":
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration instance
        """
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.dict(exclude_defaults=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    @property
    def config_dir(self) -> Path:
        """Configuration directory."""
        return self.home_dir / self.paths.configs_path
    
    @property
    def datasets_dir(self) -> Path:
        """Datasets directory."""
        if self.data_dir:
            return self.data_dir
        return self.home_dir / self.paths.datasets_path
    
    @property
    def cache_dir(self) -> Path:
        """Cache directory."""
        return self.home_dir / self.paths.cache_path
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory."""
        return self.home_dir / self.paths.logs_path
    
    @property
    def dataset_registry_dir(self) -> Path:
        """Dataset registry directory."""
        return self.config_dir
    
    @property
    def custom_features_dir(self) -> Path:
        """Custom features directory."""
        return self.home_dir / self.paths.custom_features_path
    
    # Compatibility properties
    @property
    def default_backend(self) -> str:
        """Default database backend."""
        return self.database.default_backend
    
    @property
    def batch_size(self) -> int:
        """Batch processing size."""
        return self.performance.batch_size
    
    @property
    def chunk_size(self) -> int:
        """Chunk size for processing."""
        return self.performance.chunk_size
    
    @property
    def max_workers(self) -> int:
        """Maximum worker threads."""
        return self.performance.max_workers
    
    def get_full_path(self, path_type: str) -> Path:
        """Get full path for a path type.
        
        Args:
            path_type: Type of path
            
        Returns:
            Full path
        """
        mapping = {
            "datasets_path": self.datasets_dir,
            "config_path": self.config_dir,
            "configs_path": self.config_dir,  # Legacy compatibility
            "cache_path": self.cache_dir,
            "logs_path": self.logs_dir,
            "custom_features_path": self.custom_features_dir,
        }
        
        if path_type in mapping:
            return mapping[path_type]
        raise ValueError(f"Unknown path type: {path_type}")
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        dirs = [
            self.home_dir,
            self.config_dir, 
            self.datasets_dir,
            self.cache_dir,
            self.logs_dir,
            self.custom_features_dir,
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def get_backend_config(self, backend: str) -> Dict[str, Any]:
        """Get configuration for specific backend.
        
        Args:
            backend: Backend name
            
        Returns:
            Backend configuration dict
        """
        if backend == "sqlite":
            return {
                "journal_mode": self.database.sqlite_journal_mode,
                "synchronous": self.database.sqlite_synchronous,
                "cache_size": self.database.sqlite_cache_size,
                "temp_store": self.database.sqlite_temp_store,
                "mmap_size": self.database.sqlite_mmap_size,
            }
        elif backend == "duckdb":
            return {
                "memory_limit": self.database.duckdb_memory_limit,
                "threads": self.database.duckdb_threads,
                "temp_directory": self.database.duckdb_temp_directory,
                "access_mode": self.database.duckdb_access_mode,
            }
        elif backend == "postgresql":
            return {
                "host": self.database.postgresql_host,
                "port": self.database.postgresql_port,
                "user": self.database.postgresql_user,
                "password": self.database.postgresql_password,
                "database_prefix": self.database.postgresql_database_prefix,
                "sslmode": self.database.postgresql_sslmode,
                "pool_size": self.database.postgresql_pool_size,
            }
        else:
            raise ValueError(f"Unknown backend: {backend}")


# Global instance management
_new_config: Optional[NewMDMConfig] = None


def get_new_config() -> NewMDMConfig:
    """Get new configuration instance.
    
    Returns:
        Configuration instance
    """
    global _new_config
    if _new_config is None:
        # Always create from constructor to pick up env vars
        # YAML loading is handled by pydantic_settings if file exists
        logger.info("Loading configuration")
        _new_config = NewMDMConfig()
        _new_config.ensure_directories()
    
    return _new_config


def reset_new_config() -> None:
    """Reset configuration (for testing)."""
    global _new_config
    _new_config = None