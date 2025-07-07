"""MDM Configuration management."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MDMConfig(BaseSettings):
    """MDM Configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="MDM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Base directories
    home_dir: Path = Field(
        default_factory=lambda: Path.home() / ".mdm",
        description="MDM home directory"
    )
    data_dir: Optional[Path] = Field(
        default=None,
        description="Default data directory"
    )

    # Database settings
    default_backend: str = Field(
        default="duckdb",
        description="Default database backend (sqlite, duckdb, postgresql)"
    )

    # Feature settings
    enable_auto_detect: bool = Field(
        default=True,
        description="Enable automatic dataset detection"
    )
    enable_validation: bool = Field(
        default=True,
        description="Enable data validation during registration"
    )

    # Performance settings
    chunk_size: int = Field(
        default=10000,
        description="Default chunk size for data processing"
    )
    max_workers: int = Field(
        default=4,
        description="Maximum number of worker threads"
    )

    @property
    def config_dir(self) -> Path:
        """Get configuration directory."""
        return self.home_dir / "config"

    @property
    def datasets_dir(self) -> Path:
        """Get datasets directory."""
        if self.data_dir:
            return self.data_dir
        return self.home_dir / "datasets"

    @property
    def dataset_registry_dir(self) -> Path:
        """Get dataset registry directory."""
        return self.config_dir / "datasets"

    @property
    def cache_dir(self) -> Path:
        """Get cache directory."""
        return self.home_dir / "cache"

    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        return self.home_dir / "logs"

    def get_full_path(self, path_type: str) -> Path:
        """Get full path for a path type.

        Args:
            path_type: Type of path ('datasets_path', 'config_path', etc.)

        Returns:
            Full path
        """
        if path_type == "datasets_path":
            return self.datasets_dir
        elif path_type == "config_path":
            return self.config_dir
        elif path_type == "cache_path":
            return self.cache_dir
        elif path_type == "logs_path":
            return self.logs_dir
        else:
            raise ValueError(f"Unknown path type: {path_type}")

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for dir_path in [
            self.home_dir,
            self.config_dir,
            self.datasets_dir,
            self.dataset_registry_dir,
            self.cache_dir,
            self.logs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Optional[MDMConfig] = None


def get_config() -> MDMConfig:
    """Get MDM configuration instance.
    
    Returns:
        MDMConfig instance
    """
    global _config
    if _config is None:
        _config = MDMConfig()
        _config.ensure_directories()
    return _config


def reset_config() -> None:
    """Reset configuration (mainly for testing)."""
    global _config
    _config = None
