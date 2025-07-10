"""Configuration interface for MDM.

This module defines the protocol for configuration systems, allowing
both old and new implementations to coexist during migration.
"""
from typing import Protocol, Any, Dict, Optional, runtime_checkable
from pathlib import Path


@runtime_checkable
class IConfiguration(Protocol):
    """Protocol for configuration systems."""
    
    # Core paths
    @property
    def home_dir(self) -> Path:
        """MDM home directory."""
        ...
    
    @property
    def config_dir(self) -> Path:
        """Configuration directory."""
        ...
    
    @property
    def datasets_dir(self) -> Path:
        """Datasets storage directory."""
        ...
    
    @property
    def cache_dir(self) -> Path:
        """Cache directory."""
        ...
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory."""
        ...
    
    # Database settings
    @property
    def default_backend(self) -> str:
        """Default database backend (sqlite, duckdb, postgresql)."""
        ...
    
    @property
    def batch_size(self) -> int:
        """Batch processing size."""
        ...
    
    # Feature settings
    @property
    def enable_auto_detect(self) -> bool:
        """Enable automatic dataset detection."""
        ...
    
    @property
    def enable_validation(self) -> bool:
        """Enable data validation during registration."""
        ...
    
    # Methods
    def get_full_path(self, path_type: str) -> Path:
        """Get full path for a path type."""
        ...
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        ...


@runtime_checkable 
class IConfigurationManager(Protocol):
    """Protocol for configuration managers."""
    
    def load(self) -> IConfiguration:
        """Load configuration from source."""
        ...
    
    def save(self, config: IConfiguration, path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        ...
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        ...
    
    def reload(self) -> None:
        """Reload configuration from source."""
        ...