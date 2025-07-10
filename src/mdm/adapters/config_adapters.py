"""Configuration adapters for bridging old and new systems.

This module provides adapters that allow the old configuration system
to work with the new interface during migration.
"""
from typing import Any, Dict, Optional
from pathlib import Path
import logging

from mdm.interfaces.config import IConfiguration, IConfigurationManager
from mdm.config.config import ConfigManager, get_config as get_legacy_config
from mdm.models.config import MDMConfig as LegacyMDMConfig
from mdm.core.config_new import NewMDMConfig, get_new_config
from mdm.core import metrics_collector, feature_flags

logger = logging.getLogger(__name__)


class LegacyConfigAdapter(IConfiguration):
    """Adapter for legacy MDMConfig to match IConfiguration interface."""
    
    def __init__(self, config: LegacyMDMConfig, base_path: Path):
        """Initialize adapter with legacy config.
        
        Args:
            config: Legacy MDMConfig instance
            base_path: Base path for resolving relative paths
        """
        self._config = config
        self._base_path = base_path
        
    @property
    def home_dir(self) -> Path:
        """MDM home directory."""
        return self._base_path
    
    @property
    def config_dir(self) -> Path:
        """Configuration directory."""
        return self._config.get_full_path("configs_path", self._base_path)
    
    @property 
    def datasets_dir(self) -> Path:
        """Datasets storage directory."""
        return self._config.get_full_path("datasets_path", self._base_path)
    
    @property
    def cache_dir(self) -> Path:
        """Cache directory."""
        # Legacy doesn't have cache_dir, create one
        cache_path = self._base_path / "cache"
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory."""
        return self._config.get_full_path("logs_path", self._base_path)
    
    @property
    def default_backend(self) -> str:
        """Default database backend."""
        return self._config.database.default_backend
    
    @property
    def batch_size(self) -> int:
        """Batch processing size."""
        return self._config.performance.batch_size
    
    @property
    def enable_auto_detect(self) -> bool:
        """Enable automatic dataset detection."""
        # Legacy doesn't have this, default to True
        return True
    
    @property
    def enable_validation(self) -> bool:
        """Enable data validation during registration."""
        return self._config.validation.before_features.check_duplicates
    
    def get_full_path(self, path_type: str) -> Path:
        """Get full path for a path type."""
        # Map new path types to legacy
        mapping = {
            "datasets_path": "datasets_path",
            "config_path": "configs_path", 
            "cache_path": None,  # Not in legacy
            "logs_path": "logs_path",
        }
        
        legacy_key = mapping.get(path_type)
        if legacy_key is None:
            if path_type == "cache_path":
                return self.cache_dir
            raise ValueError(f"Unknown path type: {path_type}")
            
        return self._config.get_full_path(legacy_key, self._base_path)
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for path in [self.home_dir, self.config_dir, self.datasets_dir, 
                     self.cache_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.model_dump()


class NewConfigAdapter(IConfiguration):
    """Adapter for new MDMConfig to match IConfiguration interface."""
    
    def __init__(self, config: NewMDMConfig):
        """Initialize adapter with new config.
        
        Args:
            config: New MDMConfig instance
        """
        self._config = config
        
    @property
    def home_dir(self) -> Path:
        """MDM home directory."""
        return self._config.home_dir
    
    @property
    def config_dir(self) -> Path:
        """Configuration directory."""
        return self._config.config_dir
    
    @property
    def datasets_dir(self) -> Path:
        """Datasets storage directory."""
        return self._config.datasets_dir
    
    @property
    def cache_dir(self) -> Path:
        """Cache directory."""
        return self._config.cache_dir
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory."""
        return self._config.logs_dir
    
    @property
    def default_backend(self) -> str:
        """Default database backend."""
        return self._config.default_backend
    
    @property 
    def batch_size(self) -> int:
        """Batch processing size."""
        return self._config.chunk_size  # New uses chunk_size
    
    @property
    def enable_auto_detect(self) -> bool:
        """Enable automatic dataset detection."""
        return self._config.enable_auto_detect
    
    @property
    def enable_validation(self) -> bool:
        """Enable data validation during registration."""
        return self._config.enable_validation
    
    def get_full_path(self, path_type: str) -> Path:
        """Get full path for a path type."""
        return self._config.get_full_path(path_type)
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self._config.ensure_directories()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.model_dump()


class LegacyConfigManagerAdapter(IConfigurationManager):
    """Adapter for legacy ConfigManager."""
    
    def __init__(self, manager: Optional[ConfigManager] = None):
        """Initialize adapter.
        
        Args:
            manager: Optional ConfigManager instance
        """
        self._manager = manager or ConfigManager()
        self._config_cache: Optional[IConfiguration] = None
        
    def load(self) -> IConfiguration:
        """Load configuration from source."""
        with metrics_collector.timer("config.load", tags={"implementation": "legacy"}):
            legacy_config = self._manager.load()
            adapter = LegacyConfigAdapter(legacy_config, self._manager.base_path)
            self._config_cache = adapter
            return adapter
    
    def save(self, config: IConfiguration, path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        with metrics_collector.timer("config.save", tags={"implementation": "legacy"}):
            # Convert back to legacy format if needed
            if isinstance(config, LegacyConfigAdapter):
                self._manager.save(config._config, path)
            else:
                # Convert from dict
                config_dict = config.to_dict()
                legacy_config = LegacyMDMConfig(**config_dict)
                self._manager.save(legacy_config, path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        if self._config_cache is None:
            self._config_cache = self.load()
            
        # Navigate nested keys
        parts = key.split(".")
        value = self._config_cache.to_dict()
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        # Load current config
        if self._config_cache is None:
            self._config_cache = self.load()
            
        # This is a simplified implementation
        # In practice, we'd need to update the underlying config
        logger.warning(f"Legacy config adapter: set({key}, {value}) not fully implemented")
    
    def reload(self) -> None:
        """Reload configuration from source."""
        self._config_cache = None
        self.load()


class NewConfigManagerAdapter(IConfigurationManager):
    """Adapter for new configuration system."""
    
    def __init__(self):
        """Initialize adapter."""
        self._config_cache: Optional[IConfiguration] = None
        
    def load(self) -> IConfiguration:
        """Load configuration from source."""
        with metrics_collector.timer("config.load", tags={"implementation": "new"}):
            # Force reload to pick up env vars
            from mdm.core.config_new import reset_new_config
            reset_new_config()
            new_config = get_new_config()
            adapter = NewConfigAdapter(new_config)
            self._config_cache = adapter
            return adapter
    
    def save(self, config: IConfiguration, path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        with metrics_collector.timer("config.save", tags={"implementation": "new"}):
            # New system doesn't have save functionality yet
            # This would be implemented when needed
            logger.info(f"New config save not implemented yet: {path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        if self._config_cache is None:
            self._config_cache = self.load()
            
        # Navigate nested keys
        parts = key.split(".")
        value = self._config_cache.to_dict()
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        # New system uses environment variables
        # This is a placeholder
        logger.info(f"New config set: {key}={value} (use env vars)")
    
    def reload(self) -> None:
        """Reload configuration from source."""
        from mdm.core.config import reset_config
        reset_config()
        self._config_cache = None
        self.load()


def get_config_manager() -> IConfigurationManager:
    """Get configuration manager based on feature flag.
    
    Returns:
        Configuration manager adapter
    """
    if feature_flags.get("use_new_config", False):
        logger.debug("Using new configuration system")
        return NewConfigManagerAdapter()
    else:
        logger.debug("Using legacy configuration system") 
        return LegacyConfigManagerAdapter()


def get_config() -> IConfiguration:
    """Get current configuration.
    
    Returns:
        Configuration adapter
    """
    manager = get_config_manager()
    return manager.load()