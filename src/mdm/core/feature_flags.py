"""
Feature flags for gradual migration to new implementations.

This module provides a simple feature flag system to control
which implementations are used during the migration period.
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class FeatureFlags:
    """
    Manages feature flags for MDM refactoring.
    
    Flags can be set via:
    1. Environment variables (highest priority)
    2. Config file (~/.mdm/feature_flags.yaml)
    3. Default values (lowest priority)
    """
    
    # Default feature flag values
    DEFAULTS = {
        'use_new_backend': False,  # Use stateless backends
        'use_new_registrar': False,  # Use refactored registrar
        'use_new_features': False,  # Use new feature engineering
        'enable_compatibility_warnings': True,  # Log compatibility method usage
        'enable_performance_tracking': False,  # Track performance metrics
    }
    
    def __init__(self):
        """Initialize feature flags."""
        self._flags = self.DEFAULTS.copy()
        self._load_from_file()
        self._load_from_env()
        
        logger.info(f"Feature flags initialized: {self._flags}")
    
    def _load_from_file(self):
        """Load flags from config file if it exists."""
        config_path = Path.home() / '.mdm' / 'feature_flags.yaml'
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    file_flags = yaml.safe_load(f) or {}
                
                # Update with valid flags from file
                for key, value in file_flags.items():
                    if key in self.DEFAULTS:
                        self._flags[key] = bool(value)
                        logger.debug(f"Loaded flag from file: {key}={value}")
                    else:
                        logger.warning(f"Unknown feature flag in config: {key}")
                        
            except Exception as e:
                logger.error(f"Failed to load feature flags from {config_path}: {e}")
    
    def _load_from_env(self):
        """Load flags from environment variables."""
        # Environment variables override file settings
        # Format: MDM_FEATURE_USE_NEW_BACKEND=true
        
        for key in self.DEFAULTS:
            env_var = f"MDM_FEATURE_{key.upper()}"
            value = os.environ.get(env_var)
            
            if value is not None:
                # Parse boolean from string
                self._flags[key] = value.lower() in ('true', '1', 'yes', 'on')
                logger.debug(f"Loaded flag from env: {key}={self._flags[key]}")
    
    def get(self, flag_name: str, default: Optional[bool] = None) -> bool:
        """
        Get value of a feature flag.
        
        Args:
            flag_name: Name of the feature flag
            default: Default value if flag not found
            
        Returns:
            Boolean value of the flag
        """
        return self._flags.get(flag_name, default if default is not None else False)
    
    def set(self, flag_name: str, value: bool):
        """
        Set feature flag value (runtime only, not persisted).
        
        Args:
            flag_name: Name of the feature flag
            value: Boolean value to set
        """
        if flag_name in self.DEFAULTS:
            old_value = self._flags[flag_name]
            self._flags[flag_name] = bool(value)
            logger.info(f"Feature flag '{flag_name}' changed: {old_value} -> {value}")
        else:
            logger.warning(f"Attempt to set unknown feature flag: {flag_name}")
    
    def get_all(self) -> Dict[str, bool]:
        """Get all feature flag values."""
        return self._flags.copy()
    
    def save_to_file(self, path: Optional[Path] = None):
        """
        Save current flags to file.
        
        Args:
            path: Optional path to save to (default: ~/.mdm/feature_flags.yaml)
        """
        if path is None:
            path = Path.home() / '.mdm' / 'feature_flags.yaml'
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self._flags, f, default_flow_style=False)
        
        logger.info(f"Feature flags saved to {path}")
    
    def __repr__(self) -> str:
        """String representation of feature flags."""
        return f"FeatureFlags({self._flags})"


# Global instance
feature_flags = FeatureFlags()


# Convenience functions
def is_new_backend_enabled() -> bool:
    """Check if new stateless backends are enabled."""
    return feature_flags.get('use_new_backend')


def is_new_registrar_enabled() -> bool:
    """Check if new registrar is enabled."""
    return feature_flags.get('use_new_registrar')


def is_new_features_enabled() -> bool:
    """Check if new feature engineering is enabled."""
    return feature_flags.get('use_new_features')


def enable_new_backend():
    """Enable new backend for testing."""
    feature_flags.set('use_new_backend', True)


def disable_new_backend():
    """Disable new backend (use legacy)."""
    feature_flags.set('use_new_backend', False)