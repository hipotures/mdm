"""
Feature flags for gradual migration to new implementations.

This module provides a comprehensive feature flag system to control
which implementations are used during the migration period, including
support for gradual rollouts and A/B testing.
"""
import os
import json
import hashlib
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
from functools import wraps
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
        'enable_comparison_tests': True,  # Enable comparison testing
        'enable_memory_profiling': False,  # Enable memory profiling
        'auto_fallback': True,  # Automatically fallback on errors
        'rollout_percentage': {  # Gradual rollout percentages
            'new_backend': 0,
            'new_registrar': 0,
            'new_features': 0
        }
    }
    
    def __init__(self):
        """Initialize feature flags."""
        self._flags = self.DEFAULTS.copy()
        self._callbacks: Dict[str, List[Callable]] = {}
        self._history: List[Dict[str, Any]] = []
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
    
    def set(self, flag_name: str, value: Any):
        """
        Set feature flag value with history tracking.
        
        Args:
            flag_name: Name of the feature flag
            value: Value to set (bool, int, dict, etc.)
        """
        old_value = self._flags.get(flag_name)
        self._flags[flag_name] = value
        
        # Track change in history
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "flag": flag_name,
            "old_value": old_value,
            "new_value": value
        })
        
        # Notify callbacks
        for callback in self._callbacks.get(flag_name, []):
            try:
                callback(flag_name, old_value, value)
            except Exception as e:
                logger.error(f"Error in feature flag callback: {e}")
        
        logger.info(f"Feature flag '{flag_name}' changed: {old_value} -> {value}")
    
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
        
        data = {
            'flags': self._flags,
            'history': self._history[-100:],  # Keep last 100 changes
            'last_updated': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"Feature flags saved to {path}")
    
    def register_callback(self, flag_name: str, callback: Callable):
        """Register callback for flag changes."""
        if flag_name not in self._callbacks:
            self._callbacks[flag_name] = []
        self._callbacks[flag_name].append(callback)
        logger.debug(f"Registered callback for flag: {flag_name}")
    
    def is_enabled_for_user(self, feature: str, user_id: str) -> bool:
        """
        Check if feature is enabled for specific user (gradual rollout).
        
        Args:
            feature: Feature name (e.g., 'new_backend')
            user_id: User identifier
            
        Returns:
            True if feature should be enabled for this user
        """
        # Check if feature is fully enabled
        flag_name = f"use_{feature}"
        if self.get(flag_name, False):
            return True
        
        # Check percentage rollout
        rollout = self.get('rollout_percentage', {})
        percentage = rollout.get(feature, 0)
        
        if percentage <= 0:
            return False
        if percentage >= 100:
            return True
        
        # Use consistent hashing for deterministic assignment
        user_hash = int(hashlib.md5(f"{feature}:{user_id}".encode()).hexdigest()[:8], 16)
        return (user_hash % 100) < percentage
    
    def get_history(self, flag_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get flag change history."""
        history = self._history
        if flag_name:
            history = [h for h in history if h["flag"] == flag_name]
        return history[-limit:]
    
    def __repr__(self) -> str:
        """String representation of feature flags."""
        return f"FeatureFlags({self._flags})"


# Global instance
feature_flags = FeatureFlags()


# Decorator for feature-flagged functions
def feature_flag(flag_name: str, fallback: Optional[Callable] = None):
    """
    Decorator to conditionally execute based on feature flag.
    
    Args:
        flag_name: Name of the feature flag to check
        fallback: Optional fallback function if flag is disabled
        
    Example:
        @feature_flag('use_new_backend', fallback=old_backend_func)
        def new_backend_func():
            pass
    """
    def decorator(new_impl: Callable) -> Callable:
        @wraps(new_impl)
        def wrapper(*args, **kwargs):
            if feature_flags.get(flag_name):
                try:
                    return new_impl(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in new implementation ({flag_name}): {e}")
                    if fallback and feature_flags.get("auto_fallback", True):
                        logger.warning(f"Falling back to old implementation")
                        return fallback(*args, **kwargs)
                    raise
            elif fallback:
                return fallback(*args, **kwargs)
            else:
                raise NotImplementedError(
                    f"Feature '{flag_name}' is disabled and no fallback provided"
                )
        return wrapper
    return decorator


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