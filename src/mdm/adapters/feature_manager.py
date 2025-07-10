"""
Feature engineering manager with feature flag support.

This module provides centralized management of feature engineering components
with support for switching between legacy and new implementations.
"""
from typing import Dict, Any, Optional
import logging

from mdm.core import feature_flags
from mdm.interfaces.features import IFeatureGenerator
from mdm.adapters.feature_adapters import FeatureGeneratorAdapter

logger = logging.getLogger(__name__)

# Global cache for feature generator instances
_generator_cache: Dict[str, IFeatureGenerator] = {}


class FeatureEngineeringManager:
    """Manages feature engineering components with feature flag support."""
    
    def __init__(self):
        """Initialize the feature engineering manager."""
        self._generators: Dict[str, IFeatureGenerator] = {}
        logger.info("Initialized FeatureEngineeringManager")
    
    def get_generator(self, force_new: Optional[bool] = None) -> IFeatureGenerator:
        """Get feature generator instance.
        
        Args:
            force_new: Force use of new implementation (overrides feature flag)
            
        Returns:
            Feature generator instance
        """
        # Determine which implementation to use
        use_new = force_new if force_new is not None else feature_flags.get("use_new_features", False)
        
        cache_key = "new" if use_new else "legacy"
        
        # Check cache
        if cache_key in self._generators:
            return self._generators[cache_key]
        
        # Create appropriate generator
        if use_new:
            logger.info("Creating new feature generator implementation")
            # Import here to avoid circular imports
            from mdm.core.features.generator import NewFeatureGenerator
            generator = NewFeatureGenerator()
        else:
            logger.info("Creating legacy feature generator adapter")
            generator = FeatureGeneratorAdapter()
        
        # Cache and return
        self._generators[cache_key] = generator
        return generator
    
    def clear_cache(self):
        """Clear cached generator instances."""
        self._generators.clear()
        _generator_cache.clear()
        logger.debug("Cleared feature generator cache")


# Global manager instance
_manager = FeatureEngineeringManager()


def get_feature_generator(force_new: Optional[bool] = None) -> IFeatureGenerator:
    """Get feature generator instance with feature flag support.
    
    Args:
        force_new: Force use of new implementation (overrides feature flag)
        
    Returns:
        Feature generator instance (legacy or new based on feature flag)
    """
    return _manager.get_generator(force_new)


def clear_feature_cache():
    """Clear cached feature generator instances."""
    _manager.clear_cache()