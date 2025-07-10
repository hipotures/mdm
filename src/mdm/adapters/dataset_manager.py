"""Dataset registration management with feature flag support.

This module provides the main entry points for dataset registration operations,
automatically switching between legacy and new implementations based on feature flags.
"""
import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache

from ..core import feature_flags
from ..interfaces.dataset import IDatasetRegistrar, IDatasetManager
from .dataset_adapters import DatasetRegistrarAdapter, DatasetManagerAdapter

logger = logging.getLogger(__name__)


class DatasetRegistrationManager:
    """Manages dataset registration instances with feature flag support."""
    
    def __init__(self):
        self._legacy_registrar: Optional[IDatasetRegistrar] = None
        self._new_registrar: Optional[IDatasetRegistrar] = None
        self._legacy_manager: Optional[IDatasetManager] = None
        self._new_manager: Optional[IDatasetManager] = None
        logger.info("Initialized DatasetRegistrationManager")
    
    def get_registrar(self, force_new: Optional[bool] = None) -> IDatasetRegistrar:
        """Get dataset registrar instance based on feature flags.
        
        Args:
            force_new: Force use of new implementation (overrides feature flag)
            
        Returns:
            Dataset registrar instance
        """
        # Determine which implementation to use
        use_new = force_new if force_new is not None else feature_flags.get("use_new_dataset_registration", False)
        
        if use_new:
            if self._new_registrar is None:
                # Import here to avoid circular imports
                try:
                    from ..core.dataset.registrar import NewDatasetRegistrar
                    self._new_registrar = NewDatasetRegistrar()
                    logger.info("Created new dataset registrar instance")
                except ImportError:
                    logger.warning("New dataset registrar not implemented yet, falling back to legacy")
                    use_new = False
            
            if use_new and self._new_registrar:
                logger.debug("Using new dataset registrar")
                return self._new_registrar
        
        # Use legacy implementation
        if self._legacy_registrar is None:
            self._legacy_registrar = DatasetRegistrarAdapter()
            logger.info("Created legacy dataset registrar adapter")
        
        logger.debug("Using legacy dataset registrar")
        return self._legacy_registrar
    
    def get_manager(self, force_new: Optional[bool] = None) -> IDatasetManager:
        """Get dataset manager instance based on feature flags.
        
        Args:
            force_new: Force use of new implementation (overrides feature flag)
            
        Returns:
            Dataset manager instance
        """
        # Determine which implementation to use
        use_new = force_new if force_new is not None else feature_flags.get("use_new_dataset_registration", False)
        
        if use_new:
            if self._new_manager is None:
                # Import here to avoid circular imports
                try:
                    from ..core.dataset.manager import NewDatasetManager
                    self._new_manager = NewDatasetManager()
                    logger.info("Created new dataset manager instance")
                except ImportError:
                    logger.warning("New dataset manager not implemented yet, falling back to legacy")
                    use_new = False
            
            if use_new and self._new_manager:
                logger.debug("Using new dataset manager")
                return self._new_manager
        
        # Use legacy implementation
        if self._legacy_manager is None:
            self._legacy_manager = DatasetManagerAdapter()
            logger.info("Created legacy dataset manager adapter")
        
        logger.debug("Using legacy dataset manager")
        return self._legacy_manager
    
    def clear_cache(self):
        """Clear cached instances."""
        self._legacy_registrar = None
        self._new_registrar = None
        self._legacy_manager = None
        self._new_manager = None
        logger.info("Cleared dataset registration cache")


# Global manager instance
_manager = DatasetRegistrationManager()


def get_dataset_registrar(force_new: Optional[bool] = None) -> IDatasetRegistrar:
    """Get dataset registrar instance with feature flag support.
    
    This is the main entry point for getting a dataset registrar.
    
    Args:
        force_new: Force use of new implementation (overrides feature flag)
        
    Returns:
        Dataset registrar instance (legacy or new based on feature flags)
        
    Example:
        ```python
        from mdm.adapters import get_dataset_registrar
        
        # Use implementation based on feature flag
        registrar = get_dataset_registrar()
        
        # Force new implementation
        registrar = get_dataset_registrar(force_new=True)
        ```
    """
    return _manager.get_registrar(force_new)


def get_dataset_manager(force_new: Optional[bool] = None) -> IDatasetManager:
    """Get dataset manager instance with feature flag support.
    
    This is the main entry point for getting a dataset manager.
    
    Args:
        force_new: Force use of new implementation (overrides feature flag)
        
    Returns:
        Dataset manager instance (legacy or new based on feature flags)
        
    Example:
        ```python
        from mdm.adapters import get_dataset_manager
        
        # Use implementation based on feature flag
        manager = get_dataset_manager()
        
        # Force new implementation
        manager = get_dataset_manager(force_new=True)
        ```
    """
    return _manager.get_manager(force_new)


def clear_dataset_cache():
    """Clear cached dataset instances.
    
    This forces recreation of instances on next access.
    """
    _manager.clear_cache()
    logger.info("Dataset cache cleared")


def get_registration_metrics() -> Dict[str, Any]:
    """Get metrics from dataset registration operations.
    
    Returns:
        Dict with metrics from both registrar and manager
    """
    metrics = {
        "registrar": {},
        "manager": {}
    }
    
    # Get registrar metrics if available
    registrar = _manager._legacy_registrar or _manager._new_registrar
    if registrar and hasattr(registrar, 'get_metrics'):
        metrics["registrar"] = registrar.get_metrics()
    
    # Get manager metrics if available
    manager = _manager._legacy_manager or _manager._new_manager
    if manager and hasattr(manager, 'get_metrics'):
        metrics["manager"] = manager.get_metrics()
    
    return metrics
