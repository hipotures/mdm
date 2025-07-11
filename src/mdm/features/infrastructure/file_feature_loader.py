"""File-based implementation of custom feature loader."""

import importlib.util
from pathlib import Path
from typing import Optional

from loguru import logger

from mdm.config import get_config_manager
from mdm.features.custom.base import BaseDomainFeatures
from mdm.features.domain.interfaces import ICustomFeatureLoader


class FileFeatureLoader(ICustomFeatureLoader):
    """Loads custom feature implementations from Python files."""
    
    def __init__(self, custom_features_path: Optional[Path] = None):
        """Initialize the file feature loader.
        
        Args:
            custom_features_path: Path to custom features directory
        """
        if custom_features_path is None:
            config_manager = get_config_manager()
            self.base_path = config_manager.base_path
            self.custom_features_path = (
                self.base_path / config_manager.config.paths.custom_features_path
            )
        else:
            self.custom_features_path = custom_features_path
    
    def load(self, dataset_name: str) -> Optional[BaseDomainFeatures]:
        """Load custom feature implementation for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Custom feature instance or None if not available
        """
        custom_features_file = self.custom_features_path / f"{dataset_name}.py"
        
        if not custom_features_file.exists():
            logger.debug(f"No custom features file found at {custom_features_file}")
            return None
        
        logger.info(f"Loading custom features from {custom_features_file}")
        
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(
                f"custom_features_{dataset_name}", 
                custom_features_file
            )
            
            if not spec or not spec.loader:
                logger.error(f"Failed to create module spec for {custom_features_file}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the custom feature class
            custom_feature_class = self._find_custom_feature_class(module)
            
            if custom_feature_class:
                # Create and return instance
                instance = custom_feature_class(dataset_name)
                
                # Log registered operations if available
                if hasattr(instance, '_operation_registry'):
                    operations = list(instance._operation_registry.keys())
                    logger.debug(f"Custom operations for {dataset_name}: {operations}")
                
                return instance
            else:
                logger.warning(f"No BaseDomainFeatures subclass found in {custom_features_file}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load custom features for {dataset_name}: {e}")
            return None
    
    def _find_custom_feature_class(self, module) -> Optional[type]:
        """Find the custom feature class in a module.
        
        Args:
            module: Python module to search
            
        Returns:
            Custom feature class or None
        """
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            # Check if it's a class that inherits from BaseDomainFeatures
            if (isinstance(attr, type) and 
                issubclass(attr, BaseDomainFeatures) and 
                attr is not BaseDomainFeatures):
                
                logger.debug(f"Found custom feature class: {attr_name}")
                return attr
        
        return None