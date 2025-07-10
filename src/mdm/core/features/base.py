"""
Base classes for new feature engineering system.

This module provides the foundation for the plugin-based feature engineering
architecture with improved separation of concerns.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util
import logging

from mdm.interfaces.features import IFeatureTransformer

logger = logging.getLogger(__name__)


class FeatureTransformer(ABC, IFeatureTransformer):
    """Base class for all feature transformers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize transformer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._fitted = False
        self._feature_names: List[str] = []
        self._fit_state: Dict[str, Any] = {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get transformer name."""
        pass
    
    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """Get list of supported data types."""
        pass
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit transformer to data.
        
        Args:
            data: Input DataFrame
        """
        logger.debug(f"Fitting {self.name} transformer")
        self._fit_state = self._fit_impl(data)
        self._fitted = True
    
    @abstractmethod
    def _fit_impl(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Implementation-specific fit logic.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Fit state dictionary
        """
        pass
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self._fitted:
            logger.warning(f"{self.name} transformer not fitted, fitting now")
            self.fit(data)
        
        logger.debug(f"Transforming data with {self.name}")
        result = self._transform_impl(data)
        
        # Track generated feature names
        new_cols = [col for col in result.columns if col not in data.columns]
        self._feature_names = new_cols
        
        return result
    
    @abstractmethod
    def _transform_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Implementation-specific transform logic.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        self.fit(data)
        return self.transform(data)
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names.
        
        Returns:
            List of feature names
        """
        return self._feature_names.copy()
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters.
        
        Returns:
            Parameter dictionary
        """
        return {
            "name": self.name,
            "config": self.config,
            "fitted": self._fitted,
            "fit_state": self._fit_state
        }


class TransformerRegistry:
    """Registry for feature transformers."""
    
    def __init__(self):
        """Initialize registry."""
        self._transformers: Dict[str, Type[FeatureTransformer]] = {}
        self._instances: Dict[str, FeatureTransformer] = {}
        logger.info("Initialized TransformerRegistry")
    
    def register(self, transformer_class: Type[FeatureTransformer]) -> None:
        """Register a transformer class.
        
        Args:
            transformer_class: Transformer class to register
        """
        name = transformer_class().name
        self._transformers[name] = transformer_class
        logger.debug(f"Registered transformer: {name}")
    
    def get(self, name: str, config: Optional[Dict[str, Any]] = None) -> FeatureTransformer:
        """Get transformer instance by name.
        
        Args:
            name: Transformer name
            config: Optional configuration
            
        Returns:
            Transformer instance
            
        Raises:
            KeyError: If transformer not found
        """
        if name not in self._transformers:
            raise KeyError(f"Transformer '{name}' not registered")
        
        # Create new instance with config
        key = f"{name}_{hash(str(config))}"
        if key not in self._instances:
            self._instances[key] = self._transformers[name](config)
        
        return self._instances[key]
    
    def list_transformers(self) -> List[str]:
        """List all registered transformer names.
        
        Returns:
            List of transformer names
        """
        return list(self._transformers.keys())
    
    def get_by_type(self, data_type: str) -> List[FeatureTransformer]:
        """Get all transformers that support a data type.
        
        Args:
            data_type: Data type to filter by
            
        Returns:
            List of compatible transformers
        """
        compatible = []
        
        for name, transformer_class in self._transformers.items():
            instance = self.get(name)
            if data_type in instance.supported_types:
                compatible.append(instance)
        
        return compatible
    
    def load_plugins(self, plugin_dir: Path) -> None:
        """Load transformer plugins from directory.
        
        Args:
            plugin_dir: Directory containing plugin modules
        """
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            return
        
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            try:
                # Load module
                spec = importlib.util.spec_from_file_location(
                    f"mdm_plugin_{plugin_file.stem}",
                    plugin_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find and register transformer classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, FeatureTransformer) and
                            attr != FeatureTransformer):
                            self.register(attr)
                            logger.info(f"Loaded plugin transformer: {attr().name}")
            
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
    
    def clear(self) -> None:
        """Clear all registered transformers and instances."""
        self._transformers.clear()
        self._instances.clear()
        logger.debug("Cleared transformer registry")


# Global registry instance
transformer_registry = TransformerRegistry()