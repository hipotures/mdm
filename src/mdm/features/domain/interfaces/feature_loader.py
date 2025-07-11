"""Feature loader interface for domain layer."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd

from mdm.features.custom.base import BaseDomainFeatures


class IFeatureLoader(ABC):
    """Interface for loading features from various sources."""
    
    @abstractmethod
    def load_data(self, dataset_name: str, table_name: str) -> pd.DataFrame:
        """Load data for feature generation.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the table to load
            
        Returns:
            DataFrame with loaded data
        """
        pass


class ICustomFeatureLoader(ABC):
    """Interface for loading custom feature implementations."""
    
    @abstractmethod
    def load(self, dataset_name: str) -> Optional[BaseDomainFeatures]:
        """Load custom feature implementation for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Custom feature instance or None if not available
        """
        pass