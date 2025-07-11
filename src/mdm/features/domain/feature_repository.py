"""Repository interface for feature domain."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

from .value_objects import FeatureSet, FeatureMetadata


class IFeatureRepository(ABC):
    """Repository interface for feature storage and retrieval."""
    
    @abstractmethod
    def save_features(self, dataset_name: str, table_name: str, 
                     feature_set: FeatureSet) -> None:
        """Save a feature set to storage.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the feature table
            feature_set: FeatureSet to save
        """
        pass
    
    @abstractmethod
    def load_features(self, dataset_name: str, table_name: str) -> Optional[FeatureSet]:
        """Load a feature set from storage.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the feature table
            
        Returns:
            FeatureSet if found, None otherwise
        """
        pass
    
    @abstractmethod
    def save_metadata(self, dataset_name: str, metadata: Dict[str, FeatureMetadata]) -> None:
        """Save feature metadata.
        
        Args:
            dataset_name: Name of the dataset
            metadata: Dictionary of feature metadata
        """
        pass
    
    @abstractmethod
    def load_metadata(self, dataset_name: str) -> Dict[str, FeatureMetadata]:
        """Load feature metadata.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary of feature metadata
        """
        pass
    
    @abstractmethod
    def list_feature_tables(self, dataset_name: str) -> List[str]:
        """List all feature tables for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of feature table names
        """
        pass
    
    @abstractmethod
    def delete_features(self, dataset_name: str, table_name: str) -> bool:
        """Delete a feature table.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the feature table
            
        Returns:
            True if deleted, False if not found
        """
        pass