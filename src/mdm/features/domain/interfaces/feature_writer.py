"""Feature writer interface for domain layer."""

from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd


class IFeatureWriter(ABC):
    """Interface for writing generated features to storage."""
    
    @abstractmethod
    def write_features(self, dataset_name: str, table_name: str, 
                      features: pd.DataFrame, mode: str = "replace") -> None:
        """Write generated features to storage.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the feature table
            features: DataFrame containing features
            mode: Write mode ('replace' or 'append')
        """
        pass
    
    @abstractmethod
    def write_features_batch(self, dataset_name: str, table_name: str,
                           features: pd.DataFrame, batch_number: int,
                           is_first_batch: bool = False) -> None:
        """Write features in batches for large datasets.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the feature table
            features: DataFrame containing features for this batch
            batch_number: Current batch number
            is_first_batch: Whether this is the first batch (for table creation)
        """
        pass