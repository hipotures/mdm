"""Base class for global feature operations that work on entire datasets."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd


class GlobalFeatureOperation(ABC):
    """Base class for feature operations that work on entire datasets.
    
    Unlike column-based operations, these work on the entire DataFrame
    to capture relationships between columns.
    """
    
    def __init__(self, min_signal_ratio: float = 0.01):
        """Initialize global feature operation.
        
        Args:
            min_signal_ratio: Minimum unique value ratio for signal detection
        """
        self.min_signal_ratio = min_signal_ratio
    
    @abstractmethod
    def get_applicable_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns this transformer can process.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of applicable column names
        """
        pass
    
    @abstractmethod
    def generate_features(
        self, 
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate features for the entire dataset.
        
        Args:
            df: Input dataframe
            target_column: Target column to exclude
            id_columns: ID columns to exclude
            
        Returns:
            Tuple of (features dataframe, feature descriptions)
        """
        pass