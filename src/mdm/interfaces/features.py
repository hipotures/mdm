"""
Feature engineering interfaces based on actual usage analysis.

This interface is designed to match the existing FeatureGenerator API
to ensure compatibility during migration.
"""
from typing import Protocol, Dict, Any, List, Optional, runtime_checkable
import pandas as pd


@runtime_checkable
class IFeatureTransformer(Protocol):
    """Individual feature transformer interface."""
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit transformer to data."""
        ...
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        ...
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        ...
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        ...


@runtime_checkable
class IFeatureGenerator(Protocol):
    """Feature generation interface based on actual implementation."""
    
    def generate_features(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
        datetime_columns: Optional[List[str]] = None,
        id_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate all features for dataset.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            problem_type: Type of ML problem
            datetime_columns: List of datetime columns
            id_columns: List of ID columns to exclude
            
        Returns:
            DataFrame with generated features
        """
        ...
    
    def generate_numeric_features(
        self, 
        data: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate numeric features."""
        ...
    
    def generate_categorical_features(
        self, 
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate categorical features."""
        ...
    
    def generate_datetime_features(
        self, 
        data: pd.DataFrame,
        datetime_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate datetime features."""
        ...
    
    def generate_text_features(
        self, 
        data: pd.DataFrame,
        text_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate text features."""
        ...
    
    def generate_interaction_features(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        max_interactions: int = 2
    ) -> pd.DataFrame:
        """Generate interaction features between columns."""
        ...
    
    def get_feature_importance(
        self, 
        features: pd.DataFrame,
        target: pd.Series,
        problem_type: str
    ) -> pd.DataFrame:
        """Calculate feature importance scores."""
        ...