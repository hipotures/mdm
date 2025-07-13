"""Utility functions for feature generation."""

from typing import Dict, Tuple
import pandas as pd
from loguru import logger


def check_signal(features: pd.DataFrame, descriptions: Dict[str, str], 
                min_signal_ratio: float = 0.01) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Check features for signal and remove low-variance features.
    
    Args:
        features: DataFrame with generated features
        descriptions: Dictionary mapping feature names to descriptions
        min_signal_ratio: Minimum unique value ratio for signal detection
        
    Returns:
        Tuple of (filtered features, filtered descriptions)
    """
    valid_features = []
    valid_descriptions = {}
    
    for col in features.columns:
        series = features[col]
        
        # Check for null values
        non_null_count = series.count()
        if non_null_count == 0:
            logger.debug(f"Dropping {col}: all null values")
            continue
        
        # Check if all values are the same (no signal)
        if series.nunique() == 1:
            logger.debug(f"Dropping {col}: all values are the same")
            continue
        
        # For numeric features, check variance
        if pd.api.types.is_numeric_dtype(series) and series.std() == 0:
            logger.debug(f"Dropping {col}: zero variance")
            continue
        
        # Keep feature
        valid_features.append(col)
        if col in descriptions:
            valid_descriptions[col] = descriptions[col]
    
    return features[valid_features], valid_descriptions