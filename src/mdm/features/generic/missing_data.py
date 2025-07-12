"""Missing data pattern features generator."""

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from mdm.features.base_global import GlobalFeatureOperation
from mdm.features.utils import check_signal


class MissingDataFeatures(GlobalFeatureOperation):
    """Generate features from missing data patterns.
    
    This feature generator extracts valuable information from the patterns
    of missing values in the dataset, which can be predictive in many
    real-world scenarios.
    """
    
    def __init__(self, min_missing_ratio: float = 0.01, max_missing_ratio: float = 0.95):
        """Initialize missing data feature generator.
        
        Args:
            min_missing_ratio: Minimum ratio of missing values to consider column
            max_missing_ratio: Maximum ratio of missing values to consider column
        """
        super().__init__()
        self.min_missing_ratio = min_missing_ratio
        self.max_missing_ratio = max_missing_ratio
        
    def get_applicable_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns with missing values within specified range.
        
        Args:
            df: Input dataframe
            
        Returns:
            List of column names with missing values
        """
        missing_ratios = df.isnull().mean()
        applicable = missing_ratios[
            (missing_ratios >= self.min_missing_ratio) & 
            (missing_ratios <= self.max_missing_ratio)
        ].index.tolist()
        
        logger.debug(f"Found {len(applicable)} columns with missing values in range "
                    f"[{self.min_missing_ratio}, {self.max_missing_ratio}]")
        return applicable
    
    def _generate_column_features(
        self, 
        df: pd.DataFrame, 
        column: str
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate missing data features for a single column.
        
        Args:
            df: Input dataframe
            column: Column name
            
        Returns:
            Tuple of (features dataframe, feature descriptions)
        """
        features = pd.DataFrame(index=df.index)
        descriptions = {}
        
        # Binary indicator for missing values
        is_missing = df[column].isnull().astype(int)
        features[f'{column}_is_missing'] = is_missing
        descriptions[f'{column}_is_missing'] = f"Binary indicator for missing {column}"
        
        return features, descriptions
    
    def generate_features(
        self, 
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate all missing data features.
        
        Args:
            df: Input dataframe
            target_column: Target column to exclude
            id_columns: ID columns to exclude
            
        Returns:
            Tuple of (features dataframe, feature descriptions)
        """
        features = pd.DataFrame(index=df.index)
        descriptions = {}
        
        # Get columns with missing values
        missing_cols = self.get_applicable_columns(df)
        
        # Generate column-specific features
        for col in missing_cols:
            if col != target_column and col not in (id_columns or []):
                col_features, col_descriptions = self._generate_column_features(df, col)
                features = pd.concat([features, col_features], axis=1)
                descriptions.update(col_descriptions)
        
        # Row-level missing features
        if missing_cols:
            # Total missing count per row
            features['total_missing_count'] = df[missing_cols].isnull().sum(axis=1)
            descriptions['total_missing_count'] = "Total count of missing values per row"
            
            # Missing ratio per row
            features['missing_ratio'] = features['total_missing_count'] / len(missing_cols)
            descriptions['missing_ratio'] = "Proportion of missing values per row"
            
            # Consecutive missing count
            def count_consecutive_missing(row):
                """Count maximum consecutive missing values."""
                missing = row.isnull().astype(int).values
                if missing.sum() == 0:
                    return 0
                # Find consecutive groups
                groups = np.split(missing, np.where(np.diff(missing) != 0)[0] + 1)
                max_consecutive = max(group.sum() for group in groups)
                return max_consecutive
            
            features['max_consecutive_missing'] = df[missing_cols].apply(
                count_consecutive_missing, axis=1
            )
            descriptions['max_consecutive_missing'] = "Maximum consecutive missing values per row"
            
            # Missing pattern hash (which columns are missing)
            def hash_missing_pattern(row):
                """Create hash of missing pattern."""
                pattern = ''.join(['1' if pd.isnull(v) else '0' for v in row])
                return int(hashlib.md5(pattern.encode()).hexdigest()[:8], 16)
            
            features['missing_pattern_id'] = df[missing_cols].apply(
                hash_missing_pattern, axis=1
            )
            descriptions['missing_pattern_id'] = "Hash identifier of missing value pattern"
            
            # Missing correlation features (if multiple columns)
            if len(missing_cols) > 1:
                # Count of columns that are missing together
                missing_matrix = df[missing_cols].isnull()
                for i, col1 in enumerate(missing_cols[:min(5, len(missing_cols))]):  # Limit to top 5
                    for col2 in missing_cols[i+1:min(5, len(missing_cols))]:
                        feat_name = f'missing_together_{col1}_{col2}'
                        features[feat_name] = (
                            missing_matrix[col1] & missing_matrix[col2]
                        ).astype(int)
                        descriptions[feat_name] = f"Both {col1} and {col2} are missing"
        
        # Apply signal check
        features, descriptions = check_signal(features, descriptions, self.min_signal_ratio)
        
        logger.info(f"Generated {len(features.columns)} missing data features")
        return features, descriptions