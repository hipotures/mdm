"""Column interaction features generator."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from mdm.features.base_global import GlobalFeatureOperation
from mdm.features.utils import check_signal


class InteractionFeatures(GlobalFeatureOperation):
    """Generate features from column interactions.
    
    Creates arithmetic and statistical interactions between numeric columns
    to capture non-linear relationships.
    """
    
    def __init__(
        self, 
        max_interactions: int = 20,
        min_correlation: float = 0.1,
        operations: Optional[List[str]] = None
    ):
        """Initialize interaction feature generator.
        
        Args:
            max_interactions: Maximum number of interaction pairs to generate
            min_correlation: Minimum absolute correlation to consider pair
            operations: List of operations to perform ['add', 'subtract', 'multiply', 'divide', 'max', 'min']
        """
        super().__init__()
        self.max_interactions = max_interactions
        self.min_correlation = min_correlation
        self.operations = operations or ['add', 'subtract', 'multiply', 'divide']
        
    def get_applicable_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numeric columns suitable for interactions.
        
        Args:
            df: Input dataframe
            
        Returns:
            List of numeric column names
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out likely ID columns and binary columns
        applicable = []
        for col in numeric_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.01 and df[col].nunique() > 2:  # Not ID or binary
                applicable.append(col)
        
        logger.debug(f"Found {len(applicable)} numeric columns suitable for interactions")
        return applicable
    
    def _select_top_pairs(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        target_column: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Select top column pairs based on correlation.
        
        Args:
            df: Input dataframe
            columns: List of column names
            target_column: Target column for supervised selection
            
        Returns:
            List of column pairs
        """
        # Calculate correlation matrix
        corr_matrix = df[columns].corr().abs()
        
        # Get pairs sorted by correlation
        pairs = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr = corr_matrix.iloc[i, j]
                if corr >= self.min_correlation and corr < 0.95:  # Avoid perfect correlation
                    pairs.append((columns[i], columns[j], corr))
        
        # Sort by correlation and take top pairs
        pairs.sort(key=lambda x: x[2], reverse=True)
        selected_pairs = [(p[0], p[1]) for p in pairs[:self.max_interactions]]
        
        logger.debug(f"Selected {len(selected_pairs)} column pairs for interactions")
        return selected_pairs
    
    def _generate_pair_features(
        self, 
        df: pd.DataFrame, 
        col1: str, 
        col2: str
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate interaction features for a column pair.
        
        Args:
            df: Input dataframe
            col1: First column name
            col2: Second column name
            
        Returns:
            Tuple of (features dataframe, feature descriptions)
        """
        features = pd.DataFrame(index=df.index)
        descriptions = {}
        
        # Get column values
        x = df[col1].values
        y = df[col2].values
        
        # Addition
        if 'add' in self.operations:
            features[f'{col1}_plus_{col2}'] = x + y
            descriptions[f'{col1}_plus_{col2}'] = f"Sum of {col1} and {col2}"
        
        # Subtraction
        if 'subtract' in self.operations:
            features[f'{col1}_minus_{col2}'] = x - y
            descriptions[f'{col1}_minus_{col2}'] = f"Difference between {col1} and {col2}"
        
        # Multiplication
        if 'multiply' in self.operations:
            features[f'{col1}_times_{col2}'] = x * y
            descriptions[f'{col1}_times_{col2}'] = f"Product of {col1} and {col2}"
        
        # Division (with zero handling)
        if 'divide' in self.operations:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(y != 0, x / y, 0)
            features[f'{col1}_div_{col2}'] = ratio
            descriptions[f'{col1}_div_{col2}'] = f"Ratio of {col1} to {col2}"
        
        # Maximum
        if 'max' in self.operations:
            features[f'max_{col1}_{col2}'] = np.maximum(x, y)
            descriptions[f'max_{col1}_{col2}'] = f"Maximum of {col1} and {col2}"
        
        # Minimum
        if 'min' in self.operations:
            features[f'min_{col1}_{col2}'] = np.minimum(x, y)
            descriptions[f'min_{col1}_{col2}'] = f"Minimum of {col1} and {col2}"
        
        # Ratio to sum
        if 'ratio_to_sum' in self.operations:
            with np.errstate(divide='ignore', invalid='ignore'):
                total = x + y
                ratio = np.where(total != 0, x / total, 0.5)
            features[f'{col1}_ratio_to_sum_{col2}'] = ratio
            descriptions[f'{col1}_ratio_to_sum_{col2}'] = f"Ratio of {col1} to sum with {col2}"
        
        return features, descriptions
    
    def generate_features(
        self, 
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate all interaction features.
        
        Args:
            df: Input dataframe
            target_column: Target column to exclude
            id_columns: ID columns to exclude
            
        Returns:
            Tuple of (features dataframe, feature descriptions)
        """
        features = pd.DataFrame(index=df.index)
        descriptions = {}
        
        # Get applicable columns
        columns = self.get_applicable_columns(df)
        
        # Exclude target and ID columns
        columns = [col for col in columns 
                  if col != target_column and col not in (id_columns or [])]
        
        if len(columns) < 2:
            logger.warning("Not enough numeric columns for interactions")
            return features, descriptions
        
        # Select top pairs
        pairs = self._select_top_pairs(df, columns, target_column)
        
        # Generate features for each pair
        for col1, col2 in pairs:
            pair_features, pair_descriptions = self._generate_pair_features(df, col1, col2)
            features = pd.concat([features, pair_features], axis=1)
            descriptions.update(pair_descriptions)
        
        # Apply signal check
        features, descriptions = check_signal(features, descriptions, self.min_signal_ratio)
        
        logger.info(f"Generated {len(features.columns)} interaction features from {len(pairs)} pairs")
        return features, descriptions