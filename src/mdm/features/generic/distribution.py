"""Distribution-based features generator."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from mdm.features.base_global import GlobalFeatureOperation
from mdm.features.utils import check_signal


class DistributionFeatures(GlobalFeatureOperation):
    """Generate features based on statistical distributions.
    
    Extracts distribution characteristics like skewness, kurtosis, entropy,
    and other statistical properties of numeric columns.
    """
    
    def __init__(
        self, 
        compute_normality: bool = True,
        entropy_bins: int = 10
    ):
        """Initialize distribution feature generator.
        
        Args:
            compute_normality: Whether to compute normality test statistics
            entropy_bins: Number of bins for entropy calculation
        """
        super().__init__()
        self.compute_normality = compute_normality
        self.entropy_bins = entropy_bins
        
    def get_applicable_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numeric columns suitable for distribution analysis.
        
        Args:
            df: Input dataframe
            
        Returns:
            List of numeric column names
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter columns with enough unique values
        applicable = []
        for col in numeric_cols:
            if df[col].nunique() > 10 and df[col].std() > 0:
                applicable.append(col)
        
        logger.debug(f"Found {len(applicable)} columns suitable for distribution features")
        return applicable
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """Calculate Shannon entropy of values.
        
        Args:
            values: Array of values
            
        Returns:
            Entropy value
        """
        # Remove NaN values
        clean_values = values[~np.isnan(values)]
        if len(clean_values) == 0:
            return np.nan
        
        # Create histogram
        counts, _ = np.histogram(clean_values, bins=self.entropy_bins)
        
        # Calculate probabilities
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _calculate_gini(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient.
        
        Args:
            values: Array of values
            
        Returns:
            Gini coefficient
        """
        # Remove NaN values and ensure positive
        clean_values = values[~np.isnan(values)]
        clean_values = np.abs(clean_values)  # Gini requires non-negative values
        
        if len(clean_values) == 0:
            return np.nan
        
        # Sort values
        sorted_values = np.sort(clean_values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
    
    def _generate_column_features(
        self, 
        df: pd.DataFrame, 
        column: str
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate distribution features for a single column.
        
        Args:
            df: Input dataframe
            column: Column name
            
        Returns:
            Tuple of (features dataframe, feature descriptions)
        """
        features = pd.DataFrame(index=df.index)
        descriptions = {}
        
        values = df[column].values
        
        # Global distribution metrics (same for all rows)
        # Skewness
        skewness = stats.skew(values[~np.isnan(values)])
        features[f'{column}_skewness'] = skewness
        descriptions[f'{column}_skewness'] = f"Skewness of {column} distribution"
        
        # Kurtosis
        kurtosis = stats.kurtosis(values[~np.isnan(values)])
        features[f'{column}_kurtosis'] = kurtosis
        descriptions[f'{column}_kurtosis'] = f"Kurtosis of {column} distribution"
        
        # Entropy
        entropy = self._calculate_entropy(values)
        features[f'{column}_entropy'] = entropy
        descriptions[f'{column}_entropy'] = f"Shannon entropy of {column}"
        
        # Gini coefficient
        gini = self._calculate_gini(values)
        features[f'{column}_gini'] = gini
        descriptions[f'{column}_gini'] = f"Gini coefficient of {column}"
        
        # Row-wise distribution features
        # Quantile position
        clean_values = values[~np.isnan(values)]
        if len(clean_values) > 0:
            # Calculate deciles
            deciles = np.percentile(clean_values, np.arange(10, 100, 10))
            
            def get_decile_position(x):
                if np.isnan(x):
                    return np.nan
                return np.searchsorted(deciles, x)
            
            features[f'{column}_decile_position'] = df[column].apply(get_decile_position)
            descriptions[f'{column}_decile_position'] = f"Decile position of {column} value"
            
            # IQR-based features
            q1, q3 = np.percentile(clean_values, [25, 75])
            iqr = q3 - q1
            features[f'{column}_iqr_normalized'] = (df[column] - q1) / (iqr + 1e-8)
            descriptions[f'{column}_iqr_normalized'] = f"IQR-normalized {column}"
            
            # Tail ratio
            p5, p50, p95 = np.percentile(clean_values, [5, 50, 95])
            upper_tail = p95 - p50
            lower_tail = p50 - p5
            tail_ratio = upper_tail / (lower_tail + 1e-8)
            features[f'{column}_tail_ratio'] = tail_ratio
            descriptions[f'{column}_tail_ratio'] = f"Tail ratio (p95-p50)/(p50-p5) of {column}"
        
        # Normality test (if requested)
        if self.compute_normality and len(clean_values) > 20:
            try:
                # Shapiro-Wilk test
                statistic, p_value = stats.shapiro(clean_values[:5000])  # Limit sample size
                features[f'{column}_shapiro_stat'] = statistic
                descriptions[f'{column}_shapiro_stat'] = f"Shapiro-Wilk statistic for {column}"
                
                features[f'{column}_is_normal'] = int(p_value > 0.05)
                descriptions[f'{column}_is_normal'] = f"Binary normality indicator for {column} (p>0.05)"
            except:
                logger.debug(f"Could not compute normality test for {column}")
        
        return features, descriptions
    
    def generate_features(
        self, 
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate all distribution features.
        
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
        
        if not columns:
            logger.warning("No suitable columns for distribution features")
            return features, descriptions
        
        # Generate features for each column
        for col in columns[:20]:  # Limit to prevent explosion
            try:
                col_features, col_descriptions = self._generate_column_features(df, col)
                features = pd.concat([features, col_features], axis=1)
                descriptions.update(col_descriptions)
            except Exception as e:
                logger.warning(f"Failed to generate distribution features for {col}: {e}")
        
        # Apply signal check
        features, descriptions = check_signal(features, descriptions, self.min_signal_ratio)
        
        logger.info(f"Generated {len(features.columns)} distribution features")
        return features, descriptions