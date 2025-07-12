"""Sequential and time-based features generator."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from mdm.features.base_global import GlobalFeatureOperation
from mdm.features.utils import check_signal


class SequentialFeatures(GlobalFeatureOperation):
    """Generate sequential features for ordered data.
    
    Creates lag, lead, rolling statistics, and trend features for datasets
    with natural ordering (time series, sequential IDs, etc).
    """
    
    def __init__(
        self, 
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        trend_window: int = 10
    ):
        """Initialize sequential feature generator.
        
        Args:
            lag_periods: List of lag periods to generate (default: [1, 2, 3])
            rolling_windows: List of rolling window sizes (default: [3, 7])
            trend_window: Window size for trend calculation
        """
        super().__init__()
        self.lag_periods = lag_periods or [1, 2, 3]
        self.rolling_windows = rolling_windows or [3, 7]
        self.trend_window = trend_window
        
    def get_applicable_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numeric columns suitable for sequential features.
        
        Args:
            df: Input dataframe
            
        Returns:
            List of numeric column names
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out likely ID columns
        applicable = []
        for col in numeric_cols:
            # Check if column has enough variation
            if df[col].std() > 0 and df[col].nunique() > 10:
                applicable.append(col)
        
        logger.debug(f"Found {len(applicable)} columns suitable for sequential features")
        return applicable
    
    def _generate_column_features(
        self, 
        df: pd.DataFrame, 
        column: str
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate sequential features for a single column.
        
        Args:
            df: Input dataframe
            column: Column name
            
        Returns:
            Tuple of (features dataframe, feature descriptions)
        """
        features = pd.DataFrame(index=df.index)
        descriptions = {}
        
        # Lag features
        for lag in self.lag_periods:
            feat_name = f'{column}_lag_{lag}'
            features[feat_name] = df[column].shift(lag)
            descriptions[feat_name] = f"Lag {lag} of {column}"
        
        # Lead features (only lag 1)
        feat_name = f'{column}_lead_1'
        features[feat_name] = df[column].shift(-1)
        descriptions[feat_name] = f"Lead 1 of {column}"
        
        # Rolling statistics
        for window in self.rolling_windows:
            # Rolling mean
            feat_name = f'{column}_roll_mean_{window}'
            features[feat_name] = df[column].rolling(window=window, min_periods=1).mean()
            descriptions[feat_name] = f"Rolling mean (window={window}) of {column}"
            
            # Rolling std
            feat_name = f'{column}_roll_std_{window}'
            features[feat_name] = df[column].rolling(window=window, min_periods=2).std()
            descriptions[feat_name] = f"Rolling std (window={window}) of {column}"
            
            # Rolling min
            feat_name = f'{column}_roll_min_{window}'
            features[feat_name] = df[column].rolling(window=window, min_periods=1).min()
            descriptions[feat_name] = f"Rolling min (window={window}) of {column}"
            
            # Rolling max
            feat_name = f'{column}_roll_max_{window}'
            features[feat_name] = df[column].rolling(window=window, min_periods=1).max()
            descriptions[feat_name] = f"Rolling max (window={window}) of {column}"
        
        # Change features
        # First difference
        feat_name = f'{column}_diff_1'
        features[feat_name] = df[column].diff(1)
        descriptions[feat_name] = f"First difference of {column}"
        
        # Percentage change
        feat_name = f'{column}_pct_change'
        with np.errstate(divide='ignore', invalid='ignore'):
            features[feat_name] = df[column].pct_change(1)
        features[feat_name] = features[feat_name].replace([np.inf, -np.inf], np.nan)
        descriptions[feat_name] = f"Percentage change of {column}"
        
        # Cumulative sum
        feat_name = f'{column}_cumsum'
        features[feat_name] = df[column].cumsum()
        descriptions[feat_name] = f"Cumulative sum of {column}"
        
        # Linear trend
        feat_name = f'trend_{column}'
        trend_values = self._calculate_trend(df[column], self.trend_window)
        features[feat_name] = trend_values
        descriptions[feat_name] = f"Linear trend coefficient (window={self.trend_window}) of {column}"
        
        return features, descriptions
    
    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling linear trend coefficient.
        
        Args:
            series: Input series
            window: Window size for trend calculation
            
        Returns:
            Series with trend coefficients
        """
        def get_trend(values):
            """Calculate linear trend coefficient."""
            if len(values) < 2 or values.isnull().all():
                return np.nan
            
            valid_values = values.dropna()
            if len(valid_values) < 2:
                return np.nan
            
            x = np.arange(len(valid_values))
            try:
                # Simple linear regression coefficient
                coef = np.polyfit(x, valid_values, 1)[0]
                return coef
            except:
                return np.nan
        
        return series.rolling(window=window, min_periods=2).apply(get_trend)
    
    def generate_features(
        self, 
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate all sequential features.
        
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
            logger.warning("No suitable columns for sequential features")
            return features, descriptions
        
        # Limit to top columns by variance to avoid explosion
        if len(columns) > 10:
            variances = df[columns].var()
            columns = variances.nlargest(10).index.tolist()
            logger.info(f"Limited to top 10 columns by variance for sequential features")
        
        # Generate features for each column
        for col in columns:
            col_features, col_descriptions = self._generate_column_features(df, col)
            features = pd.concat([features, col_features], axis=1)
            descriptions.update(col_descriptions)
        
        # Apply signal check
        features, descriptions = check_signal(features, descriptions, self.min_signal_ratio)
        
        logger.info(f"Generated {len(features.columns)} sequential features")
        return features, descriptions