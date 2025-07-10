"""
Built-in feature transformers for the new feature engineering system.

This module provides standard transformers for common data types.
"""
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import re

from .base import FeatureTransformer


class NumericTransformer(FeatureTransformer):
    """Transformer for numeric features."""
    
    @property
    def name(self) -> str:
        return "numeric"
    
    @property
    def supported_types(self) -> List[str]:
        return ["numeric", "int", "float"]
    
    def _fit_impl(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit numeric transformer.
        
        Computes statistics for normalization and scaling.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        fit_state = {}
        for col in numeric_cols:
            fit_state[col] = {
                "mean": data[col].mean(),
                "std": data[col].std(),
                "min": data[col].min(),
                "max": data[col].max(),
                "median": data[col].median(),
                "q1": data[col].quantile(0.25),
                "q3": data[col].quantile(0.75)
            }
        
        return fit_state
    
    def _transform_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform numeric data."""
        result = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in self._fit_state:
                continue
            
            stats = self._fit_state[col]
            
            # Log transform (for positive values)
            if (data[col] > 0).all():
                result[f"{col}_log"] = np.log1p(data[col])
            
            # Square root transform (for non-negative values)
            if (data[col] >= 0).all():
                result[f"{col}_sqrt"] = np.sqrt(data[col])
            
            # Squared transform
            result[f"{col}_squared"] = data[col] ** 2
            
            # Normalized (z-score)
            if stats["std"] > 0:
                result[f"{col}_zscore"] = (data[col] - stats["mean"]) / stats["std"]
            
            # Min-max scaled
            range_val = stats["max"] - stats["min"]
            if range_val > 0:
                result[f"{col}_minmax"] = (data[col] - stats["min"]) / range_val
            
            # Binned features
            if self.config.get("create_bins", True):
                result[f"{col}_bin"] = pd.qcut(
                    data[col], 
                    q=self.config.get("n_bins", 5), 
                    labels=False,
                    duplicates='drop'
                )
        
        return result


class CategoricalTransformer(FeatureTransformer):
    """Transformer for categorical features."""
    
    @property
    def name(self) -> str:
        return "categorical"
    
    @property
    def supported_types(self) -> List[str]:
        return ["categorical", "object", "string"]
    
    def _fit_impl(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit categorical transformer.
        
        Computes value counts and encodings.
        """
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        
        fit_state = {}
        for col in cat_cols:
            value_counts = data[col].value_counts()
            fit_state[col] = {
                "value_counts": value_counts.to_dict(),
                "unique_values": data[col].unique().tolist(),
                "n_unique": data[col].nunique(),
                "mode": data[col].mode()[0] if not data[col].mode().empty else None
            }
        
        return fit_state
    
    def _transform_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical data."""
        result = data.copy()
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            if col not in self._fit_state:
                continue
            
            stats = self._fit_state[col]
            
            # Count encoding
            count_map = stats["value_counts"]
            result[f"{col}_count"] = data[col].map(count_map).fillna(0)
            
            # Frequency encoding
            total_count = sum(count_map.values())
            freq_map = {k: v/total_count for k, v in count_map.items()}
            result[f"{col}_frequency"] = data[col].map(freq_map).fillna(0)
            
            # Is rare category (appears less than threshold)
            threshold = self.config.get("rare_threshold", 0.01)
            rare_categories = {k for k, v in freq_map.items() if v < threshold}
            result[f"{col}_is_rare"] = data[col].isin(rare_categories).astype(int)
            
            # Number of unique values ratio
            if stats["n_unique"] > 1:
                result[f"{col}_unique_ratio"] = stats["n_unique"] / len(data)
        
        return result


class DatetimeTransformer(FeatureTransformer):
    """Transformer for datetime features."""
    
    @property
    def name(self) -> str:
        return "datetime"
    
    @property
    def supported_types(self) -> List[str]:
        return ["datetime", "date", "time"]
    
    def _fit_impl(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit datetime transformer.
        
        Computes reference dates and ranges.
        """
        dt_cols = data.select_dtypes(include=['datetime64']).columns
        
        fit_state = {}
        for col in dt_cols:
            fit_state[col] = {
                "min_date": data[col].min(),
                "max_date": data[col].max(),
                "reference_date": datetime.now()
            }
        
        return fit_state
    
    def _transform_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform datetime data."""
        result = data.copy()
        dt_cols = data.select_dtypes(include=['datetime64']).columns
        
        for col in dt_cols:
            if col not in self._fit_state:
                # Convert to datetime if needed
                try:
                    result[col] = pd.to_datetime(data[col])
                except:
                    continue
            
            dt_series = result[col]
            
            # Extract components
            result[f"{col}_year"] = dt_series.dt.year
            result[f"{col}_month"] = dt_series.dt.month
            result[f"{col}_day"] = dt_series.dt.day
            result[f"{col}_dayofweek"] = dt_series.dt.dayofweek
            result[f"{col}_dayofyear"] = dt_series.dt.dayofyear
            result[f"{col}_weekofyear"] = dt_series.dt.isocalendar().week
            result[f"{col}_quarter"] = dt_series.dt.quarter
            result[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
            result[f"{col}_is_month_start"] = dt_series.dt.is_month_start.astype(int)
            result[f"{col}_is_month_end"] = dt_series.dt.is_month_end.astype(int)
            
            # Time components if timestamp
            if dt_series.dt.time.astype(str).str.contains(':').any():
                result[f"{col}_hour"] = dt_series.dt.hour
                result[f"{col}_minute"] = dt_series.dt.minute
                result[f"{col}_second"] = dt_series.dt.second
                
                # Time of day categories
                hour = dt_series.dt.hour
                result[f"{col}_time_of_day"] = pd.cut(
                    hour,
                    bins=[0, 6, 12, 18, 24],
                    labels=["night", "morning", "afternoon", "evening"],
                    include_lowest=True
                )
            
            # Days since reference date
            if col in self._fit_state:
                ref_date = self._fit_state[col]["reference_date"]
                result[f"{col}_days_since_ref"] = (dt_series - ref_date).dt.days
        
        return result


class TextTransformer(FeatureTransformer):
    """Transformer for text features."""
    
    @property
    def name(self) -> str:
        return "text"
    
    @property
    def supported_types(self) -> List[str]:
        return ["text", "string"]
    
    def _fit_impl(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit text transformer.
        
        Computes vocabulary and statistics.
        """
        text_cols = []
        
        # Identify text columns
        for col in data.select_dtypes(include=['object']).columns:
            sample = data[col].dropna().head(100)
            if not sample.empty:
                avg_length = sample.astype(str).str.len().mean()
                if avg_length > self.config.get("min_text_length", 20):
                    text_cols.append(col)
        
        fit_state = {}
        for col in text_cols:
            # Get sample for pattern detection
            sample_text = " ".join(data[col].dropna().astype(str).head(1000))
            
            fit_state[col] = {
                "avg_length": data[col].astype(str).str.len().mean(),
                "max_length": data[col].astype(str).str.len().max(),
                "has_urls": bool(re.search(r'https?://\S+', sample_text)),
                "has_emails": bool(re.search(r'\S+@\S+', sample_text))
            }
        
        return fit_state
    
    def _transform_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform text data."""
        result = data.copy()
        
        # Process columns identified during fit
        for col, stats in self._fit_state.items():
            if col not in data.columns:
                continue
            
            text = data[col].astype(str)
            
            # Length features
            result[f"{col}_length"] = text.str.len()
            result[f"{col}_word_count"] = text.str.split().str.len()
            result[f"{col}_char_count"] = text.str.replace(r'\s', '', regex=True).str.len()
            
            # Character type features
            result[f"{col}_digit_count"] = text.str.count(r'\d')
            result[f"{col}_upper_count"] = text.str.count(r'[A-Z]')
            result[f"{col}_lower_count"] = text.str.count(r'[a-z]')
            result[f"{col}_special_count"] = text.str.count(r'[^a-zA-Z0-9\s]')
            result[f"{col}_space_count"] = text.str.count(r'\s')
            
            # Ratios
            length = result[f"{col}_length"].replace(0, 1)  # Avoid division by zero
            result[f"{col}_digit_ratio"] = result[f"{col}_digit_count"] / length
            result[f"{col}_upper_ratio"] = result[f"{col}_upper_count"] / length
            result[f"{col}_special_ratio"] = result[f"{col}_special_count"] / length
            
            # Pattern features
            if stats.get("has_urls"):
                result[f"{col}_has_url"] = text.str.contains(r'https?://\S+', regex=True).astype(int)
            
            if stats.get("has_emails"):
                result[f"{col}_has_email"] = text.str.contains(r'\S+@\S+', regex=True).astype(int)
            
            # Sentence features
            result[f"{col}_sentence_count"] = text.str.count(r'[.!?]+')
            result[f"{col}_avg_word_length"] = (
                result[f"{col}_char_count"] / result[f"{col}_word_count"].replace(0, 1)
            )
        
        return result


class InteractionTransformer(FeatureTransformer):
    """Transformer for creating interaction features."""
    
    @property
    def name(self) -> str:
        return "interaction"
    
    @property
    def supported_types(self) -> List[str]:
        return ["numeric", "categorical"]
    
    def _fit_impl(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit interaction transformer.
        
        Identifies columns for interaction.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Limit columns to prevent explosion
        max_cols = self.config.get("max_interaction_cols", 10)
        numeric_cols = numeric_cols[:max_cols]
        cat_cols = cat_cols[:max_cols]
        
        return {
            "numeric_cols": numeric_cols,
            "categorical_cols": cat_cols
        }
    
    def _transform_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data with interactions."""
        result = data.copy()
        
        numeric_cols = self._fit_state.get("numeric_cols", [])
        cat_cols = self._fit_state.get("categorical_cols", [])
        
        # Numeric interactions
        if len(numeric_cols) >= 2:
            from itertools import combinations
            
            max_interactions = self.config.get("max_numeric_interactions", 20)
            
            for i, (col1, col2) in enumerate(combinations(numeric_cols, 2)):
                if i >= max_interactions:
                    break
                
                # Multiplication
                result[f"{col1}_x_{col2}"] = data[col1] * data[col2]
                
                # Division (with safety)
                denominator = data[col2].replace(0, np.nan)
                result[f"{col1}_div_{col2}"] = data[col1] / denominator
                
                # Difference
                result[f"{col1}_minus_{col2}"] = data[col1] - data[col2]
                
                # Sum
                result[f"{col1}_plus_{col2}"] = data[col1] + data[col2]
        
        # Categorical interactions
        if len(cat_cols) >= 2:
            max_cat_interactions = self.config.get("max_categorical_interactions", 10)
            
            for i, (col1, col2) in enumerate(combinations(cat_cols[:5], 2)):
                if i >= max_cat_interactions:
                    break
                
                # Concatenated feature
                result[f"{col1}__{col2}"] = (
                    data[col1].astype(str) + "_" + data[col2].astype(str)
                )
        
        # Numeric-categorical interactions
        if numeric_cols and cat_cols:
            max_mixed = self.config.get("max_mixed_interactions", 10)
            
            for i, (num_col, cat_col) in enumerate(zip(numeric_cols[:5], cat_cols[:5])):
                if i >= max_mixed:
                    break
                
                # Group statistics
                group_mean = data.groupby(cat_col)[num_col].transform('mean')
                group_std = data.groupby(cat_col)[num_col].transform('std').fillna(0)
                
                result[f"{num_col}_by_{cat_col}_mean"] = group_mean
                result[f"{num_col}_by_{cat_col}_std"] = group_std
                result[f"{num_col}_by_{cat_col}_zscore"] = (
                    (data[num_col] - group_mean) / group_std.replace(0, 1)
                )
        
        return result


# Register built-in transformers
from .base import transformer_registry

transformer_registry.register(NumericTransformer)
transformer_registry.register(CategoricalTransformer)
transformer_registry.register(DatetimeTransformer)
transformer_registry.register(TextTransformer)
transformer_registry.register(InteractionTransformer)