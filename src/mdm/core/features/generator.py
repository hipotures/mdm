"""
New feature generator implementation.

This module provides the refactored feature generation system with:
- Clean plugin-based architecture
- Better memory efficiency with batch processing
- Comprehensive type support
- Feature importance calculation
"""
from typing import Dict, List, Optional, Any, Set
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from mdm.interfaces.features import IFeatureGenerator
from .base import transformer_registry, FeatureTransformer

logger = logging.getLogger(__name__)


class NewFeatureGenerator(IFeatureGenerator):
    """New implementation of feature generator with improved architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature generator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.registry = transformer_registry
        self._metrics = {
            "features_generated": 0,
            "datasets_processed": 0,
            "processing_time": 0.0,
            "transformers_used": set()
        }
        
        # Load custom plugins if configured
        plugin_dir = self.config.get("plugin_dir")
        if plugin_dir:
            self.registry.load_plugins(Path(plugin_dir))
        
        logger.info("Initialized NewFeatureGenerator")
    
    def generate_features(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
        datetime_columns: Optional[List[str]] = None,
        id_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate all features for dataset.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            problem_type: Type of ML problem
            datetime_columns: List of datetime columns
            id_columns: List of ID columns to exclude
            
        Returns:
            DataFrame with generated features
        """
        start_time = datetime.now()
        logger.info(f"Generating features for dataset with {len(data)} rows")
        
        # Prepare exclusion set
        exclude_cols: Set[str] = set()
        if id_columns:
            exclude_cols.update(id_columns)
        if target_column:
            exclude_cols.add(target_column)
        
        # Convert datetime columns if specified
        if datetime_columns:
            for col in datetime_columns:
                if col in data.columns and col not in exclude_cols:
                    try:
                        data[col] = pd.to_datetime(data[col])
                    except Exception as e:
                        logger.warning(f"Failed to convert {col} to datetime: {e}")
        
        # Determine which transformers to use
        transformers_to_use = self._select_transformers(data, exclude_cols)
        
        # Apply transformers
        result = data.copy()
        
        for transformer_name, columns in transformers_to_use.items():
            if not columns:
                continue
            
            try:
                transformer = self.registry.get(transformer_name, self.config)
                
                # Apply transformer to relevant columns
                subset = data[list(columns)]
                transformed = transformer.fit_transform(subset)
                
                # Add new features to result
                new_cols = [col for col in transformed.columns if col not in result.columns]
                for col in new_cols:
                    result[col] = transformed[col]
                
                self._metrics["transformers_used"].add(transformer_name)
                logger.debug(f"Applied {transformer_name} transformer, added {len(new_cols)} features")
                
            except Exception as e:
                logger.error(f"Error applying {transformer_name} transformer: {e}")
        
        # Update metrics
        features_added = len(result.columns) - len(data.columns)
        self._metrics["features_generated"] += features_added
        self._metrics["datasets_processed"] += 1
        self._metrics["processing_time"] += (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Generated {features_added} features in {(datetime.now() - start_time).total_seconds():.2f}s")
        
        return result
    
    def generate_numeric_features(
        self, 
        data: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate numeric features.
        
        Args:
            data: Input DataFrame
            numeric_columns: Columns to process (auto-detect if None)
            
        Returns:
            DataFrame with numeric features
        """
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return data
        
        transformer = self.registry.get("numeric", self.config)
        subset = data[numeric_columns]
        
        return transformer.fit_transform(subset)
    
    def generate_categorical_features(
        self, 
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate categorical features.
        
        Args:
            data: Input DataFrame
            categorical_columns: Columns to process (auto-detect if None)
            
        Returns:
            DataFrame with categorical features
        """
        if categorical_columns is None:
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_columns:
            return data
        
        transformer = self.registry.get("categorical", self.config)
        subset = data[categorical_columns]
        
        return transformer.fit_transform(subset)
    
    def generate_datetime_features(
        self, 
        data: pd.DataFrame,
        datetime_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate datetime features.
        
        Args:
            data: Input DataFrame
            datetime_columns: Columns to process (auto-detect if None)
            
        Returns:
            DataFrame with datetime features
        """
        if datetime_columns is None:
            datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        if not datetime_columns:
            return data
        
        transformer = self.registry.get("datetime", self.config)
        subset = data[datetime_columns]
        
        return transformer.fit_transform(subset)
    
    def generate_text_features(
        self, 
        data: pd.DataFrame,
        text_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate text features.
        
        Args:
            data: Input DataFrame
            text_columns: Columns to process (auto-detect if None)
            
        Returns:
            DataFrame with text features
        """
        transformer = self.registry.get("text", self.config)
        
        # Text transformer does its own column detection
        if text_columns:
            subset = data[text_columns]
        else:
            subset = data
        
        return transformer.fit_transform(subset)
    
    def generate_interaction_features(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        max_interactions: int = 2
    ) -> pd.DataFrame:
        """Generate interaction features between columns.
        
        Args:
            data: Input DataFrame
            columns: Columns to create interactions from
            max_interactions: Maximum interaction order
            
        Returns:
            DataFrame with interaction features
        """
        config = self.config.copy()
        config["max_interactions"] = max_interactions
        
        transformer = self.registry.get("interaction", config)
        
        if columns:
            subset = data[columns]
        else:
            subset = data
        
        return transformer.fit_transform(subset)
    
    def get_feature_importance(
        self, 
        features: pd.DataFrame,
        target: pd.Series,
        problem_type: str
    ) -> pd.DataFrame:
        """Calculate feature importance scores.
        
        Args:
            features: Feature DataFrame
            target: Target series
            problem_type: Type of ML problem
            
        Returns:
            DataFrame with feature importance scores
        """
        importance_scores = []
        
        # Ensure target is numeric for correlation
        if problem_type == "classification" and target.dtype == 'object':
            # Convert to numeric codes
            target_numeric = pd.Categorical(target).codes
        else:
            target_numeric = target
        
        for column in features.columns:
            try:
                if pd.api.types.is_numeric_dtype(features[column]):
                    # Pearson correlation for numeric features
                    correlation = features[column].corr(target_numeric)
                    
                    # Also calculate Spearman correlation
                    spearman_corr = features[column].corr(target_numeric, method='spearman')
                    
                    importance_scores.append({
                        'feature': column,
                        'importance': abs(correlation),
                        'correlation': correlation,
                        'spearman_correlation': spearman_corr,
                        'type': 'numeric'
                    })
                    
                elif pd.api.types.is_categorical_dtype(features[column]) or features[column].dtype == 'object':
                    # For categorical features, use chi-square or mutual information
                    # Simplified: use value counts correlation
                    value_counts = features[column].value_counts()
                    if len(value_counts) < 100:  # Reasonable cardinality
                        # Create dummy variables
                        dummies = pd.get_dummies(features[column], prefix=column)
                        
                        # Calculate correlation for each dummy
                        max_corr = 0
                        for dummy_col in dummies.columns:
                            corr = abs(dummies[dummy_col].corr(target_numeric))
                            max_corr = max(max_corr, corr)
                        
                        importance_scores.append({
                            'feature': column,
                            'importance': max_corr,
                            'correlation': max_corr,
                            'spearman_correlation': 0,
                            'type': 'categorical'
                        })
                    else:
                        # High cardinality categorical
                        importance_scores.append({
                            'feature': column,
                            'importance': 0,
                            'correlation': 0,
                            'spearman_correlation': 0,
                            'type': 'high_cardinality_categorical'
                        })
                
            except Exception as e:
                logger.warning(f"Failed to calculate importance for {column}: {e}")
                importance_scores.append({
                    'feature': column,
                    'importance': 0,
                    'correlation': 0,
                    'spearman_correlation': 0,
                    'type': 'error'
                })
        
        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def _select_transformers(
        self, 
        data: pd.DataFrame, 
        exclude_cols: Set[str]
    ) -> Dict[str, List[str]]:
        """Select appropriate transformers for each column.
        
        Args:
            data: Input DataFrame
            exclude_cols: Columns to exclude
            
        Returns:
            Mapping of transformer names to column lists
        """
        transformers_map = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "text": []
        }
        
        for col in data.columns:
            if col in exclude_cols:
                continue
            
            if pd.api.types.is_numeric_dtype(data[col]):
                transformers_map["numeric"].append(col)
                
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                transformers_map["datetime"].append(col)
                
            elif pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object':
                # Check if it's text
                sample = data[col].dropna().head(100)
                if not sample.empty:
                    avg_length = sample.astype(str).str.len().mean()
                    unique_ratio = len(sample.unique()) / len(sample)
                    
                    if avg_length > 50 and unique_ratio > 0.8:
                        transformers_map["text"].append(col)
                    else:
                        transformers_map["categorical"].append(col)
                else:
                    transformers_map["categorical"].append(col)
        
        # Add interaction transformer if we have suitable columns
        if (len(transformers_map["numeric"]) >= 2 or 
            (transformers_map["numeric"] and transformers_map["categorical"])):
            transformers_map["interaction"] = (
                transformers_map["numeric"] + transformers_map["categorical"]
            )
        
        return transformers_map
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get generator metrics.
        
        Returns:
            Metrics dictionary
        """
        return {
            "features_generated": self._metrics["features_generated"],
            "datasets_processed": self._metrics["datasets_processed"],
            "processing_time": self._metrics["processing_time"],
            "transformers_used": list(self._metrics["transformers_used"]),
            "avg_time_per_dataset": (
                self._metrics["processing_time"] / self._metrics["datasets_processed"]
                if self._metrics["datasets_processed"] > 0 else 0
            )
        }