"""
Feature engineering adapters for existing implementations.

These adapters wrap the legacy feature generator to provide the IFeatureGenerator
interface while maintaining full backward compatibility.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import time
import logging

from ..interfaces.features import IFeatureGenerator
from ..features.generator import FeatureGenerator
# Remove transformer imports - use generator directly

logger = logging.getLogger(__name__)


class FeatureGeneratorAdapter(IFeatureGenerator):
    """Adapter for existing feature generator with metrics tracking."""
    
    def __init__(self):
        self._generator = FeatureGenerator()
        self._metrics = {
            "features_generated": 0,
            "processing_time": 0.0,
            "datasets_processed": 0,
            "call_count": {}
        }
        logger.info("Initialized FeatureGeneratorAdapter")
    
    def _track_call(self, method: str, features_added: int = 0, time_taken: float = 0.0):
        """Track method calls and metrics."""
        self._metrics["call_count"][method] = self._metrics["call_count"].get(method, 0) + 1
        self._metrics["features_generated"] += features_added
        self._metrics["processing_time"] += time_taken
        
        logger.debug(f"Feature method called: {method} "
                    f"(features added: {features_added}, time: {time_taken:.2f}s)")
    
    def generate_features(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
        datetime_columns: Optional[List[str]] = None,
        id_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate all features with metrics tracking."""
        start_time = time.time()
        initial_cols = len(data.columns)
        
        # Call the original method with parameters it expects
        result = self._generator.generate_features(
            data=data,
            target_column=target_column,
            problem_type=problem_type,
            datetime_columns=datetime_columns,
            id_columns=id_columns
        )
        
        time_taken = time.time() - start_time
        features_added = len(result.columns) - initial_cols
        
        self._track_call("generate_features", features_added, time_taken)
        self._metrics["datasets_processed"] += 1
        
        return result
    
    def generate_numeric_features(
        self, 
        data: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate numeric features."""
        start_time = time.time()
        initial_cols = len(data.columns)
        
        # Detect numeric columns if not provided
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=['int', 'float']).columns.tolist()
        
        # Generate numeric features using the generator
        result = data.copy()
        
        if numeric_columns and hasattr(self._generator, 'generate_numeric_features'):
            numeric_features = self._generator.generate_numeric_features(data[numeric_columns])
            # Merge new features
            for col in numeric_features.columns:
                if col not in result.columns:
                    result[col] = numeric_features[col]
        
        time_taken = time.time() - start_time
        features_added = len(result.columns) - initial_cols
        
        self._track_call("generate_numeric_features", features_added, time_taken)
        
        return result
    
    def generate_categorical_features(
        self, 
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate categorical features."""
        start_time = time.time()
        initial_cols = len(data.columns)
        
        # Detect categorical columns if not provided
        if categorical_columns is None:
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Generate categorical features using the generator
        result = data.copy()
        
        if categorical_columns and hasattr(self._generator, 'generate_categorical_features'):
            categorical_features = self._generator.generate_categorical_features(data[categorical_columns])
            # Merge new features
            for col in categorical_features.columns:
                if col not in result.columns:
                    result[col] = categorical_features[col]
        
        time_taken = time.time() - start_time
        features_added = len(result.columns) - initial_cols
        
        self._track_call("generate_categorical_features", features_added, time_taken)
        
        return result
    
    def generate_datetime_features(
        self, 
        data: pd.DataFrame,
        datetime_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate datetime features."""
        start_time = time.time()
        initial_cols = len(data.columns)
        
        # Check if the generator has this method
        if hasattr(self._generator, 'generate_datetime_features'):
            result = self._generator.generate_datetime_features(data, datetime_columns)
        else:
            # Fallback implementation for older versions
            result = data.copy()
            
            if datetime_columns:
                for col in datetime_columns:
                    if col in data.columns:
                        # Convert to datetime if not already
                        dt_col = pd.to_datetime(data[col], errors='coerce')
                        
                        # Extract basic datetime features
                        result[f"{col}_year"] = dt_col.dt.year
                        result[f"{col}_month"] = dt_col.dt.month
                        result[f"{col}_day"] = dt_col.dt.day
                        result[f"{col}_dayofweek"] = dt_col.dt.dayofweek
                        result[f"{col}_hour"] = dt_col.dt.hour
                        result[f"{col}_minute"] = dt_col.dt.minute
        
        time_taken = time.time() - start_time
        features_added = len(result.columns) - initial_cols
        
        self._track_call("generate_datetime_features", features_added, time_taken)
        
        return result
    
    def generate_text_features(
        self, 
        data: pd.DataFrame,
        text_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate text features."""
        start_time = time.time()
        initial_cols = len(data.columns)
        
        # Check if the generator has this method
        if hasattr(self._generator, 'generate_text_features'):
            result = self._generator.generate_text_features(data, text_columns)
        else:
            # Basic text feature implementation
            result = data.copy()
            
            if text_columns is None:
                # Detect text columns (string columns with avg length > 20)
                text_columns = []
                for col in data.select_dtypes(include=['object']).columns:
                    avg_len = data[col].astype(str).str.len().mean()
                    if avg_len > 20:
                        text_columns.append(col)
            
            for col in text_columns:
                if col in data.columns:
                    # Basic text features
                    result[f"{col}_length"] = data[col].astype(str).str.len()
                    result[f"{col}_word_count"] = data[col].astype(str).str.split().str.len()
                    result[f"{col}_has_digits"] = data[col].astype(str).str.contains(r'\d', regex=True).astype(int)
                    result[f"{col}_has_special"] = data[col].astype(str).str.contains(r'[^a-zA-Z0-9\s]', regex=True).astype(int)
        
        time_taken = time.time() - start_time
        features_added = len(result.columns) - initial_cols
        
        self._track_call("generate_text_features", features_added, time_taken)
        
        return result
    
    def generate_interaction_features(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        max_interactions: int = 2
    ) -> pd.DataFrame:
        """Generate interaction features between columns."""
        start_time = time.time()
        initial_cols = len(data.columns)
        
        result = data.copy()
        
        # Simple interaction features for numeric columns
        if columns is None:
            columns = data.select_dtypes(include=['int', 'float']).columns.tolist()
        
        if max_interactions >= 2 and len(columns) >= 2:
            # Generate pairwise products for top numeric columns
            for i, col1 in enumerate(columns[:5]):  # Limit to prevent explosion
                for col2 in columns[i+1:6]:
                    result[f"{col1}_x_{col2}"] = data[col1] * data[col2]
        
        time_taken = time.time() - start_time
        features_added = len(result.columns) - initial_cols
        
        self._track_call("generate_interaction_features", features_added, time_taken)
        
        return result
    
    def get_feature_importance(
        self, 
        features: pd.DataFrame,
        target: pd.Series,
        problem_type: str
    ) -> pd.DataFrame:
        """Calculate feature importance scores."""
        start_time = time.time()
        
        # Check if the generator has this method
        if hasattr(self._generator, 'get_feature_importance'):
            result = self._generator.get_feature_importance(features, target, problem_type)
        else:
            # Simple correlation-based importance
            importances = []
            
            for col in features.columns:
                if features[col].dtype in ['int64', 'float64']:
                    # Numeric features - use correlation
                    correlation = features[col].corr(target)
                    importances.append({
                        'feature': col,
                        'importance': abs(correlation),
                        'type': 'correlation'
                    })
                else:
                    # Categorical features - use simple metric
                    importances.append({
                        'feature': col,
                        'importance': 0.0,
                        'type': 'categorical'
                    })
            
            result = pd.DataFrame(importances).sort_values('importance', ascending=False)
        
        time_taken = time.time() - start_time
        self._track_call("get_feature_importance", 0, time_taken)
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        return self._metrics.copy()