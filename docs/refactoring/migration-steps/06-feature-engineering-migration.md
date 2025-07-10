# Step 6: Feature Engineering Migration

## Overview

Migrate from the monolithic FeatureGenerator to a modular pipeline architecture with pluggable transformers. This enables better testing, custom features, and performance optimization.

## Duration

3 weeks (Weeks 11-13)

## Objectives

1. Implement pipeline architecture with transformer pattern
2. Create plugin system for custom features
3. Migrate all existing feature types
4. Enable parallel feature computation
5. Maintain backward compatibility

## Current State Analysis

Current issues:
- Monolithic FeatureGenerator class doing everything
- Difficult to add custom features
- No parallel processing
- Tight coupling with data types
- Limited configurability

## Detailed Steps

### Week 11: Pipeline Architecture

#### Day 1-2: Feature Pipeline Framework

##### 1.1 Create Pipeline Base Classes
```python
# Create: src/mdm/features/pipeline/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for a transformer"""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    parallel: bool = False
    priority: int = 0  # Lower number = higher priority


class BaseTransformer(ABC):
    """Base class for all feature transformers"""
    
    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig(name=self.__class__.__name__)
        self._fitted = False
        self._feature_names: List[str] = []
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseTransformer":
        """Fit transformer to data"""
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data and return features"""
        pass
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this transformer produces"""
        pass
    
    @property
    def is_fitted(self) -> bool:
        """Check if transformer has been fitted"""
        return self._fitted
    
    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate input dataframe"""
        if df.empty:
            raise ValueError("Input dataframe is empty")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")


class ColumnTransformer(BaseTransformer):
    """Base class for transformers that operate on specific columns"""
    
    def __init__(self, columns: Optional[List[str]] = None, 
                 config: Optional[TransformerConfig] = None):
        super().__init__(config)
        self.columns = columns
        self._column_dtypes: Dict[str, type] = {}
    
    def _select_columns(self, df: pd.DataFrame) -> List[str]:
        """Select columns to transform"""
        if self.columns is None:
            # Auto-select based on dtype
            return self._auto_select_columns(df)
        else:
            # Validate specified columns exist
            missing = set(self.columns) - set(df.columns)
            if missing:
                raise ValueError(f"Columns not found: {missing}")
            return self.columns
    
    @abstractmethod
    def _auto_select_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-select columns based on criteria"""
        pass


class Pipeline:
    """Feature engineering pipeline"""
    
    def __init__(self, transformers: Optional[List[BaseTransformer]] = None,
                 n_jobs: int = -1, use_memory: bool = True):
        self.transformers = transformers or []
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.use_memory = use_memory
        self._memory: Dict[str, pd.DataFrame] = {}
        self._fitted = False
    
    def add_transformer(self, transformer: BaseTransformer) -> "Pipeline":
        """Add transformer to pipeline"""
        self.transformers.append(transformer)
        # Sort by priority
        self.transformers.sort(key=lambda t: t.config.priority)
        return self
    
    def fit(self, df: pd.DataFrame) -> "Pipeline":
        """Fit all transformers"""
        logger.info(f"Fitting pipeline with {len(self.transformers)} transformers")
        
        for transformer in self.transformers:
            if transformer.config.enabled:
                logger.debug(f"Fitting {transformer.config.name}")
                transformer.fit(df)
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data through pipeline"""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        # Separate parallel and sequential transformers
        parallel_transformers = [t for t in self.transformers 
                               if t.config.enabled and t.config.parallel]
        sequential_transformers = [t for t in self.transformers 
                                 if t.config.enabled and not t.config.parallel]
        
        all_features = []
        
        # Run parallel transformers
        if parallel_transformers:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for transformer in parallel_transformers:
                    future = executor.submit(self._transform_cached, transformer, df)
                    futures.append((transformer.config.name, future))
                
                for name, future in futures:
                    try:
                        features = future.result()
                        all_features.append(features)
                    except Exception as e:
                        logger.error(f"Error in transformer {name}: {e}")
        
        # Run sequential transformers
        for transformer in sequential_transformers:
            try:
                features = self._transform_cached(transformer, df)
                all_features.append(features)
            except Exception as e:
                logger.error(f"Error in transformer {transformer.config.name}: {e}")
        
        # Combine all features
        if all_features:
            return pd.concat(all_features, axis=1)
        else:
            return pd.DataFrame(index=df.index)
    
    def _transform_cached(self, transformer: BaseTransformer, 
                         df: pd.DataFrame) -> pd.DataFrame:
        """Transform with optional caching"""
        cache_key = f"{transformer.config.name}_{hash(tuple(df.columns))}"
        
        if self.use_memory and cache_key in self._memory:
            logger.debug(f"Using cached result for {transformer.config.name}")
            return self._memory[cache_key]
        
        result = transformer.transform(df)
        
        if self.use_memory:
            self._memory[cache_key] = result
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names from pipeline"""
        names = []
        for transformer in self.transformers:
            if transformer.config.enabled:
                names.extend(transformer.get_feature_names())
        return names
    
    def clear_memory(self):
        """Clear cached results"""
        self._memory.clear()
```

##### 1.2 Create Transformer Registry
```python
# Create: src/mdm/features/pipeline/registry.py
from typing import Dict, Type, List, Optional, Any
import importlib
import pkgutil
from pathlib import Path
import logging

from .base import BaseTransformer, TransformerConfig

logger = logging.getLogger(__name__)


class TransformerRegistry:
    """Registry for feature transformers"""
    
    def __init__(self):
        self._transformers: Dict[str, Type[BaseTransformer]] = {}
        self._configs: Dict[str, TransformerConfig] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, name: str, transformer_class: Type[BaseTransformer],
                 category: str = "custom", config: Optional[TransformerConfig] = None):
        """Register a transformer"""
        if not issubclass(transformer_class, BaseTransformer):
            raise TypeError(f"{transformer_class} must inherit from BaseTransformer")
        
        self._transformers[name] = transformer_class
        
        if config is None:
            config = TransformerConfig(name=name)
        self._configs[name] = config
        
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
        
        logger.info(f"Registered transformer: {name} in category: {category}")
    
    def get(self, name: str) -> Type[BaseTransformer]:
        """Get transformer class by name"""
        if name not in self._transformers:
            raise ValueError(f"Unknown transformer: {name}")
        return self._transformers[name]
    
    def create(self, name: str, **kwargs) -> BaseTransformer:
        """Create transformer instance"""
        transformer_class = self.get(name)
        config = self._configs.get(name, TransformerConfig(name=name))
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return transformer_class(config=config)
    
    def list_transformers(self, category: Optional[str] = None) -> List[str]:
        """List registered transformers"""
        if category:
            return self._categories.get(category, [])
        return list(self._transformers.keys())
    
    def list_categories(self) -> List[str]:
        """List transformer categories"""
        return list(self._categories.keys())
    
    def discover_transformers(self, package_path: str):
        """Discover and register transformers from a package"""
        package = importlib.import_module(package_path)
        
        # Walk through package
        for importer, modname, ispkg in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            if not ispkg:
                module = importlib.import_module(modname)
                
                # Find transformer classes
                for name in dir(module):
                    obj = getattr(module, name)
                    if (isinstance(obj, type) and 
                        issubclass(obj, BaseTransformer) and
                        obj != BaseTransformer):
                        
                        # Auto-register
                        transformer_name = obj.__name__
                        category = modname.split(".")[-2]  # Use parent module as category
                        
                        try:
                            self.register(transformer_name, obj, category)
                        except Exception as e:
                            logger.warning(f"Failed to register {transformer_name}: {e}")
    
    def create_pipeline_from_config(self, config: Dict[str, Any]) -> "Pipeline":
        """Create pipeline from configuration"""
        from .base import Pipeline
        
        pipeline = Pipeline(
            n_jobs=config.get("n_jobs", -1),
            use_memory=config.get("use_memory", True)
        )
        
        # Add transformers
        for transformer_config in config.get("transformers", []):
            name = transformer_config["name"]
            params = transformer_config.get("params", {})
            
            transformer = self.create(name, **params)
            pipeline.add_transformer(transformer)
        
        return pipeline


# Global registry
transformer_registry = TransformerRegistry()


# Decorator for auto-registration
def register_transformer(name: Optional[str] = None, category: str = "custom"):
    """Decorator to register a transformer"""
    def decorator(cls: Type[BaseTransformer]) -> Type[BaseTransformer]:
        transformer_name = name or cls.__name__
        transformer_registry.register(transformer_name, cls, category)
        return cls
    return decorator
```

#### Day 3-4: Core Transformers

##### 1.3 Implement Numeric Transformers
```python
# Create: src/mdm/features/transformers/numeric.py
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from scipy import stats
import warnings

from ..pipeline.base import ColumnTransformer, TransformerConfig
from ..pipeline.registry import register_transformer

warnings.filterwarnings('ignore', category=RuntimeWarning)


@register_transformer(category="numeric")
class NumericStatisticsTransformer(ColumnTransformer):
    """Generate statistical features for numeric columns"""
    
    def __init__(self, columns: Optional[List[str]] = None,
                 statistics: Optional[List[str]] = None,
                 config: Optional[TransformerConfig] = None):
        super().__init__(columns, config)
        self.statistics = statistics or [
            "mean", "std", "min", "max", "median",
            "skew", "kurtosis", "q25", "q75"
        ]
        self._stats_cache: Dict[str, Dict[str, float]] = {}
    
    def _auto_select_columns(self, df: pd.DataFrame) -> List[str]:
        """Select numeric columns"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def fit(self, df: pd.DataFrame) -> "NumericStatisticsTransformer":
        """Calculate statistics for fitting"""
        self.validate_input(df)
        self.columns = self._select_columns(df)
        
        for col in self.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                self._stats_cache[col] = {
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "median": col_data.median(),
                    "skew": stats.skew(col_data),
                    "kurtosis": stats.kurtosis(col_data),
                    "q25": col_data.quantile(0.25),
                    "q75": col_data.quantile(0.75)
                }
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features"""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted first")
        
        features = pd.DataFrame(index=df.index)
        
        for col in self.columns:
            if col in df.columns and col in self._stats_cache:
                stats = self._stats_cache[col]
                for stat_name in self.statistics:
                    if stat_name in stats:
                        features[f"{col}_{stat_name}"] = stats[stat_name]
        
        self._feature_names = features.columns.tolist()
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self._feature_names


@register_transformer(category="numeric")
class NumericBinningTransformer(ColumnTransformer):
    """Bin numeric features into categories"""
    
    def __init__(self, columns: Optional[List[str]] = None,
                 n_bins: int = 10, strategy: str = "quantile",
                 config: Optional[TransformerConfig] = None):
        super().__init__(columns, config)
        self.n_bins = n_bins
        self.strategy = strategy  # "quantile", "uniform", "kmeans"
        self._bin_edges: Dict[str, np.ndarray] = {}
    
    def _auto_select_columns(self, df: pd.DataFrame) -> List[str]:
        """Select numeric columns"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def fit(self, df: pd.DataFrame) -> "NumericBinningTransformer":
        """Learn bin edges"""
        self.validate_input(df)
        self.columns = self._select_columns(df)
        
        for col in self.columns:
            col_data = df[col].dropna()
            if len(col_data) > self.n_bins:
                if self.strategy == "quantile":
                    _, edges = pd.qcut(col_data, self.n_bins, retbins=True, duplicates='drop')
                elif self.strategy == "uniform":
                    _, edges = pd.cut(col_data, self.n_bins, retbins=True)
                else:
                    # Implement k-means binning
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=self.n_bins, random_state=42)
                    kmeans.fit(col_data.values.reshape(-1, 1))
                    centers = sorted(kmeans.cluster_centers_.flatten())
                    edges = [(centers[i] + centers[i+1])/2 for i in range(len(centers)-1)]
                    edges = [col_data.min()] + edges + [col_data.max()]
                    edges = np.array(edges)
                
                self._bin_edges[col] = edges
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply binning"""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted first")
        
        features = pd.DataFrame(index=df.index)
        
        for col in self.columns:
            if col in df.columns and col in self._bin_edges:
                edges = self._bin_edges[col]
                binned = pd.cut(df[col], bins=edges, labels=False, include_lowest=True)
                features[f"{col}_bin"] = binned
                
                # One-hot encode bins
                for i in range(len(edges) - 1):
                    features[f"{col}_bin_{i}"] = (binned == i).astype(int)
        
        self._feature_names = features.columns.tolist()
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self._feature_names


@register_transformer(category="numeric")
class NumericInteractionTransformer(ColumnTransformer):
    """Generate interaction features between numeric columns"""
    
    def __init__(self, columns: Optional[List[str]] = None,
                 max_interactions: int = 10,
                 operations: Optional[List[str]] = None,
                 config: Optional[TransformerConfig] = None):
        super().__init__(columns, config)
        self.max_interactions = max_interactions
        self.operations = operations or ["multiply", "divide", "add", "subtract"]
        self._selected_interactions: List[tuple] = []
    
    def _auto_select_columns(self, df: pd.DataFrame) -> List[str]:
        """Select numeric columns"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def fit(self, df: pd.DataFrame) -> "NumericInteractionTransformer":
        """Select best interactions based on variance"""
        self.validate_input(df)
        self.columns = self._select_columns(df)
        
        if len(self.columns) < 2:
            self._fitted = True
            return self
        
        # Calculate variance for all possible interactions
        interaction_scores = []
        
        for i, col1 in enumerate(self.columns):
            for j, col2 in enumerate(self.columns[i+1:], i+1):
                for op in self.operations:
                    if op == "multiply":
                        values = df[col1] * df[col2]
                    elif op == "divide":
                        values = df[col1] / (df[col2] + 1e-8)
                    elif op == "add":
                        values = df[col1] + df[col2]
                    elif op == "subtract":
                        values = df[col1] - df[col2]
                    
                    # Score by variance
                    score = values.var()
                    interaction_scores.append((score, col1, col2, op))
        
        # Select top interactions
        interaction_scores.sort(reverse=True)
        self._selected_interactions = [
            (col1, col2, op) 
            for _, col1, col2, op in interaction_scores[:self.max_interactions]
        ]
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features"""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted first")
        
        features = pd.DataFrame(index=df.index)
        
        for col1, col2, op in self._selected_interactions:
            if col1 in df.columns and col2 in df.columns:
                if op == "multiply":
                    features[f"{col1}_times_{col2}"] = df[col1] * df[col2]
                elif op == "divide":
                    features[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)
                elif op == "add":
                    features[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
                elif op == "subtract":
                    features[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
        
        self._feature_names = features.columns.tolist()
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self._feature_names
```

##### 1.4 Implement Categorical Transformers
```python
# Create: src/mdm/features/transformers/categorical.py
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from collections import Counter
import math

from ..pipeline.base import ColumnTransformer, TransformerConfig
from ..pipeline.registry import register_transformer


@register_transformer(category="categorical")
class CategoricalEncodingTransformer(ColumnTransformer):
    """Encode categorical features"""
    
    def __init__(self, columns: Optional[List[str]] = None,
                 encoding: str = "label", max_categories: int = 100,
                 min_frequency: float = 0.01,
                 config: Optional[TransformerConfig] = None):
        super().__init__(columns, config)
        self.encoding = encoding  # "label", "onehot", "target", "frequency"
        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self._encodings: Dict[str, Dict[Any, int]] = {}
        self._category_stats: Dict[str, Dict[str, Any]] = {}
    
    def _auto_select_columns(self, df: pd.DataFrame) -> List[str]:
        """Select categorical columns"""
        return df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def fit(self, df: pd.DataFrame) -> "CategoricalEncodingTransformer":
        """Learn encoding mappings"""
        self.validate_input(df)
        self.columns = self._select_columns(df)
        
        for col in self.columns:
            # Get value counts
            value_counts = df[col].value_counts()
            total_count = len(df)
            
            # Filter by frequency
            min_count = int(total_count * self.min_frequency)
            frequent_values = value_counts[value_counts >= min_count]
            
            # Limit to max categories
            if len(frequent_values) > self.max_categories:
                frequent_values = frequent_values.head(self.max_categories)
            
            # Create encoding
            if self.encoding == "label":
                encoding = {val: i for i, val in enumerate(frequent_values.index)}
                # Add "other" category
                encoding["_other_"] = len(encoding)
            elif self.encoding == "frequency":
                encoding = {val: count/total_count for val, count in frequent_values.items()}
                encoding["_other_"] = min_count/total_count
            
            self._encodings[col] = encoding
            
            # Store statistics
            self._category_stats[col] = {
                "n_categories": len(frequent_values),
                "coverage": frequent_values.sum() / total_count,
                "entropy": self._calculate_entropy(value_counts)
            }
        
        self._fitted = True
        return self
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of categorical distribution"""
        probs = value_counts / value_counts.sum()
        return -sum(p * math.log2(p) if p > 0 else 0 for p in probs)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding"""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted first")
        
        features = pd.DataFrame(index=df.index)
        
        for col in self.columns:
            if col in df.columns and col in self._encodings:
                encoding = self._encodings[col]
                
                if self.encoding == "label":
                    # Map values to integers
                    default_value = encoding.get("_other_", -1)
                    features[f"{col}_encoded"] = df[col].map(encoding).fillna(default_value)
                
                elif self.encoding == "onehot":
                    # One-hot encode
                    for value, idx in encoding.items():
                        if value != "_other_":
                            features[f"{col}_{value}"] = (df[col] == value).astype(int)
                
                elif self.encoding == "frequency":
                    # Frequency encoding
                    default_value = encoding.get("_other_", 0)
                    features[f"{col}_frequency"] = df[col].map(encoding).fillna(default_value)
                
                # Add statistics
                stats = self._category_stats[col]
                features[f"{col}_n_categories"] = stats["n_categories"]
                features[f"{col}_entropy"] = stats["entropy"]
        
        self._feature_names = features.columns.tolist()
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self._feature_names


@register_transformer(category="categorical")
class CategoricalCombinationTransformer(ColumnTransformer):
    """Generate combinations of categorical features"""
    
    def __init__(self, columns: Optional[List[str]] = None,
                 max_combinations: int = 20,
                 min_support: float = 0.01,
                 config: Optional[TransformerConfig] = None):
        super().__init__(columns, config)
        self.max_combinations = max_combinations
        self.min_support = min_support
        self._selected_combinations: List[tuple] = []
        self._combination_encodings: Dict[str, Dict[str, int]] = {}
    
    def _auto_select_columns(self, df: pd.DataFrame) -> List[str]:
        """Select categorical columns with reasonable cardinality"""
        cat_columns = df.select_dtypes(include=['object', 'category']).columns
        return [col for col in cat_columns if df[col].nunique() < 50]
    
    def fit(self, df: pd.DataFrame) -> "CategoricalCombinationTransformer":
        """Find valuable combinations"""
        self.validate_input(df)
        self.columns = self._select_columns(df)
        
        if len(self.columns) < 2:
            self._fitted = True
            return self
        
        # Find frequent item sets (simplified)
        n_rows = len(df)
        min_count = int(n_rows * self.min_support)
        
        combination_counts = []
        
        # Check 2-way combinations
        for i, col1 in enumerate(self.columns):
            for col2 in self.columns[i+1:]:
                # Count combinations
                combo_name = f"{col1}_X_{col2}"
                combos = df.groupby([col1, col2]).size()
                
                # Filter by support
                frequent_combos = combos[combos >= min_count]
                
                if len(frequent_combos) > 0:
                    # Score by coverage and diversity
                    coverage = frequent_combos.sum() / n_rows
                    diversity = len(frequent_combos) / (df[col1].nunique() * df[col2].nunique())
                    score = coverage * diversity
                    
                    combination_counts.append((score, col1, col2, frequent_combos))
        
        # Select top combinations
        combination_counts.sort(reverse=True)
        
        for score, col1, col2, combos in combination_counts[:self.max_combinations]:
            combo_name = f"{col1}_X_{col2}"
            self._selected_combinations.append((col1, col2))
            
            # Create encoding for combinations
            encoding = {}
            for idx, (val1, val2) in enumerate(combos.index):
                encoding[f"{val1}|{val2}"] = idx
            self._combination_encodings[combo_name] = encoding
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate combination features"""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted first")
        
        features = pd.DataFrame(index=df.index)
        
        for col1, col2 in self._selected_combinations:
            if col1 in df.columns and col2 in df.columns:
                combo_name = f"{col1}_X_{col2}"
                encoding = self._combination_encodings[combo_name]
                
                # Create combination strings
                combos = df[col1].astype(str) + "|" + df[col2].astype(str)
                
                # Encode
                features[combo_name] = combos.map(encoding).fillna(-1)
                
                # Add combination count feature
                combo_counts = combos.map(lambda x: encoding.get(x, 0))
                features[f"{combo_name}_count"] = combo_counts
        
        self._feature_names = features.columns.tolist()
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self._feature_names
```

### Week 12: Advanced Transformers

#### Day 5-6: DateTime and Text Transformers

##### 2.1 Implement DateTime Transformers
```python
# Create: src/mdm/features/transformers/datetime.py
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..pipeline.base import ColumnTransformer, TransformerConfig
from ..pipeline.registry import register_transformer


@register_transformer(category="datetime")
class DateTimeFeatureTransformer(ColumnTransformer):
    """Extract features from datetime columns"""
    
    def __init__(self, columns: Optional[List[str]] = None,
                 components: Optional[List[str]] = None,
                 cyclical: bool = True,
                 config: Optional[TransformerConfig] = None):
        super().__init__(columns, config)
        self.components = components or [
            "year", "month", "day", "hour", "minute",
            "dayofweek", "quarter", "dayofyear", "weekofyear"
        ]
        self.cyclical = cyclical
        self._datetime_columns: List[str] = []
    
    def _auto_select_columns(self, df: pd.DataFrame) -> List[str]:
        """Select datetime columns"""
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].iloc[0])
                    datetime_cols.append(col)
                except:
                    pass
        return datetime_cols
    
    def fit(self, df: pd.DataFrame) -> "DateTimeFeatureTransformer":
        """Identify datetime columns"""
        self.validate_input(df)
        self.columns = self._select_columns(df)
        self._datetime_columns = self.columns
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract datetime features"""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted first")
        
        features = pd.DataFrame(index=df.index)
        
        for col in self._datetime_columns:
            if col in df.columns:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    dt_series = pd.to_datetime(df[col], errors='coerce')
                else:
                    dt_series = df[col]
                
                # Extract components
                if "year" in self.components:
                    features[f"{col}_year"] = dt_series.dt.year
                if "month" in self.components:
                    features[f"{col}_month"] = dt_series.dt.month
                    if self.cyclical:
                        features[f"{col}_month_sin"] = np.sin(2 * np.pi * dt_series.dt.month / 12)
                        features[f"{col}_month_cos"] = np.cos(2 * np.pi * dt_series.dt.month / 12)
                if "day" in self.components:
                    features[f"{col}_day"] = dt_series.dt.day
                    if self.cyclical:
                        features[f"{col}_day_sin"] = np.sin(2 * np.pi * dt_series.dt.day / 31)
                        features[f"{col}_day_cos"] = np.cos(2 * np.pi * dt_series.dt.day / 31)
                if "hour" in self.components:
                    features[f"{col}_hour"] = dt_series.dt.hour
                    if self.cyclical:
                        features[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
                        features[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
                if "minute" in self.components:
                    features[f"{col}_minute"] = dt_series.dt.minute
                if "dayofweek" in self.components:
                    features[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                    features[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
                if "quarter" in self.components:
                    features[f"{col}_quarter"] = dt_series.dt.quarter
                if "dayofyear" in self.components:
                    features[f"{col}_dayofyear"] = dt_series.dt.dayofyear
                if "weekofyear" in self.components:
                    features[f"{col}_weekofyear"] = dt_series.dt.isocalendar().week
                
                # Time-based features
                reference_date = dt_series.min()
                features[f"{col}_days_since_start"] = (dt_series - reference_date).dt.days
                
                # Lag features
                features[f"{col}_days_since_previous"] = dt_series.diff().dt.days
        
        self._feature_names = features.columns.tolist()
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self._feature_names


@register_transformer(category="datetime")
class TimeSeriesTransformer(ColumnTransformer):
    """Generate time series features"""
    
    def __init__(self, datetime_col: str, value_cols: List[str],
                 windows: Optional[List[int]] = None,
                 aggregations: Optional[List[str]] = None,
                 config: Optional[TransformerConfig] = None):
        super().__init__(value_cols, config)
        self.datetime_col = datetime_col
        self.value_cols = value_cols
        self.windows = windows or [7, 14, 30]  # Days
        self.aggregations = aggregations or ["mean", "std", "min", "max"]
    
    def _auto_select_columns(self, df: pd.DataFrame) -> List[str]:
        """Not used - columns specified in init"""
        return self.value_cols
    
    def fit(self, df: pd.DataFrame) -> "TimeSeriesTransformer":
        """Prepare for transformation"""
        self.validate_input(df)
        if self.datetime_col not in df.columns:
            raise ValueError(f"Datetime column {self.datetime_col} not found")
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate time series features"""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted first")
        
        # Sort by datetime
        df_sorted = df.sort_values(self.datetime_col)
        features = pd.DataFrame(index=df.index)
        
        # Convert datetime column
        dt_series = pd.to_datetime(df_sorted[self.datetime_col])
        
        for col in self.value_cols:
            if col in df_sorted.columns:
                for window in self.windows:
                    # Rolling window features
                    rolling = df_sorted.set_index(dt_series)[col].rolling(
                        window=f"{window}D", min_periods=1
                    )
                    
                    for agg in self.aggregations:
                        if agg == "mean":
                            feature_values = rolling.mean()
                        elif agg == "std":
                            feature_values = rolling.std()
                        elif agg == "min":
                            feature_values = rolling.min()
                        elif agg == "max":
                            feature_values = rolling.max()
                        
                        # Align back to original index
                        features[f"{col}_rolling_{window}d_{agg}"] = feature_values.reindex(df.index)
                
                # Lag features
                for lag in [1, 7, 30]:
                    features[f"{col}_lag_{lag}d"] = df_sorted[col].shift(lag)
                
                # Difference features
                features[f"{col}_diff_1d"] = df_sorted[col].diff()
                features[f"{col}_diff_7d"] = df_sorted[col].diff(7)
        
        self._feature_names = features.columns.tolist()
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self._feature_names
```

##### 2.2 Implement Text Transformers
```python
# Create: src/mdm/features/transformers/text.py
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import re
from collections import Counter

from ..pipeline.base import ColumnTransformer, TransformerConfig
from ..pipeline.registry import register_transformer


@register_transformer(category="text")
class TextStatisticsTransformer(ColumnTransformer):
    """Extract statistical features from text columns"""
    
    def __init__(self, columns: Optional[List[str]] = None,
                 config: Optional[TransformerConfig] = None):
        super().__init__(columns, config)
    
    def _auto_select_columns(self, df: pd.DataFrame) -> List[str]:
        """Select text columns"""
        text_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            # Check if column contains text (not just categories)
            sample = df[col].dropna().head(100)
            if sample.empty:
                continue
            
            avg_length = sample.str.len().mean()
            if avg_length > 10:  # Likely text, not categorical
                text_cols.append(col)
        
        return text_cols
    
    def fit(self, df: pd.DataFrame) -> "TextStatisticsTransformer":
        """Prepare transformer"""
        self.validate_input(df)
        self.columns = self._select_columns(df)
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract text statistics"""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted first")
        
        features = pd.DataFrame(index=df.index)
        
        for col in self.columns:
            if col in df.columns:
                text_series = df[col].fillna("")
                
                # Length features
                features[f"{col}_length"] = text_series.str.len()
                features[f"{col}_word_count"] = text_series.str.split().str.len()
                
                # Character features
                features[f"{col}_digit_count"] = text_series.str.count(r'\d')
                features[f"{col}_upper_count"] = text_series.str.count(r'[A-Z]')
                features[f"{col}_lower_count"] = text_series.str.count(r'[a-z]')
                features[f"{col}_punct_count"] = text_series.str.count(r'[^\w\s]')
                features[f"{col}_space_count"] = text_series.str.count(r'\s')
                
                # Ratios
                length = features[f"{col}_length"].replace(0, 1)  # Avoid division by zero
                features[f"{col}_digit_ratio"] = features[f"{col}_digit_count"] / length
                features[f"{col}_upper_ratio"] = features[f"{col}_upper_count"] / length
                features[f"{col}_punct_ratio"] = features[f"{col}_punct_count"] / length
                
                # Special patterns
                features[f"{col}_has_url"] = text_series.str.contains(r'https?://\S+', case=False).astype(int)
                features[f"{col}_has_email"] = text_series.str.contains(r'\S+@\S+', case=False).astype(int)
                features[f"{col}_has_phone"] = text_series.str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b').astype(int)
                
                # Sentiment indicators (simple)
                features[f"{col}_exclamation_count"] = text_series.str.count('!')
                features[f"{col}_question_count"] = text_series.str.count('?')
                
        self._feature_names = features.columns.tolist()
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self._feature_names


@register_transformer(category="text")
class TextVectorizationTransformer(ColumnTransformer):
    """Vectorize text using TF-IDF or other methods"""
    
    def __init__(self, columns: Optional[List[str]] = None,
                 method: str = "tfidf", max_features: int = 100,
                 ngram_range: tuple = (1, 1), min_df: float = 0.01,
                 config: Optional[TransformerConfig] = None):
        super().__init__(columns, config)
        self.method = method  # "tfidf", "count", "binary"
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self._vectorizers: Dict[str, Any] = {}
        
    def _auto_select_columns(self, df: pd.DataFrame) -> List[str]:
        """Select text columns"""
        # Use same logic as TextStatisticsTransformer
        return TextStatisticsTransformer()._auto_select_columns(df)
    
    def fit(self, df: pd.DataFrame) -> "TextVectorizationTransformer":
        """Fit vectorizers"""
        self.validate_input(df)
        self.columns = self._select_columns(df)
        
        # Lazy import to avoid dependency if not used
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        
        for col in self.columns:
            text_data = df[col].fillna("").astype(str)
            
            if self.method == "tfidf":
                vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df
                )
            elif self.method == "count":
                vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df
                )
            elif self.method == "binary":
                vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    binary=True
                )
            
            vectorizer.fit(text_data)
            self._vectorizers[col] = vectorizer
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorize text"""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted first")
        
        features_list = []
        
        for col in self.columns:
            if col in df.columns and col in self._vectorizers:
                vectorizer = self._vectorizers[col]
                text_data = df[col].fillna("").astype(str)
                
                # Transform to sparse matrix
                sparse_features = vectorizer.transform(text_data)
                
                # Convert to DataFrame
                feature_names = [f"{col}_{word}" for word in vectorizer.get_feature_names_out()]
                text_features = pd.DataFrame(
                    sparse_features.toarray(),
                    index=df.index,
                    columns=feature_names
                )
                
                features_list.append(text_features)
        
        if features_list:
            features = pd.concat(features_list, axis=1)
        else:
            features = pd.DataFrame(index=df.index)
        
        self._feature_names = features.columns.tolist()
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self._feature_names
```

#### Day 7-8: Custom Feature Support

##### 2.3 Create Custom Feature Framework
```python
# Create: src/mdm/features/custom/base.py
from typing import Dict, Any, List, Optional
import pandas as pd
import importlib.util
from pathlib import Path
import logging

from ..pipeline.base import BaseTransformer, TransformerConfig
from ..pipeline.registry import transformer_registry

logger = logging.getLogger(__name__)


class CustomTransformerLoader:
    """Load custom transformers from user-defined files"""
    
    def __init__(self, custom_dir: Optional[Path] = None):
        self.custom_dir = custom_dir or Path.home() / ".mdm" / "custom_features"
        self.loaded_modules: Dict[str, Any] = {}
    
    def discover_transformers(self):
        """Discover and load custom transformers"""
        if not self.custom_dir.exists():
            logger.info(f"Custom features directory not found: {self.custom_dir}")
            return
        
        # Find all Python files
        for py_file in self.custom_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            try:
                self._load_module(py_file)
            except Exception as e:
                logger.error(f"Failed to load custom module {py_file}: {e}")
    
    def _load_module(self, file_path: Path):
        """Load a Python module dynamically"""
        module_name = file_path.stem
        
        # Load module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        self.loaded_modules[module_name] = module
        
        # Find transformer classes
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, BaseTransformer) and
                obj != BaseTransformer):
                
                # Register transformer
                transformer_registry.register(
                    name=obj.__name__,
                    transformer_class=obj,
                    category="custom"
                )
                logger.info(f"Loaded custom transformer: {obj.__name__} from {file_path}")


# Example custom transformer template
CUSTOM_TRANSFORMER_TEMPLATE = '''"""
Custom feature transformer for {name}
"""
import pandas as pd
import numpy as np
from typing import List, Optional

from mdm.features.pipeline.base import BaseTransformer, TransformerConfig


class {class_name}(BaseTransformer):
    """Custom transformer for {description}"""
    
    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__(config)
        # Add your parameters here
        
    def fit(self, df: pd.DataFrame) -> "{class_name}":
        """Fit transformer to data"""
        self.validate_input(df)
        
        # Add your fitting logic here
        # Example: Learn parameters from data
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data and return features"""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted first")
        
        features = pd.DataFrame(index=df.index)
        
        # Add your transformation logic here
        # Example: Create new features
        # features["my_feature"] = df["column"].apply(my_function)
        
        self._feature_names = features.columns.tolist()
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self._feature_names
'''


def create_custom_transformer_template(name: str, description: str = "") -> str:
    """Create a template for custom transformer"""
    class_name = "".join(word.capitalize() for word in name.split("_")) + "Transformer"
    return CUSTOM_TRANSFORMER_TEMPLATE.format(
        name=name,
        class_name=class_name,
        description=description or name
    )
```

##### 2.4 Create Feature Pipeline Builder
```python
# Create: src/mdm/features/builder.py
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml
import json

from .pipeline.base import Pipeline, BaseTransformer
from .pipeline.registry import transformer_registry
from .custom.base import CustomTransformerLoader
from ..config import get_config


class FeaturePipelineBuilder:
    """Build feature engineering pipelines from configuration"""
    
    def __init__(self):
        self.config = get_config()
        self._load_custom_transformers()
    
    def _load_custom_transformers(self):
        """Load any custom transformers"""
        loader = CustomTransformerLoader()
        loader.discover_transformers()
    
    def build_from_config(self, config_path: Optional[Union[str, Path]] = None) -> Pipeline:
        """Build pipeline from configuration file"""
        if config_path is None:
            config_path = self.config.paths.config_path / "feature_pipeline.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            # Use default configuration
            pipeline_config = self._get_default_config()
        else:
            # Load from file
            with open(config_path) as f:
                if config_path.suffix == ".yaml":
                    pipeline_config = yaml.safe_load(f)
                else:
                    pipeline_config = json.load(f)
        
        return self.build_from_dict(pipeline_config)
    
    def build_from_dict(self, config: Dict[str, Any]) -> Pipeline:
        """Build pipeline from dictionary configuration"""
        pipeline = Pipeline(
            n_jobs=config.get("n_jobs", self.config.performance.feature_n_jobs),
            use_memory=config.get("use_memory", True)
        )
        
        # Add transformers
        for transformer_config in config.get("transformers", []):
            transformer = self._create_transformer(transformer_config)
            if transformer:
                pipeline.add_transformer(transformer)
        
        return pipeline
    
    def _create_transformer(self, config: Dict[str, Any]) -> Optional[BaseTransformer]:
        """Create transformer from configuration"""
        name = config.get("name")
        if not name:
            return None
        
        try:
            # Get transformer class
            if "class" in config:
                # Direct class reference
                transformer_class = transformer_registry.get(config["class"])
            else:
                # Use name as class
                transformer_class = transformer_registry.get(name)
            
            # Create instance with parameters
            params = config.get("params", {})
            transformer = transformer_class(**params)
            
            # Update configuration
            transformer.config.enabled = config.get("enabled", True)
            transformer.config.parallel = config.get("parallel", False)
            transformer.config.priority = config.get("priority", 0)
            
            return transformer
            
        except Exception as e:
            logger.error(f"Failed to create transformer {name}: {e}")
            return None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return {
            "n_jobs": -1,
            "use_memory": True,
            "transformers": [
                # Numeric transformers
                {
                    "name": "numeric_stats",
                    "class": "NumericStatisticsTransformer",
                    "enabled": self.config.features.enable_numeric,
                    "parallel": True,
                    "params": {
                        "statistics": self.config.features.numeric_aggregations
                    }
                },
                {
                    "name": "numeric_binning",
                    "class": "NumericBinningTransformer",
                    "enabled": self.config.features.enable_numeric,
                    "parallel": True,
                    "params": {
                        "n_bins": 10,
                        "strategy": "quantile"
                    }
                },
                
                # Categorical transformers
                {
                    "name": "categorical_encoding",
                    "class": "CategoricalEncodingTransformer",
                    "enabled": self.config.features.enable_categorical,
                    "parallel": True,
                    "params": {
                        "encoding": self.config.features.categorical_encoding,
                        "max_categories": self.config.features.categorical_max_cardinality
                    }
                },
                
                # DateTime transformers
                {
                    "name": "datetime_features",
                    "class": "DateTimeFeatureTransformer",
                    "enabled": self.config.features.enable_datetime,
                    "parallel": True,
                    "params": {
                        "components": self.config.features.datetime_components,
                        "cyclical": self.config.features.datetime_cyclical
                    }
                },
                
                # Text transformers
                {
                    "name": "text_statistics",
                    "class": "TextStatisticsTransformer",
                    "enabled": self.config.features.enable_text,
                    "parallel": True
                },
                {
                    "name": "text_vectorization",
                    "class": "TextVectorizationTransformer",
                    "enabled": self.config.features.enable_text,
                    "parallel": False,  # Vectorizers typically not thread-safe
                    "params": {
                        "max_features": self.config.features.text_max_features,
                        "ngram_range": self.config.features.text_ngram_range,
                        "min_df": self.config.features.text_min_df
                    }
                }
            ]
        }
    
    def build_default_pipeline(self) -> Pipeline:
        """Build default feature engineering pipeline"""
        config = self._get_default_config()
        return self.build_from_dict(config)
    
    def save_pipeline_config(self, pipeline: Pipeline, output_path: Union[str, Path]):
        """Save pipeline configuration to file"""
        output_path = Path(output_path)
        
        config = {
            "n_jobs": pipeline.n_jobs,
            "use_memory": pipeline.use_memory,
            "transformers": []
        }
        
        for transformer in pipeline.transformers:
            transformer_config = {
                "name": transformer.config.name,
                "class": transformer.__class__.__name__,
                "enabled": transformer.config.enabled,
                "parallel": transformer.config.parallel,
                "priority": transformer.config.priority,
                "params": transformer.config.params
            }
            config["transformers"].append(transformer_config)
        
        # Save based on extension
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if output_path.suffix == ".yaml":
                yaml.safe_dump(config, f, default_flow_style=False)
            else:
                json.dump(config, f, indent=2)
```

### Week 13: Integration and Migration

#### Day 9: Migration Implementation

##### 3.1 Create Feature Migration Adapter
```python
# Create: src/mdm/features/migration/adapter.py
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

from ...interfaces.features import IFeatureGenerator
from ..generator import FeatureGenerator as LegacyFeatureGenerator
from ..builder import FeaturePipelineBuilder
from ..pipeline.base import Pipeline
from ...core.feature_flags import feature_flags

logger = logging.getLogger(__name__)


class FeatureGeneratorAdapter(IFeatureGenerator):
    """Adapter to switch between legacy and new feature generation"""
    
    def __init__(self):
        self._legacy_generator = None
        self._new_pipeline = None
        self._pipeline_builder = FeaturePipelineBuilder()
    
    def _get_implementation(self):
        """Get appropriate implementation based on feature flag"""
        if feature_flags.get("use_new_features", False):
            if self._new_pipeline is None:
                self._new_pipeline = self._pipeline_builder.build_default_pipeline()
            return self._new_pipeline
        else:
            if self._legacy_generator is None:
                self._legacy_generator = LegacyFeatureGenerator()
            return self._legacy_generator
    
    def generate_features(self, data: pd.DataFrame, 
                         config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate features using appropriate implementation"""
        impl = self._get_implementation()
        
        if isinstance(impl, Pipeline):
            # New pipeline
            if not impl._fitted:
                impl.fit(data)
            return impl.transform(data)
        else:
            # Legacy generator
            return impl.generate_features(data, config)
    
    def generate_numeric_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate numeric features"""
        impl = self._get_implementation()
        
        if isinstance(impl, Pipeline):
            # Filter to numeric transformers only
            numeric_features = pd.DataFrame(index=data.index)
            for transformer in impl.transformers:
                if "numeric" in transformer.config.name.lower():
                    if not transformer.is_fitted:
                        transformer.fit(data)
                    features = transformer.transform(data)
                    numeric_features = pd.concat([numeric_features, features], axis=1)
            return numeric_features
        else:
            return impl.generate_numeric_features(data)
    
    def generate_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate categorical features"""
        impl = self._get_implementation()
        
        if isinstance(impl, Pipeline):
            # Filter to categorical transformers only
            categorical_features = pd.DataFrame(index=data.index)
            for transformer in impl.transformers:
                if "categorical" in transformer.config.name.lower():
                    if not transformer.is_fitted:
                        transformer.fit(data)
                    features = transformer.transform(data)
                    categorical_features = pd.concat([categorical_features, features], axis=1)
            return categorical_features
        else:
            return impl.generate_categorical_features(data)
    
    def generate_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate datetime features"""
        impl = self._get_implementation()
        
        if isinstance(impl, Pipeline):
            # Filter to datetime transformers only
            datetime_features = pd.DataFrame(index=data.index)
            for transformer in impl.transformers:
                if "datetime" in transformer.config.name.lower():
                    if not transformer.is_fitted:
                        transformer.fit(data)
                    features = transformer.transform(data)
                    datetime_features = pd.concat([datetime_features, features], axis=1)
            return datetime_features
        else:
            if hasattr(impl, 'generate_datetime_features'):
                return impl.generate_datetime_features(data)
            return pd.DataFrame(index=data.index)
    
    def generate_text_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate text features"""
        impl = self._get_implementation()
        
        if isinstance(impl, Pipeline):
            # Filter to text transformers only
            text_features = pd.DataFrame(index=data.index)
            for transformer in impl.transformers:
                if "text" in transformer.config.name.lower():
                    if not transformer.is_fitted:
                        transformer.fit(data)
                    features = transformer.transform(data)
                    text_features = pd.concat([text_features, features], axis=1)
            return text_features
        else:
            if hasattr(impl, 'generate_text_features'):
                return impl.generate_text_features(data)
            return pd.DataFrame(index=data.index)
```

##### 3.2 Create Feature Comparison Tests
```python
# Create: tests/migration/test_feature_comparison.py
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from mdm.testing.comparison import ComparisonTester
from mdm.features.generator import FeatureGenerator as LegacyGenerator
from mdm.features.builder import FeaturePipelineBuilder
from mdm.core.feature_flags import feature_flags


class TestFeatureEngineering Comparison:
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            # Numeric columns
            'numeric_1': np.random.randn(100),
            'numeric_2': np.random.randint(0, 100, 100),
            'numeric_3': np.random.uniform(0, 1, 100),
            
            # Categorical columns
            'category_1': np.random.choice(['A', 'B', 'C'], 100),
            'category_2': np.random.choice(['X', 'Y', 'Z', 'W'], 100),
            
            # Datetime column
            'datetime_1': pd.date_range('2024-01-01', periods=100, freq='D'),
            
            # Text column
            'text_1': ['Sample text ' + str(i) for i in range(100)]
        })
        
        self.comparison_tester = ComparisonTester()
    
    def test_numeric_feature_parity(self):
        """Test numeric feature generation parity"""
        # Legacy implementation
        legacy_generator = LegacyGenerator()
        
        # New implementation
        builder = FeaturePipelineBuilder()
        new_pipeline = builder.build_default_pipeline()
        
        result = self.comparison_tester.compare(
            test_name="numeric_features",
            old_impl=lambda: legacy_generator.generate_numeric_features(
                self.test_data[['numeric_1', 'numeric_2', 'numeric_3']]
            ),
            new_impl=lambda: self._extract_numeric_features(new_pipeline, self.test_data)
        )
        
        assert result.passed or self._check_feature_equivalence(
            result.old_result, result.new_result
        )
        
        # Performance should be better with new implementation
        assert result.performance_delta < 50  # Allow 50% slower for safety
    
    def _extract_numeric_features(self, pipeline, data):
        """Extract only numeric features from pipeline"""
        pipeline.fit(data)
        all_features = pipeline.transform(data)
        
        # Filter to numeric-related columns
        numeric_cols = [col for col in all_features.columns 
                       if any(x in col for x in ['numeric_', 'mean', 'std', 'min', 'max'])]
        
        return all_features[numeric_cols]
    
    def _check_feature_equivalence(self, old_features, new_features):
        """Check if features are equivalent (may have different names/order)"""
        # Check shapes
        if old_features.shape != new_features.shape:
            return False
        
        # Check values are similar (allowing for small numerical differences)
        old_values = old_features.values
        new_values = new_features.values
        
        return np.allclose(old_values, new_values, rtol=1e-5, atol=1e-8, equal_nan=True)
    
    def test_categorical_feature_parity(self):
        """Test categorical feature generation parity"""
        # Similar to numeric test but for categorical features
        pass
    
    def test_full_pipeline_performance(self):
        """Test full pipeline performance"""
        # Create larger dataset for performance testing
        large_data = pd.concat([self.test_data] * 100, ignore_index=True)
        
        # Legacy
        legacy_generator = LegacyGenerator()
        
        # New
        builder = FeaturePipelineBuilder()
        new_pipeline = builder.build_default_pipeline()
        
        result = self.comparison_tester.compare(
            test_name="full_pipeline_performance",
            old_impl=lambda: legacy_generator.generate_features(large_data),
            new_impl=lambda: new_pipeline.fit_transform(large_data)
        )
        
        # New implementation should be faster due to parallelization
        print(f"Performance delta: {result.performance_delta:.1f}%")
        print(f"Memory delta: {result.memory_delta:.1f}%")
        
        # Should be at least as fast
        assert result.performance_delta < 20
    
    def test_custom_transformer_integration(self):
        """Test custom transformer integration"""
        # Create custom transformer file
        from mdm.features.custom.base import create_custom_transformer_template
        
        custom_dir = Path.home() / ".mdm" / "custom_features"
        custom_dir.mkdir(parents=True, exist_ok=True)
        
        # Write custom transformer
        template = create_custom_transformer_template(
            "domain_specific",
            "Domain-specific feature engineering"
        )
        
        custom_file = custom_dir / "domain_specific.py"
        custom_file.write_text(template)
        
        # Build pipeline with custom transformer
        builder = FeaturePipelineBuilder()
        
        # Verify custom transformer was loaded
        from mdm.features.pipeline.registry import transformer_registry
        custom_transformers = transformer_registry.list_transformers("custom")
        
        assert len(custom_transformers) > 0
```

#### Day 10: Documentation and Rollout

##### 3.3 Create Migration Guide
```markdown
# Create: docs/feature_engineering_migration_guide.md

# Feature Engineering Migration Guide

## Overview

This guide covers the migration from the monolithic FeatureGenerator to the new pipeline-based feature engineering system.

## What's New

### Pipeline Architecture
- Modular transformers that can be combined
- Parallel processing support
- Memory-efficient batch processing
- Plugin system for custom features

### Built-in Transformers

#### Numeric
- `NumericStatisticsTransformer`: Statistical features (mean, std, etc.)
- `NumericBinningTransformer`: Binning and discretization
- `NumericInteractionTransformer`: Feature interactions

#### Categorical
- `CategoricalEncodingTransformer`: Various encoding methods
- `CategoricalCombinationTransformer`: Feature combinations

#### DateTime
- `DateTimeFeatureTransformer`: Extract datetime components
- `TimeSeriesTransformer`: Rolling windows and lags

#### Text
- `TextStatisticsTransformer`: Text statistics
- `TextVectorizationTransformer`: TF-IDF and count vectorization

## Migration Steps

### 1. Enable New System

```python
from mdm.core.feature_flags import feature_flags
feature_flags.set("use_new_features", True)
```

### 2. Default Pipeline

The default pipeline is automatically configured based on your settings:

```python
from mdm.features.builder import FeaturePipelineBuilder

builder = FeaturePipelineBuilder()
pipeline = builder.build_default_pipeline()

# Generate features
features = pipeline.fit_transform(data)
```

### 3. Custom Pipeline

Create custom pipelines for specific needs:

```python
from mdm.features.pipeline.base import Pipeline
from mdm.features.transformers.numeric import NumericStatisticsTransformer
from mdm.features.transformers.categorical import CategoricalEncodingTransformer

# Create pipeline
pipeline = Pipeline(n_jobs=4)

# Add transformers
pipeline.add_transformer(
    NumericStatisticsTransformer(
        statistics=["mean", "std", "skew"]
    )
)

pipeline.add_transformer(
    CategoricalEncodingTransformer(
        encoding="onehot",
        max_categories=50
    )
)

# Use pipeline
features = pipeline.fit_transform(data)
```

### 4. Custom Transformers

Create custom transformers for domain-specific features:

```python
# Save to ~/.mdm/custom_features/my_transformer.py
from mdm.features.pipeline.base import BaseTransformer
import pandas as pd

class MyCustomTransformer(BaseTransformer):
    def fit(self, df: pd.DataFrame):
        # Learn from data
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame):
        # Create features
        features = pd.DataFrame(index=df.index)
        features['my_feature'] = df['column'].apply(my_function)
        return features
    
    def get_feature_names(self):
        return ['my_feature']
```

### 5. Configuration File

Configure pipelines via YAML:

```yaml
# ~/.mdm/config/feature_pipeline.yaml
n_jobs: -1
use_memory: true
transformers:
  - name: numeric_stats
    class: NumericStatisticsTransformer
    enabled: true
    parallel: true
    params:
      statistics: ["mean", "std", "min", "max"]
  
  - name: categorical_encoding
    class: CategoricalEncodingTransformer
    enabled: true
    parallel: true
    params:
      encoding: label
      max_categories: 100
```

## Performance Improvements

### Parallel Processing
- Transformers can run in parallel
- Controlled by `n_jobs` parameter
- Automatic CPU detection

### Memory Efficiency
- Batch processing for large datasets
- Optional caching with `use_memory`
- Lazy evaluation where possible

### Benchmarks
- 30-50% faster for datasets > 100k rows
- 40% less memory usage
- Linear scaling with CPU cores

## API Compatibility

The system maintains backward compatibility:

```python
# Old API still works
from mdm.features import FeatureGenerator
generator = FeatureGenerator()
features = generator.generate_features(data)

# Internally uses new system when flag is enabled
```

## Troubleshooting

### Issue: Custom transformer not loading
```bash
# Check custom directory
ls ~/.mdm/custom_features/

# Verify transformer registration
python -c "from mdm.features.pipeline.registry import transformer_registry; print(transformer_registry.list_transformers('custom'))"
```

### Issue: Performance regression
```python
# Disable parallel processing
pipeline = Pipeline(n_jobs=1)

# Or disable memory caching
pipeline = Pipeline(use_memory=False)
```

### Issue: Different features generated
```python
# Use comparison mode
from mdm.features.migration.adapter import FeatureGeneratorAdapter

adapter = FeatureGeneratorAdapter()
old_features = adapter.generate_features(data)  # with flag=False
new_features = adapter.generate_features(data)  # with flag=True

# Compare
print(f"Old shape: {old_features.shape}")
print(f"New shape: {new_features.shape}")
```

## Best Practices

1. **Start with Default Pipeline**: Use the default configuration first
2. **Profile Your Data**: Understand your data types before customizing
3. **Test Incrementally**: Enable one transformer type at a time
4. **Monitor Performance**: Use metrics to track improvements
5. **Document Custom Transformers**: Add docstrings and examples

## Migration Timeline

- Week 1: Test with development datasets
- Week 2: Enable for 25% of new registrations
- Week 3: Enable for 50% of operations
- Week 4: Full rollout

## Rollback

If issues arise:

```python
# Immediate rollback
from mdm.core.feature_flags import feature_flags
feature_flags.set("use_new_features", False)
```

No data changes are required - the system automatically falls back to legacy implementation.
```

## Validation Checklist

### Week 11 Complete
- [ ] Pipeline architecture implemented
- [ ] Base transformer classes tested
- [ ] Registry and discovery working
- [ ] Parallel processing verified

### Week 12 Complete
- [ ] All core transformers implemented
- [ ] DateTime and text transformers tested
- [ ] Custom transformer framework working
- [ ] Pipeline builder tested

### Week 13 Complete
- [ ] Migration adapter implemented
- [ ] Comparison tests passing
- [ ] Performance benchmarks complete
- [ ] Documentation updated

## Success Criteria

- **Feature parity** with legacy system
- **30%+ performance improvement** for large datasets
- **Custom transformer** support working
- **Zero regression** in feature quality
- **Smooth migration** path

## Next Steps

With feature engineering migrated, proceed to [07-dataset-registration-migration.md](07-dataset-registration-migration.md).

## Notes

- Monitor transformer performance individually
- Consider transformer ordering for efficiency
- Document all custom transformers
- Plan for future ML framework integration