# Feature Engineering System Redesign

## Overview

The current feature engineering system suffers from tight coupling, lack of extensibility, and mixed responsibilities. This guide details the transformation to a flexible, plugin-based architecture.

## Current Problems

### 1. God Class - FeatureGenerator
```python
# CURRENT - Too many responsibilities
class FeatureGenerator:
    def generate_features(self, df, target_column, column_types, problem_type, tables_dict, db_info, dataset_name, tmp_path):
        # 200+ lines doing everything!
        # - Type detection
        # - Feature generation  
        # - Database operations
        # - Progress tracking
        # - Error handling
```

### 2. Hard-coded Feature Types
```python
# CURRENT - Not extensible
self.registry = FeatureRegistry()
# Only these types, no way to add more
```

### 3. Mixed Concerns
- Feature generation mixed with database operations
- Progress tracking intertwined with business logic
- No clear separation of feature definition and execution

## Target Architecture

### 1. Feature Pipeline Architecture
```python
# NEW - Clean pipeline architecture
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Protocol

@dataclass
class FeatureContext:
    """Context passed through feature pipeline."""
    dataset_name: str
    target_column: Optional[str]
    problem_type: Optional[str]
    column_types: Dict[str, ColumnType]
    metadata: Dict[str, Any]

class FeatureTransformer(Protocol):
    """Protocol for feature transformers."""
    
    @property
    def name(self) -> str:
        """Transformer name."""
        ...
    
    @property
    def version(self) -> str:
        """Transformer version."""
        ...
    
    def can_transform(self, df: pd.DataFrame, context: FeatureContext) -> bool:
        """Check if transformer can handle this data."""
        ...
    
    def transform(self, df: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
        """Transform dataframe to add features."""
        ...
    
    def get_feature_names(self) -> List[str]:
        """Get list of features this transformer creates."""
        ...
```

### 2. Feature Pipeline
```python
# NEW - Pipeline pattern
class FeaturePipeline:
    """Manages feature transformation pipeline."""
    
    def __init__(self):
        self.transformers: List[FeatureTransformer] = []
        self.hooks: Dict[str, List[Callable]] = {
            'before_transform': [],
            'after_transform': [],
            'on_error': []
        }
    
    def add_transformer(self, transformer: FeatureTransformer) -> 'FeaturePipeline':
        """Add transformer to pipeline."""
        self.transformers.append(transformer)
        return self
    
    def add_hook(self, event: str, hook: Callable) -> 'FeaturePipeline':
        """Add event hook."""
        self.hooks[event].append(hook)
        return self
    
    def execute(self, df: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
        """Execute pipeline on dataframe."""
        result_df = df.copy()
        
        for transformer in self.transformers:
            try:
                # Before transform hooks
                for hook in self.hooks['before_transform']:
                    hook(transformer, result_df, context)
                
                # Check if transformer applies
                if transformer.can_transform(result_df, context):
                    result_df = transformer.transform(result_df, context)
                
                # After transform hooks
                for hook in self.hooks['after_transform']:
                    hook(transformer, result_df, context)
                    
            except Exception as e:
                # Error hooks
                for hook in self.hooks['on_error']:
                    hook(transformer, e, context)
                raise
        
        return result_df
```

### 3. Transformer Implementations
```python
# NEW - Clean transformer implementations
class StatisticalTransformer:
    """Generates statistical features."""
    
    name = "statistical"
    version = "1.0.0"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.features_created: List[str] = []
    
    def can_transform(self, df: pd.DataFrame, context: FeatureContext) -> bool:
        """Check if we have numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return len(numeric_cols) > 0
    
    def transform(self, df: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
        """Add statistical features."""
        result = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Z-score normalization
            if self._has_variance(df[col]):
                feature_name = f"{col}_zscore"
                result[feature_name] = (df[col] - df[col].mean()) / df[col].std()
                self.features_created.append(feature_name)
            
            # Log transformation for positive values
            if (df[col] > 0).all():
                feature_name = f"{col}_log"
                result[feature_name] = np.log1p(df[col])
                self.features_created.append(feature_name)
            
            # Percentile rank
            feature_name = f"{col}_percentile"
            result[feature_name] = df[col].rank(pct=True)
            self.features_created.append(feature_name)
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get created feature names."""
        return self.features_created
    
    def _has_variance(self, series: pd.Series) -> bool:
        """Check if series has variance."""
        return series.nunique() > 1
```

### 4. Plugin System
```python
# NEW - Plugin architecture
class FeaturePluginManager:
    """Manages feature transformer plugins."""
    
    def __init__(self, plugin_paths: List[Path]):
        self.plugin_paths = plugin_paths
        self.plugins: Dict[str, Type[FeatureTransformer]] = {}
        self._discover_plugins()
    
    def _discover_plugins(self) -> None:
        """Discover plugins in plugin paths."""
        for path in self.plugin_paths:
            if not path.exists():
                continue
                
            for file in path.glob("*.py"):
                if file.name.startswith("_"):
                    continue
                    
                module = self._load_module(file)
                transformers = self._find_transformers(module)
                
                for name, transformer in transformers.items():
                    self.plugins[name] = transformer
    
    def _load_module(self, file: Path) -> Any:
        """Dynamically load Python module."""
        spec = importlib.util.spec_from_file_location(file.stem, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def _find_transformers(self, module: Any) -> Dict[str, Type[FeatureTransformer]]:
        """Find transformer classes in module."""
        transformers = {}
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, FeatureTransformer) and
                obj != FeatureTransformer):
                transformers[name] = obj
        
        return transformers
    
    def get_transformer(self, name: str) -> Optional[Type[FeatureTransformer]]:
        """Get transformer by name."""
        return self.plugins.get(name)
    
    def list_transformers(self) -> List[str]:
        """List available transformers."""
        return list(self.plugins.keys())
```

### 5. Feature Store
```python
# NEW - Feature metadata store
@dataclass
class FeatureMetadata:
    """Metadata for a generated feature."""
    name: str
    source_column: str
    transformer: str
    transformer_version: str
    parameters: Dict[str, Any]
    created_at: datetime
    statistics: Dict[str, float]

class FeatureStore:
    """Stores feature metadata and lineage."""
    
    def __init__(self, storage: StorageBackend):
        self.storage = storage
    
    def save_feature(self, feature: FeatureMetadata) -> None:
        """Save feature metadata."""
        with self.storage.get_connection() as conn:
            # Save to _mdm_features table
            pass
    
    def get_features_for_dataset(self, dataset_name: str) -> List[FeatureMetadata]:
        """Get all features for dataset."""
        with self.storage.get_connection() as conn:
            # Query features
            pass
    
    def get_feature_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get feature lineage information."""
        # Track how feature was created
        pass
```

## Migration Strategy

### Phase 1: Create New Architecture
```python
# features/pipeline.py
class FeaturePipeline:
    # New pipeline implementation

# features/transformers/base.py
class BaseTransformer:
    # Base transformer class

# features/plugins.py  
class FeaturePluginManager:
    # Plugin manager
```

### Phase 2: Implement Core Transformers
```python
# features/transformers/statistical.py
class StatisticalTransformer(BaseTransformer):
    # Statistical features

# features/transformers/temporal.py
class TemporalTransformer(BaseTransformer):
    # Date/time features

# features/transformers/text.py
class TextTransformer(BaseTransformer):
    # Text features

# features/transformers/categorical.py
class CategoricalTransformer(BaseTransformer):
    # Categorical encoding
```

### Phase 3: Adapter for Old System
```python
# features/legacy_adapter.py
class LegacyFeatureAdapter:
    """Adapts new pipeline to old interface."""
    
    def __init__(self, pipeline: FeaturePipeline):
        self.pipeline = pipeline
    
    def generate_features(self, df, target_column, column_types, problem_type, *args, **kwargs):
        """Old interface compatibility."""
        context = FeatureContext(
            dataset_name=kwargs.get('dataset_name', 'unknown'),
            target_column=target_column,
            problem_type=problem_type,
            column_types=column_types,
            metadata={}
        )
        
        return self.pipeline.execute(df, context)
```

## Plugin Development Guide

### Creating a Custom Transformer
```python
# ~/.mdm/plugins/features/domain_features.py
from mdm.features import BaseTransformer, FeatureContext
import pandas as pd

class DomainSpecificTransformer(BaseTransformer):
    """Custom domain-specific features."""
    
    name = "domain_specific"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.threshold = config.get('threshold', 0.5)
    
    def can_transform(self, df: pd.DataFrame, context: FeatureContext) -> bool:
        """Check if this transformer applies."""
        # Custom logic
        return 'price' in df.columns and 'quantity' in df.columns
    
    def transform(self, df: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
        """Add domain features."""
        result = df.copy()
        
        # Revenue feature
        result['revenue'] = df['price'] * df['quantity']
        
        # High value flag
        result['is_high_value'] = result['revenue'] > result['revenue'].quantile(self.threshold)
        
        return result
```

### Registering Plugins
```yaml
# ~/.mdm/mdm.yaml
features:
  plugin_paths:
    - ~/.mdm/plugins/features
    - ./custom_features
  enabled_transformers:
    - statistical
    - temporal
    - text
    - categorical
    - domain_specific
  transformer_config:
    domain_specific:
      threshold: 0.75
```

## Testing Strategy

### Unit Tests
```python
# tests/unit/features/test_transformers.py
class TestStatisticalTransformer:
    def test_zscore_transformation(self):
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        context = FeatureContext(dataset_name="test", column_types={'value': ColumnType.NUMERIC})
        
        transformer = StatisticalTransformer()
        result = transformer.transform(df, context)
        
        assert 'value_zscore' in result.columns
        assert result['value_zscore'].mean() == pytest.approx(0, abs=1e-10)
        assert result['value_zscore'].std() == pytest.approx(1, abs=1e-10)
```

### Integration Tests
```python
# tests/integration/features/test_pipeline.py
class TestFeaturePipeline:
    def test_full_pipeline(self):
        pipeline = FeaturePipeline()
        pipeline.add_transformer(StatisticalTransformer())
        pipeline.add_transformer(TemporalTransformer())
        
        df = create_test_dataframe()
        context = create_test_context()
        
        result = pipeline.execute(df, context)
        
        # Verify all expected features
        assert len(result.columns) > len(df.columns)
```

## Performance Optimization

### Parallel Processing
```python
# features/parallel_pipeline.py
class ParallelFeaturePipeline(FeaturePipeline):
    """Pipeline with parallel execution."""
    
    def execute(self, df: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
        """Execute transformers in parallel."""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            
            for transformer in self.transformers:
                if transformer.can_transform(df, context):
                    future = executor.submit(transformer.transform, df.copy(), context)
                    futures.append((transformer, future))
            
            # Combine results
            result_df = df.copy()
            for transformer, future in futures:
                transformed = future.result()
                # Merge new columns
                new_cols = [col for col in transformed.columns if col not in df.columns]
                result_df[new_cols] = transformed[new_cols]
        
        return result_df
```

### Caching
```python
# features/cached_transformer.py
class CachedTransformer:
    """Decorator for caching transformer results."""
    
    def __init__(self, transformer: FeatureTransformer):
        self.transformer = transformer
        self.cache = {}
    
    def transform(self, df: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
        """Transform with caching."""
        cache_key = self._compute_cache_key(df, context)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.transformer.transform(df, context)
        self.cache[cache_key] = result
        
        return result
```

## Monitoring and Observability

### Feature Quality Metrics
```python
# features/monitoring.py
class FeatureMonitor:
    """Monitors feature quality."""
    
    def analyze_feature(self, feature_series: pd.Series) -> Dict[str, Any]:
        """Analyze feature quality metrics."""
        return {
            'null_rate': feature_series.isnull().mean(),
            'unique_ratio': feature_series.nunique() / len(feature_series),
            'variance': feature_series.var(),
            'skewness': feature_series.skew(),
            'kurtosis': feature_series.kurtosis(),
            'has_signal': self._has_signal(feature_series)
        }
    
    def _has_signal(self, series: pd.Series) -> bool:
        """Check if feature has predictive signal."""
        return series.nunique() > 1 and series.var() > 0
```

## Success Criteria

1. **Modularity**: Each transformer is independent
2. **Extensibility**: Easy to add new transformers via plugins
3. **Performance**: No degradation vs current system
4. **Testability**: 95%+ test coverage
5. **Observability**: Full feature lineage tracking