# MDM API Reference

## Overview

The MDM (ML Data Manager) API provides a comprehensive interface for managing machine learning datasets. This document covers the new architecture introduced in the 2025 refactoring.

## Table of Contents

1. [Core Interfaces](#core-interfaces)
2. [Storage API](#storage-api)
3. [Dataset API](#dataset-api)
4. [Feature Engineering API](#feature-engineering-api)
5. [Configuration API](#configuration-api)
6. [Performance API](#performance-api)
7. [Migration API](#migration-api)

## Core Interfaces

### IStorageBackend

The foundation interface for all storage backends.

```python
from mdm.interfaces import IStorageBackend

class IStorageBackend(ABC):
    """Abstract interface for storage backends."""
    
    @abstractmethod
    def get_engine(self, database_path: str) -> Engine:
        """Get SQLAlchemy engine for database operations."""
        pass
    
    @abstractmethod
    def initialize_database(self, engine: Engine) -> None:
        """Initialize database schema."""
        pass
    
    @abstractmethod
    def create_table_from_dataframe(
        self, df: pd.DataFrame, table_name: str, 
        engine: Engine, if_exists: str = "fail"
    ) -> None:
        """Create table from pandas DataFrame."""
        pass
```

### IDatasetManager

Interface for dataset management operations.

```python
from mdm.interfaces import IDatasetManager

class IDatasetManager(ABC):
    """Abstract interface for dataset management."""
    
    @abstractmethod
    def register_dataset(self, dataset_info: DatasetInfo) -> None:
        """Register a new dataset."""
        pass
    
    @abstractmethod
    def get_dataset(self, name: str) -> Optional[DatasetInfo]:
        """Retrieve dataset information."""
        pass
    
    @abstractmethod
    def list_datasets(self) -> List[DatasetInfo]:
        """List all registered datasets."""
        pass
```

### IFeatureGenerator

Interface for feature engineering operations.

```python
from mdm.interfaces import IFeatureGenerator

class IFeatureGenerator(ABC):
    """Abstract interface for feature generation."""
    
    @abstractmethod
    def generate_features(
        self, df: pd.DataFrame, 
        column_types: Dict[str, str]
    ) -> pd.DataFrame:
        """Generate features from DataFrame."""
        pass
    
    @abstractmethod
    def get_available_transformers(self) -> List[str]:
        """Get list of available feature transformers."""
        pass
```

## Storage API

### Getting a Storage Backend

```python
from mdm.adapters import get_storage_backend

# Get storage backend based on feature flags
backend = get_storage_backend("sqlite", config={
    "pragmas": {"journal_mode": "WAL"},
    "timeout": 30
})

# Use the backend
engine = backend.get_engine("/path/to/database.db")
```

### Available Backends

1. **SQLite** - Lightweight, file-based storage
2. **DuckDB** - Columnar storage for analytics
3. **PostgreSQL** - Enterprise-grade relational database

### Storage Configuration

```python
# SQLite configuration
sqlite_config = {
    "pragmas": {
        "journal_mode": "WAL",
        "synchronous": "NORMAL",
        "cache_size": -64000,
        "temp_store": "MEMORY"
    },
    "timeout": 30,
    "enable_performance_optimizations": True
}

# DuckDB configuration
duckdb_config = {
    "pragmas": {
        "memory_limit": "4GB",
        "threads": 4
    },
    "extensions": ["parquet", "json"]
}

# PostgreSQL configuration
postgresql_config = {
    "host": "localhost",
    "port": 5432,
    "user": "mdm_user",
    "password": "secure_password",
    "database": "mdm_db",
    "pool_size": 5,
    "max_overflow": 10
}
```

## Dataset API

### Dataset Registration

```python
from mdm.adapters import get_dataset_registrar
from mdm.models import DatasetInfo

# Get registrar
registrar = get_dataset_registrar()

# Register dataset
dataset_info = registrar.register_dataset(
    name="iris_dataset",
    path="/data/iris.csv",
    target_column="species",
    problem_type="classification"
)
```

### Dataset Management

```python
from mdm.adapters import get_dataset_manager

# Get manager
manager = get_dataset_manager()

# List datasets
datasets = manager.list_datasets()

# Get specific dataset
dataset = manager.get_dataset("iris_dataset")

# Update dataset
manager.update_dataset("iris_dataset", {
    "description": "Classic Iris flower dataset",
    "tags": ["classification", "flowers", "example"]
})

# Search datasets
results = manager.search_datasets("iris")
results_by_tag = manager.search_datasets_by_tag("classification")

# Delete dataset
manager.delete_dataset("iris_dataset", force=True)
```

### Dataset Statistics

```python
# Get statistics
stats = manager.get_statistics("iris_dataset")

# Save new statistics
from mdm.models import DatasetStatistics
new_stats = DatasetStatistics(
    row_count=150,
    column_count=5,
    memory_usage_bytes=12000,
    # ... other fields
)
manager.save_statistics("iris_dataset", new_stats)
```

## Feature Engineering API

### Basic Feature Generation

```python
from mdm.adapters import get_feature_generator

# Get generator
generator = get_feature_generator()

# Generate features
df_with_features = generator.generate_features(
    df=original_df,
    column_types={
        "age": "numeric",
        "category": "categorical",
        "text": "text"
    }
)
```

### Custom Feature Transformers

```python
from mdm.interfaces import IFeatureTransformer
from mdm.core.features import register_transformer

@register_transformer("custom")
class CustomTransformer(IFeatureTransformer):
    """Custom feature transformer."""
    
    def fit(self, df: pd.DataFrame, columns: List[str]) -> None:
        """Fit transformer on data."""
        # Implementation
        pass
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        # Implementation
        pass
```

### Feature Pipeline

```python
from mdm.core.features import FeaturePipeline

# Create pipeline
pipeline = FeaturePipeline()

# Add transformers
pipeline.add_transformer("numeric", ["age", "salary"])
pipeline.add_transformer("categorical", ["department", "role"])
pipeline.add_transformer("text", ["description"])

# Fit and transform
pipeline.fit(train_df)
transformed_df = pipeline.transform(test_df)
```

## Configuration API

### Configuration Management

```python
from mdm.adapters import get_config_manager, get_config

# Get configuration manager
config_manager = get_config_manager()

# Get current configuration
config = get_config()

# Update configuration
config_manager.update_config({
    "database": {
        "default_backend": "duckdb"
    },
    "performance": {
        "batch_size": 5000
    }
})

# Environment variable override
# MDM_DATABASE_DEFAULT_BACKEND=postgresql
# MDM_PERFORMANCE_BATCH_SIZE=10000
```

### Configuration Schema

```python
from mdm.models import MDMConfig

config = MDMConfig(
    database=DatabaseConfig(
        default_backend="sqlite",
        sqlite=SQLiteConfig(...),
        duckdb=DuckDBConfig(...),
        postgresql=PostgreSQLConfig(...)
    ),
    paths=PathsConfig(
        base_path="~/.mdm",
        datasets_path="datasets",
        configs_path="config/datasets"
    ),
    logging=LoggingConfig(
        level="INFO",
        file=None,
        format="..."
    ),
    performance=PerformanceConfig(
        batch_size=10000,
        max_workers=4,
        enable_profiling=False
    )
)
```

## Performance API

### Query Optimization

```python
from mdm.performance import QueryOptimizer

# Create optimizer
optimizer = QueryOptimizer(cache_query_plans=True)

# Optimize query
optimized_query, plan = optimizer.optimize_query(
    "SELECT * FROM large_table WHERE category = 'A'",
    connection
)

# Check optimization hints
if plan.optimization_hints:
    for hint in plan.optimization_hints:
        print(f"Hint: {hint}")
```

### Caching

```python
from mdm.performance import CacheManager, CachePolicy

# Create cache manager
cache = CacheManager(
    max_size_mb=100,
    policy=CachePolicy.LRU,
    default_ttl=300
)

# Use cache decorator
@cache.cached(ttl=60)
def expensive_operation(dataset_name: str):
    # Expensive computation
    return result

# Manual cache operations
cache.set("key", value, ttl=120)
value = cache.get("key")
cache.delete("key")
```

### Batch Processing

```python
from mdm.performance import BatchOptimizer, BatchConfig

# Configure batch processing
config = BatchConfig(
    batch_size=10000,
    max_workers=4,
    enable_parallel=True
)
optimizer = BatchOptimizer(config)

# Process dataframe in batches
result_df = optimizer.process_dataframe_batches(
    df=large_df,
    process_func=lambda batch: batch.apply(transform),
    progress_callback=lambda done, total: print(f"{done}/{total}")
)
```

### Connection Pooling

```python
from mdm.performance import ConnectionPool, PoolConfig

# Configure pool
pool = ConnectionPool(
    connection_string="postgresql://...",
    config=PoolConfig(
        pool_size=5,
        max_overflow=10,
        timeout=30.0,
        recycle=3600
    )
)

# Use pooled connection
with pool.get_connection() as conn:
    result = conn.execute("SELECT * FROM datasets")
```

### Performance Monitoring

```python
from mdm.performance import get_monitor

# Get global monitor
monitor = get_monitor()

# Track operations
with monitor.track_operation("dataset_registration") as timer:
    # Perform operation
    pass
    print(f"Duration: {timer.duration}s")

# Track queries
monitor.track_query("select", duration=0.05, rows=100)

# Get performance report
report = monitor.get_report()
print(f"Total operations: {report['summary']['total_operations']}")
```

## Migration API

### Feature Flags

```python
from mdm.core import feature_flags

# Check feature flag
if feature_flags.get("use_new_storage"):
    # Use new implementation
    pass
else:
    # Use legacy implementation
    pass

# Set feature flag
feature_flags.set("use_new_storage", True)

# Bulk update
feature_flags.set_multiple({
    "use_new_storage": True,
    "use_new_features": True,
    "use_new_config": True
})
```

### Migration Validation

```python
from mdm.migration import MigrationValidator

# Create validator
validator = MigrationValidator()

# Validate storage migration
results = validator.validate_storage_migration(
    legacy_backend=legacy_storage,
    new_backend=new_storage,
    test_data=sample_df
)

# Check compatibility
if validator.check_compatibility("dataset_name"):
    print("Dataset is compatible with new backend")
```

### Progressive Rollout

```python
from mdm.migration import RolloutManager

# Create rollout manager
rollout = RolloutManager()

# Configure rollout percentage
rollout.set_rollout_percentage("use_new_storage", 25)

# Check if enabled for specific context
if rollout.is_enabled("use_new_storage", context={"user_id": 123}):
    # Use new implementation
    pass
```

## Error Handling

### Exception Hierarchy

```python
from mdm.core.exceptions import (
    MDMError,           # Base exception
    DatasetError,       # Dataset-related errors
    StorageError,       # Storage backend errors
    ConfigError,        # Configuration errors
    ValidationError,    # Data validation errors
    MigrationError      # Migration-related errors
)

try:
    # Perform operations
    pass
except DatasetError as e:
    # Handle dataset-specific error
    logger.error(f"Dataset error: {e}")
except StorageError as e:
    # Handle storage error
    logger.error(f"Storage error: {e}")
except MDMError as e:
    # Handle general MDM error
    logger.error(f"MDM error: {e}")
```

### Error Context

```python
from mdm.core.exceptions import add_error_context

try:
    # Risky operation
    pass
except Exception as e:
    # Add context and re-raise
    add_error_context(e, {
        "dataset": "iris_dataset",
        "operation": "feature_generation",
        "backend": "sqlite"
    })
    raise
```

## Best Practices

### 1. Resource Management

Always use context managers for database connections:

```python
# Good
with backend.session(db_path) as session:
    # Perform operations
    pass

# Avoid
session = backend.get_session(db_path)
# Operations without proper cleanup
```

### 2. Batch Processing

For large datasets, always use batch processing:

```python
# Good
optimizer = BatchOptimizer(BatchConfig(batch_size=10000))
result = optimizer.process_dataframe_batches(large_df, process_func)

# Avoid
result = large_df.apply(process_func)  # May cause memory issues
```

### 3. Feature Flags

Use feature flags for gradual migration:

```python
# Good
if feature_flags.get("use_new_implementation"):
    result = new_implementation()
else:
    result = legacy_implementation()

# Avoid
result = new_implementation()  # No fallback
```

### 4. Error Handling

Always provide meaningful error context:

```python
# Good
try:
    register_dataset(name, path)
except Exception as e:
    raise DatasetError(
        f"Failed to register dataset '{name}' from '{path}': {e}"
    ) from e

# Avoid
try:
    register_dataset(name, path)
except:
    raise  # No context
```

### 5. Performance Monitoring

Track performance-critical operations:

```python
# Good
with monitor.track_operation("expensive_operation", dataset=name):
    result = perform_expensive_operation()

# Avoid
result = perform_expensive_operation()  # No monitoring
```

## Appendix

### Environment Variables

Complete list of supported environment variables:

- `MDM_DATABASE_DEFAULT_BACKEND`: Default storage backend
- `MDM_DATABASE_SQLITE_TIMEOUT`: SQLite connection timeout
- `MDM_DATABASE_DUCKDB_MEMORY_LIMIT`: DuckDB memory limit
- `MDM_DATABASE_POSTGRESQL_HOST`: PostgreSQL host
- `MDM_PATHS_BASE_PATH`: Base directory for MDM
- `MDM_LOGGING_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MDM_LOGGING_FILE`: Log file path
- `MDM_PERFORMANCE_BATCH_SIZE`: Default batch size
- `MDM_PERFORMANCE_MAX_WORKERS`: Maximum parallel workers

### Type Definitions

Common type definitions used throughout the API:

```python
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd

DataFrameDict = Dict[str, pd.DataFrame]
ColumnTypes = Dict[str, str]
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, Union[int, float]]
```