# MDM API Reference

## Overview

The MDM (ML Data Manager) API provides a comprehensive interface for managing machine learning datasets. This document reflects the actual implementation after the 2025 refactoring.

## Table of Contents

1. [High-Level API (MDMClient)](#high-level-api-mdmclient)
2. [Storage Backends](#storage-backends)
3. [Dataset Management](#dataset-management)
4. [Feature Engineering](#feature-engineering)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Exceptions](#exceptions)

## High-Level API (MDMClient)

The `MDMClient` provides the simplest way to interact with MDM programmatically.

### Basic Usage

```python
from mdm.api import MDMClient

# Initialize client
client = MDMClient()

# Register a dataset
dataset_info = client.register_dataset(
    name="iris",
    path="data/iris.csv",
    target_column="species",
    problem_type="multiclass_classification"
)

# List all datasets
datasets = client.list_datasets()
for ds in datasets:
    print(f"{ds.name}: {ds.row_count} rows, {ds.problem_type}")

# Load dataset as DataFrame
df = client.load_dataset("iris")
print(df.head())

# Get dataset information
info = client.get_dataset_info("iris")
print(f"Target: {info.target_column}")
print(f"Features: {info.feature_columns}")

# Export dataset
client.export_dataset("iris", format="parquet", output_dir="./exports")

# Remove dataset
client.remove_dataset("iris", force=True)
```

### Advanced Operations

```python
# Update dataset metadata
client.update_dataset(
    "iris",
    description="Classic ML dataset for classification",
    tags=["classification", "flowers", "tutorial"]
)

# Search datasets
results = client.search_datasets("classification")
results = client.search_datasets_by_tag("tutorial")

# Get dataset statistics
stats = client.get_dataset_stats("iris")
print(f"Missing values: {stats['missing_percentage']}%")
```

## Storage Backends

MDM supports multiple storage backends through a unified interface.

### StorageBackend Base Class

```python
from mdm.storage.base import StorageBackend

# The base class that all backends inherit from
class StorageBackend:
    """Base class for all storage backends."""
    
    def get_engine(self, database_path: str) -> Engine:
        """Get SQLAlchemy engine."""
        pass
    
    def initialize_database(self, engine: Engine) -> None:
        """Initialize database schema."""
        pass
    
    def create_table_from_dataframe(
        self, df: pd.DataFrame, table_name: str, 
        engine: Engine, if_exists: str = "fail"
    ) -> None:
        """Create table from DataFrame."""
        pass
```

### Available Backends

```python
from mdm.storage.factory import BackendFactory

# Get a backend instance
backend = BackendFactory.create("sqlite", config={})
backend = BackendFactory.create("duckdb", config={})
backend = BackendFactory.create("postgresql", config={
    "host": "localhost",
    "port": 5432,
    "user": "mdm_user",
    "password": "secret"
})
```

## Dataset Management

### DatasetManager

```python
from mdm.dataset.manager import DatasetManager

manager = DatasetManager()

# List datasets
datasets = manager.list_datasets()

# Get specific dataset
dataset_info = manager.get_dataset("iris")

# Check if dataset exists
exists = manager.dataset_exists("iris")

# Load dataset configuration
config = manager._load_dataset_config("iris")
```

### DatasetRegistrar

```python
from mdm.dataset.registrar import DatasetRegistrar

registrar = DatasetRegistrar()

# Register with auto-detection
dataset_info = registrar.register(
    name="titanic",
    path=Path("./titanic_data/"),
    auto_detect=True,
    target_column="survived",
    generate_features=True
)

# Register with specific options
dataset_info = registrar.register(
    name="sales",
    path=Path("./sales.csv"),
    auto_detect=False,
    target_column="revenue",
    id_columns=["transaction_id"],
    problem_type="regression",
    description="Monthly sales data",
    tags=["sales", "revenue", "timeseries"]
)
```

## Feature Engineering

### FeatureGenerator

```python
from mdm.features.generator import FeatureGenerator

generator = FeatureGenerator()

# Generate features for a DataFrame
df_with_features = generator.generate_features(
    df=original_df,
    column_types={
        "date": "datetime",
        "category": "categorical",
        "price": "numeric",
        "description": "text"
    }
)

# Get available transformers
transformers = generator.generic_transformers
# Includes: TemporalFeatures, CategoricalFeatures, 
#          StatisticalFeatures, TextFeatures, etc.
```

### Custom Transformers

```python
# Create custom transformer in ~/.mdm/config/custom_features/sales.py
from mdm.features.base import BaseTransformer

class SalesFeatures(BaseTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add custom features
        df['revenue_per_item'] = df['revenue'] / df['quantity']
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])
        return df
```

## Configuration

### ConfigManager

```python
from mdm.config import get_config_manager

# Get configuration manager
config_manager = get_config_manager()

# Access configuration
config = config_manager.config
print(f"Default backend: {config.database.default_backend}")
print(f"Batch size: {config.performance.batch_size}")

# Get paths
base_path = config_manager.base_path
datasets_path = base_path / config.paths.datasets_path
```

### Settings Model

```python
from mdm.models.config import Settings

# Configuration is validated with Pydantic
settings = Settings()

# Access nested configuration
db_config = settings.database
perf_config = settings.performance
log_config = settings.logging
```

## Monitoring

### SimpleMonitor

```python
from mdm.monitoring import SimpleMonitor

monitor = SimpleMonitor()

# Record metrics
monitor.record_metric("registration_time", 2.5)
monitor.record_metric("query_time", 0.05)
monitor.record_metric("export_size_mb", 125.3)

# Get metrics
metrics = monitor.get_metrics()
print(f"Avg registration time: {metrics['registration_time']['avg']}")
```

## Exceptions

MDM uses specific exceptions for different error types:

```python
from mdm.core.exceptions import (
    MDMError,          # Base exception
    DatasetError,      # Dataset-specific errors
    StorageError,      # Storage backend errors
    ConfigError,       # Configuration errors
    ValidationError    # Data validation errors
)

try:
    client.register_dataset("test", "data.csv")
except DatasetError as e:
    print(f"Dataset error: {e}")
except StorageError as e:
    print(f"Storage error: {e}")
except MDMError as e:
    print(f"General MDM error: {e}")
```

## CLI Integration

### Using Typer

```python
import typer
from mdm.cli.utils import console

app = typer.Typer()

@app.command()
def my_command(name: str):
    """Custom MDM command."""
    console.print(f"[green]Processing {name}[/green]")
    
    # Use MDM functionality
    client = MDMClient()
    datasets = client.list_datasets()
    
    # Rich output
    from rich.table import Table
    table = Table(title="Datasets")
    table.add_column("Name")
    table.add_column("Rows")
    
    for ds in datasets:
        table.add_row(ds.name, str(ds.row_count))
    
    console.print(table)
```

## Complete Example

```python
"""Example: ML Pipeline with MDM"""

from mdm.api import MDMClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Initialize MDM
client = MDMClient()

# Register dataset if not exists
if not client.dataset_exists("customer_churn"):
    client.register_dataset(
        name="customer_churn",
        path="data/customers.csv",
        target_column="churned",
        problem_type="binary_classification"
    )

# Load and prepare data
df = client.load_dataset("customer_churn")
info = client.get_dataset_info("customer_churn")

# Use MDM's feature columns
X = df[info.feature_columns]
y = df[info.target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.3f}")

# Export predictions
predictions_df = pd.DataFrame({
    'id': X_test.index,
    'prediction': model.predict(X_test),
    'probability': model.predict_proba(X_test)[:, 1]
})

# Register predictions as new dataset
client.register_dataset(
    name="customer_churn_predictions",
    dataframe=predictions_df,
    description=f"Model predictions with {score:.3f} accuracy"
)
```