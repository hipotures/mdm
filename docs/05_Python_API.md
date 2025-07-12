# Python API Reference

MDM provides a comprehensive Python API for programmatic dataset management and ML integration.

## Quick Start

```python
from mdm import MDMClient

# Initialize client
client = MDMClient()

# Load a dataset
df = client.load_dataset("titanic")

# Get dataset information
info = client.get_dataset_info("titanic")
print(f"Dataset has {info.row_count} rows and {info.column_count} columns")
```

## MDMClient

The main entry point for all MDM operations.

### Initialization

```python
from mdm import MDMClient

# Default initialization (uses ~/.mdm/mdm.yaml)
client = MDMClient()

# With custom config path
client = MDMClient(config_path="/path/to/config.yaml")

# With config overrides
client = MDMClient(
    backend="duckdb",
    batch_size=50000,
    log_level="DEBUG"
)
```

### Core Methods

#### load_dataset

Load a dataset as a pandas DataFrame.

```python
df = client.load_dataset(
    name: str,
    table: str = "train",
    columns: list[str] = None,
    limit: int = None,
    include_features: bool = True
)
```

**Parameters:**
- `name`: Dataset name
- `table`: Table to load ("train", "test", etc.)
- `columns`: Specific columns to load (None = all)
- `limit`: Maximum rows to load
- `include_features`: Include generated features

**Examples:**
```python
# Load training data
train = client.load_dataset("sales")

# Load test data without features
test = client.load_dataset("sales", table="test", include_features=False)

# Load specific columns
df = client.load_dataset("sales", columns=["date", "product", "price"])

# Load sample
sample = client.load_dataset("large_dataset", limit=1000)
```

#### register_dataset

Register a new dataset.

```python
dataset_info = client.register_dataset(
    name: str,
    path: str,
    target_column: str = None,
    id_columns: list[str] = None,
    problem_type: str = None,
    tags: list[str] = None,
    description: str = None,
    force: bool = False,
    no_features: bool = False,
    **column_types
)
```

**Parameters:**
- `name`: Unique dataset name
- `path`: Path to data file/directory
- `target_column`: Target column for ML
- `id_columns`: List of ID columns
- `problem_type`: "classification", "regression", etc.
- `tags`: List of tags
- `description`: Dataset description
- `force`: Overwrite if exists
- `no_features`: Skip feature generation
- `**column_types`: Force column types

**Examples:**
```python
# Basic registration
info = client.register_dataset("iris", "/data/iris.csv")

# With ML configuration
info = client.register_dataset(
    name="titanic",
    path="/data/titanic/",
    target_column="Survived",
    id_columns=["PassengerId"],
    problem_type="classification",
    tags=["kaggle", "binary-classification"],
    description="Titanic survival prediction dataset"
)

# Force column types
info = client.register_dataset(
    name="sales",
    path="/data/sales.csv",
    categorical_columns=["region", "product_type"],
    datetime_columns=["order_date", "ship_date"],
    text_columns=["notes"]
)
```

#### get_dataset_info

Get detailed dataset information.

```python
info = client.get_dataset_info(name: str)
```

**Returns:** `DatasetInfo` object with attributes:
- `name`: Dataset name
- `path`: Storage path
- `backend`: Storage backend
- `created_at`: Registration timestamp
- `row_count`: Number of rows
- `column_count`: Number of columns
- `target_column`: Target column name
- `id_columns`: List of ID columns
- `problem_type`: ML problem type
- `tags`: List of tags
- `description`: Description
- `tables`: Available tables
- `columns`: Column information
- `size_mb`: Database size in MB

**Example:**
```python
info = client.get_dataset_info("titanic")
print(f"Dataset: {info.name}")
print(f"Target: {info.target_column}")
print(f"Tables: {', '.join(info.tables)}")
print(f"Size: {info.size_mb:.2f} MB")
```

#### list_datasets

List all registered datasets.

```python
datasets = client.list_datasets(
    limit: int = None,
    sort_by: str = "name",
    reverse: bool = False,
    filter_backend: str = None
)
```

**Returns:** List of `DatasetInfo` objects

**Example:**
```python
# Get all datasets
datasets = client.list_datasets()

# Get 10 largest datasets
large_datasets = client.list_datasets(
    limit=10,
    sort_by="size",
    reverse=True
)

# Filter by backend
duckdb_datasets = client.list_datasets(filter_backend="duckdb")
```

#### search_datasets

Search datasets by pattern or attributes.

```python
results = client.search_datasets(
    pattern: str = None,
    tag: str = None,
    problem_type: str = None,
    min_rows: int = None,
    max_rows: int = None
)
```

**Examples:**
```python
# Search by name pattern
sales_datasets = client.search_datasets(pattern="sales")

# Find classification datasets
classifiers = client.search_datasets(problem_type="classification")

# Find by tag
kaggle_data = client.search_datasets(tag="kaggle")

# Complex search
large_clean = client.search_datasets(
    tag="validated",
    min_rows=10000
)
```

#### export_dataset

Export dataset to file.

```python
export_path = client.export_dataset(
    name: str,
    format: str = "csv",
    output_path: str = None,
    compression: str = None,
    table: str = None,
    include_features: bool = True,
    columns: list[str] = None
)
```

**Parameters:**
- `format`: "csv", "parquet", "json", "excel"
- `compression`: "gzip", "zip", "bz2", "xz"

**Example:**
```python
# Export as compressed parquet
path = client.export_dataset(
    "sales",
    format="parquet",
    compression="gzip",
    output_path="/exports/sales_2024.parquet.gz"
)
```

#### remove_dataset

Remove a dataset.

```python
client.remove_dataset(
    name: str,
    force: bool = False
)
```

**Example:**
```python
# Remove with confirmation
client.remove_dataset("old_data")

# Force removal
client.remove_dataset("test_data", force=True)
```

## Specialized Clients

MDMClient provides specialized clients for specific use cases:

### RegistrationClient

Advanced dataset registration operations.

```python
# Access via main client
reg_client = client.registration

# Direct initialization
from mdm.api.registration import RegistrationClient
reg_client = RegistrationClient()
```

**Methods:**

```python
# Register with custom validation
info = reg_client.register_with_validation(
    name="validated_data",
    path="/data/input.csv",
    validation_rules={
        "min_rows": 1000,
        "required_columns": ["id", "target"],
        "max_null_percentage": 0.1
    }
)

# Batch registration
results = reg_client.register_batch([
    {"name": "train_2023", "path": "/data/2023/train.csv"},
    {"name": "train_2024", "path": "/data/2024/train.csv"}
])

# Register from URL
info = reg_client.register_from_url(
    name="remote_data",
    url="https://example.com/data.csv"
)
```

### QueryClient

Advanced data querying capabilities.

```python
# Access via main client
query_client = client.query

# Direct initialization
from mdm.api.query import QueryClient
query_client = QueryClient()
```

**Methods:**

```python
# Execute SQL query
result_df = query_client.execute_sql(
    dataset="sales",
    query="SELECT product, SUM(price) as revenue FROM train GROUP BY product"
)

# Query with parameters
result_df = query_client.query(
    dataset="sales",
    table="train",
    filters={"category": "Electronics", "price": {"$gt": 100}},
    columns=["product", "price", "quantity"],
    sort_by="price",
    limit=100
)

# Aggregate query
stats = query_client.aggregate(
    dataset="sales",
    table="train",
    group_by=["category"],
    aggregations={
        "total_revenue": ("price", "sum"),
        "avg_quantity": ("quantity", "mean"),
        "product_count": ("product", "count")
    }
)
```

### MLIntegrationClient

Machine learning framework integration.

```python
# Access via main client
ml_client = client.ml

# Direct initialization
from mdm.api.ml_integration import MLIntegrationClient
ml_client = MLIntegrationClient()
```

**Methods:**

```python
# Load train/test split
X_train, X_test, y_train, y_test = ml_client.load_train_test_split(
    dataset="titanic",
    test_size=0.2,
    random_state=42,
    stratify=True
)

# Create scikit-learn compatible dataset
sklearn_dataset = ml_client.to_sklearn_dataset("titanic")
X, y = sklearn_dataset.data, sklearn_dataset.target

# Create PyTorch DataLoader
train_loader, test_loader = ml_client.to_pytorch_loaders(
    dataset="titanic",
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Create TensorFlow dataset
tf_dataset = ml_client.to_tensorflow_dataset(
    dataset="titanic",
    batch_size=32,
    prefetch=True
)

# Create submission file
ml_client.create_submission(
    dataset="titanic",
    predictions=model_predictions,
    output_path="submission.csv",
    id_column="PassengerId"
)

# Cross-validation splits
cv_splits = ml_client.create_cv_splits(
    dataset="titanic",
    n_splits=5,
    strategy="stratified"  # or "time_series"
)
```

### ExportClient

Advanced export operations.

```python
# Access via main client
export_client = client.export

# Direct initialization
from mdm.api.export import ExportClient
export_client = ExportClient()
```

**Methods:**

```python
# Export with transformations
path = export_client.export_with_transform(
    dataset="sales",
    transformations=[
        {"type": "normalize", "columns": ["price", "quantity"]},
        {"type": "encode", "columns": ["category"], "method": "onehot"}
    ],
    format="parquet"
)

# Export multiple formats
paths = export_client.export_multiple_formats(
    dataset="sales",
    formats=["csv", "parquet", "json"],
    output_dir="/exports"
)

# Export for specific ML framework
path = export_client.export_for_framework(
    dataset="titanic",
    framework="tensorflow",  # Creates TFRecord files
    split_ratio=0.8
)
```

## Working with Features

### Accessing Generated Features

```python
# Load with features
df_with_features = client.load_dataset("sales", include_features=True)

# Get feature columns
info = client.get_dataset_info("sales")
original_cols = info.columns["original"]
feature_cols = info.columns["features"]

print(f"Original columns: {len(original_cols)}")
print(f"Generated features: {len(feature_cols)}")

# Access specific feature types
temporal_features = [col for col in feature_cols if col.startswith("date_")]
statistical_features = [col for col in feature_cols if "_zscore" in col]
```

### Custom Feature Integration

```python
from mdm.features.custom.base import BaseDomainFeatures
import pandas as pd

class SalesFeatures(BaseDomainFeatures):
    """Custom features for sales dataset."""
    
    def __init__(self):
        super().__init__('sales')
    
    def _register_operations(self):
        self._operation_registry = {
            'revenue_features': self.calculate_revenue_features,
            'customer_features': self.calculate_customer_features
        }
    
    def calculate_revenue_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            'revenue': df['price'] * df['quantity'],
            'high_value_order': ((df['price'] * df['quantity']) > 1000).astype(int)
        }
    
    def calculate_customer_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        customer_totals = df.groupby('customer_id')['price'].transform('sum')
        return {
            'customer_total_spent': customer_totals,
            'is_vip_customer': (customer_totals > 5000).astype(int)
        }

# Save to ~/.mdm/config/custom_features/sales.py
# Then re-register dataset to apply custom features
```

## Advanced Querying

### SQL Interface

```python
# Direct SQL execution
with client.get_connection("sales") as conn:
    # Read query
    df = pd.read_sql("""
        SELECT 
            DATE_TRUNC('month', date) as month,
            category,
            SUM(price * quantity) as revenue,
            COUNT(DISTINCT customer_id) as unique_customers
        FROM train
        WHERE date >= '2024-01-01'
        GROUP BY 1, 2
        ORDER BY 1, 2
    """, conn)
    
    # Write query (be careful!)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sales_summary AS
        SELECT category, SUM(price) as total
        FROM train
        GROUP BY category
    """)
```

### Query Builder

```python
from mdm.api.query import QueryBuilder

# Build complex queries programmatically
qb = QueryBuilder("sales", "train")
result = (qb
    .select(["product", "category", "price", "quantity"])
    .where("price", ">", 100)
    .where("category", "in", ["Electronics", "Computers"])
    .group_by(["category", "product"])
    .having("SUM(quantity)", ">", 10)
    .order_by("price", desc=True)
    .limit(50)
    .execute()
)
```

## Performance Optimization

### Batch Processing

```python
# Process large dataset in batches
def process_in_batches(dataset_name, batch_size=10000):
    info = client.get_dataset_info(dataset_name)
    total_rows = info.row_count
    
    results = []
    for offset in range(0, total_rows, batch_size):
        # Load batch
        batch = client.query.query(
            dataset=dataset_name,
            table="train",
            offset=offset,
            limit=batch_size
        )
        
        # Process batch
        processed = your_processing_function(batch)
        results.append(processed)
        
        # Progress
        print(f"Processed {min(offset + batch_size, total_rows)}/{total_rows} rows")
    
    return pd.concat(results)
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def parallel_feature_computation(dataset_name, n_workers=4):
    df = client.load_dataset(dataset_name)
    
    # Split data for parallel processing
    chunks = np.array_split(df, n_workers)
    
    def process_chunk(chunk):
        # Your feature computation
        return compute_features(chunk)
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    return pd.concat(results)
```

### Memory-Efficient Loading

```python
# Iterator for large datasets
def iterate_dataset(dataset_name, chunk_size=1000):
    info = client.get_dataset_info(dataset_name)
    
    for offset in range(0, info.row_count, chunk_size):
        chunk = client.query.query(
            dataset=dataset_name,
            offset=offset,
            limit=chunk_size
        )
        yield chunk

# Use iterator
for chunk in iterate_dataset("large_dataset"):
    # Process chunk without loading entire dataset
    process_chunk(chunk)
```

## Error Handling

### Exception Types

```python
from mdm.core.exceptions import (
    MDMError,          # Base exception
    DatasetError,      # Dataset-specific errors
    BackendError,      # Storage backend errors
    ValidationError,   # Data validation errors
    ConfigError        # Configuration errors
)

# Example error handling
try:
    df = client.load_dataset("nonexistent")
except DatasetError as e:
    print(f"Dataset error: {e}")
except BackendError as e:
    print(f"Backend error: {e}")
    # Maybe try different backend
except MDMError as e:
    print(f"General MDM error: {e}")
```

### Validation and Checks

```python
# Check dataset exists
if client.dataset_exists("maybe_data"):
    df = client.load_dataset("maybe_data")
else:
    print("Dataset not found")

# Validate before operations
def safe_load_dataset(name, required_columns=None):
    try:
        info = client.get_dataset_info(name)
        
        if required_columns:
            missing = set(required_columns) - set(info.columns["original"])
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        
        return client.load_dataset(name)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
```

## Integration Examples

### Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data
X_train, X_test, y_train, y_test = client.ml.load_train_test_split(
    "titanic",
    test_size=0.2
)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
score = pipeline.score(X_test, y_test)
print(f"Accuracy: {score:.3f}")

# Create submission
test_data = client.load_dataset("titanic", table="test", include_features=False)
predictions = pipeline.predict(test_data)
client.ml.create_submission("titanic", predictions)
```

### PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class MDMDataset(Dataset):
    def __init__(self, dataset_name, client, table="train"):
        self.data = client.load_dataset(dataset_name, table=table)
        self.info = client.get_dataset_info(dataset_name)
        
        # Separate features and target
        self.X = self.data.drop(columns=[self.info.target_column])
        self.y = self.data[self.info.target_column]
        
        # Convert to tensors
        self.X = torch.FloatTensor(self.X.values)
        self.y = torch.LongTensor(self.y.values)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Usage
dataset = MDMDataset("titanic", client)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Pandas Profiling Integration

```python
from ydata_profiling import ProfileReport

# Load dataset
df = client.load_dataset("sales")

# Generate profile
profile = ProfileReport(
    df,
    title="Sales Dataset Analysis",
    explorative=True
)

# Save report
profile.to_file("sales_profile.html")
```

## Best Practices

### 1. Resource Management

```python
# Use context managers for connections
with client.get_connection("sales") as conn:
    df = pd.read_sql("SELECT * FROM train LIMIT 1000", conn)

# Close clients when done (if not using context manager)
client.close()
```

### 2. Configuration Management

```python
# Development config
dev_client = MDMClient(
    backend="sqlite",
    log_level="DEBUG",
    batch_size=1000
)

# Production config
prod_client = MDMClient(
    backend="postgresql",
    log_level="WARNING",
    batch_size=50000
)
```

### 3. Error Recovery

```python
# Implement retry logic
from time import sleep

def resilient_load(dataset_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.load_dataset(dataset_name)
        except BackendError as e:
            if attempt < max_retries - 1:
                sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
```

### 4. Logging

```python
import logging

# Configure MDM logging
logging.getLogger("mdm").setLevel(logging.DEBUG)

# Or use client parameter
client = MDMClient(log_level="DEBUG")
```

## API Reference Summary

### Core Classes
- `MDMClient`: Main client interface
- `DatasetInfo`: Dataset metadata container
- `RegistrationClient`: Dataset registration operations
- `QueryClient`: Data querying operations
- `MLIntegrationClient`: ML framework integration
- `ExportClient`: Export operations

### Key Methods
- `load_dataset()`: Load data as DataFrame
- `register_dataset()`: Register new dataset
- `get_dataset_info()`: Get dataset metadata
- `list_datasets()`: List all datasets
- `search_datasets()`: Search datasets
- `export_dataset()`: Export to file
- `remove_dataset()`: Delete dataset

### Exceptions
- `MDMError`: Base exception
- `DatasetError`: Dataset not found/invalid
- `BackendError`: Storage backend issues
- `ValidationError`: Data validation failures
- `ConfigError`: Configuration problems