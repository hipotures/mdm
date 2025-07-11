# Programmatic API

MDM provides a comprehensive Python API for programmatic dataset management. The API is organized into two levels: the high-level MDMClient API for common operations and the advanced Dataset Manager/Service APIs for full control.

## Installation

```python
# High-level API (recommended)
from mdm.api import MDMClient

# Advanced APIs
from mdm import DatasetManager, load_config
from mdm.services import DatasetService
from mdm.database.engine_factory import DatabaseFactory
```

## MDMClient API (Recommended)

The MDMClient provides the simplest and most intuitive interface for working with MDM.

### Quick Start

```python
from mdm.api import MDMClient

# Initialize client (auto-loads configuration)
client = MDMClient()

# Register a dataset
dataset_info = client.register_dataset(
    name="sales_2024",
    path="data/sales.csv",
    target_column="revenue"
)

# Load and work with data
df = client.load_dataset("sales_2024")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Get dataset information
info = client.get_dataset_info("sales_2024")
print(f"Target: {info.target_column}")
print(f"Problem type: {info.problem_type}")
```

### Complete Example: ML Workflow

```python
from mdm.api import MDMClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize MDM client
client = MDMClient()

# Register Titanic dataset
client.register_dataset(
    name="titanic",
    path="data/titanic.csv",
    target_column="Survived",
    problem_type="binary_classification",
    id_columns=["PassengerId"]
)

# Load data with automatic feature engineering
df = client.load_dataset("titanic")

# Get dataset info
info = client.get_dataset_info("titanic")
print(f"Features: {len(info.feature_columns)} columns")
print(f"Target distribution: {df[info.target_column].value_counts().to_dict()}")

# Prepare data for modeling
X = df[info.feature_columns]
y = df[info.target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Export results
client.export_dataset(
    "titanic",
    format="parquet",
    output_dir="./exports/titanic"
)
```

### Dataset Registration Examples

```python
# Basic CSV registration
client.register_dataset(
    name="iris",
    path="data/iris.csv",
    target_column="species"
)

# Kaggle competition format (auto-detects train/test structure)
client.register_dataset(
    name="house_prices",
    path="data/kaggle/house-prices/",  # Contains train.csv, test.csv
    target_column="SalePrice"
)

# Multi-file dataset with specific configurations
client.register_dataset(
    name="customer_churn",
    path="data/churn/",
    target_column="Churn",
    problem_type="binary_classification",
    id_columns=["customerID"],
    description="Telco customer churn prediction dataset"
)

# Time series dataset
client.register_dataset(
    name="stock_prices",
    path="data/stocks.csv",
    target_column="Close",
    problem_type="regression",
    description="Daily stock prices with technical indicators"
)

# Force overwrite existing dataset
client.register_dataset(
    name="iris",
    path="data/iris_updated.csv",
    target_column="species",
    force=True
)
```

### Data Loading and Querying

```python
# Load full dataset
df = client.load_dataset("house_prices")

# Load with sampling
df_sample = client.load_dataset("house_prices", sample_size=1000)

# Load specific columns
df = client.load_dataset(
    "house_prices",
    columns=["SalePrice", "GrLivArea", "YearBuilt"]
)

# Execute custom SQL query
df = client.query_dataset(
    "house_prices",
    "SELECT * FROM train WHERE SalePrice > 200000 AND YearBuilt > 2000"
)

# Load train/test splits separately
train_df = client.load_table("house_prices", "train")
test_df = client.load_table("house_prices", "test")

# Load with features if available
df = client.load_dataset("house_prices", include_features=True)
```

### Dataset Information and Statistics

```python
# Get comprehensive dataset info
info = client.get_dataset_info("titanic")
print(f"Name: {info.name}")
print(f"Description: {info.description}")
print(f"Target: {info.target_column}")
print(f"Problem type: {info.problem_type}")
print(f"Row count: {info.row_count}")
print(f"Column count: {info.column_count}")
print(f"Features: {info.feature_columns}")
print(f"ID columns: {info.id_columns}")
print(f"Tables: {info.tables}")
print(f"Size: {info.size_mb:.2f} MB")
print(f"Backend: {info.backend}")

# Get statistical summary
stats = client.get_dataset_stats("titanic")
print(f"Missing values: {stats['missing_percentage']:.2f}%")
print(f"Numeric columns: {stats['numeric_columns']}")
print(f"Categorical columns: {stats['categorical_columns']}")
print(f"Target distribution: {stats['target_distribution']}")

# Get column-level statistics
column_stats = client.get_column_stats("titanic", "Age")
print(f"Mean: {column_stats['mean']:.2f}")
print(f"Std: {column_stats['std']:.2f}")
print(f"Missing: {column_stats['missing_count']} ({column_stats['missing_percentage']:.2f}%)")
```

### Dataset Management

```python
# List all datasets
datasets = client.list_datasets()
for ds in datasets:
    print(f"{ds.name}: {ds.row_count} rows, {ds.problem_type}")

# List with filtering
classification_datasets = client.list_datasets(
    problem_type="classification"
)

large_datasets = client.list_datasets(
    min_rows=10000
)

# Search datasets
results = client.search_datasets("customer")
results = client.search_datasets_by_tag("production")

# Update dataset metadata
client.update_dataset(
    "titanic",
    description="Titanic passenger survival prediction",
    tags=["classification", "tutorial", "binary"],
    problem_type="binary_classification"
)

# Remove dataset
client.remove_dataset("old_dataset", force=True)
```

### Export and Backup

```python
# Export to CSV
client.export_dataset(
    "titanic",
    format="csv",
    output_dir="./exports"
)

# Export to Parquet with compression
client.export_dataset(
    "house_prices",
    format="parquet",
    output_dir="./exports",
    compression="gzip"
)

# Export specific tables
client.export_dataset(
    "house_prices",
    format="csv",
    output_dir="./exports",
    tables=["train"]
)

# Export with metadata
client.export_dataset(
    "titanic",
    format="parquet",
    output_dir="./exports",
    include_metadata=True
)

# Batch export multiple datasets
datasets_to_export = ["titanic", "iris", "house_prices"]
for dataset_name in datasets_to_export:
    client.export_dataset(
        dataset_name,
        format="parquet",
        output_dir=f"./backups/{dataset_name}"
    )
```

### Working with Features

```python
# Load dataset with generated features
df = client.load_dataset("house_prices", include_features=True)

# Get feature information
feature_info = client.get_feature_info("house_prices")
print(f"Total features: {feature_info['total_features']}")
print(f"Numeric features: {feature_info['numeric_features']}")
print(f"Categorical features: {feature_info['categorical_features']}")
print(f"Generated features: {feature_info['generated_features']}")

# Access feature importance (if available)
if 'feature_importance' in feature_info:
    for feature, importance in feature_info['feature_importance'].items():
        print(f"{feature}: {importance:.4f}")
```

### Performance Optimization

```python
# Use chunked processing for large datasets
def process_in_chunks(dataset_name, chunk_size=10000):
    info = client.get_dataset_info(dataset_name)
    total_rows = info.row_count
    
    for offset in range(0, total_rows, chunk_size):
        # Query chunk
        chunk = client.query_dataset(
            dataset_name,
            f"SELECT * FROM train LIMIT {chunk_size} OFFSET {offset}"
        )
        
        # Process chunk
        yield process_chunk(chunk)

# Parallel processing example
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def analyze_dataset(dataset_name):
    df = client.load_dataset(dataset_name)
    return {
        'name': dataset_name,
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
    }

# Analyze multiple datasets in parallel
dataset_names = client.list_datasets()[:5]  # First 5 datasets
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(analyze_dataset, [ds.name for ds in dataset_names]))

for result in results:
    print(f"{result['name']}: {result['shape']}, {result['memory_usage']:.2f} MB")
```

### Error Handling

```python
from mdm.exceptions import (
    DatasetNotFoundError,
    DatasetAlreadyExistsError,
    ValidationError,
    StorageError
)

try:
    # Try to load non-existent dataset
    df = client.load_dataset("nonexistent")
except DatasetNotFoundError as e:
    print(f"Dataset not found: {e}")

try:
    # Try to register existing dataset without force
    client.register_dataset(
        name="iris",
        path="data/iris.csv",
        target_column="species"
    )
except DatasetAlreadyExistsError as e:
    print(f"Dataset already exists: {e}")
    # Use force=True to overwrite

try:
    # Invalid registration
    client.register_dataset(
        name="invalid dataset name!",  # Invalid characters
        path="/nonexistent/path",
        target_column="target"
    )
except ValidationError as e:
    print(f"Validation error: {e}")

# Graceful error handling with fallback
def safe_load_dataset(name, fallback_path=None):
    try:
        return client.load_dataset(name)
    except DatasetNotFoundError:
        if fallback_path:
            print(f"Dataset {name} not found, registering from {fallback_path}")
            client.register_dataset(name, fallback_path)
            return client.load_dataset(name)
        else:
            raise
```

### Integration Examples

#### Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Load house prices data
df = client.load_dataset("house_prices")
info = client.get_dataset_info("house_prices")

# Prepare features and target
X = df[info.feature_columns]
y = df[info.target_column]

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std() * 2:.2f})")
```

#### Pandas Profiling Integration

```python
import pandas_profiling

# Load dataset
df = client.load_dataset("titanic")

# Generate profile report
profile = pandas_profiling.ProfileReport(
    df,
    title="Titanic Dataset Profile",
    explorative=True
)

# Save report
profile.to_file("titanic_profile.html")
```

#### DuckDB Analytics

```python
# Direct DuckDB queries for analytics
result = client.query_dataset(
    "house_prices",
    """
    SELECT 
        YearBuilt,
        AVG(SalePrice) as avg_price,
        COUNT(*) as num_houses,
        STDDEV(SalePrice) as price_stddev
    FROM train
    WHERE YearBuilt >= 2000
    GROUP BY YearBuilt
    ORDER BY YearBuilt
    """
)

# Visualize trends
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(result['YearBuilt'], result['avg_price'], marker='o')
plt.title('Average House Price by Year Built')
plt.xlabel('Year Built')
plt.ylabel('Average Price ($)')

plt.subplot(1, 2, 2)
plt.bar(result['YearBuilt'], result['num_houses'])
plt.title('Number of Houses by Year Built')
plt.xlabel('Year Built')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
```

## Dataset Manager API

The Dataset Manager provides a high-level interface for common dataset operations.

### Initialization

```python
from mdm import DatasetManager, load_config

# Load configuration from mdm.yaml
config = load_config()

# Initialize manager
dataset_manager = DatasetManager(config)
```

### Dataset Registration

```python
# Register dataset programmatically
result = dataset_manager.register_dataset(
    name="my_dataset",
    dataset_path="/path/to/data",
    target_column="target",
    description="My ML dataset",
    auto_analyze=True
)

# Register with additional options
result = dataset_manager.register_dataset(
    name="sales_data",
    dataset_path="/data/sales",
    target_column="revenue",
    id_columns=["transaction_id", "customer_id"],
    problem_type="regression",
    datetime_columns=["date", "timestamp"],
    categorical_columns=["category", "region"],
    numeric_columns=["amount", "quantity"],
    auto_analyze=True,
    force=False  # Don't overwrite existing
)
```

### Dataset Information

```python
# Get dataset information
dataset_info = dataset_manager.get_dataset("my_dataset")

# Access dataset properties
print(f"Target: {dataset_info.target_column}")
print(f"Problem Type: {dataset_info.problem_type}")
print(f"Tables: {dataset_info.tables}")
print(f"Row Count: {dataset_info.num_rows}")

# List all datasets
datasets = dataset_manager.list_datasets()
for dataset in datasets:
    print(f"{dataset.name}: {dataset.description}")

# Filter datasets
regression_datasets = dataset_manager.list_datasets(
    filter_func=lambda d: d.problem_type == "regression"
)
```

### Loading Data

```python
# Load dataset files as DataFrames
train_df, test_df = dataset_manager.load_dataset_files(
    "my_dataset",
    sample_size=10000  # Optional sampling
)

# Load specific tables
train_df = dataset_manager.load_table("my_dataset", "train")
test_df = dataset_manager.load_table("my_dataset", "test")

# Load with query
df = dataset_manager.query_dataset(
    "my_dataset",
    "SELECT * FROM train WHERE target > 100 LIMIT 1000"
)
```

### Direct Database Connection

```python
# Get dataset connection (DuckDB)
conn = dataset_manager.get_dataset_connection("my_dataset")

# Execute custom queries
result = conn.execute("SELECT COUNT(*) FROM train").fetchone()

# Use with context manager
with dataset_manager.get_dataset_connection("my_dataset") as conn:
    df = conn.execute("SELECT * FROM train").fetch_df()
```

## Dataset Service API (Advanced)

The Dataset Service provides low-level access to all MDM functionality.

### Initialization

```python
from mdm.services.dataset_service import DatasetService
from mdm.database.engine_factory import DatabaseFactory

# Initialize service
db_manager = DatabaseFactory.create_manager(config)
dataset_service = DatasetService(db_manager)
```

### Advanced Registration

```python
# Auto-register with feature generation
result = dataset_service.register_dataset_auto(
    name="kaggle_competition",
    path="/data/kaggle/competition",
    target_column="target",
    id_column="id",
    competition_name="house-prices",
    description="Kaggle House Prices Competition",
    force_update=False
)

# Backend is determined by mdm.yaml configuration at registration time
result = dataset_service.register_dataset(
    name="large_dataset",
    train_path="/data/train.parquet",
    test_path="/data/test.parquet",
    target_column="label"
)
```

### Dataset Operations

```python
# Search datasets
matching = dataset_service.search_datasets("kaggle")
matching_regression = dataset_service.search_datasets(
    query="regression",
    filters={"problem_type": "regression"}
)

# Get statistics
stats = dataset_service.get_dataset_stats()
print(f"Total datasets: {stats['total']}")
print(f"Total size: {stats['total_size_gb']} GB")

# Get detailed statistics for a dataset
dataset_stats = dataset_service.get_dataset_statistics(
    "my_dataset",
    include_correlations=True,
    include_distributions=True
)

# Update dataset metadata
updates = {
    "description": "Updated description",
    "target_column": "new_target",
    "tags": ["production", "v2"]
}
dataset_service.update_dataset("my_dataset", updates)
```

### Export and Import

```python
# Export dataset
dataset_service.export_dataset(
    "my_dataset",
    output_dir="./exports",
    format="parquet",
    include_metadata=True
)

# Export metadata only
metadata = dataset_service.export_metadata(
    "my_dataset",
    output_path="./metadata.json"
)
```

## Working with DataFrames

### Pandas Integration

```python
import pandas as pd

# Load as pandas DataFrame
train_df = dataset_manager.load_table("titanic", "train")

# Process with pandas
train_df['Age_Group'] = pd.cut(train_df['Age'], bins=[0, 18, 60, 100])

# Save processed data back
dataset_manager.save_processed_data(
    "titanic",
    train_df,
    table_name="train_processed"
)
```

### DuckDB Integration

```python
# Direct DuckDB operations
with dataset_manager.get_dataset_connection("house_prices") as conn:
    # Create view
    conn.execute("""
        CREATE VIEW expensive_houses AS
        SELECT * FROM train WHERE SalePrice > 500000
    """)
    
    # Advanced analytics
    result = conn.execute("""
        SELECT 
            Neighborhood,
            AVG(SalePrice) as avg_price,
            COUNT(*) as count
        FROM train
        GROUP BY Neighborhood
        ORDER BY avg_price DESC
    """).fetch_df()
```

## Batch Operations

### Processing Multiple Datasets

```python
# Batch export
export_configs = [
    {"name": "titanic", "format": "parquet"},
    {"name": "house_prices", "format": "csv"},
    {"name": "customer_churn", "format": "parquet"}
]

for config in export_configs:
    dataset_service.export_dataset(
        config["name"],
        output_dir=f"./exports/{config['name']}",
        format=config["format"]
    )
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def process_dataset(dataset_name):
    """Process a single dataset"""
    df = dataset_manager.load_table(dataset_name, "train")
    stats = df.describe()
    return dataset_name, stats

# Process multiple datasets in parallel
dataset_names = ["titanic", "house_prices", "customer_churn"]
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_dataset, dataset_names))
```

## Error Handling

```python
from mdm.exceptions import (
    DatasetNotFoundError,
    RegistrationError,
    ValidationError
)

try:
    dataset = dataset_manager.get_dataset("nonexistent")
except DatasetNotFoundError:
    print("Dataset not found")

try:
    result = dataset_manager.register_dataset(
        name="invalid_data",
        dataset_path="/invalid/path"
    )
except RegistrationError as e:
    print(f"Registration failed: {e}")
```

## Custom Extensions

### Creating Custom Loaders

```python
class CustomDatasetLoader:
    def __init__(self, dataset_manager):
        self.dm = dataset_manager
    
    def load_with_preprocessing(self, dataset_name):
        """Load dataset with custom preprocessing"""
        # Load raw data
        train_df = self.dm.load_table(dataset_name, "train")
        
        # Apply preprocessing
        train_df = self.preprocess(train_df)
        
        return train_df
    
    def preprocess(self, df):
        """Custom preprocessing logic"""
        # Remove outliers, encode categoricals, etc.
        return df

# Use custom loader
loader = CustomDatasetLoader(dataset_manager)
processed_df = loader.load_with_preprocessing("titanic")
```

### Integration with ML Frameworks

```python
# Scikit-learn integration
from sklearn.model_selection import train_test_split

def get_sklearn_data(dataset_name, test_size=0.2):
    """Get data ready for scikit-learn"""
    df = dataset_manager.load_table(dataset_name, "train")
    info = dataset_manager.get_dataset(dataset_name)
    
    # Separate features and target
    X = df.drop(columns=[info.target_column] + info.id_columns)
    y = df[info.target_column]
    
    # Split data
    return train_test_split(X, y, test_size=test_size, random_state=42)

# PyTorch integration
import torch
from torch.utils.data import Dataset, DataLoader

class MDMDataset(Dataset):
    def __init__(self, dataset_name, table="train"):
        self.df = dataset_manager.load_table(dataset_name, table)
        self.info = dataset_manager.get_dataset(dataset_name)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = row.drop([self.info.target_column])
        target = row[self.info.target_column]
        return torch.tensor(features.values), torch.tensor(target)

# Create PyTorch DataLoader
dataset = MDMDataset("titanic")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Best Practices

1. **Use context managers** for database connections
2. **Handle exceptions** appropriately
3. **Use batch operations** for multiple datasets
4. **Cache frequently accessed data** in memory
5. **Clean up connections** after use

## Advanced MDMClient Features

### Configuration Override

```python
# Initialize with custom configuration
from mdm.config import MDMConfig

config = MDMConfig(
    database_backend="duckdb",
    batch_size=50000,
    feature_engineering_enabled=True
)

client = MDMClient(config=config)

# Or use environment variables
import os
os.environ['MDM_DATABASE_DEFAULT_BACKEND'] = 'postgresql'
os.environ['MDM_PERFORMANCE_BATCH_SIZE'] = '100000'

client = MDMClient()  # Will use env vars
```

### Custom Feature Engineering

```python
# Define custom feature transformer
def create_custom_features(df, dataset_info):
    """Add domain-specific features."""
    if 'price' in df.columns and 'quantity' in df.columns:
        df['total_value'] = df['price'] * df['quantity']
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df

# Register dataset with custom features
client.register_dataset(
    name="sales",
    path="data/sales.csv",
    target_column="revenue",
    custom_feature_function=create_custom_features
)
```

### Monitoring and Logging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor registration progress
from mdm.utils import ProgressCallback

class CustomProgress(ProgressCallback):
    def on_progress(self, current, total, message):
        print(f"Progress: {current}/{total} - {message}")

client.register_dataset(
    name="large_dataset",
    path="data/large.csv",
    progress_callback=CustomProgress()
)

# Get operation metrics
metrics = client.get_metrics()
print(f"Total datasets: {metrics['total_datasets']}")
print(f"Total size: {metrics['total_size_gb']:.2f} GB")
print(f"Average query time: {metrics['avg_query_time_ms']:.2f} ms")
```

### Direct Backend Access

```python
# Get direct access to storage backend
backend = client.get_backend("house_prices")

# Execute raw SQL
with backend.get_connection() as conn:
    result = conn.execute(
        "SELECT COUNT(DISTINCT Neighborhood) FROM train"
    ).fetchone()
    print(f"Unique neighborhoods: {result[0]}")

# Create custom indexes for performance
with backend.get_connection() as conn:
    conn.execute(
        "CREATE INDEX idx_year_price ON train(YearBuilt, SalePrice)"
    )
```

### Batch Operations

```python
# Register multiple datasets
datasets_to_register = [
    {"name": "iris", "path": "data/iris.csv", "target": "species"},
    {"name": "wine", "path": "data/wine.csv", "target": "quality"},
    {"name": "digits", "path": "data/digits.csv", "target": "digit"}
]

for ds in datasets_to_register:
    try:
        client.register_dataset(
            name=ds["name"],
            path=ds["path"],
            target_column=ds["target"]
        )
        print(f"✓ Registered {ds['name']}")
    except Exception as e:
        print(f"✗ Failed to register {ds['name']}: {e}")

# Bulk operations
dataset_names = [ds.name for ds in client.list_datasets()]

# Export all datasets
for name in dataset_names:
    client.export_dataset(name, format="parquet", output_dir=f"./backups/{name}")

# Get statistics for all datasets
all_stats = {}
for name in dataset_names:
    all_stats[name] = client.get_dataset_stats(name)

# Find datasets with missing values
datasets_with_missing = [
    name for name, stats in all_stats.items()
    if stats['missing_percentage'] > 0
]
print(f"Datasets with missing values: {datasets_with_missing}")
```

## Next Steps

- Explore [Advanced Features](09_Advanced_Features.md)
- Learn [Best Practices](10_Best_Practices.md)
- See [Troubleshooting](11_Troubleshooting.md) for common issues
- Check [API Reference](../api/API_Reference.md) for complete API documentation