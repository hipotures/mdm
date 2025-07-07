# Programmatic API

MDM provides a comprehensive Python API for programmatic dataset management. The API is organized into two levels: a high-level Dataset Manager API for common operations and an advanced Dataset Service API for full control.

## Installation

```python
# Import MDM components
from mdm import DatasetManager, load_config
from mdm.services import DatasetService
from mdm.database.engine_factory import DatabaseFactory
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

## Next Steps

- Explore [Advanced Features](09_Advanced_Features.md)
- Learn [Best Practices](10_Best_Practices.md)
- See [Troubleshooting](11_Troubleshooting.md) for common issues