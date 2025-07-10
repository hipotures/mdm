# MDM API Reference

## Table of Contents
1. [MDMClient](#mdmclient)
2. [Dataset Operations](#dataset-operations)
3. [Data Access](#data-access)
4. [Export/Import](#exportimport)
5. [Search and Filter](#search-and-filter)
6. [Batch Operations](#batch-operations)
7. [Exceptions](#exceptions)
8. [Models](#models)

## MDMClient

The main entry point for programmatic access to MDM.

### Constructor

```python
from mdm import MDMClient

client = MDMClient(config: Optional[MDMConfig] = None)
```

**Parameters:**
- `config` (Optional[MDMConfig]): Custom configuration. If None, uses default configuration from `~/.mdm/mdm.yaml`

**Example:**
```python
# Use default configuration
client = MDMClient()

# Use custom configuration
from mdm.config import MDMConfig
custom_config = MDMConfig(
    database={"default_backend": "postgresql"},
    performance={"batch_size": 50000}
)
client = MDMClient(config=custom_config)
```

## Dataset Operations

### register_dataset

Register a new dataset in MDM.

```python
def register_dataset(
    self,
    name: str,
    dataset_path: Union[str, Path],
    target_column: Optional[str] = None,
    problem_type: Optional[str] = None,
    id_columns: Optional[List[str]] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    source: Optional[str] = None,
    force: bool = False,
    generate_features: bool = True
) -> DatasetInfo
```

**Parameters:**
- `name` (str): Unique dataset name
- `dataset_path` (Union[str, Path]): Path to dataset file or directory
- `target_column` (Optional[str]): Name of target column for ML
- `problem_type` (Optional[str]): One of: 'binary_classification', 'multiclass_classification', 'regression', 'time_series', 'clustering'
- `id_columns` (Optional[List[str]]): List of ID columns
- `description` (Optional[str]): Dataset description
- `tags` (Optional[List[str]]): Tags for categorization
- `source` (Optional[str]): Data source information
- `force` (bool): Overwrite if dataset exists
- `generate_features` (bool): Generate engineered features

**Returns:**
- `DatasetInfo`: Dataset information object

**Raises:**
- `DatasetError`: If registration fails
- `ValueError`: If parameters are invalid

**Example:**
```python
# Basic registration
info = client.register_dataset(
    name="sales_2024",
    dataset_path="/data/sales.csv"
)

# Full registration with metadata
info = client.register_dataset(
    name="customer_churn",
    dataset_path="/data/customers/",
    target_column="churned",
    problem_type="binary_classification",
    id_columns=["customer_id"],
    description="Customer churn prediction dataset",
    tags=["classification", "customer", "2024"],
    source="CRM System Export",
    generate_features=True
)
```

### get_dataset

Retrieve dataset information.

```python
def get_dataset(self, name: str) -> Optional[DatasetInfo]
```

**Parameters:**
- `name` (str): Dataset name

**Returns:**
- `Optional[DatasetInfo]`: Dataset information or None if not found

**Example:**
```python
info = client.get_dataset("sales_2024")
if info:
    print(f"Dataset: {info.name}")
    print(f"Rows: {info.metadata.get('row_count', 'Unknown')}")
    print(f"Target: {info.target_column}")
```

### list_datasets

List all registered datasets.

```python
def list_datasets(
    self,
    limit: Optional[int] = None,
    offset: int = 0,
    sort_by: str = "name",
    tags: Optional[List[str]] = None,
    problem_type: Optional[str] = None
) -> List[DatasetInfo]
```

**Parameters:**
- `limit` (Optional[int]): Maximum number of datasets to return
- `offset` (int): Number of datasets to skip
- `sort_by` (str): Sort field ('name', 'created_at', 'size')
- `tags` (Optional[List[str]]): Filter by tags
- `problem_type` (Optional[str]): Filter by problem type

**Returns:**
- `List[DatasetInfo]`: List of dataset information objects

**Example:**
```python
# List all datasets
datasets = client.list_datasets()

# List with filters
ml_datasets = client.list_datasets(
    limit=10,
    sort_by="created_at",
    tags=["ml", "production"],
    problem_type="classification"
)
```

### update_dataset

Update dataset metadata.

```python
def update_dataset(
    self,
    name: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    problem_type: Optional[str] = None,
    id_columns: Optional[List[str]] = None
) -> DatasetInfo
```

**Parameters:**
- `name` (str): Dataset name
- `description` (Optional[str]): New description
- `tags` (Optional[List[str]]): New tags (replaces existing)
- `target_column` (Optional[str]): New target column
- `problem_type` (Optional[str]): New problem type
- `id_columns` (Optional[List[str]]): New ID columns

**Returns:**
- `DatasetInfo`: Updated dataset information

**Example:**
```python
updated = client.update_dataset(
    "sales_2024",
    description="Updated sales data with Q4 included",
    tags=["sales", "2024", "complete"]
)
```

### remove_dataset

Remove a dataset from MDM.

```python
def remove_dataset(self, name: str, force: bool = False) -> None
```

**Parameters:**
- `name` (str): Dataset name
- `force` (bool): Skip confirmation (always True internally)

**Raises:**
- `DatasetError`: If dataset not found or removal fails

**Example:**
```python
# Remove dataset
client.remove_dataset("old_dataset")
```

## Data Access

### load_dataset

Load dataset into memory as pandas DataFrame(s).

```python
def load_dataset(
    self,
    name: str,
    tables: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    as_iterator: bool = False,
    chunk_size: int = 10000
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], Iterator[pd.DataFrame]]
```

**Parameters:**
- `name` (str): Dataset name
- `tables` (Optional[List[str]]): Tables to load (default: ['train', 'test'])
- `columns` (Optional[List[str]]): Columns to load
- `sample_size` (Optional[int]): Random sample size
- `as_iterator` (bool): Return iterator for memory efficiency
- `chunk_size` (int): Chunk size for iterator

**Returns:**
- Single DataFrame (if one table)
- Tuple of DataFrames (if multiple tables)
- Iterator of DataFrames (if as_iterator=True)

**Example:**
```python
# Load full dataset
train_df, test_df = client.load_dataset("sales_2024")

# Load specific columns
df = client.load_dataset(
    "sales_2024",
    tables=["train"],
    columns=["date", "amount", "category"]
)

# Load as iterator for large datasets
for chunk in client.load_dataset("huge_dataset", as_iterator=True):
    # Process chunk
    process_chunk(chunk)

# Load sample
sample_df = client.load_dataset("sales_2024", sample_size=1000)
```

### load_dataset_files

Load specific files from a dataset.

```python
def load_dataset_files(
    self,
    name: str,
    sample_size: Optional[int] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]
```

**Parameters:**
- `name` (str): Dataset name
- `sample_size` (Optional[int]): Sample size per file

**Returns:**
- `Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]`: (train_df, test_df)

**Example:**
```python
train_df, test_df = client.load_dataset_files("sales_2024")
if test_df is not None:
    print(f"Test set size: {len(test_df)}")
```

### get_dataset_stats

Get detailed statistics for a dataset.

```python
def get_dataset_stats(self, name: str) -> Dict[str, Any]
```

**Parameters:**
- `name` (str): Dataset name

**Returns:**
- `Dict[str, Any]`: Statistics including row counts, memory usage, column types

**Example:**
```python
stats = client.get_dataset_stats("sales_2024")
print(f"Total rows: {stats['row_count']}")
print(f"Memory usage: {stats['memory_size_mb']:.2f} MB")
print(f"Numeric columns: {stats['numeric_columns']}")
```

## Export/Import

### export_dataset

Export dataset to files.

```python
def export_dataset(
    self,
    name: str,
    output_dir: str,
    format: str = "csv",
    tables: Optional[List[str]] = None,
    compression: Optional[str] = None
) -> List[str]
```

**Parameters:**
- `name` (str): Dataset name
- `output_dir` (str): Output directory path
- `format` (str): Export format ('csv', 'parquet', 'json')
- `tables` (Optional[List[str]]): Tables to export
- `compression` (Optional[str]): Compression type ('gzip', 'zip', 'snappy')

**Returns:**
- `List[str]`: List of exported file paths

**Example:**
```python
# Export as CSV
files = client.export_dataset(
    "sales_2024",
    output_dir="/exports/",
    format="csv"
)

# Export as compressed Parquet
files = client.export_dataset(
    "sales_2024",
    output_dir="/exports/",
    format="parquet",
    compression="snappy"
)
```

## Search and Filter

### search_datasets

Search datasets by name or metadata.

```python
def search_datasets(
    self,
    query: str,
    search_in: List[str] = ["name", "description", "tags"],
    limit: Optional[int] = None
) -> List[DatasetInfo]
```

**Parameters:**
- `query` (str): Search query
- `search_in` (List[str]): Fields to search
- `limit` (Optional[int]): Maximum results

**Returns:**
- `List[DatasetInfo]`: Matching datasets

**Example:**
```python
# Search by name
sales_datasets = client.search_datasets("sales")

# Search in specific fields
customer_datasets = client.search_datasets(
    "customer",
    search_in=["name", "tags"]
)
```

### filter_datasets

Filter datasets by criteria.

```python
def filter_datasets(
    self,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
    has_target: Optional[bool] = None,
    backend: Optional[str] = None
) -> List[DatasetInfo]
```

**Parameters:**
- `min_rows` (Optional[int]): Minimum row count
- `max_rows` (Optional[int]): Maximum row count
- `created_after` (Optional[datetime]): Created after date
- `created_before` (Optional[datetime]): Created before date
- `has_target` (Optional[bool]): Has target column
- `backend` (Optional[str]): Storage backend type

**Returns:**
- `List[DatasetInfo]`: Filtered datasets

**Example:**
```python
# Find large datasets
large_datasets = client.filter_datasets(min_rows=1000000)

# Find recent datasets with targets
ml_ready = client.filter_datasets(
    created_after=datetime(2024, 1, 1),
    has_target=True
)
```

## Batch Operations

### batch_export

Export multiple datasets at once.

```python
def batch_export(
    self,
    dataset_names: List[str],
    output_dir: str,
    format: str = "csv",
    compression: Optional[str] = None,
    parallel: bool = True
) -> Dict[str, List[str]]
```

**Parameters:**
- `dataset_names` (List[str]): Dataset names to export
- `output_dir` (str): Output directory
- `format` (str): Export format
- `compression` (Optional[str]): Compression type
- `parallel` (bool): Export in parallel

**Returns:**
- `Dict[str, List[str]]`: Map of dataset name to exported files

**Example:**
```python
# Export multiple datasets
results = client.batch_export(
    ["sales_2023", "sales_2024", "sales_forecast"],
    output_dir="/backups/",
    format="parquet",
    compression="snappy"
)
```

### batch_update

Update multiple datasets at once.

```python
def batch_update(
    self,
    updates: Dict[str, Dict[str, Any]]
) -> Dict[str, DatasetInfo]
```

**Parameters:**
- `updates` (Dict[str, Dict[str, Any]]): Map of dataset name to update fields

**Returns:**
- `Dict[str, DatasetInfo]`: Map of dataset name to updated info

**Example:**
```python
updates = {
    "sales_2023": {"tags": ["historical", "sales"]},
    "sales_2024": {"tags": ["current", "sales"]},
}
results = client.batch_update(updates)
```

## Exceptions

### DatasetError

Base exception for dataset-related errors.

```python
from mdm.core.exceptions import DatasetError

try:
    client.get_dataset("nonexistent")
except DatasetError as e:
    print(f"Dataset error: {e}")
```

### StorageError

Storage backend errors.

```python
from mdm.core.exceptions import StorageError

try:
    client.load_dataset("corrupted_dataset")
except StorageError as e:
    print(f"Storage error: {e}")
```

### ConfigurationError

Configuration-related errors.

```python
from mdm.core.exceptions import ConfigurationError

try:
    client = MDMClient(config=invalid_config)
except ConfigurationError as e:
    print(f"Config error: {e}")
```

## Models

### DatasetInfo

Dataset information model.

```python
@dataclass
class DatasetInfo:
    name: str
    description: Optional[str]
    source: str
    created_at: datetime
    updated_at: datetime
    tables: Dict[str, str]
    target_column: Optional[str]
    id_columns: List[str]
    problem_type: Optional[ProblemType]
    tags: List[str]
    database: Dict[str, Any]
    metadata: Dict[str, Any]
```

### ProblemType

Enumeration of supported problem types.

```python
from enum import Enum

class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
```

### ColumnType

Column type enumeration.

```python
from enum import Enum

class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BINARY = "binary"
    ID = "id"
    TARGET = "target"
```

## Advanced Usage

### Custom Feature Engineering

```python
# Register dataset with custom features
from mdm.features import CustomTransformer

class PriceFeatures(CustomTransformer):
    def transform(self, df):
        df['price_log'] = np.log1p(df['price'])
        df['price_squared'] = df['price'] ** 2
        return df

# Register with custom transformer
client.register_dataset(
    "sales_advanced",
    dataset_path="/data/sales.csv",
    custom_transformers=[PriceFeatures()]
)
```

### Memory-Efficient Processing

```python
# Process large dataset in chunks
def process_large_dataset(name: str):
    total_processed = 0
    
    for chunk in client.load_dataset(name, as_iterator=True, chunk_size=50000):
        # Process chunk
        processed = process_chunk(chunk)
        total_processed += len(processed)
        
        # Save results incrementally
        save_results(processed)
    
    return total_processed
```

### Integration with ML Frameworks

```python
# scikit-learn integration
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
X, _ = client.load_dataset("customer_churn", tables=["train"])
y = X.pop("churned")

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# PyTorch integration
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load and convert to tensors
df = client.load_dataset("image_features", tables=["train"])
X = torch.tensor(df.drop("label", axis=1).values, dtype=torch.float32)
y = torch.tensor(df["label"].values, dtype=torch.long)

# Create DataLoader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```