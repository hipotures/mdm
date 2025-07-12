# 5. Programmatic API

MDM provides a high-level `MDMClient` for programmatic access to its features. This allows you to integrate dataset management directly into your Python applications and machine learning pipelines.

## Getting Started

First, instantiate the client:

```python
from mdm.api.mdm_client import MDMClient

client = MDMClient()
```

## Core Methods

### Dataset Registration

```python
client.register_dataset(
    name="my-new-dataset",
    dataset_path="/path/to/my/data",
    target_column="price",
    problem_type="regression",
    tags=["housing", "structured"]
)
```

### Listing and Querying Datasets

```python
# List all datasets
datasets = client.list_datasets()

# Get information about a specific dataset
info = client.get_dataset("my-new-dataset")

# Check if a dataset exists
exists = client.dataset_exists("my-new-dataset")

# Load a dataset into a pandas DataFrame
df = client.load_dataset("my-new-dataset")

# Execute a raw SQL query against the dataset's database
results = client.query_dataset(
    name="my-new-dataset",
    query="SELECT * FROM train WHERE price > 100000"
)
```

### Dataset Management

```python
# Update a dataset's metadata
client.update_dataset(
    name="my-new-dataset",
    description="A new, more detailed description."
)

# Remove a dataset
client.remove_dataset("my-new-dataset", force=True) # Use force=True to skip confirmation

# Export a dataset to CSV files
client.export_dataset(
    name="my-new-dataset",
    output_dir="/path/to/export",
    format="csv"
)
```

### Statistics

```python
# Compute (or re-compute) statistics for a dataset
client.compute_statistics("my-new-dataset")

# Retrieve pre-computed statistics
stats = client.get_statistics("my-new-dataset")
```

### Search

```python
# Search for datasets by name or description
results = client.search_datasets("housing")

# Search for datasets by tag
results = client.search_datasets_by_tag("structured")
```
