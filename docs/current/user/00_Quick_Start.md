# MDM Quick Start Guide

Get started with ML Data Manager in 5 minutes!

## Installation

```bash
# Using pip
pip install mdm

# Or for development
git clone https://github.com/mdm-project/mdm.git
cd mdm
uv pip install -e .
```

## Basic Usage

### 1. Register Your First Dataset

```bash
# Register a CSV file
mdm dataset register iris data/iris.csv --target species

# Register a directory with multiple files
mdm dataset register titanic ./titanic_data/
```

### 2. List Your Datasets

```bash
# See all registered datasets
mdm dataset list

# Output:
# ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
# ┃ Name    ┃ Problem Type            ┃ Target  ┃ Tables  ┃ Total Rows ┃ MEM Size ┃ Backend ┃
# ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
# │ iris    │ multiclass_classification│ species │ 1       │ 150        │ 8.5 KB   │ sqlite  │
# │ titanic │ binary_classification   │ survived│ 3       │ 1,309      │ 126.4 KB │ sqlite  │
# └─────────┴─────────────────────────┴─────────┴─────────┴────────────┴──────────┴─────────┘
```

### 3. Get Dataset Information

```bash
# View detailed information
mdm dataset info iris

# View statistics
mdm dataset stats iris
```

### 4. Export Dataset

```bash
# Export to CSV
mdm dataset export iris --format csv

# Export to Parquet with compression
mdm dataset export iris --format parquet --compression gzip
```

## Python API

```python
from mdm.api import MDMClient

# Initialize client
client = MDMClient()

# Register dataset
client.register_dataset(
    name="iris",
    path="data/iris.csv",
    target_column="species"
)

# List datasets
datasets = client.list_datasets()
for ds in datasets:
    print(f"{ds.name}: {ds.row_count} rows")

# Load dataset as DataFrame
df = client.load_dataset("iris")
print(df.head())
```

## Key Features

- **Auto-detection**: MDM automatically detects file formats, delimiters, and data types
- **Multi-backend**: Choose between SQLite (default), DuckDB, or PostgreSQL
- **Rich metadata**: Automatic statistics and quality metrics
- **Fast startup**: Optimized CLI loads in ~0.1 seconds

## Next Steps

- Read the [Project Overview](01_Project_Overview.md) for detailed information
- Learn about [Dataset Registration](04_Dataset_Registration.md) options
- Explore [Advanced Features](09_Advanced_Features.md)
- Check out the [tutorials](tutorials/) for step-by-step guides

## Getting Help

```bash
# View all commands
mdm --help

# Get help for specific command
mdm dataset register --help

# Check MDM version
mdm version
```

## Common Tasks

### Search for Datasets
```bash
mdm dataset search titanic
```

### Update Dataset Metadata
```bash
mdm dataset update iris --description "Classic ML dataset" --tags "classification,flowers"
```

### Remove Dataset
```bash
mdm dataset remove old_dataset --force
```

### Change Storage Backend
Edit `~/.mdm/mdm.yaml`:
```yaml
database:
  default_backend: duckdb  # Options: sqlite, duckdb, postgresql
```

That's it! You're ready to manage your ML datasets with MDM. 🚀