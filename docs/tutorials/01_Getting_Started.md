# Tutorial 1: Getting Started with MDM

## Introduction

Welcome to MDM (ML Data Manager)! This tutorial will guide you through the basics of using MDM to manage your machine learning datasets.

## Prerequisites

- Python 3.8 or higher
- Basic knowledge of pandas and machine learning concepts
- Command line familiarity

## Installation

```bash
# Install MDM
pip install mdm-refactor

# Verify installation
mdm version

# Create MDM directory structure
mdm init
```

## Your First Dataset

### 1. Prepare Your Data

Create a sample CSV file `iris.csv`:

```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
```

### 2. Register the Dataset

```bash
# Register dataset with MDM
mdm dataset register iris_dataset ./iris.csv \
    --target species \
    --problem-type classification
```

Output:
```
✓ Dataset 'iris_dataset' registered successfully
  Location: ~/.mdm/datasets/iris_dataset
  Rows: 5
  Columns: 5
  Target: species
  Problem Type: classification
```

### 3. View Dataset Information

```bash
# Get detailed information
mdm dataset info iris_dataset
```

Output:
```
Dataset: iris_dataset
├── Created: 2025-01-10 10:30:00
├── Size: 150 rows × 5 columns
├── Target: species
├── Problem Type: classification
└── Features:
    ├── Numeric: 4 (sepal_length, sepal_width, petal_length, petal_width)
    └── Categorical: 0
```

### 4. Explore Dataset Statistics

```bash
# View statistics
mdm dataset stats iris_dataset
```

## Working with Datasets

### List All Datasets

```bash
mdm dataset list
```

### Search Datasets

```bash
# Search by name
mdm dataset search iris

# Search by tag
mdm dataset list --tag classification
```

### Update Dataset Metadata

```bash
# Add description and tags
mdm dataset update iris_dataset \
    --description "Classic Iris flower dataset" \
    --tags "classification,flowers,tutorial"
```

### Export Dataset

```bash
# Export to different formats
mdm dataset export iris_dataset --format parquet
mdm dataset export iris_dataset --format json --output ./exports/
```

## Feature Engineering

### Generate Basic Features

```python
from mdm.adapters import get_dataset_manager, get_feature_generator
import pandas as pd

# Load dataset
manager = get_dataset_manager()
dataset_info = manager.get_dataset("iris_dataset")

# Load data
df = pd.read_csv("iris.csv")

# Generate features
generator = get_feature_generator()
df_with_features = generator.generate_features(
    df=df,
    column_types={
        "sepal_length": "numeric",
        "sepal_width": "numeric",
        "petal_length": "numeric",
        "petal_width": "numeric",
        "species": "categorical"
    }
)

print(f"Original columns: {len(df.columns)}")
print(f"After feature engineering: {len(df_with_features.columns)}")
```

### View Generated Features

```python
# List new features
new_features = [col for col in df_with_features.columns if col not in df.columns]
print("Generated features:")
for feature in new_features:
    print(f"  - {feature}")
```

## Configuration

### View Current Configuration

```bash
mdm config show
```

### Change Default Backend

```bash
# Switch to DuckDB for better analytics performance
mdm config set database.default_backend duckdb

# Or use environment variable
export MDM_DATABASE_DEFAULT_BACKEND=duckdb
```

### Performance Tuning

```bash
# Increase batch size for large datasets
mdm config set performance.batch_size 50000

# Enable parallel processing
mdm config set performance.max_workers 8
```

## Best Practices

### 1. Dataset Naming

Use descriptive, lowercase names with underscores:
- ✅ `customer_churn_2024`
- ✅ `sales_forecast_daily`
- ❌ `MyDataset`
- ❌ `test-data-1`

### 2. Problem Type Selection

Choose the appropriate problem type:
- `classification`: Predicting categories
- `regression`: Predicting continuous values
- `clustering`: Grouping similar items
- `time_series`: Time-based predictions

### 3. Data Organization

```
project/
├── data/
│   ├── raw/           # Original data files
│   ├── processed/     # Cleaned data
│   └── features/      # Feature-engineered data
├── notebooks/         # Jupyter notebooks
└── scripts/          # Processing scripts
```

### 4. Regular Backups

```bash
# Backup MDM configuration and metadata
mdm backup create ~/mdm_backups/backup_$(date +%Y%m%d)

# Restore from backup
mdm backup restore ~/mdm_backups/backup_20250110
```

## Next Steps

Congratulations! You've learned the basics of MDM. Here's what to explore next:

1. **[Tutorial 2: Advanced Dataset Management](./02_Advanced_Dataset_Management.md)**
   - Working with multiple data files
   - Time series datasets
   - Handling large datasets

2. **[Tutorial 3: Custom Feature Engineering](./03_Custom_Feature_Engineering.md)**
   - Creating custom transformers
   - Feature pipelines
   - Domain-specific features

3. **[Tutorial 4: Performance Optimization](./04_Performance_Optimization.md)**
   - Caching strategies
   - Batch processing
   - Query optimization

## Troubleshooting

### Dataset Registration Fails

```bash
# Check file path
ls -la iris.csv

# Validate CSV format
head -5 iris.csv

# Check permissions
mdm debug permissions
```

### Performance Issues

```bash
# Enable debug logging
export MDM_LOGGING_LEVEL=DEBUG

# Run with profiling
mdm dataset register large_dataset.csv --profile
```

### Getting Help

```bash
# Built-in help
mdm --help
mdm dataset --help

# System information
mdm info

# Check for updates
mdm version --check-updates
```

## Summary

In this tutorial, you learned how to:
- Install and configure MDM
- Register and manage datasets
- Generate basic features
- Configure MDM for your needs

Continue to the next tutorial to learn about advanced features!