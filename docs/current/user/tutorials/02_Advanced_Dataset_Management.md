# Tutorial 2: Advanced Dataset Management

## Introduction

This tutorial covers advanced dataset management features in MDM, including working with multiple files, time series data, and large datasets.

## Working with Multiple Data Files

### Kaggle-Style Datasets

Many datasets come as multiple files (train.csv, test.csv, etc.). MDM automatically detects and manages these:

```bash
# Dataset structure
titanic/
├── train.csv
├── test.csv
└── gender_submission.csv

# Register the dataset
mdm dataset register titanic ./titanic/ \
    --target survived \
    --problem-type classification
```

### Accessing Individual Files

```python
from mdm.adapters import get_dataset_manager

manager = get_dataset_manager()
dataset = manager.get_dataset("titanic")

# Access individual files
print("Data files:", dataset.data_files)
# Output: {'train': 'train.csv', 'test': 'test.csv', 'submission': 'gender_submission.csv'}

# Load specific file
train_df = manager.load_data_file("titanic", "train")
test_df = manager.load_data_file("titanic", "test")
```

### Merging Data Files

```python
import pandas as pd

# Merge multiple files
def merge_dataset_files(dataset_name):
    manager = get_dataset_manager()
    dataset = manager.get_dataset(dataset_name)
    
    # Load all data files
    dataframes = {}
    for file_key, file_path in dataset.data_files.items():
        df = manager.load_data_file(dataset_name, file_key)
        dataframes[file_key] = df
    
    # Custom merge logic
    if 'train' in dataframes and 'test' in dataframes:
        # Add source column
        dataframes['train']['source'] = 'train'
        dataframes['test']['source'] = 'test'
        
        # Combine
        combined = pd.concat([dataframes['train'], dataframes['test']], 
                           ignore_index=True)
        return combined
    
    return None

# Use the function
combined_df = merge_dataset_files("titanic")
print(f"Combined shape: {combined_df.shape}")
```

## Time Series Datasets

### Registering Time Series Data

```bash
# Register with time column
mdm dataset register stock_prices ./stocks.csv \
    --time-column date \
    --target close_price \
    --problem-type time_series
```

### Working with Time-Based Features

```python
from mdm.core.features import TimeSeriesTransformer

# Create time series features
ts_transformer = TimeSeriesTransformer(
    time_column='date',
    features=['lag', 'rolling_mean', 'rolling_std']
)

# Configure lag features
ts_transformer.add_lag_features(
    columns=['close_price', 'volume'],
    lags=[1, 7, 30]
)

# Configure rolling features
ts_transformer.add_rolling_features(
    columns=['close_price'],
    windows=[7, 14, 30],
    functions=['mean', 'std', 'min', 'max']
)

# Apply transformations
df_with_ts_features = ts_transformer.transform(df)
```

### Time-Based Validation Splits

```python
from mdm.utils import TimeSeriesSplitter

# Create time-based splits
splitter = TimeSeriesSplitter(
    time_column='date',
    n_splits=5,
    test_size=0.2
)

# Generate splits
for train_idx, test_idx in splitter.split(df):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    print(f"Train: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test: {test_df['date'].min()} to {test_df['date'].max()}")
```

## Handling Large Datasets

### Chunked Processing

```python
from mdm.adapters import get_dataset_registrar

# Register large dataset with chunking
registrar = get_dataset_registrar()
registrar.register_dataset(
    name="large_dataset",
    path="./very_large_file.csv",
    chunk_size=50000,  # Process 50k rows at a time
    dtype_optimization=True  # Optimize column types for memory
)
```

### Memory-Efficient Loading

```python
from mdm.utils import DataLoader

# Create efficient loader
loader = DataLoader(
    backend="duckdb",  # Use DuckDB for better memory management
    cache_enabled=True
)

# Load subset of data
subset = loader.load_subset(
    dataset_name="large_dataset",
    columns=['id', 'feature1', 'target'],
    sample_size=10000,
    random_state=42
)

# Load with filters
filtered = loader.load_filtered(
    dataset_name="large_dataset",
    filters={
        'date': {'>=': '2024-01-01', '<=': '2024-12-31'},
        'category': {'in': ['A', 'B', 'C']}
    }
)
```

### Batch Processing

```python
from mdm.performance import BatchProcessor

# Process in batches
processor = BatchProcessor(batch_size=10000)

def process_batch(batch_df):
    # Your processing logic
    batch_df['new_feature'] = batch_df['feature1'] * 2
    return batch_df

# Process entire dataset
results = processor.process_dataset(
    dataset_name="large_dataset",
    process_func=process_batch,
    output_path="./processed_data.parquet"
)

print(f"Processed {results['total_rows']} rows in {results['duration']:.2f}s")
```

## Advanced Search and Filtering

### Complex Dataset Searches

```python
from mdm.adapters import get_dataset_manager

manager = get_dataset_manager()

# Search with multiple criteria
results = manager.search_datasets(
    query="customer",
    filters={
        'problem_type': 'classification',
        'min_rows': 1000,
        'created_after': '2024-01-01'
    }
)

# Search in metadata
results = manager.search_in_metadata(
    query="churn",
    search_in=['description', 'tags', 'columns']
)
```

### Dataset Versioning

```python
# Create dataset version
manager.create_version(
    dataset_name="customer_churn",
    version_name="v2_balanced",
    description="Balanced dataset with SMOTE"
)

# List versions
versions = manager.list_versions("customer_churn")
for version in versions:
    print(f"{version.name}: {version.description}")

# Load specific version
df = manager.load_version("customer_churn", "v2_balanced")
```

## Data Quality Management

### Automated Quality Checks

```python
from mdm.quality import DataQualityChecker

# Create quality checker
checker = DataQualityChecker()

# Run comprehensive checks
quality_report = checker.check_dataset("customer_data")

# Review issues
if quality_report.has_issues():
    print("Data quality issues found:")
    for issue in quality_report.issues:
        print(f"  - {issue.severity}: {issue.description}")
        print(f"    Affected columns: {issue.columns}")
        print(f"    Rows affected: {issue.row_count}")
```

### Data Profiling

```python
from mdm.profiling import DataProfiler

# Create detailed profile
profiler = DataProfiler()
profile = profiler.profile_dataset("customer_data")

# Generate HTML report
profile.to_html("customer_data_profile.html")

# Get specific insights
print("Missing values:", profile.missing_values)
print("Cardinality:", profile.cardinality)
print("Correlations:", profile.correlations)
```

## Dataset Relationships

### Linking Related Datasets

```python
# Define relationships
manager.add_relationship(
    parent_dataset="customers",
    child_dataset="transactions",
    join_keys={
        'parent': 'customer_id',
        'child': 'customer_id'
    },
    relationship_type='one_to_many'
)

# Query related data
related_data = manager.get_related_data(
    dataset="customers",
    include=['transactions'],
    filters={'customer_status': 'active'}
)
```

### Dataset Lineage

```python
from mdm.lineage import LineageTracker

# Track dataset transformations
tracker = LineageTracker()

# Record transformation
tracker.record_transformation(
    source_datasets=["raw_sales", "raw_customers"],
    target_dataset="sales_analysis",
    transformation_type="join_and_aggregate",
    transformation_code="scripts/create_sales_analysis.py"
)

# View lineage
lineage = tracker.get_lineage("sales_analysis")
lineage.visualize("lineage_graph.png")
```

## Performance Optimization for Large Datasets

### Indexing

```python
# Create indexes for faster queries
manager.create_index(
    dataset_name="large_transactions",
    columns=['date', 'customer_id'],
    index_type='btree'
)

# Verify index usage
query_plan = manager.explain_query(
    dataset_name="large_transactions",
    query="SELECT * FROM data WHERE date >= '2024-01-01'"
)
print("Uses index:", query_plan.uses_index)
```

### Partitioning

```python
# Partition by date for time series
manager.partition_dataset(
    dataset_name="sensor_data",
    partition_column="date",
    partition_type="monthly"
)

# Query specific partition
partition_data = manager.query_partition(
    dataset_name="sensor_data",
    partition="2024-03"
)
```

## Advanced CLI Usage

### Batch Operations

```bash
# Register multiple datasets
for file in data/*.csv; do
    name=$(basename "$file" .csv)
    mdm dataset register "$name" "$file" --auto-detect
done

# Export all datasets
mdm dataset export-all --format parquet --output ./exports/

# Update multiple datasets
mdm dataset update-batch --tag machine-learning --add-tag production
```

### Custom Scripts

```bash
# Create custom MDM script
cat > mdm_weekly_update.sh << 'EOF'
#!/bin/bash
# Weekly dataset update script

echo "Starting weekly MDM update..."

# Update all time series datasets
mdm dataset list --tag time_series | while read dataset; do
    echo "Updating $dataset..."
    mdm dataset refresh "$dataset"
done

# Generate quality reports
mdm quality check-all --output ./quality_reports/

# Backup metadata
mdm backup create ./backups/weekly_$(date +%Y%m%d)

echo "Weekly update complete!"
EOF

chmod +x mdm_weekly_update.sh
```

## Integration with ML Workflows

### Scikit-learn Integration

```python
from mdm.integrations import SKLearnHelper

# Create helper
helper = SKLearnHelper()

# Load dataset ready for sklearn
X, y = helper.load_for_training("titanic")

# Automatic train/test split with stratification
X_train, X_test, y_train, y_test = helper.train_test_split(
    dataset_name="titanic",
    test_size=0.2,
    stratify=True
)

# Get preprocessing pipeline
preprocessor = helper.get_preprocessor("titanic")
```

### PyTorch Integration

```python
from mdm.integrations import PyTorchDataset

# Create PyTorch dataset
dataset = PyTorchDataset(
    dataset_name="image_classification",
    transform=transforms.ToTensor()
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

## Summary

In this tutorial, you learned:
- Managing multi-file datasets
- Working with time series data
- Handling large datasets efficiently
- Advanced search and filtering
- Data quality management
- Performance optimization techniques
- Integration with ML frameworks

Next: [Tutorial 3: Custom Feature Engineering](./03_Custom_Feature_Engineering.md)