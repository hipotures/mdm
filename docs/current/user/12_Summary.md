# Summary

## What is MDM?

MDM (ML Data Manager) is a standalone, enterprise-grade dataset management system designed specifically for machine learning workflows. It provides a unified interface for managing, versioning, and accessing datasets across different storage backends.

## Key Features

### 1. **Multi-Backend Support**
- **DuckDB** (default): Best for analytical queries and medium to large datasets
- **SQLite**: Lightweight option for small datasets and maximum portability
- **PostgreSQL**: Enterprise option with multi-user support and advanced features

### 2. **Intelligent Dataset Registration**
- Auto-detection of data structure (train/test/validation splits)
- Automatic file format detection (CSV, Parquet, JSON, Excel)
- Kaggle competition structure recognition
- Target column and problem type inference

### 3. **Advanced Feature Engineering**
- Two-tier system: generic transformers + custom features
- Automatic feature generation for common patterns
- Signal detection to filter low-quality features
- Support for custom dataset-specific transformers

### 4. **Comprehensive CLI**
- Dataset management: register, list, info, search, update, remove
- Export/import with multiple format support
- Batch operations for multiple datasets
- Time series specific commands
- Rich terminal output with progress indicators

### 5. **Programmatic API**
- High-level `MDMClient` for common operations
- Low-level `DatasetService` for advanced use cases
- ML framework integration (sklearn, PyTorch, TensorFlow)
- Efficient chunk processing for large datasets

### 6. **Performance Optimization**
- Configurable batch processing
- Memory-efficient operations
- Query optimization
- Parallel processing support

### 7. **Time Series Support**
- Time-based train/test/validation splits
- Cross-validation fold generation
- Frequency and seasonality detection
- Trend analysis

## Quick Start

### Installation
```bash
pip install mdm
```

### Basic Workflow
```bash
# 1. Register a dataset
mdm dataset register titanic /path/to/kaggle/titanic

# 2. Get dataset information
mdm dataset info titanic

# 3. Search datasets
mdm dataset search kaggle

# 4. Export dataset
mdm dataset export titanic --format parquet

# 5. Use in Python
from mdm import load_dataset
train_df, test_df = load_dataset("titanic")
```

## Architecture Overview

```
MDM System Architecture
├── Configuration Layer
│   ├── YAML-based configuration
│   ├── Environment variable overrides
│   └── Sensible defaults
├── Storage Layer
│   ├── Backend abstraction (DuckDB/SQLite/PostgreSQL)
│   ├── Efficient data loading
│   └── Query optimization
├── Dataset Management
│   ├── Registration with auto-detection
│   ├── Metadata tracking
│   └── Version management
├── Feature Engineering
│   ├── Generic transformers
│   ├── Custom transformers
│   └── Signal detection
├── API Layer
│   ├── CLI (Typer-based)
│   ├── High-level Python API
│   └── Low-level service API
└── Utilities
    ├── Performance monitoring
    ├── Time series operations
    └── ML framework integration
```

## Design Principles

1. **Immutability**: Datasets are immutable once registered
2. **Discoverability**: Easy search and metadata access
3. **Flexibility**: Multiple backends and formats supported
4. **Performance**: Optimized for ML workflows
5. **Simplicity**: Intuitive CLI and API
6. **Extensibility**: Custom transformers and backends

## Common Use Cases

### 1. Kaggle Competition Workflow
```bash
# Download and register
kaggle competitions download -c house-prices
mdm dataset register house_prices ./house-prices-advanced-regression-techniques

# Explore
mdm dataset info house_prices --details

# Work with data
from mdm import MDMClient
client = MDMClient()
train_df, test_df = client.load_dataset_files("house_prices")
```

### 2. Time Series Analysis
```bash
# Register time series data
mdm dataset register sales_data /data/sales \
    --time-column date \
    --target revenue \
    --problem-type time_series

# Analyze patterns
mdm timeseries analyze sales_data

# Create CV folds
mdm timeseries validate sales_data --folds 5 --gap 7
```

### 3. Feature Engineering Pipeline
```python
# Create custom features
# ~/.mdm/custom_features/titanic.py
from mdm.features.custom.base import BaseDomainFeatures

class CustomFeatureOperations(BaseDomainFeatures):
    def get_domain_features(self, df):
        features = {}
        features['family_size'] = df['SibSp'] + df['Parch'] + 1
        features['is_alone'] = (features['family_size'] == 1).astype(int)
        return features

# Regenerate with custom features
mdm dataset register titanic /path/to/data --force
```

### 4. Large Dataset Processing
```python
from mdm import MDMClient

client = MDMClient()

# Process in chunks
def process_chunk(chunk_df):
    # Your processing logic
    return chunk_df.describe()

results = client.process_in_chunks(
    "large_dataset",
    process_func=process_chunk,
    chunk_size=50000
)
```

## Best Practices Summary

1. **Use meaningful dataset names** with versioning
2. **Keep datasets immutable** - create new versions instead of modifying
3. **Choose the right backend** based on dataset size and access patterns
4. **Use Parquet format** for better performance and type preservation
5. **Leverage batch operations** for multiple datasets
6. **Monitor performance** with built-in tools
7. **Write custom transformers** for domain-specific features
8. **Use chunk processing** for memory efficiency

## Troubleshooting Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| Dataset already exists | Use `--force` flag |
| Out of memory | Reduce batch size or switch backend |
| Slow queries | Create indexes or optimize query |
| Feature generation fails | Check logs or skip with `--no-features` |
| Command not found | Check installation and PATH |

## What's Next?

1. **Explore the CLI**: Run `mdm --help` to see all commands
2. **Read the API docs**: Check the programmatic API guide
3. **Try the examples**: Use sample data in `data/sample/`
4. **Write custom features**: Create domain-specific transformers
5. **Run the tests**: Verify your setup with test scripts

## Resources

- **Documentation**: Full documentation in `docs/`
- **Test Scripts**: End-to-end tests in `scripts/`
- **Examples**: Sample data in `data/sample/`
- **Configuration**: Example configs in `config/`
- **Tests**: Unit and integration tests in `tests/`

## Key Takeaways

1. MDM makes ML dataset management **simple and consistent**
2. **Case-insensitive** dataset names for convenience
3. **Auto-detection** reduces manual configuration
4. **Feature engineering** is built-in and extensible
5. **Multiple backends** provide flexibility
6. **Performance optimized** for ML workflows
7. **Comprehensive testing** ensures reliability

MDM is designed to be the foundation of your ML data pipeline, providing reliable, efficient, and scalable dataset management for projects of any size.