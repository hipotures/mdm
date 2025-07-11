# MDM Frequently Asked Questions (FAQ)

## General Questions

### What is MDM?
MDM (ML Data Manager) is a standalone, enterprise-grade dataset management system designed specifically for machine learning workflows. It provides a unified solution for registering, storing, and managing datasets.

### Why should I use MDM instead of just folders?
- **Rich metadata**: Automatic statistics, quality metrics, and profiling
- **Fast search**: Find datasets by name, tags, or content
- **Version tracking**: Know when datasets were updated
- **Multi-format support**: Work with CSV, Parquet, JSON seamlessly
- **Reproducibility**: Consistent dataset handling across projects

### What backends does MDM support?
MDM supports three backends via SQLAlchemy:
- **SQLite** (default) - Best for single-user, local datasets
- **DuckDB** - Best for analytical workloads and larger datasets
- **PostgreSQL** - Best for multi-user, enterprise environments

## Installation & Setup

### How do I install MDM?
```bash
pip install mdm
```

### Where does MDM store data?
By default, MDM stores all data in `~/.mdm/`:
- `~/.mdm/datasets/` - Dataset databases
- `~/.mdm/config/datasets/` - Dataset configurations
- `~/.mdm/mdm.yaml` - Main configuration

### Can I change the storage location?
Yes, set the `MDM_HOME_DIR` environment variable:
```bash
export MDM_HOME_DIR=/custom/path
```

## Dataset Registration

### How do I register a dataset?
```bash
# Simple CSV
mdm dataset register mydata data.csv

# With target column
mdm dataset register mydata data.csv --target label

# Directory with multiple files
mdm dataset register kaggle_comp ./competition_data/
```

### What file formats are supported?
- CSV (.csv, .csv.gz)
- Parquet (.parquet)
- JSON (.json)
- Excel (.xlsx) - basic support
- Compressed CSV (.csv.gz)

### How does auto-detection work?
MDM automatically detects:
- File encoding (UTF-8, ISO-8859-1, etc.)
- CSV delimiters (comma, semicolon, tab, pipe)
- Compression (gzip)
- Kaggle competition structure (train/test/submission files)
- ID columns and target columns
- Data types using ydata-profiling

### Can I skip feature generation for faster registration?
Yes, use the `--no-features` flag:
```bash
mdm dataset register large_dataset data.csv --no-features
```

## Backend & Performance

### How do I change the backend?
Edit `~/.mdm/mdm.yaml`:
```yaml
database:
  default_backend: duckdb  # or sqlite, postgresql
```
**Note**: You must set this BEFORE registering datasets. Changing backends makes datasets from other backends invisible.

### Why is registration slow?
Registration runs ydata-profiling for type detection, which can be slow for large datasets. Solutions:
- Use `--no-features` to skip feature generation
- Adjust batch size in configuration
- Use DuckDB backend for better performance

### How do I improve performance?
- Set appropriate batch size: `MDM_PERFORMANCE_BATCH_SIZE=50000`
- Use DuckDB for analytical queries
- Enable parallel processing in config
- Use `--no-features` for large datasets

## Common Issues

### "Dataset already exists" error
Use the `--force` flag to overwrite:
```bash
mdm dataset register mydata data.csv --force
```

### "No datasets found" after changing backend
This is expected behavior. MDM only shows datasets matching the current backend. To see SQLite datasets, set `default_backend: sqlite`.

### Command takes too long to start
This was fixed in v0.2.0. The CLI now starts in ~0.1 seconds. Update to the latest version:
```bash
pip install --upgrade mdm
```

### Cannot find my datasets
Check:
1. Current backend matches dataset backend
2. Correct MDM_HOME_DIR if using custom location
3. Dataset YAML files exist in `~/.mdm/config/datasets/`

## Advanced Usage

### How do I use MDM in Python?
```python
from mdm.api import MDMClient

client = MDMClient()
df = client.load_dataset("mydata")
```

### Can I use custom feature engineering?
Yes, place custom transformers in `~/.mdm/config/custom_features/`. (Note: This feature has known issues in current version)

### How do I export datasets?
```bash
# To CSV
mdm dataset export mydata --format csv

# To Parquet with compression
mdm dataset export mydata --format parquet --compression gzip

# To specific location
mdm dataset export mydata --output ./exports/
```

### Can I update dataset metadata?
```bash
mdm dataset update mydata \
  --description "Updated description" \
  --tags "tag1,tag2" \
  --problem-type regression
```

## Migration & Compatibility

### How do I migrate from an older version?
MDM uses feature flags for gradual migration. See the Migration Guide for details.

### Is MDM backwards compatible?
The v0.2.0 refactoring maintains compatibility while adding new features. Existing datasets continue to work.

### Can I use multiple backends simultaneously?
No, MDM uses a single-backend architecture. All datasets must use the same backend type at any given time.

## Troubleshooting

### How do I enable debug logging?
```bash
export MDM_LOGGING_LEVEL=DEBUG
mdm dataset list
```

### Where are the logs?
By default, logs are only shown in console. To enable file logging:
```bash
export MDM_LOGGING_FILE=/path/to/mdm.log
```

### How do I report bugs?
1. Check existing issues on GitHub
2. Enable debug logging and capture output
3. Create a minimal reproducible example
4. Submit issue with details

## Best Practices

### Dataset Naming
- Use lowercase letters, numbers, and underscores
- Avoid spaces and special characters
- Use descriptive names: `customer_churn_2024` not `data1`

### Organization
- Use consistent naming conventions
- Tag datasets for easy discovery
- Document datasets with descriptions
- Regular cleanup of unused datasets

### Performance
- Use appropriate backend for your use case
- Adjust batch size for your system
- Use `--no-features` for initial exploration
- Monitor disk space in `~/.mdm/`

## Getting Help

- **Documentation**: Start with `00_Table_of_Contents.md`
- **CLI Help**: `mdm --help` or `mdm dataset register --help`
- **Issues**: GitHub issue tracker
- **Examples**: See tutorials in `docs/current/user/tutorials/`