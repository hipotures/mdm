# Frequently Asked Questions

## General Questions

### What is MDM?

MDM (ML Data Manager) is a standalone dataset management system designed specifically for machine learning workflows. It provides:
- Centralized dataset storage with database backends
- Automatic feature engineering
- Simple CLI and Python API
- Integration with ML frameworks

### How is MDM different from DVC or MLflow?

| Feature | MDM | DVC | MLflow |
|---------|-----|-----|--------|
| **Focus** | Dataset management | Version control | Experiment tracking |
| **Storage** | Database-backed | File-based | Metadata only |
| **Features** | Auto-generation | No | No |
| **SQL Access** | Yes | No | Limited |
| **Local-first** | Yes | Yes | No |

MDM focuses on making datasets "ML-ready" with features and easy access, while DVC focuses on versioning and MLflow on experiment tracking.

### Can I use MDM in production?

Yes! MDM v1.0.0 is production-ready. For production use:
- Use PostgreSQL backend for multi-user access
- Configure appropriate resource limits
- Set up regular backups
- Use environment variables for sensitive config

### What size datasets can MDM handle?

MDM is optimized for datasets from megabytes to gigabytes:
- **Small datasets** (<100MB): Any backend works well
- **Medium datasets** (100MB-10GB): DuckDB or PostgreSQL recommended
- **Large datasets** (10GB-100GB): PostgreSQL with proper indexing
- **Very large datasets** (>100GB): Consider partitioning or use MDM for samples

## Installation and Setup

### Which Python versions are supported?

MDM requires Python 3.9 or higher. Recommended versions:
- Python 3.11 (best performance)
- Python 3.10 (good balance)
- Python 3.9 (minimum supported)

### Do I need to install database servers?

No for SQLite and DuckDB - they're embedded. Yes for PostgreSQL:
- **SQLite**: Included with Python
- **DuckDB**: Automatically installed with MDM
- **PostgreSQL**: Requires separate PostgreSQL server

### How do I upgrade MDM?

```bash
# For pip installation
pip install --upgrade mdm

# For development installation
cd /path/to/mdm
git pull
pip install -e ".[dev]"
```

Always check the [CHANGELOG](https://github.com/hipotures/mdm/blob/main/CHANGELOG.md) for breaking changes.

### Can I use MDM on Windows?

Yes, MDM supports Windows 10+. For best experience:
- Use WSL2 (Windows Subsystem for Linux)
- Or use Windows Terminal with UTF-8 encoding
- Install Visual C++ Build Tools for some dependencies

## Dataset Management

### How do I handle multiple CSV files?

MDM automatically detects multiple files:
```bash
# Directory structure:
data/
├── train.csv
├── test.csv
└── sample_submission.csv

# Register all at once
mdm dataset register kaggle_comp ./data
```

### Can I update data after registration?

Currently, you need to re-register:
```bash
# Update the source file, then:
mdm dataset register mydata updated_data.csv --force
```

Future versions will support incremental updates.

### How does MDM detect Kaggle datasets?

MDM looks for these patterns:
- Files named `train.csv` and `test.csv`
- A `sample_submission.csv` file
- Common Kaggle directory structures

If detected, MDM automatically:
- Identifies the target column from sample submission
- Keeps train/test files separate
- Preserves submission format

### Can I register compressed files?

Yes, MDM supports compressed CSV files:
```bash
# Gzipped files
mdm dataset register compressed data.csv.gz

# Will support in future: .zip, .tar.gz, .bz2
```

### How do I handle large datasets?

For large datasets:
```bash
# 1. Increase batch size for faster loading
export MDM_PERFORMANCE_BATCH_SIZE=50000

# 2. Skip features initially
mdm dataset register large_data big_file.csv --no-features

# 3. Use DuckDB for better compression
export MDM_DATABASE_DEFAULT_BACKEND=duckdb

# 4. Process in chunks in your code
for chunk in client.iterate_dataset("large_data", chunk_size=100000):
    process(chunk)
```

## Feature Engineering

### How do I disable feature generation?

Three ways:
```bash
# 1. During registration
mdm dataset register mydata data.csv --no-features

# 2. In configuration
features:
  enable_at_registration: false

# 3. Via environment
export MDM_FEATURES_ENABLE_AT_REGISTRATION=false
```

### Can I use my own features?

Yes! Create `~/.mdm/config/custom_features/{dataset_name}.py`:
```python
from mdm.features.custom.base import BaseDomainFeatures

class CustomFeatureOperations(BaseDomainFeatures):
    def __init__(self):
        super().__init__('dataset_name')
    
    def calculate_features(self, df):
        return {
            'custom_feature': df['col1'] * df['col2']
        }
```

### Which features are generated automatically?

MDM generates features based on column types:
- **Numeric**: statistics, log transform, outliers, binning
- **Datetime**: year, month, day, weekday, is_weekend
- **Categorical**: one-hot encoding, frequency encoding
- **Text**: length, word count, special characters

### How do I see what features were generated?

```python
# In Python
info = client.get_dataset_info("mydata")
print("Original columns:", info.columns["original"])
print("Generated features:", info.columns["features"])

# Or check the logs during registration
export MDM_LOGGING_LEVEL=INFO
mdm dataset register mydata data.csv
```

## Storage and Backends

### Which backend should I use?

Quick decision guide:
- **SQLite**: Development, small datasets, single user
- **DuckDB**: Analytics, medium-large datasets, complex queries
- **PostgreSQL**: Production, multi-user, need ACID compliance

### Can I change backends after registration?

No, datasets are tied to their backend. To migrate:
```bash
# 1. Export from old backend
mdm dataset export mydata --format parquet

# 2. Change backend
export MDM_DATABASE_DEFAULT_BACKEND=duckdb

# 3. Re-register
mdm dataset register mydata mydata_export.parquet
```

### How much disk space do I need?

Rule of thumb:
- **SQLite**: ~1.5x original CSV size
- **DuckDB**: ~0.3-0.5x original CSV size (compressed)
- **PostgreSQL**: ~2x original CSV size (with indexes)

Plus space for features (typically 2-3x original column count).

### Can multiple users access the same dataset?

Depends on backend:
- **SQLite**: Read-only for multiple users
- **DuckDB**: Single writer, multiple readers
- **PostgreSQL**: Full multi-user support

## Performance

### Why is registration slow?

Common causes and solutions:
1. **Large file**: Use larger batch size
2. **Many features**: Disable some feature types
3. **Complex features**: Skip custom features
4. **Type detection**: Force column types

### How can I speed up queries?

```python
# 1. Create indexes
with client.get_connection("dataset") as conn:
    conn.execute("CREATE INDEX idx_col ON train(column)")

# 2. Use database filtering
# Don't do: df = load_all(); df[df.col == 'A']
# Do: client.query.query(dataset, filters={'col': 'A'})

# 3. Cache results
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_stats(dataset_name, category):
    return client.query.aggregate(...)
```

### What's the optimal batch size?

Depends on available memory:
```python
import psutil

# Simple formula
available_gb = psutil.virtual_memory().available / 1e9
batch_size = int(available_gb * 1e6 / 100)  # ~100 bytes per row
batch_size = max(1000, min(batch_size, 100000))  # Clamp

print(f"Recommended batch size: {batch_size}")
```

## Troubleshooting

### How do I debug issues?

Enable debug mode:
```bash
export MDM_LOGGING_LEVEL=DEBUG
export MDM_CLI_SHOW_TRACEBACK=true
mdm dataset register problem_data.csv
```

Check logs at `~/.mdm/logs/mdm.log`

### What does "Backend not available" mean?

The backend isn't installed or configured:
```bash
# Install backend
pip install duckdb  # or psycopg2-binary

# Check it's available
python -c "import duckdb; print('OK')"
```

### Why are my datetime columns stored as TEXT?

This is a known limitation. Workaround:
```python
# Force datetime during registration
mdm dataset register mydata data.csv \
    --datetime-columns "date,created_at"

# Or convert when loading
df = client.load_dataset("mydata")
df['date'] = pd.to_datetime(df['date'])
```

## API and Integration

### Can I use MDM without the CLI?

Yes, MDM has a full Python API:
```python
from mdm import MDMClient

client = MDMClient()
# All CLI operations available programmatically
```

### Does MDM integrate with Jupyter?

Yes! MDM works great in notebooks:
```python
# In Jupyter
from mdm import MDMClient
client = MDMClient()

# Load with nice display
df = client.load_dataset("titanic")
df.head()  # Rendered as nice HTML table
```

### Can I use MDM with cloud storage?

Not directly, but you can:
```python
# Download from cloud
import boto3
s3 = boto3.client('s3')
s3.download_file('bucket', 'data.csv', '/tmp/data.csv')

# Register local copy
client.register_dataset("cloud_data", "/tmp/data.csv")
```

### How do I export for Kaggle submission?

```python
# Make predictions
predictions = model.predict(test_data)

# Create submission
client.ml.create_submission(
    dataset="competition",
    predictions=predictions,
    output_path="submission.csv"
)
```

## Best Practices

### Should I commit MDM data to git?

No! Add to `.gitignore`:
```
# .gitignore
~/.mdm/
*.db
*.duckdb
```

Instead, commit:
- Your custom feature definitions
- Configuration files (without passwords)
- Scripts for dataset registration

### How do I handle sensitive data?

1. Use PostgreSQL with proper access control
2. Encrypt data at rest
3. Don't commit sensitive config:
```yaml
postgresql:
  password: ${MDM_PG_PASSWORD}  # From environment
```

### What's the recommended workflow?

1. **Development**: SQLite, small samples
2. **Analysis**: DuckDB, full datasets
3. **Production**: PostgreSQL, access control
4. **Deployment**: Export models with metadata

### How do I version datasets?

Simple approach using naming:
```bash
mdm dataset register sales_v1 sales_jan.csv
mdm dataset register sales_v2 sales_jan_feb.csv
mdm dataset register sales_v3 sales_q1.csv
```

Future versions will support native versioning.

## Common Errors

### "Multiple values for keyword argument"

This is a known CLI bug. Use Python API instead:
```python
# Instead of CLI with comma-separated values
client.register_dataset(
    "mydata",
    "data.csv", 
    id_columns=["id1", "id2"]  # List in Python
)
```

### "Dataset X not visible"

You changed backends. Datasets are backend-specific:
```bash
# Check current backend
mdm info

# List shows only current backend datasets
mdm dataset list

# To see all datasets, check config files
ls ~/.mdm/config/datasets/
```

### "MemoryError during registration"

Dataset too large for memory:
1. Reduce batch size
2. Use DuckDB (better memory management)
3. Skip features initially
4. Process in chunks

## Future Features

### What's planned for future versions?

Based on roadmap:
- Native dataset versioning
- Incremental updates
- Cloud storage backends
- More ML framework integrations
- Web UI for dataset exploration
- Distributed processing support

### Can I request features?

Yes! Open an issue on [GitHub](https://github.com/hipotures/mdm/issues) with:
- Use case description
- Current workaround (if any)
- Proposed solution

### How can I contribute?

See [CONTRIBUTING.md](https://github.com/hipotures/mdm/blob/main/CONTRIBUTING.md). We welcome:
- Bug reports with reproducible examples
- Documentation improvements
- New feature transformers
- Backend implementations
- Integration examples