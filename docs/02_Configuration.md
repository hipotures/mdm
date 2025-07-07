# Configuration

MDM uses a hierarchical configuration system with YAML files to provide flexible and organized settings management.

## Configuration Structure

The configuration system consists of:
1. **Main configuration file** (`mdm.yaml`) - System-wide settings
2. **Dataset configuration files** - Individual dataset settings
3. **Environment variables** - Runtime overrides

## MDM Home Directory Structure

The `~/.mdm/` directory contains all MDM data and configuration:

```
~/.mdm/
├── mdm.yaml                    # Main system configuration
├── config/
│   └── datasets/              # Dataset configuration files
│       ├── titanic.yaml
│       ├── house_prices.yaml
│       └── ...
├── datasets/                  # Self-contained dataset directories
│   ├── titanic/
│   │   └── dataset.sqlite     # Default SQLite database (via SQLAlchemy)
│   ├── house_prices/
│   │   └── dataset.duckdb     # DuckDB backend (via SQLAlchemy)
│   └── ...
├── config/
│   ├── datasets/              # Dataset configuration files
│   │   ├── titanic.yaml
│   │   └── house_prices.yaml
│   └── custom_features/       # Optional custom feature transformers
│       ├── titanic.py        # Dataset-specific features for titanic
│       └── house_prices.py   # Dataset-specific features for house_prices
├── logs/                      # Application log files
│   ├── mdm.log               # Main application log
│   └── mdm.log.{date}        # Rotated logs
└── tmp/                      # Temporary files during operations
```

## Main Configuration (mdm.yaml)

The main configuration file `~/.mdm/mdm.yaml` defines system-wide settings. MDM automatically loads this configuration on startup and uses it to determine settings like the default database backend, storage paths, and logging configuration.

```yaml
# Storage paths (relative to ~/.mdm/ or absolute)
storage:
  datasets_path: datasets/            # Where dataset databases are stored
  configs_path: config/datasets/      # Where dataset configs are stored
  logs_path: logs/                    # Log files

# Database settings
database:
  default_backend: sqlite             # Default: sqlite (recommended), duckdb, postgresql
                                     # IMPORTANT: All datasets must use the same backend
                                     # Changing this hides datasets with different backends
  
  # SQLAlchemy settings (common for all backends)
  sqlalchemy:
    echo: false                       # Log SQL queries
    pool_size: 5                      # Connection pool size
    max_overflow: 10                  # Maximum overflow connections
    pool_timeout: 30                  # Pool timeout in seconds
  
  # SQLite settings (default)
  sqlite:
    journal_mode: "WAL"               # Write-Ahead Logging for better concurrency
    synchronous: "NORMAL"             # Balance between safety and speed
    
  # DuckDB settings
  duckdb:
    memory_limit: "8GB"
    threads: 4
    
  # PostgreSQL settings
  postgresql:
    host: localhost
    port: 5432
    user: mdm_user
    password: mdm_pass
    database_prefix: mdm_             # Prefix for dataset databases

# Performance settings
performance:
  batch_size: 10000                   # Rows per batch for bulk operations
  max_concurrent_operations: 5        # Max concurrent async operations

# Logging
logging:
  level: INFO      # File logging level
  file: mdm.log    # Log file name (in logs_path directory)

# Feature Engineering (optional)
# See 09_Advanced_Features.md for complete configuration options
feature_engineering:
  enabled: true    # Enable automatic feature generation
  # Additional options documented in Advanced Features
```

**Console Logging Behavior**:
- Console output shows only WARNING and ERROR messages by default
- INFO level messages are written to log files but hidden from console
- Use `--verbose` flag or set `MDM_LOG_LEVEL=DEBUG` to see detailed output
- Log files are stored in `~/.mdm/logs/`

### System Paths Configuration

The `paths` section defines where MDM stores different types of files:

- **datasets_path**: Root directory for dataset database files (SQLite/DuckDB files)
- **configs_path**: Directory for dataset configuration YAML files
- **logs_path**: Directory for application log files
- **custom_features_path**: Directory for custom feature transformer Python files

Paths can be:
- Relative (resolved from `~/.mdm/`)
- Absolute (use full system paths)

**Note**: Despite the historical name 'storage' in some examples, this section configures system paths, not database storage backends.

### Database Configuration

Choose and configure your database backend:

- **default_backend**: Sets the backend for ALL datasets (sqlite, duckdb, postgresql)
- **Single Backend Architecture**: MDM uses one backend type for all datasets at a time
- **Backend Switching**: Changing this setting makes datasets with different backends invisible
- **Backend-specific settings**: Configure memory, threads, connections per backend

**Important**: Backend selection is a critical concept in MDM. For the complete, authoritative explanation, see [Backend Selection and Configuration](03_Database_Architecture.md#backend-selection-and-configuration-authoritative).

### Optional Configuration Sections

MDM supports additional optional configuration sections for advanced features:

#### Feature Engineering Configuration

When enabled, MDM can automatically generate features during dataset registration:

```yaml
feature_engineering:
  enabled: true
  # Full configuration options in 09_Advanced_Features.md
```

For complete feature engineering documentation, including:
- Generic and custom transformers
- Signal detection parameters
- Custom feature development

See [Advanced Features - Feature Engineering System](09_Advanced_Features.md#1-feature-engineering-system)

#### Export Settings

Configure default export behavior for datasets:

```yaml
export:
  default_format: csv       # Default: csv (options: csv, parquet, json)
  compression: zip          # Default: zip (options: none, zip, gzip, snappy)
  include_metadata: true    # Default: true - Include metadata in exports
```

**Export Configuration Details**:
- **default_format**: Default file format when exporting datasets
  - `csv`: Comma-separated values (most compatible)
  - `parquet`: Apache Parquet (efficient for large datasets)
  - `json`: JSON format (good for nested data)
- **compression**: Compression method for exported files
  - `none`: No compression
  - `zip`: ZIP compression (default, good compatibility)
  - `gzip`: GZIP compression (good for streaming)
  - `snappy`: Snappy compression (fast, requires parquet format)
- **include_metadata**: Whether to export dataset metadata alongside data

These defaults can be overridden per export operation using CLI flags.

### Logging Configuration

- **level**: File logging verbosity (DEBUG, INFO, WARNING, ERROR)
- **file**: Log file name (relative to logs_path)

**Important Logging Details**:
- File logging captures all messages at the configured level and above
- Console output is filtered separately - only WARNING and ERROR shown by default
- Operational messages (dataset not found, already exists) are logged as INFO, not ERROR
- Use environment variables to override: `export MDM_LOG_LEVEL=DEBUG`

### Performance Configuration

The performance section controls how MDM handles large-scale operations:

#### Batch Size

The `batch_size` parameter controls how many rows are processed at once during bulk operations:

```yaml
performance:
  batch_size: 10000  # Default: 10000, Range: 1000-100000
```

**When batch_size is used:**
- **Dataset Registration**: When importing CSV/Parquet files into the database
- **Table Creation**: When creating new tables from DataFrames
- **Data Export**: When exporting large datasets to files
- **Feature Generation**: When processing rows for feature engineering

**Progress Indicators:**
- MDM displays progress bars during batch operations showing:
  - Current batch number and total batches
  - Rows processed and total rows
  - Estimated time remaining
  - Processing speed (rows/second)

**Example progress output:**
```
Importing data: ━━━━━━━━━━━━━━━━━━━━ 75% 75,000/100,000 rows [2,500 rows/s, ETA: 10s]
```

**Performance Tuning:**
- **Smaller batches (1,000-5,000)**: Lower memory usage, safer for limited RAM
- **Default (10,000)**: Good balance for most datasets
- **Larger batches (25,000-100,000)**: Faster for large datasets with sufficient memory

**Environment Variable Override:**
```bash
export MDM_BATCH_SIZE=25000  # Use larger batches for better performance
mdm dataset register large_data /path/to/data.csv
```

**Current Limitations:**
- SQLite backend currently uses a hardcoded batch size of 1,000 rows
- DuckDB backend respects the configured batch_size
- PostgreSQL backend (when implemented) will use the configured value

#### Max Concurrent Operations

Controls parallelism for async operations:

```yaml
performance:
  max_concurrent_operations: 5  # Default: 5
```

Used by:
- Async batch processors
- Parallel dataset operations
- Concurrent file exports

### Dataset Validation Configuration

MDM performs automatic validation at two stages during dataset processing:

```yaml
validation:
  # Validation before feature generation
  before_features:
    check_duplicates: true      # Check for duplicate rows
    drop_duplicates: true       # Drop duplicates (logs warning if found)
    signal_detection: false     # Check signal ratio (logs warning if low)
    max_missing_ratio: 0.99     # Maximum allowed missing data ratio
    min_rows: 10                # Minimum rows required
  
  # Validation after feature generation
  after_features:
    check_duplicates: false     # Check for duplicates in features
    signal_detection: true      # Verify features have signal
    max_missing_ratio: 0.99     # Maximum missing in features
    min_rows: 10                # Minimum rows after processing
```

**Validation Stages:**

1. **Before Features (`before_features`)**:
   - Runs immediately after data is loaded, before any feature generation
   - Ensures data quality for feature engineering
   - `check_duplicates`: Identifies duplicate rows in the raw data
   - `drop_duplicates`: Automatically removes duplicates (logs WARNING if any found)
   - `signal_detection`: Checks if columns have sufficient signal (variance)
   - Logs WARNING for low signal but continues processing

2. **After Features (`after_features`)**:
   - Runs after all feature generation is complete
   - Validates the quality of generated features
   - `signal_detection`: Ensures generated features contain meaningful signal
   - Typically doesn't check duplicates as features should be unique

**Signal Detection:**
- Calculates the ratio of unique values to total rows
- Warns if ratio is below configured threshold (default: 1%)
- Helps identify constant or near-constant features

**Validation Behavior:**
- All validations run automatically during dataset operations
- Warnings are logged but don't stop processing
- Failed validations (like min_rows) will raise errors
- No manual validation commands needed

**Environment Variable Overrides:**
```bash
export MDM_VALIDATION_BEFORE_FEATURES_CHECK_DUPLICATES=false
export MDM_VALIDATION_AFTER_FEATURES_SIGNAL_DETECTION=true
```

## Directory Structure as Registry

MDM uses a directory-based registry approach instead of a central database:

```
~/.mdm/
├── mdm.yaml                          # Main configuration
├── datasets/                         # Dataset storage
│   ├── titanic/                     # Each dataset in its own directory
│   │   └── dataset.duckdb           # Data + local metadata
│   └── house_prices/
│       └── dataset.duckdb
├── config/
│   └── datasets/                     # Dataset configurations (the "registry")
│       ├── titanic.yaml             # Discovery pointer for titanic
│       └── house_prices.yaml        # Discovery pointer for house_prices
└── config/
    ├── datasets/                     # Dataset configurations (the "registry")
    │   ├── titanic.yaml             # Discovery pointer for titanic
    │   └── house_prices.yaml        # Discovery pointer for house_prices
    └── custom_features/              # Optional custom feature transformers
        ├── titanic.py               # Dataset-specific features for titanic
        └── house_prices.py          # Dataset-specific features for house_prices
└── tmp/                             # Temporary files
```

**Key points:**
- No central registry database
- Dataset discovery works by scanning `config/datasets/` directory
- Each YAML file serves as a lightweight pointer to the actual dataset
- Adding a dataset = creating directory + YAML file
- Removing a dataset = deleting directory + YAML file

## Dataset Configuration

Each dataset has its own YAML configuration file in `~/.mdm/config/datasets/{dataset_name}.yaml`.

### YAML File Schema (Authoritative)

**This section defines exactly what is stored in dataset YAML files vs. the dataset's database.**

#### Fields in YAML File

The following fields are stored in the dataset's YAML configuration file:

**Required fields (set during registration):**
- `name`: Dataset identifier (string)
- `description`: Dataset description (string, can be empty)
- `database`: Connection information
  - For file-based backends: `path` (absolute path to .duckdb or .sqlite file)
  - For PostgreSQL: `connection_string`
- `registered_at`: ISO timestamp of registration (automatically set)

**Optional fields (set during registration if applicable):**
- `tables`: Mapping of logical table names to physical table names
  - `train`: Training data table name
  - `test`: Test data table name  
  - `validation`: Validation table name (if exists)
  - `submission`: Submission template table name (if exists)
- `target_column`: Name of target column for ML (if supervised)
- `id_columns`: List of ID columns to exclude from features
- `problem_type`: ML problem type (regression, classification, etc.)

**Example YAML file:**

```yaml
name: house_prices
description: Kaggle House Prices - Advanced Regression Techniques
registered_at: "2024-01-15T10:30:00Z"

# Database connection
database:
  path: ~/.mdm/datasets/house_prices/dataset.duckdb
  
# Table names
tables:
  train: train
  test: test
  submission: sample_submission

# ML configuration
target_column: SalePrice
id_columns: [Id]
problem_type: regression
```

#### Database-Only Information

The following information is stored exclusively in the dataset's database:

- **Row counts** and **table sizes**
- **Column statistics** (types, nulls, distributions)
- **Feature engineering metadata**
- **Last updated timestamps**
- **Detailed schema information**

This data is accessed when needed through database queries.

#### Performance Implications

The `mdm dataset list` command has two modes:

1. **Default mode** (fast): Reads only YAML files
   - Shows: name, description, registered_at, problem_type
   - Does NOT show: row counts, DB size

2. **Full mode** (`--full` flag): Queries each dataset's database
   - Shows all information including current row counts and sizes
   - Slower for many datasets

**Note**: Some examples in documentation may show row counts in default mode for illustration. In practice, this information requires `--full` flag or is shown as "?" when not available.

### For PostgreSQL Backend

```yaml
name: large_dataset
description: Large scale customer data

database:
  connection_string: postgresql://user:pass@localhost/mdm_large_dataset
  
tables:
  train: train
  test: test

target_column: customer_value
id_columns: [customer_id, transaction_id]
problem_type: regression
```


**Important**: Dataset configuration files do NOT contain backend settings. See [Backend Selection and Configuration](03_Database_Architecture.md#backend-selection-and-configuration-authoritative) for the authoritative explanation of how backends work.

## Environment Variables

Override configuration settings using environment variables:

```bash
# Override log level
export MDM_LOG_LEVEL=DEBUG

# Override datasets path
export MDM_DATASETS_PATH=/data/ml/datasets

# Override default backend
export MDM_DEFAULT_BACKEND=postgresql
```

## Configuration Loading

MDM loads configuration in the following way:

1. **Initialization**: On first use, MDM creates `~/.mdm/mdm.yaml` with default settings
2. **Loading**: The configuration is loaded and parsed on each MDM invocation
3. **Mapping**: YAML settings are mapped to internal configuration:
   - `database.default_backend` → Backend for new datasets
   - `storage.*` paths → Resolved relative to `~/.mdm/`
   - `logging.level` → File logging verbosity

## Configuration Precedence

Configuration values are resolved in this order (highest to lowest):
1. Environment variables
2. Main configuration file (mdm.yaml)
3. Default values (built into MDM)

**Note**: Dataset YAML files contain only connection information and metadata, not backend configuration. The backend type is determined globally by `mdm.yaml`.

## Viewing Configuration Values (Debug Mode)

To see which configuration values MDM has loaded, use debug mode:

```bash
# Enable debug mode via environment variable
export MDM_LOG_LEVEL=DEBUG
mdm info

# Or use the --debug flag
mdm info --debug
```

In debug mode, MDM shows:
- Path to configuration file being loaded
- All configuration values (YAML, environment, and defaults)
- Which values were overridden by environment variables
- Which values are using defaults (when not set in YAML)

**Example debug output:**
```
[DEBUG] Loading configuration from: /home/user/.mdm/mdm.yaml
[DEBUG] Configuration values:
[DEBUG]   database.default_backend: sqlite (from mdm.yaml)
[DEBUG]   database.connection_timeout: 30 (default)
[DEBUG]   performance.batch_size: 10000 (overridden by MDM_BATCH_SIZE)
[DEBUG]   performance.max_concurrent_operations: 5 (from mdm.yaml)
[DEBUG]   logging.level: DEBUG (overridden by MDM_LOG_LEVEL)
[DEBUG]   logging.format: json (default)
[DEBUG]   export.default_format: csv (from mdm.yaml)
[DEBUG]   export.compression: zip (default)
...
```

This is particularly useful for:
- Verifying configuration changes take effect
- Debugging which settings are being used
- Understanding precedence when values are set in multiple places
- Testing configuration before running operations

## Best Practices

1. **Directory organization** - Keep all MDM files under `~/.mdm/` for easy management
2. **Store sensitive data** (passwords, API keys) in environment variables
3. **Version control** the YAML configuration files (except sensitive data)
4. **Document** any custom configuration in your project README
5. **Manual dataset addition** - You can manually create YAML files to add datasets

## Example Configurations

### Configuration Examples

Available configuration examples:
- [`mdm.yaml.minimal`](mdm.yaml.minimal) - Empty file showing MDM works with defaults
- [`mdm.yaml.default`](mdm.yaml.default) - ALL options with their DEFAULT values and comments
- [`mdm.yaml.sqlite-example`](mdm.yaml.sqlite-example) - Ready-to-use SQLite configuration
- [`mdm.yaml.duckdb-example`](mdm.yaml.duckdb-example) - Ready-to-use DuckDB configuration
- [`mdm.yaml.production`](mdm.yaml.production) - Production PostgreSQL with security

### Development Environment

```yaml
# Simple development configuration
paths:
  datasets_path: ./dev/datasets/
  configs_path: ./dev/configs/
  
database:
  default_backend: sqlite  # Lightweight for development
  
  sqlite:
    journal_mode: "WAL"
  
logging:
  level: DEBUG
  format: text           # Human-readable logs
  
feature_engineering:
  enabled: true          # Test feature generation
```

### Production Environment

```yaml
# Production with PostgreSQL
paths:
  datasets_path: /data/production/ml/datasets/
  configs_path: /data/production/ml/configs/
  logs_path: /var/log/mdm/
  
database:
  default_backend: postgresql
  
  sqlalchemy:
    pool_size: 20
    pool_pre_ping: true  # Verify connections
    
  postgresql:
    host: ${MDM_DB_HOST}          # Use env vars for secrets
    port: ${MDM_DB_PORT:-5432}
    user: ${MDM_DB_USER}
    password: ${MDM_DB_PASSWORD}
    database_prefix: prod_mdm_
    sslmode: require
    
performance:
  batch_size: 50000
    
logging:
  level: WARNING
  format: json           # For log aggregation
  max_bytes: 52428800    # 50MB
  backup_count: 10
```

### Research Environment

```yaml
# Research with DuckDB for analytics
paths:
  datasets_path: /research/shared/datasets/
  configs_path: ./configs/
  
database:
  default_backend: duckdb  # Best for analytical queries
  
  duckdb:
    memory_limit: "32GB"
    threads: 16
    
feature_engineering:
  enabled: true
  generic:
    temporal:
      include_lag: true      # Time series features
    statistical:
      outlier_threshold: 4.0 # Less strict for research
```

## Next Steps

- Learn about the [Database Architecture](03_Database_Architecture.md)
- Start [Registering Datasets](04_Dataset_Registration.md)
- Explore [Database Backends](06_Database_Backends.md) options