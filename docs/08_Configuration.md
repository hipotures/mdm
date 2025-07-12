# Configuration Guide

MDM uses a hierarchical configuration system with sensible defaults, YAML configuration files, and environment variable overrides. This guide covers all configuration options and best practices.

## Configuration Hierarchy

MDM resolves configuration in the following order (highest priority first):

1. **Environment Variables** - Override any setting
2. **Configuration File** - `~/.mdm/mdm.yaml`
3. **Default Values** - Built-in sensible defaults

```
┌─────────────────────────────────────┐
│     Environment Variables           │ ← Highest Priority
│  MDM_DATABASE_DEFAULT_BACKEND=...   │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│     Configuration File              │
│    ~/.mdm/mdm.yaml                  │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│     Default Values                  │ ← Lowest Priority
│    (Built into MDM)                 │
└─────────────────────────────────────┘
```

## Configuration File

### Location

MDM looks for configuration in:
- Primary: `~/.mdm/mdm.yaml`
- Custom: Specified via `--config` flag or `MDM_CONFIG_PATH` environment variable

### Creating Configuration

```bash
# MDM creates a default config on first use
# Or create manually:
mkdir -p ~/.mdm
cat > ~/.mdm/mdm.yaml << 'EOF'
# MDM Configuration
database:
  default_backend: sqlite

performance:
  batch_size: 10000
  
logging:
  level: INFO
EOF
```

### Complete Configuration Reference

```yaml
# ~/.mdm/mdm.yaml - Complete configuration with all options

# Database configuration
database:
  # Default backend for all datasets (sqlite, duckdb, postgresql)
  default_backend: sqlite
  
  # SQLite specific settings
  sqlite:
    journal_mode: WAL          # WAL, DELETE, TRUNCATE, PERSIST, MEMORY
    synchronous: NORMAL        # FULL, NORMAL, OFF
    cache_size: -64000         # Negative = KB, positive = pages
    temp_store: MEMORY         # DEFAULT, FILE, MEMORY
    mmap_size: 268435456       # Memory-mapped I/O size (bytes)
    foreign_keys: true         # Enable foreign key constraints
    recursive_triggers: true   # Enable recursive triggers
    
  # DuckDB specific settings
  duckdb:
    memory_limit: 4GB          # Maximum memory usage
    threads: 4                 # Number of threads (0 = auto)
    temp_directory: null       # Temp file location (null = system default)
    enable_progress_bar: false # Disable DuckDB's progress bar
    max_memory: 8GB           # Hard memory limit
    
  # PostgreSQL specific settings
  postgresql:
    host: localhost            # Database host
    port: 5432                # Database port
    user: mdm_user            # Database user
    password: ${MDM_PG_PASSWORD}  # From environment variable
    database: mdm             # Database name
    
    # Connection pool settings
    pool_size: 10             # Core connection pool size
    max_overflow: 20          # Maximum overflow connections
    pool_timeout: 30          # Seconds to wait for connection
    pool_recycle: 3600        # Recycle connections after N seconds
    
    # SSL settings
    sslmode: prefer           # disable, allow, prefer, require, verify-ca, verify-full
    sslcert: null            # Path to client certificate
    sslkey: null             # Path to client key
    sslrootcert: null        # Path to CA certificate
    
    # Performance
    echo: false              # Log SQL statements
    connect_args:
      connect_timeout: 10
      options: "-c statement_timeout=300000"  # 5 minutes

# Performance settings
performance:
  batch_size: 10000           # Rows per batch during processing
  max_workers: 4              # Maximum parallel workers
  chunk_size: 10000          # Chunk size for data loading
  enable_progress: true       # Show progress bars
  memory_limit: null         # Memory limit for operations (e.g., "2GB")
  
  # Cache settings
  cache:
    enabled: true            # Enable caching
    ttl: 3600               # Cache TTL in seconds
    max_size: 1000          # Maximum cache entries
    
  # Optimization flags
  optimize_dtypes: true      # Optimize pandas dtypes
  use_categorical: true      # Convert strings to categorical
  
# Feature engineering settings
features:
  enable_at_registration: true  # Generate features during registration
  batch_size: 10000            # Feature generation batch size
  n_jobs: -1                   # CPU cores for parallel processing (-1 = all)
  
  # Signal detection thresholds
  min_variance: 0.01           # Minimum variance to keep feature
  max_correlation: 0.95        # Maximum correlation between features
  min_non_zero_ratio: 0.01     # Minimum non-zero value ratio
  
  # Generic feature settings
  statistical:
    enabled: true
    features: [zscore, log, sqrt, squared, outlier, percentile]
    outlier_method: iqr        # iqr or zscore
    outlier_threshold: 3.0     # For zscore method
    
  temporal:
    enabled: true
    components: [year, month, day, dayofweek, hour, quarter]
    enable_cyclical: true      # Sin/cos encoding
    enable_lag: false          # Lag features (experimental)
    
  categorical:
    enabled: true
    max_cardinality: 50        # Max unique values for one-hot
    min_frequency: 0.01        # Minimum frequency threshold
    encoding_method: onehot    # onehot, target, frequency
    enable_target_encoding: true
    
  text:
    enabled: true
    min_text_length: 20        # Minimum length to process
    max_features: 100          # For TF-IDF/BoW
    enable_sentiment: false    # Sentiment analysis
    
  distribution:
    enabled: true
    n_bins: 10                # Number of bins
    strategy: quantile        # uniform or quantile
    
  custom:
    enabled: true
    auto_discover: true       # Auto-load from custom_features/
    paths:                    # Additional paths to search
      - ~/.mdm/config/custom_features
      
# Logging configuration
logging:
  level: INFO                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: ~/.mdm/logs/mdm.log  # Log file path (null = no file logging)
  max_size: 10485760         # Max log file size (10MB)
  backup_count: 5            # Number of backup files
  format: text               # text or json
  
  # Console output
  console:
    enabled: true
    level: WARNING           # Console log level
    
  # Component-specific levels
  components:
    mdm.storage: INFO
    mdm.features: DEBUG
    mdm.cli: INFO

# Path configuration
paths:
  base_path: ~/.mdm          # Base directory for all MDM data
  datasets_path: ${paths.base_path}/datasets
  config_path: ${paths.base_path}/config
  logs_path: ${paths.base_path}/logs
  temp_path: ${paths.base_path}/temp
  cache_path: ${paths.base_path}/cache

# Export settings
export:
  default_format: csv        # csv, parquet, json, excel
  compression: null          # null, gzip, zip, bz2, xz
  include_features: true     # Include generated features
  
  # Format-specific settings
  csv:
    sep: ","
    encoding: utf-8
    index: false
    
  parquet:
    engine: pyarrow          # pyarrow or fastparquet
    compression: snappy      # snappy, gzip, brotli
    
  json:
    orient: records          # records, table, split
    lines: true             # JSON lines format
    
# Validation settings
validation:
  strict_mode: false         # Strict validation
  
  # Dataset validation
  dataset:
    min_rows: 1             # Minimum rows required
    max_null_percentage: 0.9 # Maximum null percentage
    required_columns: []     # Always required columns
    
  # Type validation
  types:
    infer_datetime: true    # Auto-detect datetime columns
    datetime_formats:       # Formats to try
      - "%Y-%m-%d"
      - "%Y-%m-%d %H:%M:%S"
      - "%d/%m/%Y"
      
# CLI settings
cli:
  default_limit: 20         # Default number of items to show
  confirm_destructive: true # Confirm before destructive operations
  show_traceback: false     # Show full traceback on errors
  progress_bar_style: rich  # rich, tqdm, or none
  
# API settings
api:
  timeout: 300              # API timeout in seconds
  retry_count: 3           # Number of retries
  retry_delay: 1           # Delay between retries

# Monitoring settings
monitoring:
  enabled: true            # Enable monitoring
  metrics_db: ${paths.base_path}/metrics.db
  retention_days: 30       # Keep metrics for N days
  
# Development settings (only in dev mode)
development:
  debug: false             # Debug mode
  profile: false           # Enable profiling
  explain_queries: false   # EXPLAIN all queries
  mock_data: false        # Use mock data for testing
```

## Environment Variables

### Naming Convention

Environment variables follow the pattern: `MDM_<SECTION>_<SUBSECTION>_<KEY>`

- Sections are separated by underscores
- All uppercase
- Nested values use additional underscores

### Common Environment Variables

```bash
# Database settings
export MDM_DATABASE_DEFAULT_BACKEND=duckdb
export MDM_DATABASE_SQLITE_JOURNAL_MODE=WAL
export MDM_DATABASE_DUCKDB_MEMORY_LIMIT=8GB
export MDM_DATABASE_POSTGRESQL_HOST=db.example.com
export MDM_DATABASE_POSTGRESQL_PASSWORD=secret

# Performance settings
export MDM_PERFORMANCE_BATCH_SIZE=50000
export MDM_PERFORMANCE_MAX_WORKERS=8
export MDM_PERFORMANCE_ENABLE_PROGRESS=false

# Feature settings
export MDM_FEATURES_ENABLE_AT_REGISTRATION=true
export MDM_FEATURES_STATISTICAL_ENABLED=false
export MDM_FEATURES_N_JOBS=4

# Logging settings
export MDM_LOGGING_LEVEL=DEBUG
export MDM_LOGGING_FILE=/var/log/mdm/mdm.log
export MDM_LOGGING_FORMAT=json

# Path settings
export MDM_PATHS_BASE_PATH=/data/mdm
export MDM_PATHS_TEMP_PATH=/tmp/mdm

# CLI settings
export MDM_CLI_CONFIRM_DESTRUCTIVE=false
export MDM_CLI_SHOW_TRACEBACK=true
```

### Using Environment Files

```bash
# Create .env file
cat > .env << 'EOF'
MDM_DATABASE_DEFAULT_BACKEND=postgresql
MDM_DATABASE_POSTGRESQL_HOST=localhost
MDM_DATABASE_POSTGRESQL_USER=mdm
MDM_DATABASE_POSTGRESQL_PASSWORD=secret
MDM_PERFORMANCE_BATCH_SIZE=25000
MDM_LOGGING_LEVEL=INFO
EOF

# Load environment
export $(cat .env | xargs)

# Or use direnv
echo "dotenv" > .envrc
direnv allow
```

## Configuration Profiles

### Development Profile

```yaml
# ~/.mdm/mdm-dev.yaml
database:
  default_backend: sqlite
  
performance:
  batch_size: 1000
  max_workers: 2
  
logging:
  level: DEBUG
  console:
    level: DEBUG
    
development:
  debug: true
  explain_queries: true
```

### Production Profile

```yaml
# ~/.mdm/mdm-prod.yaml
database:
  default_backend: postgresql
  postgresql:
    host: prod-db.example.com
    pool_size: 20
    
performance:
  batch_size: 50000
  max_workers: 8
  
logging:
  level: WARNING
  file: /var/log/mdm/production.log
  format: json
  
monitoring:
  enabled: true
  retention_days: 90
```

### Analytics Profile

```yaml
# ~/.mdm/mdm-analytics.yaml
database:
  default_backend: duckdb
  duckdb:
    memory_limit: 16GB
    threads: 16
    
performance:
  batch_size: 100000
  optimize_dtypes: true
  
features:
  enable_at_registration: true
  n_jobs: -1
```

### Using Profiles

```bash
# Via environment variable
export MDM_CONFIG_PATH=~/.mdm/mdm-dev.yaml
mdm dataset list

# Via command line
mdm --config ~/.mdm/mdm-prod.yaml dataset list

# In Python
from mdm import MDMClient
client = MDMClient(config_path="~/.mdm/mdm-analytics.yaml")
```

## Advanced Configuration

### Dynamic Configuration

```python
from mdm.core.config import get_settings, update_settings

# Get current settings
settings = get_settings()
print(f"Current backend: {settings.database.default_backend}")

# Update settings at runtime
update_settings({
    "performance": {
        "batch_size": 50000,
        "max_workers": 8
    }
})

# Context manager for temporary settings
from mdm.core.config import temporary_settings

with temporary_settings({"logging.level": "DEBUG"}):
    # Debug logging enabled here
    client = MDMClient()
    # ...
# Original settings restored
```

### Configuration Validation

```python
from mdm.core.config import validate_config

# Validate configuration file
errors = validate_config("~/.mdm/mdm.yaml")
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

### Conditional Configuration

```yaml
# Use environment variables for conditional config
database:
  default_backend: ${MDM_ENV:sqlite}  # Default to sqlite if MDM_ENV not set
  
  postgresql:
    host: ${MDM_PG_HOST:localhost}
    port: ${MDM_PG_PORT:5432}
    
logging:
  level: ${MDM_LOG_LEVEL:INFO}
  file: ${MDM_LOG_FILE:~/.mdm/logs/mdm.log}
```

### Backend-Specific Overrides

```yaml
# Override settings per backend
backends:
  sqlite:
    performance:
      batch_size: 10000
      
  duckdb:
    performance:
      batch_size: 50000
      
  postgresql:
    performance:
      batch_size: 25000
```

## Security Considerations

### Sensitive Information

1. **Never commit passwords to version control**
```yaml
# Bad
postgresql:
  password: mypassword123

# Good
postgresql:
  password: ${MDM_PG_PASSWORD}  # From environment
```

2. **Use secure file permissions**
```bash
chmod 600 ~/.mdm/mdm.yaml
```

3. **Encrypt sensitive configs**
```bash
# Encrypt config
gpg -c ~/.mdm/mdm.yaml

# Decrypt when needed
gpg -d ~/.mdm/mdm.yaml.gpg > ~/.mdm/mdm.yaml
```

### Connection Security

```yaml
postgresql:
  # Require SSL
  sslmode: require
  
  # Verify server certificate
  sslmode: verify-full
  sslrootcert: /path/to/ca-bundle.crt
  
  # Client certificates
  sslcert: /path/to/client-cert.pem
  sslkey: /path/to/client-key.pem
```

## Troubleshooting Configuration

### Debug Configuration Loading

```bash
# Show configuration loading process
MDM_DEBUG_CONFIG=true mdm info

# Output shows:
# - Config file path
# - Environment variables detected
# - Final merged configuration
```

### Common Issues

1. **Config file not found**
```bash
# Check location
echo $HOME/.mdm/mdm.yaml

# Create default
mdm info  # Creates default config
```

2. **Invalid YAML**
```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('$HOME/.mdm/mdm.yaml'))"
```

3. **Environment variable not working**
```bash
# Check variable is exported
echo $MDM_DATABASE_DEFAULT_BACKEND

# Ensure correct format
export MDM_DATABASE_DEFAULT_BACKEND=duckdb  # Correct
export MDM_DATABASE_DEFAULT-BACKEND=duckdb  # Wrong (hyphen)
```

4. **Permission denied**
```bash
# Fix permissions
chmod 755 ~/.mdm
chmod 644 ~/.mdm/mdm.yaml
```

## Best Practices

### 1. Use Environment Variables for Secrets
```yaml
postgresql:
  password: ${MDM_PG_PASSWORD}
  
api:
  key: ${MDM_API_KEY}
```

### 2. Separate Configs by Environment
```bash
~/.mdm/
├── mdm.yaml          # Default/development
├── mdm-prod.yaml     # Production
├── mdm-staging.yaml  # Staging
└── mdm-test.yaml     # Testing
```

### 3. Version Control Safe Configs
```yaml
# .gitignore
*.yaml
!mdm-example.yaml

# mdm-example.yaml (safe to commit)
database:
  default_backend: sqlite
postgresql:
  host: ${DB_HOST}
  password: ${DB_PASSWORD}
```

### 4. Document Custom Settings
```yaml
# Custom feature for our team
features:
  custom:
    # Load industry-specific features from shared location
    paths:
      - /shared/mdm/features  # NFS mount with team features
```

### 5. Monitor Configuration Changes
```python
import os
import hashlib

def get_config_hash():
    """Get hash of current configuration."""
    with open(os.path.expanduser("~/.mdm/mdm.yaml"), "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Log configuration changes
current_hash = get_config_hash()
if current_hash != previous_hash:
    logger.info(f"Configuration changed: {current_hash}")
```

## Configuration Schema

MDM uses Pydantic for configuration validation. The schema ensures:

- Type safety
- Required fields
- Valid ranges
- Enum constraints

```python
# Example: Custom validation
from mdm.core.config import Settings

def validate_custom_config(config_dict):
    """Validate configuration dictionary."""
    try:
        settings = Settings(**config_dict)
        return True, settings
    except Exception as e:
        return False, str(e)
```

## Migration from Older Versions

If upgrading from older MDM versions:

```python
# Migration script
import yaml
import os

def migrate_config():
    """Migrate old config format to new."""
    old_config = os.path.expanduser("~/.mdm/config.yml")
    new_config = os.path.expanduser("~/.mdm/mdm.yaml")
    
    if os.path.exists(old_config) and not os.path.exists(new_config):
        with open(old_config) as f:
            config = yaml.safe_load(f)
        
        # Apply migrations
        if "storage" in config:
            config["database"] = config.pop("storage")
            
        # Save new format
        with open(new_config, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"Migrated config to {new_config}")

migrate_config()
```