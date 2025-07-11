# MDM Environment Variables Reference

This document provides a complete reference of all environment variables supported by MDM.

## Overview

MDM follows a hierarchical configuration system:
1. **Defaults** (built into code)
2. **Configuration file** (`~/.mdm/mdm.yaml`)
3. **Environment variables** (highest priority)

Environment variables always override settings from the configuration file.

## Naming Convention

MDM environment variables follow the pattern: `MDM_<SECTION>_<KEY>`

For nested configuration, use underscores to separate levels:
- `MDM_DATABASE_DEFAULT_BACKEND` maps to `database.default_backend`
- `MDM_DATABASE_SQLITE_PRAGMAS_JOURNAL_MODE` maps to `database.sqlite.pragmas.journal_mode`

## Complete Variable Reference

### Core Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MDM_HOME_DIR` | Base directory for MDM data | `~/.mdm` | `/opt/mdm` |
| `MDM_TESTING` | Enable test mode | `false` | `true` |

### Database Settings

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `MDM_DATABASE_DEFAULT_BACKEND` | Storage backend to use | `sqlite` | `sqlite`, `duckdb`, `postgresql` |
| `MDM_DATABASE_SQLALCHEMY_ECHO` | Log SQL queries | `false` | `true`, `false` |
| `MDM_DATABASE_SQLALCHEMY_POOL_SIZE` | Connection pool size | `5` | Any integer |
| `MDM_DATABASE_SQLALCHEMY_MAX_OVERFLOW` | Max overflow connections | `10` | Any integer |

### SQLite Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MDM_DATABASE_SQLITE_PRAGMAS_JOURNAL_MODE` | Journal mode | `WAL` | `WAL`, `DELETE` |
| `MDM_DATABASE_SQLITE_PRAGMAS_SYNCHRONOUS` | Sync mode | `NORMAL` | `NORMAL`, `FULL`, `OFF` |
| `MDM_DATABASE_SQLITE_PRAGMAS_CACHE_SIZE` | Cache size | `-64000` | `-64000` (64MB) |
| `MDM_DATABASE_SQLITE_PRAGMAS_TEMP_STORE` | Temp storage | `MEMORY` | `MEMORY`, `FILE` |
| `MDM_DATABASE_SQLITE_PRAGMAS_MMAP_SIZE` | Memory map size | `268435456` | Bytes |

### DuckDB Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MDM_DATABASE_DUCKDB_SETTINGS_MEMORY_LIMIT` | Memory limit | `4GB` | `8GB`, `50%` |
| `MDM_DATABASE_DUCKDB_SETTINGS_THREADS` | Thread count | `4` | Any integer |
| `MDM_DATABASE_DUCKDB_SETTINGS_TEMP_DIRECTORY` | Temp directory | System temp | `/tmp/duckdb` |

### PostgreSQL Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MDM_DATABASE_POSTGRESQL_HOST` | Database host | `localhost` | `db.example.com` |
| `MDM_DATABASE_POSTGRESQL_PORT` | Database port | `5432` | `5433` |
| `MDM_DATABASE_POSTGRESQL_USER` | Database user | `mdm_user` | `admin` |
| `MDM_DATABASE_POSTGRESQL_PASSWORD` | Database password | `mdm_password` | `secret123` |
| `MDM_DATABASE_POSTGRESQL_DATABASE` | Database name | `mdm` | `ml_datasets` |

### Performance Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MDM_PERFORMANCE_BATCH_SIZE` | Batch size for data loading | `10000` | `50000` |
| `MDM_PERFORMANCE_MAX_CONCURRENT_OPERATIONS` | Max parallel operations | `4` | `8` |
| `MDM_PERFORMANCE_CACHE_SIZE_MB` | Cache size in MB | `100` | `500` |
| `MDM_PERFORMANCE_ENABLE_PROGRESS_BARS` | Show progress bars | `true` | `false` |

### Logging Settings

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `MDM_LOGGING_LEVEL` | Log level | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MDM_LOGGING_FILE` | Log file path | None | `/var/log/mdm.log` |
| `MDM_LOGGING_FORMAT` | Log format | `console` | `console`, `json` |
| `MDM_LOGGING_MAX_BYTES` | Max log file size | `10485760` | Bytes (10MB) |
| `MDM_LOGGING_BACKUP_COUNT` | Log files to keep | `5` | Any integer |

### Path Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MDM_PATHS_DATASETS_PATH` | Datasets directory | `datasets` | `data/datasets` |
| `MDM_PATHS_CONFIGS_PATH` | Configs directory | `config` | `conf` |
| `MDM_PATHS_CACHE_PATH` | Cache directory | `cache` | `tmp/cache` |
| `MDM_PATHS_LOGS_PATH` | Logs directory | `logs` | `var/logs` |

### Export Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MDM_EXPORT_DEFAULT_FORMAT` | Default export format | `csv` | `parquet` |
| `MDM_EXPORT_COMPRESSION` | Default compression | None | `gzip`, `snappy` |
| `MDM_EXPORT_INCLUDE_METADATA` | Include metadata | `true` | `false` |

## Usage Examples

### Basic Configuration

```bash
# Change backend to DuckDB
export MDM_DATABASE_DEFAULT_BACKEND=duckdb

# Increase batch size for large datasets
export MDM_PERFORMANCE_BATCH_SIZE=50000

# Enable debug logging
export MDM_LOGGING_LEVEL=DEBUG
```

### Development Setup

```bash
# Use custom MDM directory
export MDM_HOME_DIR=/tmp/mdm-dev

# Enable SQL query logging
export MDM_DATABASE_SQLALCHEMY_ECHO=true

# Set debug logging to file
export MDM_LOGGING_LEVEL=DEBUG
export MDM_LOGGING_FILE=/tmp/mdm-debug.log
```

### Production Setup

```bash
# PostgreSQL configuration
export MDM_DATABASE_DEFAULT_BACKEND=postgresql
export MDM_DATABASE_POSTGRESQL_HOST=db.prod.example.com
export MDM_DATABASE_POSTGRESQL_USER=mdm_prod
export MDM_DATABASE_POSTGRESQL_PASSWORD=secret123

# Performance tuning
export MDM_PERFORMANCE_BATCH_SIZE=100000
export MDM_PERFORMANCE_MAX_CONCURRENT_OPERATIONS=16

# Logging
export MDM_LOGGING_LEVEL=WARNING
export MDM_LOGGING_FILE=/var/log/mdm/mdm.log
export MDM_LOGGING_FORMAT=json
```

### Testing Configuration

```bash
# Isolated test environment
export MDM_HOME_DIR=/tmp/mdm-test-$$
export MDM_TESTING=true

# Fast settings for tests
export MDM_PERFORMANCE_BATCH_SIZE=100
export MDM_PERFORMANCE_ENABLE_PROGRESS_BARS=false
```

## Priority and Overrides

Environment variables have the highest priority in MDM's configuration system:

1. **Built-in defaults** (lowest priority)
2. **Configuration file** (`~/.mdm/mdm.yaml`)
3. **Environment variables** (highest priority)

Example:
```yaml
# In ~/.mdm/mdm.yaml
database:
  default_backend: sqlite
  
performance:
  batch_size: 10000
```

```bash
# This overrides the yaml config
export MDM_DATABASE_DEFAULT_BACKEND=duckdb
export MDM_PERFORMANCE_BATCH_SIZE=50000
```

## Tips and Best Practices

1. **Use `.env` files** for project-specific settings:
   ```bash
   # .env file
   MDM_HOME_DIR=/project/mdm-data
   MDM_DATABASE_DEFAULT_BACKEND=duckdb
   
   # Load it
   source .env
   ```

2. **Set permanently** in shell profile:
   ```bash
   # ~/.bashrc or ~/.zshrc
   export MDM_PERFORMANCE_BATCH_SIZE=50000
   ```

3. **Check current settings**:
   ```bash
   mdm info  # Shows current configuration
   ```

4. **Debug configuration issues**:
   ```bash
   MDM_LOGGING_LEVEL=DEBUG mdm dataset list
   ```

5. **Isolate environments**:
   ```bash
   # Development
   export MDM_HOME_DIR=~/.mdm-dev
   
   # Testing
   export MDM_HOME_DIR=~/.mdm-test
   ```

## Common Patterns

### Large Dataset Processing
```bash
export MDM_PERFORMANCE_BATCH_SIZE=100000
export MDM_DATABASE_DEFAULT_BACKEND=duckdb
export MDM_DATABASE_DUCKDB_SETTINGS_MEMORY_LIMIT=16GB
```

### Debugging Issues
```bash
export MDM_LOGGING_LEVEL=DEBUG
export MDM_LOGGING_FILE=/tmp/mdm-debug.log
export MDM_DATABASE_SQLALCHEMY_ECHO=true
```

### CI/CD Pipeline
```bash
export MDM_HOME_DIR=${CI_PROJECT_DIR}/mdm-data
export MDM_TESTING=true
export MDM_PERFORMANCE_ENABLE_PROGRESS_BARS=false
export MDM_LOGGING_FORMAT=json
```