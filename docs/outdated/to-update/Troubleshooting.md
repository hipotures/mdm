# MDM Troubleshooting Guide

This guide helps you resolve common issues with MDM.

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [Registration Errors](#registration-errors)
3. [Performance Problems](#performance-problems)
4. [Configuration Issues](#configuration-issues)
5. [Database Errors](#database-errors)
6. [Feature Engineering Issues](#feature-engineering-issues)
7. [Export/Import Problems](#export-import-problems)
8. [Known Limitations](#known-limitations)

## Installation Issues

### uv.lock File Missing
**Problem**: `uv.lock` file not found
```
error: No uv.lock file found
```

**Solution**:
```bash
# Generate lock file
uv lock

# Then install
uv pip install -e .
```

### Import Error: No module named 'mdm'
**Problem**: MDM not properly installed

**Solution**:
```bash
# Ensure you're in virtual environment
source .venv/bin/activate

# Install in development mode
pip install -e .

# Verify
python -c "import mdm; print(mdm.__version__)"
```

### DuckDB Backend Not Working
**Problem**: Missing SQLAlchemy dialect for DuckDB
```
Error: No module named 'duckdb_engine'
```

**Solution**:
```bash
# Install DuckDB engine
pip install duckdb-engine

# Or use SQLite backend instead
# In ~/.mdm/mdm.yaml:
database:
  default_backend: sqlite
```

## Registration Errors

### Dataset Already Exists
**Problem**: Dataset with same name exists
```
Error: Dataset 'sales' already exists
```

**Solution**:
```bash
# Option 1: Use different name
mdm dataset register sales_2024 data.csv

# Option 2: Remove existing (careful!)
mdm dataset remove sales --force
mdm dataset register sales data.csv

# Note: --force flag in register doesn't work currently
```

### Invalid Dataset Name
**Problem**: Spaces or special characters in name
```
Error: Dataset name can only contain alphanumeric characters, underscores, and dashes
```

**Solution**:
```bash
# Use valid characters only
# Good: sales_data, sales-2024, salesData123
# Bad: "sales data", sales@2024, sales.data

mdm dataset register sales_data data.csv
```

### Multiple Values Error
**Problem**: --time-column or --id-columns causes error
```
TypeError: register() got multiple values for keyword argument 'time_column'
```

**Solution**:
```bash
# These parameters have a bug. Workaround:
# 1. Register without them
mdm dataset register timeseries data.csv

# 2. Update after registration (if update works)
mdm dataset update timeseries --time-column timestamp
```

### Path Does Not Exist
**Problem**: File or directory not found
```
Error: Path does not exist: /path/to/data.csv
```

**Solution**:
```bash
# Check path exists
ls -la /path/to/data.csv

# Use absolute path
mdm dataset register sales /absolute/path/to/data.csv

# Or use relative path from current directory
mdm dataset register sales ./data/sales.csv
```

## Performance Problems

### Slow Registration
**Problem**: Large dataset takes too long to register

**Solution**:
```bash
# 1. Check batch size configuration
# In ~/.mdm/mdm.yaml:
performance:
  batch_size: 50000  # Increase for faster loading

# 2. Monitor progress
export MDM_LOGGING_LEVEL=DEBUG
mdm dataset register large_data data.csv

# 3. Consider sampling for initial exploration
head -n 10000 large_data.csv > sample.csv
mdm dataset register sample sample.csv
```

### Memory Errors
**Problem**: Out of memory during registration

**Solution**:
```bash
# Reduce batch size in ~/.mdm/mdm.yaml:
performance:
  batch_size: 10000  # Smaller batches

# Or set via environment:
export MDM_PERFORMANCE_BATCH_SIZE=5000
mdm dataset register large_data data.csv
```

### Timeout in Tests
**Problem**: Performance tests fail with timeout

**Solution**:
```python
# Increase timeout in tests
assert end - start < 10.0  # Instead of 5.0
```

## Configuration Issues

### Configuration Not Applied
**Problem**: Changes to mdm.yaml have no effect

**Common Issues**:
1. **SQLAlchemy echo not working**
   ```yaml
   database:
     sqlalchemy:
       echo: true  # Doesn't print SQL
   ```

2. **Log level not changing**
   ```yaml
   logging:
     level: ERROR  # Still shows INFO
   ```

3. **Export format ignored**
   ```yaml
   export:
     default_format: parquet  # Still exports CSV
   ```

**Solution**:
Many configuration options are not fully implemented. Use CLI flags instead:
```bash
# Use CLI options
mdm dataset export sales --format parquet

# Or environment variables (if supported)
export MDM_LOGGING_LEVEL=ERROR
```

### Environment Variables Not Working
**Problem**: MDM_* environment variables ignored

**Working Variables**:
```bash
# These work:
export MDM_DATABASE_DEFAULT_BACKEND=sqlite
export MDM_PERFORMANCE_BATCH_SIZE=50000

# This doesn't:
export MDM_MDM_HOME=/custom/path  # Not implemented
```

### Finding Configuration
**Problem**: Don't know current configuration

**Solution**:
```bash
# Show current configuration
mdm info

# Check config file
cat ~/.mdm/mdm.yaml

# Find MDM home
echo ~/.mdm
```

## Database Errors

### SQLite Busy Error
**Problem**: Database is locked
```
sqlite3.OperationalError: database is locked
```

**Solution**:
```bash
# 1. Ensure no other process is using database
ps aux | grep mdm

# 2. Check for stuck SQLite processes
lsof ~/.mdm/datasets/*/dataset.sqlite

# 3. Enable WAL mode (should be automatic)
sqlite3 ~/.mdm/datasets/YOUR_DATASET/dataset.sqlite "PRAGMA journal_mode=WAL;"
```

### Connection String Error
**Problem**: PostgreSQL connection fails

**Solution**:
```bash
# Check PostgreSQL configuration
# In ~/.mdm/mdm.yaml:
database:
  default_backend: postgresql
  postgresql:
    host: localhost
    port: 5432
    database: mdm
    user: your_user
    password: your_password

# Test connection
psql -h localhost -U your_user -d mdm
```

### Backend Mismatch
**Problem**: Dataset created with different backend

**Solution**:
```bash
# Check dataset's backend
cat ~/.mdm/config/datasets/YOUR_DATASET.yaml

# If switching backends, must re-register:
# 1. Export data
mdm dataset export old_dataset /tmp/backup

# 2. Change backend in mdm.yaml
# 3. Register with new backend
mdm dataset register new_dataset /tmp/backup/old_dataset_train.csv
```

## Feature Engineering Issues

### No Features Generated
**Problem**: Features table is empty or missing

**Causes**:
1. Feature engineering disabled (though config doesn't work)
2. No suitable features detected
3. All features filtered due to low signal

**Solution**:
```bash
# Check if features were generated
sqlite3 ~/.mdm/datasets/YOUR_DATASET/dataset.sqlite "SELECT * FROM train_features LIMIT 5;"

# Re-register with feature generation
mdm dataset register test data.csv --generate-features
```

### Custom Features Not Applied
**Problem**: Custom feature files ignored

**Current Status**: Custom features not implemented

**Workaround**:
```python
# Manually apply features after loading
import pandas as pd
from mdm import MDMClient

client = MDMClient()
df = client.load_dataset("sales")

# Apply custom features
df['custom_feature'] = df['column1'] * df['column2']
```

### Datetime Features Missing
**Problem**: No temporal features from date columns

**Current Status**: Datetime detection not implemented

**Workaround**:
```python
# Convert after loading
df = client.load_dataset("sales")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
```

## Export/Import Problems

### Export Creates ZIP Instead of Format
**Problem**: Always exports as ZIP regardless of format

**Solution**:
```bash
# Exports create ZIP with files inside
mdm dataset export sales /output --format parquet

# Extract files
cd /output
unzip sales_export_*.zip

# For uncompressed export:
mdm dataset export sales /output --compression none
```

### Large Export Fails
**Problem**: Memory error during export

**Solution**:
```bash
# Export in chunks (manual process)
# 1. Load with iterator
client = MDMClient()
for i, chunk in enumerate(client.load_dataset("large", as_iterator=True)):
    chunk.to_csv(f"export_chunk_{i}.csv", index=False)
```

### Import Not Supported
**Problem**: No import command

**Solution**:
```bash
# Re-register is the only way to import
mdm dataset register imported_data /path/to/exported.csv
```

## Known Limitations

### CLI Options Not Implemented
These options exist but don't work:
- `--no-auto`: Manual registration not implemented
- `--skip-analysis`: Option doesn't exist
- `--dry-run`: Option doesn't exist
- `--force` (for register): Doesn't overwrite

### Missing Column Type Options
These column specifications don't exist:
- `--datetime-columns`
- `--categorical-columns`
- `--numeric-columns`
- `--text-columns`
- `--ignore-columns`

### Type Detection Limitations
MDM currently detects only:
- ✅ Numeric types (int, float)
- ✅ Text types
- ✅ ID columns (by pattern)
- ❌ Datetime (stored as text)
- ❌ Categorical (stored as text)
- ❌ Boolean (stored as text)

### Configuration Limitations
Many config options are parsed but not used:
- SQLAlchemy echo setting
- Log file output
- Export defaults
- Feature engineering settings
- Some performance settings

## Debug Mode

### Enable Detailed Logging
```bash
# Set debug level
export MDM_LOGGING_LEVEL=DEBUG

# Run command
mdm dataset register test data.csv 2>debug.log

# Check log
grep ERROR debug.log
grep WARNING debug.log
```

### Inspect Database Directly
```bash
# Open dataset database
sqlite3 ~/.mdm/datasets/YOUR_DATASET/dataset.sqlite

# Useful commands:
.tables                    # List all tables
.schema train             # Show table structure
SELECT COUNT(*) FROM train;  # Row count
.mode column              # Better display
.headers on               # Show column names
SELECT * FROM train LIMIT 5;  # Sample data
```

### Check Dataset Configuration
```bash
# View dataset config
cat ~/.mdm/config/datasets/YOUR_DATASET.yaml

# List all datasets
ls ~/.mdm/config/datasets/

# Check dataset directory
ls -la ~/.mdm/datasets/YOUR_DATASET/
```

## Getting Help

If these solutions don't resolve your issue:

1. **Check GitHub Issues**: Look for similar problems
2. **Enable Debug Logging**: Get detailed error information
3. **Minimal Reproduction**: Create simple test case
4. **File an Issue**: Include:
   - MDM version (`mdm --version`)
   - Python version (`python --version`)
   - Error message and stack trace
   - Steps to reproduce
   - Sample data (if possible)