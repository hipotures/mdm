# MDM Troubleshooting Guide

## Overview

This guide helps diagnose and resolve common issues with MDM. Each section includes symptoms, causes, and solutions.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Dataset Registration Problems](#dataset-registration-problems)
3. [Performance Issues](#performance-issues)
4. [Storage Backend Errors](#storage-backend-errors)
5. [Feature Engineering Issues](#feature-engineering-issues)
6. [Configuration Problems](#configuration-problems)
7. [Migration Issues](#migration-issues)
8. [Common Error Messages](#common-error-messages)

## Installation Issues

### Issue: Installation Fails with Dependency Errors

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement sqlalchemy>=2.0
ERROR: No matching distribution found for sqlalchemy>=2.0
```

**Solution:**
```bash
# Update pip first
python -m pip install --upgrade pip

# Use uv for better dependency resolution
pip install uv
uv pip install mdm-refactor

# Or install with specific Python version
python3.9 -m pip install mdm-refactor
```

### Issue: Command 'mdm' Not Found

**Symptoms:**
```bash
$ mdm version
bash: mdm: command not found
```

**Solutions:**

1. **Check installation:**
```bash
pip show mdm-refactor
```

2. **Add to PATH:**
```bash
# Find where mdm is installed
python -c "import mdm; print(mdm.__file__)"

# Add scripts directory to PATH
export PATH="$HOME/.local/bin:$PATH"

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

3. **Use Python module syntax:**
```bash
python -m mdm.cli version
```

## Dataset Registration Problems

### Issue: Dataset Registration Fails

**Symptoms:**
```
Error: Failed to register dataset: [Errno 2] No such file or directory
```

**Diagnostic Steps:**

1. **Check file exists:**
```bash
ls -la /path/to/your/data.csv
```

2. **Check permissions:**
```bash
# File should be readable
chmod 644 /path/to/your/data.csv
```

3. **Validate CSV format:**
```bash
# Check first few lines
head -5 /path/to/your/data.csv

# Check for encoding issues
file -i /path/to/your/data.csv
```

**Solutions:**

1. **Use absolute path:**
```bash
mdm dataset register mydata $(pwd)/data.csv
```

2. **Fix encoding issues:**
```python
# Convert to UTF-8
import pandas as pd
df = pd.read_csv('data.csv', encoding='latin1')
df.to_csv('data_utf8.csv', encoding='utf-8', index=False)
```

3. **Handle special characters in path:**
```bash
mdm dataset register mydata "./path with spaces/data.csv"
```

### Issue: Target Column Not Found

**Symptoms:**
```
Error: Target column 'label' not found in dataset
```

**Solution:**
```bash
# List columns first
mdm dataset preview data.csv --columns

# Register with correct column name
mdm dataset register mydata data.csv --target correct_column_name
```

## Performance Issues

### Issue: Slow Dataset Registration

**Symptoms:**
- Registration takes minutes for medium-sized files
- Progress bar moves slowly

**Diagnostic:**
```bash
# Enable debug logging
export MDM_LOGGING_LEVEL=DEBUG
mdm dataset register large_data.csv --profile
```

**Solutions:**

1. **Increase batch size:**
```bash
mdm config set performance.batch_size 50000
```

2. **Use DuckDB for large files:**
```bash
mdm config set database.default_backend duckdb
```

3. **Disable feature generation during registration:**
```bash
mdm dataset register large_data.csv --skip-features
```

### Issue: High Memory Usage

**Symptoms:**
- Python process uses excessive memory
- System becomes unresponsive

**Solutions:**

1. **Limit batch size:**
```python
from mdm.adapters import get_config_manager

config_manager = get_config_manager()
config_manager.update_config({
    "performance": {
        "batch_size": 10000,
        "max_memory_mb": 2048
    }
})
```

2. **Use chunked processing:**
```python
from mdm.utils import ChunkedProcessor

processor = ChunkedProcessor(chunk_size=10000)
processor.process_file("large_file.csv", output="processed.parquet")
```

## Storage Backend Errors

### Issue: SQLite Database Locked

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**

1. **Close other connections:**
```python
from mdm.adapters import clear_storage_cache
clear_storage_cache()
```

2. **Increase timeout:**
```bash
mdm config set database.sqlite.timeout 30
```

3. **Use WAL mode:**
```bash
mdm config set database.sqlite.pragmas.journal_mode WAL
```

### Issue: DuckDB Out of Memory

**Symptoms:**
```
duckdb.OutOfMemoryException: Out of Memory Error
```

**Solutions:**

1. **Increase memory limit:**
```bash
mdm config set database.duckdb.pragmas.memory_limit "8GB"
```

2. **Enable disk spilling:**
```bash
mdm config set database.duckdb.pragmas.temp_directory "/tmp/duckdb_temp"
```

### Issue: PostgreSQL Connection Failed

**Symptoms:**
```
psycopg2.OperationalError: could not connect to server
```

**Solutions:**

1. **Check connection parameters:**
```bash
# Test connection
psql -h localhost -U mdm_user -d mdm_db

# Update MDM config
mdm config set database.postgresql.host localhost
mdm config set database.postgresql.port 5432
mdm config set database.postgresql.user mdm_user
```

2. **Check PostgreSQL service:**
```bash
# Linux
sudo systemctl status postgresql

# macOS
brew services list | grep postgresql
```

## Feature Engineering Issues

### Issue: Feature Generation Fails

**Symptoms:**
```
Error: Failed to generate features: unsupported operand type(s)
```

**Solutions:**

1. **Check data types:**
```python
from mdm.adapters import get_dataset_manager

manager = get_dataset_manager()
dataset = manager.get_dataset("mydata")

# Review column types
print(dataset.column_types)
```

2. **Handle missing values:**
```python
# Clean data before feature generation
df = df.fillna(0)  # or df.dropna()
```

3. **Skip problematic features:**
```python
from mdm.adapters import get_feature_generator

generator = get_feature_generator()
generator.set_exclude_features(['problematic_feature'])
```

## Configuration Problems

### Issue: Configuration Not Loading

**Symptoms:**
- Changes to config file not taking effect
- Environment variables ignored

**Diagnostic:**
```bash
# Show current configuration
mdm config show

# Check config file location
mdm config --show-path

# Validate configuration
mdm config validate
```

**Solutions:**

1. **Fix YAML syntax:**
```yaml
# Correct indentation
database:
  default_backend: sqlite  # 2 spaces
  sqlite:
    timeout: 30
```

2. **Check environment variable format:**
```bash
# Correct format
export MDM_DATABASE_DEFAULT_BACKEND=duckdb
export MDM_PERFORMANCE_BATCH_SIZE=10000

# Wrong format (won't work)
export MDM_BACKEND=duckdb
```

3. **Clear config cache:**
```python
from mdm.adapters import get_config_manager
config_manager = get_config_manager()
config_manager.reload()
```

## Migration Issues

### Issue: Feature Flag Not Working

**Symptoms:**
- Still using old implementation after enabling flags
- Inconsistent behavior

**Solutions:**

1. **Clear all caches:**
```python
from mdm.adapters import (
    clear_storage_cache,
    clear_feature_cache,
    clear_dataset_cache,
    clear_cli_cache
)

clear_storage_cache()
clear_feature_cache()
clear_dataset_cache()
clear_cli_cache()
```

2. **Verify flag status:**
```python
from mdm.core import feature_flags

# Check current flags
print(feature_flags.get_all())

# Force reload
feature_flags.reload()
```

### Issue: Data Loss During Migration

**Symptoms:**
- Datasets missing after migration
- Statistics not preserved

**Prevention:**
```bash
# Always backup before migration
mdm backup create ~/mdm_backup_$(date +%Y%m%d)

# Verify backup
mdm backup verify ~/mdm_backup_20250110
```

**Recovery:**
```bash
# Restore from backup
mdm backup restore ~/mdm_backup_20250110

# Or manually restore
cp -r ~/mdm_backup_20250110/.mdm ~/
```

## Common Error Messages

### "Multiple values for keyword argument"

**Cause:** Known bug with --time-column and --group-column flags

**Workaround:**
```python
# Use Python API instead of CLI
from mdm.adapters import get_dataset_registrar

registrar = get_dataset_registrar()
registrar.register_dataset(
    name="timeseries",
    path="data.csv",
    time_column="date",
    group_columns=["category"]
)
```

### "Column types mismatch"

**Cause:** Inconsistent data types in CSV

**Solution:**
```python
# Specify dtypes explicitly
import pandas as pd

dtypes = {
    'id': 'int64',
    'amount': 'float64',
    'category': 'object'
}

df = pd.read_csv('data.csv', dtype=dtypes)
```

### "Permission denied"

**Cause:** Insufficient permissions on MDM directory

**Solution:**
```bash
# Fix permissions
chmod -R u+rw ~/.mdm

# Check ownership
ls -la ~/.mdm
```

## Debug Mode

### Enable Comprehensive Debugging

```bash
# Set environment variables
export MDM_LOGGING_LEVEL=DEBUG
export MDM_LOGGING_FILE=/tmp/mdm_debug.log

# Run with profiling
mdm --profile dataset register test.csv

# Check debug log
tail -f /tmp/mdm_debug.log
```

### Performance Profiling

```python
from mdm.utils import Profiler

with Profiler() as prof:
    # Your code here
    pass

prof.print_stats()
prof.save_report("profile_report.html")
```

## Getting Help

### Collect Diagnostic Information

```bash
# Create diagnostic report
mdm debug report > mdm_diagnostic.txt

# Include in bug reports:
# - MDM version
# - Python version
# - Operating system
# - Error messages
# - Steps to reproduce
```

### Resources

1. **Documentation:** https://mdm.readthedocs.io
2. **GitHub Issues:** https://github.com/mdm/mdm-refactor/issues
3. **Community Forum:** https://discuss.mdm.io
4. **Email Support:** support@mdm.io

### Emergency Recovery

If MDM is completely broken:

```bash
# 1. Backup data
cp -r ~/.mdm ~/.mdm_backup

# 2. Clean reinstall
pip uninstall mdm-refactor
rm -rf ~/.mdm/config
pip install mdm-refactor

# 3. Restore data
cp -r ~/.mdm_backup/datasets ~/.mdm/
```

## Appendix: Useful Commands

### Health Check
```bash
# System health check
mdm doctor

# Check specific component
mdm doctor --component storage
mdm doctor --component config
```

### Reset Commands
```bash
# Reset configuration
mdm config reset

# Clear all caches
mdm cache clear --all

# Remove all feature flags
mdm flags reset
```

### Diagnostic Queries
```sql
-- Check dataset tables (SQLite)
SELECT name FROM sqlite_master WHERE type='table';

-- Check row counts
SELECT COUNT(*) FROM dataset_data;

-- Check for corruption
PRAGMA integrity_check;
```