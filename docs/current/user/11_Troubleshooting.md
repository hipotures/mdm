# MDM Troubleshooting Guide

This comprehensive guide helps you diagnose and resolve common issues with MDM. Each section includes symptoms, causes, and proven solutions.

## Table of Contents

1. [Quick Fixes](#quick-fixes)
2. [Installation Issues](#installation-issues)
3. [Dataset Registration Errors](#dataset-registration-errors)
4. [Performance Problems](#performance-problems)
5. [Backend and Database Issues](#backend-and-database-issues)
6. [Configuration Problems](#configuration-problems)
7. [Feature Engineering Issues](#feature-engineering-issues)
8. [Export and Import Issues](#export-and-import-issues)
9. [Migration and Compatibility](#migration-and-compatibility)
10. [Known Bugs and Limitations](#known-bugs-and-limitations)
11. [Debugging Tools](#debugging-tools)
12. [Getting Help](#getting-help)

## Quick Fixes

### Most Common Issues and Solutions

| Problem | Quick Fix |
|---------|-----------|
| Dataset already exists | `mdm dataset remove NAME --force` then re-register |
| Command not found | `export PATH=$PATH:~/.local/bin` |
| Slow registration | Use `--no-features` flag |
| Backend mismatch | Check `~/.mdm/mdm.yaml` for `default_backend` |
| Memory errors | `export MDM_PERFORMANCE_BATCH_SIZE=5000` |
| No datasets visible | Wrong backend selected - check config |

## Installation Issues

### Issue: MDM Command Not Found

**Symptoms:**
```bash
$ mdm version
bash: mdm: command not found
```

**Solutions:**

1. **Check installation:**
```bash
pip show mdm
pip list | grep mdm
```

2. **Add to PATH:**
```bash
# Add to current session
export PATH=$PATH:~/.local/bin

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
source ~/.bashrc
```

3. **Use Python module syntax:**
```bash
python -m mdm.cli.main version
```

### Issue: Import Error - No module named 'mdm'

**Symptoms:**
```python
ModuleNotFoundError: No module named 'mdm'
```

**Solutions:**

1. **Install in development mode:**
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e .

# Or using pip
pip install -e .
```

2. **Verify installation:**
```bash
python -c "import mdm; print(mdm.__version__)"
python -c "import sys; print(sys.path)"
```

### Issue: Missing Dependencies

**Symptoms:**
```
ERROR: No module named 'duckdb_engine'
ERROR: Could not find a version that satisfies the requirement sqlalchemy>=2.0
```

**Solutions:**

1. **Install missing dependencies:**
```bash
# DuckDB engine
pip install duckdb-engine

# Update pip first
python -m pip install --upgrade pip

# Use uv for better dependency resolution
pip install uv
uv pip install -e .
```

2. **Generate lock file (if missing):**
```bash
uv lock
```

## Dataset Registration Errors

### Issue: Dataset Already Exists

**Symptoms:**
```
Error: Dataset 'sales' already exists
```

**Solutions:**

1. **Use force flag to overwrite:**
```bash
mdm dataset register sales data.csv --force
```

2. **Remove and re-register:**
```bash
mdm dataset remove sales --force
mdm dataset register sales data.csv
```

3. **Use different name:**
```bash
mdm dataset register sales_v2 data.csv
```

### Issue: Path Not Found

**Symptoms:**
```
Error: Path does not exist: /path/to/data.csv
```

**Solutions:**

1. **Use absolute path:**
```bash
mdm dataset register mydata $(pwd)/data.csv
```

2. **Check file exists:**
```bash
ls -la data.csv
file data.csv  # Check encoding
head -5 data.csv  # Preview content
```

3. **Handle spaces in path:**
```bash
mdm dataset register mydata "./path with spaces/data.csv"
```

### Issue: Target Column Not Found

**Symptoms:**
```
Error: Target column 'label' not found in dataset
```

**Solutions:**

1. **List available columns first:**
```bash
# Check column names in CSV
head -1 data.csv | tr ',' '\n'
```

2. **Register with correct column:**
```bash
mdm dataset register mydata data.csv --target correct_column_name
```

3. **Let MDM auto-detect:**
```bash
# Don't specify target, let MDM detect it
mdm dataset register mydata data.csv
```

### Issue: Multiple Values Error (Known Bug)

**Symptoms:**
```
TypeError: register() got multiple values for keyword argument 'time_column'
```

**Workaround:**
```bash
# These flags are broken: --time-column, --group-column, --id-columns
# Register without them, then update later
mdm dataset register timeseries data.csv

# Or use Python API
from mdm.api import MDMClient
client = MDMClient()
client.register_dataset("timeseries", "data.csv", time_column="date")
```

## Performance Problems

### Issue: Slow Registration

**Symptoms:**
- Registration takes minutes for medium files
- Progress bar moves slowly
- High memory usage

**Solutions:**

1. **Skip feature generation (fastest):**
```bash
mdm dataset register large_data data.csv --no-features
```

2. **Increase batch size:**
```bash
# Set for session
export MDM_PERFORMANCE_BATCH_SIZE=50000

# Or in config
# ~/.mdm/mdm.yaml
performance:
  batch_size: 50000
```

3. **Use DuckDB for large datasets:**
```bash
# In ~/.mdm/mdm.yaml
database:
  default_backend: duckdb
```

4. **Monitor progress with debug logging:**
```bash
export MDM_LOGGING_LEVEL=DEBUG
mdm dataset register large_data data.csv
```

### Issue: CLI Startup Slow

**Note:** This was fixed in v0.2.0. CLI now starts in ~0.1 seconds.

**If still experiencing slow startup:**
```bash
# Check version
mdm version

# Update to latest
pip install --upgrade mdm
```

### Issue: Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate array
Out of memory error
```

**Solutions:**

1. **Reduce batch size:**
```bash
export MDM_PERFORMANCE_BATCH_SIZE=1000
```

2. **Use chunked processing:**
```python
from mdm.api import MDMClient
client = MDMClient()

# Process in chunks
for chunk in pd.read_csv('large.csv', chunksize=10000):
    # Process chunk
    pass
```

3. **Switch to disk-based backend:**
```yaml
# ~/.mdm/mdm.yaml
database:
  default_backend: sqlite  # Lower memory than DuckDB
```

## Backend and Database Issues

### Issue: No Datasets Visible After Backend Change

**Symptoms:**
- `mdm dataset list` shows empty
- Datasets exist but not shown

**Explanation:** MDM only shows datasets matching current backend.

**Solutions:**

1. **Check current backend:**
```bash
mdm info | grep backend
cat ~/.mdm/mdm.yaml | grep default_backend
```

2. **Switch back to see datasets:**
```yaml
# ~/.mdm/mdm.yaml
database:
  default_backend: sqlite  # or duckdb, postgresql
```

3. **Re-register for new backend:**
```bash
# Export first
mdm dataset export mydata --format csv

# Change backend, then
mdm dataset register mydata exported_data.csv
```

### Issue: SQLite Database Locked

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**

1. **Enable WAL mode (should be automatic):**
```bash
sqlite3 ~/.mdm/datasets/YOUR_DATASET/dataset.sqlite "PRAGMA journal_mode=WAL;"
```

2. **Kill stuck processes:**
```bash
# Find processes
lsof ~/.mdm/datasets/*/dataset.sqlite

# Kill if needed
kill -9 PID
```

3. **Increase timeout:**
```yaml
# ~/.mdm/mdm.yaml
database:
  sqlite:
    timeout: 30.0
```

### Issue: DuckDB Memory Errors

**Symptoms:**
```
duckdb.OutOfMemoryException: Out of Memory Error
```

**Solutions:**

1. **Increase memory limit:**
```yaml
# ~/.mdm/mdm.yaml
database:
  duckdb:
    settings:
      memory_limit: "8GB"
      threads: 4
```

2. **Enable disk spilling:**
```yaml
database:
  duckdb:
    settings:
      temp_directory: "/tmp/duckdb_temp"
```

### Issue: PostgreSQL Connection Failed

**Symptoms:**
```
psycopg2.OperationalError: could not connect to server
```

**Solutions:**

1. **Check connection parameters:**
```yaml
# ~/.mdm/mdm.yaml
database:
  postgresql:
    host: localhost
    port: 5432
    database: mdm
    user: your_user
    password: your_password
```

2. **Test connection:**
```bash
psql -h localhost -U your_user -d mdm
```

3. **Check service status:**
```bash
# Linux
sudo systemctl status postgresql

# macOS
brew services list | grep postgresql
```

## Configuration Problems

### Issue: Configuration Not Applied

**Symptoms:**
- Changes to mdm.yaml have no effect
- Environment variables ignored

**Solutions:**

1. **Check YAML syntax:**
```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('$HOME/.mdm/mdm.yaml'))"
```

2. **Use correct environment variable format:**
```bash
# Correct format
export MDM_DATABASE_DEFAULT_BACKEND=duckdb
export MDM_PERFORMANCE_BATCH_SIZE=10000
export MDM_LOGGING_LEVEL=DEBUG

# Wrong format (won't work)
export MDM_BACKEND=duckdb
```

3. **Check configuration location:**
```bash
mdm info  # Shows config location and values
```

### Issue: Some Config Options Don't Work

**Known limitations - these are parsed but not fully implemented:**
- SQLAlchemy echo setting
- Some export defaults
- Custom feature settings
- Log file output (partially works)

**Workaround:** Use CLI flags or environment variables instead.

## Feature Engineering Issues

### Issue: Feature Generation Fails

**Symptoms:**
```
Error: Failed to generate features
Features table empty or missing
```

**Solutions:**

1. **Skip features temporarily:**
```bash
mdm dataset register mydata data.csv --no-features
```

2. **Check logs for specific error:**
```bash
export MDM_LOGGING_LEVEL=DEBUG
mdm dataset register mydata data.csv 2>&1 | grep ERROR
```

3. **Disable problematic transformers:**
```yaml
# ~/.mdm/mdm.yaml (if config works)
feature_engineering:
  generic_features:
    text:
      enabled: false
```

### Issue: Custom Features Not Loaded

**Current Status:** Custom feature loading not fully implemented.

**Workaround:**
```python
# Apply features manually after loading
from mdm.api import MDMClient

client = MDMClient()
df = client.load_dataset("mydata")

# Add custom features
df['custom_feature'] = df['col1'] * df['col2']
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])
```

### Issue: Datetime Features Missing

**Problem:** Datetime columns stored as text, not detected.

**Workaround:**
```python
# Convert after loading
df = client.load_dataset("mydata")
df['date'] = pd.to_datetime(df['date'])

# Extract features manually
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
```

## Export and Import Issues

### Issue: Export Creates ZIP Instead of Raw Format

**Behavior:** Export always creates ZIP file containing the requested format.

**Solution:**
```bash
# Export creates ZIP
mdm dataset export mydata ./output --format parquet

# Extract the files
cd output
unzip mydata_export_*.zip

# Files are in requested format inside ZIP
```

### Issue: Large Export Fails

**Solutions:**

1. **Export specific tables:**
```bash
mdm dataset export large_dataset ./output --table train
```

2. **Use Python API for chunked export:**
```python
from mdm.api import MDMClient
client = MDMClient()

# Manual chunked export
df_iterator = pd.read_sql("SELECT * FROM train", 
                         con=engine, 
                         chunksize=10000)

for i, chunk in enumerate(df_iterator):
    chunk.to_csv(f"export_part_{i}.csv", index=False)
```

### Issue: No Import Command

**Current Status:** Direct import not supported.

**Workaround:**
```bash
# Re-register is the only way to "import"
mdm dataset register imported_data /path/to/exported/data.csv
```

## Migration and Compatibility

### Issue: Feature Flags Not Working

**Symptoms:**
- Still using old implementation after enabling flags
- Inconsistent behavior

**Solutions:**

1. **Check flag status:**
```python
from mdm.core import feature_flags

# View current flags
print(feature_flags.get_all())

# Enable specific flag
feature_flags.set("use_new_storage", True)

# Enable all new features
feature_flags.enable_all_new_features()
```

2. **Use environment variables:**
```bash
export MDM_FEATURE_USE_NEW_STORAGE=true
export MDM_FEATURE_USE_NEW_FEATURES=true
```

### Issue: Datasets Lost After Migration

**Prevention:**
```bash
# Always backup first
cp -r ~/.mdm ~/.mdm_backup_$(date +%Y%m%d)

# Verify backup
ls -la ~/.mdm_backup_*/datasets/
```

**Recovery:**
```bash
# Restore from backup
cp -r ~/.mdm_backup_20250111/* ~/.mdm/
```

## Known Bugs and Limitations

### CLI Options That Don't Work

These flags exist but are non-functional:
- `--no-auto` - Manual registration not implemented
- `--force` (for register) - Doesn't overwrite existing
- `--skip-analysis` - Option doesn't exist
- `--dry-run` - Not implemented

### Missing Column Type Options

These don't exist despite being in docs:
- `--datetime-columns`
- `--categorical-columns`
- `--numeric-columns`
- `--text-columns`
- `--ignore-columns`

**Workaround:** MDM auto-detects types, or specify in Python API.

### Type Detection Limitations

| Type | Detection Status |
|------|-----------------|
| Numeric (int, float) | ✅ Works |
| Text | ✅ Works |
| ID columns | ✅ Works (by pattern) |
| Datetime | ❌ Stored as text |
| Categorical | ❌ Stored as text |
| Boolean | ❌ Stored as text |

### Configuration Options Not Implemented

Many options in mdm.yaml are parsed but ignored:
- SQLAlchemy echo setting
- Some export defaults
- Custom feature paths
- Some performance settings

## Debugging Tools

### Enable Debug Logging

```bash
# For one command
MDM_LOGGING_LEVEL=DEBUG mdm dataset register test.csv

# For session
export MDM_LOGGING_LEVEL=DEBUG
export MDM_LOGGING_FILE=/tmp/mdm_debug.log

# With timing info
mdm dataset register test.csv --profile
```

### Direct Database Inspection

```bash
# Open SQLite database
sqlite3 ~/.mdm/datasets/YOUR_DATASET/dataset.sqlite

# Useful SQLite commands
.tables                      # List all tables
.schema train               # Show table structure
SELECT COUNT(*) FROM train; # Row count
SELECT * FROM train LIMIT 5; # Sample data
.mode column                # Better display
.headers on                 # Show column names
```

### Check Dataset Configuration

```bash
# View dataset YAML
cat ~/.mdm/config/datasets/YOUR_DATASET.yaml

# List all datasets
ls ~/.mdm/config/datasets/

# Check dataset directory
ls -la ~/.mdm/datasets/YOUR_DATASET/
du -sh ~/.mdm/datasets/YOUR_DATASET/  # Size
```

### System Health Check

```bash
# MDM info
mdm info

# Version check
mdm version
python -c "import mdm; print(mdm.__version__)"

# Disk space
df -h ~/.mdm/

# Python environment
pip show mdm
python --version
```

## Getting Help

### Before Reporting Issues

1. **Check version:**
```bash
mdm version  # Should be 0.2.0 or later
```

2. **Enable debug logging:**
```bash
export MDM_LOGGING_LEVEL=DEBUG
# Reproduce issue
```

3. **Create minimal example:**
```python
# minimal_repro.py
from mdm.api import MDMClient
client = MDMClient()
# Add minimal code to reproduce issue
```

### Collecting Diagnostic Information

```bash
# System info
mdm info > diagnostic.txt
python --version >> diagnostic.txt
pip freeze | grep -E "(mdm|pandas|sqlalchemy)" >> diagnostic.txt

# Error details
echo "=== ERROR OUTPUT ===" >> diagnostic.txt
MDM_LOGGING_LEVEL=DEBUG mdm [your command] 2>&1 | tail -100 >> diagnostic.txt
```

### Resources

1. **Documentation:**
   - [Table of Contents](00_Table_of_Contents.md)
   - [FAQ](15_FAQ.md)
   - [Environment Variables](16_Environment_Variables.md)

2. **Code & Issues:**
   - GitHub: https://github.com/mdm-project/mdm
   - Issue Tracker: https://github.com/mdm-project/mdm/issues

3. **When Filing Issues Include:**
   - MDM version (`mdm version`)
   - Python version (`python --version`)
   - Complete error message and stack trace
   - Minimal code to reproduce
   - Sample data (if possible)

### Emergency Recovery

If MDM is completely broken:

```bash
# 1. Backup current state
cp -r ~/.mdm ~/.mdm_emergency_backup

# 2. Clean reinstall
pip uninstall mdm
rm -rf ~/.mdm/config
rm -rf ~/.local/lib/python*/site-packages/mdm*
pip install mdm

# 3. Restore datasets only
mkdir -p ~/.mdm
cp -r ~/.mdm_emergency_backup/datasets ~/.mdm/
cp -r ~/.mdm_emergency_backup/config/datasets ~/.mdm/config/
```

## Quick Reference Card

### Most Used Commands

```bash
# Registration
mdm dataset register NAME PATH [--target COL] [--force] [--no-features]

# List and info
mdm dataset list
mdm dataset info NAME
mdm dataset stats NAME

# Export
mdm dataset export NAME [--format csv|parquet] [--output DIR]

# Remove
mdm dataset remove NAME --force

# System
mdm info
mdm version

# Debug mode
MDM_LOGGING_LEVEL=DEBUG mdm [command]
```

### Key Environment Variables

```bash
MDM_DATABASE_DEFAULT_BACKEND=sqlite|duckdb|postgresql
MDM_PERFORMANCE_BATCH_SIZE=10000
MDM_LOGGING_LEVEL=DEBUG|INFO|WARNING|ERROR
MDM_LOGGING_FILE=/path/to/log
MDM_HOME_DIR=/custom/mdm/home
```

### Performance Tips

1. Use `--no-features` for initial exploration
2. Set appropriate batch size for your system
3. Use DuckDB for analytical queries
4. Use PostgreSQL for concurrent access
5. Monitor with `MDM_LOGGING_LEVEL=DEBUG`

---

**Remember:** Many issues are due to known limitations. Check the [Known Bugs](#known-bugs-and-limitations) section first!