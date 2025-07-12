# Troubleshooting Guide

This guide helps you resolve common issues with MDM. For each problem, we provide symptoms, causes, and solutions.

## Installation Issues

### Python Version Error

**Symptom:**
```
ERROR: MDM requires Python 3.9 or higher. Current version: 3.8.10
```

**Solution:**
```bash
# Option 1: Upgrade system Python
sudo apt update && sudo apt install python3.9

# Option 2: Use pyenv
curl https://pyenv.run | bash
pyenv install 3.11.8
pyenv global 3.11.8

# Option 3: Use conda
conda create -n mdm python=3.11
conda activate mdm
```

### Missing Dependencies

**Symptom:**
```
ModuleNotFoundError: No module named 'mdm'
ImportError: cannot import name 'MDMClient' from 'mdm'
```

**Solution:**
```bash
# Reinstall with all dependencies
pip uninstall mdm-ml
pip install mdm-ml[all]

# Or for development
git clone https://github.com/hipotures/mdm.git
cd mdm
pip install -e ".[dev]"
```

### C++ Compiler Error (Windows)

**Symptom:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution:**
1. Download Visual Studio Build Tools from [Microsoft](https://visualstudio.microsoft.com/downloads/)
2. Install with "Desktop development with C++" workload
3. Restart terminal and retry installation

## Dataset Registration Issues

### Path Not Found

**Symptom:**
```
DatasetError: Path not found: /path/to/data.csv
```

**Solution:**
```bash
# Check path exists
ls -la /path/to/data.csv

# Use absolute path
mdm dataset register mydata $(pwd)/data.csv

# Check permissions
chmod 644 data.csv
```

### Dataset Already Exists

**Symptom:**
```
DatasetError: Dataset 'mydata' already exists
```

**Solution:**
```bash
# Option 1: Use different name
mdm dataset register mydata_v2 data.csv

# Option 2: Force overwrite
mdm dataset register mydata data.csv --force

# Option 3: Remove old dataset
mdm dataset remove mydata
mdm dataset register mydata data.csv
```

### Memory Error During Registration

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Reduce batch size
export MDM_PERFORMANCE_BATCH_SIZE=5000
mdm dataset register large_data data.csv

# Skip features for now
mdm dataset register large_data data.csv --no-features

# Use DuckDB for better memory handling
export MDM_DATABASE_DEFAULT_BACKEND=duckdb
mdm dataset register large_data data.csv
```

### Column Type Detection Issues

**Symptom:**
```
Warning: Column 'date' detected as TEXT, expected DATETIME
```

**Solution:**
```bash
# Force column types
mdm dataset register mydata data.csv \
    --datetime-columns "date,created_at" \
    --categorical-columns "status,category" \
    --numeric-columns "amount,quantity"
```

## Backend Issues

### SQLite Locked Error

**Symptom:**
```
sqlite3.OperationalError: database is locked
```

**Solution:**
```python
# In config
database:
  sqlite:
    journal_mode: WAL  # Better concurrency
    timeout: 30        # Wait longer for locks

# Or close other connections
client.close()
# Then retry
```

### DuckDB Memory Error

**Symptom:**
```
OutOfMemoryException: Not enough memory available
```

**Solution:**
```yaml
# Limit DuckDB memory usage
database:
  duckdb:
    memory_limit: 4GB  # Set based on available RAM
    temp_directory: /path/with/space  # For spill files
```

### PostgreSQL Connection Error

**Symptom:**
```
psycopg2.OperationalError: could not connect to server
```

**Solution:**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify connection details
psql -h localhost -U mdm_user -d mdm_db

# Update config
export MDM_DATABASE_POSTGRESQL_HOST=localhost
export MDM_DATABASE_POSTGRESQL_PORT=5432
export MDM_DATABASE_POSTGRESQL_PASSWORD=yourpassword
```

### Backend Not Available

**Symptom:**
```
BackendError: Backend 'duckdb' not available
```

**Solution:**
```bash
# Install backend-specific dependencies
pip install duckdb

# For PostgreSQL
pip install psycopg2-binary

# Verify installation
python -c "import duckdb; print('DuckDB OK')"
```

## Query and Performance Issues

### Slow Queries

**Symptom:**
Queries taking too long to execute

**Solution:**
```python
# 1. Create indexes
with client.get_connection("dataset") as conn:
    conn.execute("CREATE INDEX idx_date ON train(date)")
    conn.execute("CREATE INDEX idx_category ON train(category)")
    conn.execute("ANALYZE")  # Update statistics

# 2. Use query optimization
# Bad: Load all then filter
df = client.load_dataset("large")
filtered = df[df['category'] == 'A']

# Good: Filter in database
filtered = client.query.query(
    dataset="large",
    filters={"category": "A"}
)

# 3. Increase cache
export MDM_PERFORMANCE_CACHE_MAX_SIZE=10000
```

### Out of Memory

**Symptom:**
```
MemoryError during data loading
```

**Solution:**
```python
# 1. Process in chunks
def process_large_dataset(name, chunk_size=10000):
    for chunk in client.iterate_dataset(name, chunk_size):
        process(chunk)
        del chunk  # Free memory

# 2. Optimize dtypes
df = client.load_dataset("data")
df = optimize_dtypes(df)  # Convert to efficient types

# 3. Use specific columns
df = client.load_dataset("data", columns=["id", "value"])
```

## Feature Engineering Issues

### Features Not Generated

**Symptom:**
No features created during registration

**Solution:**
```bash
# 1. Check feature configuration
cat ~/.mdm/mdm.yaml | grep -A5 features

# 2. Enable features
export MDM_FEATURES_ENABLE_AT_REGISTRATION=true

# 3. Re-register with features
mdm dataset register mydata data.csv --force

# 4. Check for errors in logs
export MDM_LOGGING_LEVEL=DEBUG
mdm dataset register mydata data.csv
```

### Custom Features Not Loading

**Symptom:**
Custom feature transformer not applied

**Solution:**
```bash
# 1. Check file location
ls ~/.mdm/config/custom_features/mydata.py

# 2. Verify class name
grep "class CustomFeatureOperations" ~/.mdm/config/custom_features/mydata.py

# 3. Check for syntax errors
python ~/.mdm/config/custom_features/mydata.py

# 4. Enable custom features
export MDM_FEATURES_CUSTOM_ENABLED=true
export MDM_FEATURES_CUSTOM_AUTO_DISCOVER=true
```

### Feature Generation Hanging

**Symptom:**
Feature generation stuck or very slow

**Solution:**
```yaml
# 1. Reduce parallel workers
features:
  n_jobs: 2  # Instead of -1

# 2. Skip expensive features
features:
  text:
    enabled: false  # Text features are slow
  temporal:
    enable_cyclical: false  # Skip sin/cos encoding

# 3. Limit categorical encoding
features:
  categorical:
    max_cardinality: 20  # Reduce from 50
```

## CLI Issues

### Command Not Found

**Symptom:**
```
bash: mdm: command not found
```

**Solution:**
```bash
# 1. Check installation
pip show mdm-ml

# 2. Add to PATH
export PATH=$PATH:~/.local/bin

# 3. Use python -m
python -m mdm.cli dataset list

# 4. Reinstall
pip uninstall mdm-ml
pip install mdm-ml
```

### Slow CLI Startup

**Symptom:**
CLI takes several seconds to start

**Solution:**
```bash
# 1. Check for import issues
python -X importtime -m mdm.cli version

# 2. Use fast commands
mdm version  # Optimized fast path

# 3. Disable progress bars
export MDM_CLI_PROGRESS_BAR_STYLE=none
```

### Output Formatting Issues

**Symptom:**
Garbled or no color output

**Solution:**
```bash
# 1. Disable colors
mdm --no-color dataset list

# 2. Fix terminal encoding
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# 3. Use different terminal
# Some terminals don't support Rich formatting
```

## Configuration Issues

### Config Not Loading

**Symptom:**
Settings not being applied

**Solution:**
```bash
# 1. Check config location
mdm info  # Shows config path

# 2. Validate YAML
python -c "import yaml; yaml.safe_load(open('~/.mdm/mdm.yaml'))"

# 3. Check environment variables
env | grep MDM_

# 4. Debug config loading
export MDM_DEBUG_CONFIG=true
mdm info
```

### Invalid Configuration

**Symptom:**
```
ValidationError: Invalid configuration
```

**Solution:**
```yaml
# Check common issues:

# 1. Indentation (must be spaces, not tabs)
database:
  default_backend: sqlite  # 2 spaces

# 2. Valid values
database:
  default_backend: sqlite  # not 'sqllite'

# 3. Correct types
performance:
  batch_size: 10000  # number, not "10000"
```

## Data Issues

### Encoding Errors

**Symptom:**
```
UnicodeDecodeError: 'utf-8' codec can't decode
```

**Solution:**
```python
# 1. Specify encoding
df = pd.read_csv("data.csv", encoding='latin-1')

# 2. Try different encodings
encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
for enc in encodings:
    try:
        df = pd.read_csv("data.csv", encoding=enc)
        break
    except UnicodeDecodeError:
        continue

# 3. Handle errors
df = pd.read_csv("data.csv", encoding_errors='ignore')
```

### Data Type Conflicts

**Symptom:**
```
ValueError: could not convert string to float
```

**Solution:**
```python
# 1. Check problematic values
pd.read_csv("data.csv", nrows=5)

# 2. Force string type first
df = pd.read_csv("data.csv", dtype=str)
# Then convert selectively
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

# 3. Handle mixed types
df = pd.read_csv("data.csv", low_memory=False)
```

## Export Issues

### Export Fails

**Symptom:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# 1. Check directory permissions
ls -la ~/exports/

# 2. Use different location
mdm dataset export mydata --output /tmp/export.csv

# 3. Create directory first
mkdir -p ~/exports
chmod 755 ~/exports
```

### Wrong Format

**Symptom:**
Exported file is corrupted or wrong format

**Solution:**
```bash
# 1. Specify format explicitly
mdm dataset export mydata --format parquet

# 2. Check compression
mdm dataset export mydata --format csv --compression none

# 3. Verify export
file exported_data.parquet
```

## Common Error Messages

### "Multiple values for keyword argument"

**Known Bug:** Some CLI options cause this error

**Workaround:**
```bash
# Instead of:
mdm dataset register data.csv --id-columns "id1,id2"

# Use Python API:
client.register_dataset("mydata", "data.csv", id_columns=["id1", "id2"])
```

### "No module named 'ydata_profiling'"

**Solution:**
```bash
pip install ydata-profiling
# Or reinstall MDM with all dependencies
pip install mdm-ml[all]
```

### "Connection pool is full"

**Solution:**
```yaml
# Increase pool size
database:
  postgresql:
    pool_size: 20
    max_overflow: 40
```

## Debug Mode

For difficult issues, enable debug mode:

```bash
# Maximum debugging
export MDM_LOGGING_LEVEL=DEBUG
export MDM_LOGGING_FILE=/tmp/mdm_debug.log
export MDM_CLI_SHOW_TRACEBACK=true
export MDM_DEBUG_CONFIG=true

# Run problematic command
mdm dataset register problem_data.csv

# Check logs
tail -f /tmp/mdm_debug.log
```

## Getting Help

### 1. Check Logs
```bash
# Default log location
tail -f ~/.mdm/logs/mdm.log

# With debug info
tail -f ~/.mdm/logs/mdm_debug.log
```

### 2. Version Information
```bash
mdm version
mdm info  # Full system info
```

### 3. Report Issues
Include this information when reporting bugs:
- MDM version (`mdm version`)
- Python version (`python --version`)
- Operating system
- Full error message and traceback
- Steps to reproduce

### 4. Community Support
- GitHub Issues: https://github.com/hipotures/mdm/issues
- Discussions: https://github.com/hipotures/mdm/discussions

## Recovery Procedures

### Corrupted Dataset

```bash
# 1. Backup current state
cp -r ~/.mdm/datasets/corrupted ~/.mdm/datasets/corrupted_backup

# 2. Try to export what's readable
mdm dataset export corrupted --output backup.csv || true

# 3. Remove and re-register
mdm dataset remove corrupted --force
mdm dataset register corrupted original_data.csv
```

### Reset MDM

```bash
# Complete reset (removes all data!)
rm -rf ~/.mdm
mdm info  # Creates fresh config
```

### Database Repair

```bash
# SQLite
sqlite3 ~/.mdm/datasets/mydata/data.db "PRAGMA integrity_check"
sqlite3 ~/.mdm/datasets/mydata/data.db "VACUUM"

# PostgreSQL
psql -d mdm_db -c "REINDEX DATABASE mdm_db"
psql -d mdm_db -c "VACUUM ANALYZE"
```