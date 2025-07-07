# Troubleshooting

This guide helps you diagnose and resolve common issues with MDM. Each problem includes symptoms, causes, and solutions.

## Common Issues

### 1. Package Installation Issues with uv

**Symptom:**
```
# Regular pip doesn't work
$ pip install pandas
$ python -c "import pandas"
ModuleNotFoundError: No module named 'pandas'

# Or packages installed with pip are not visible
$ pip list  # shows packages
$ python -c "import mdm"  # fails
```

**Cause:**
The virtual environment was created with `uv venv`, which creates a specialized environment that is NOT compatible with regular pip. Packages must be installed using `uv pip`.

**Solutions:**

```bash
# Correct approach - use uv pip for ALL installations
source .venv/bin/activate
uv pip install pandas
uv pip install -e .

# Verify packages are installed correctly
uv pip list  # Use uv pip list, not pip list
```

**Prevention:**
- Always use `uv pip` instead of `pip` in this project
- Check that you're in the correct virtual environment: `which python` should show `.venv/bin/python`
- Use `uv pip list` to check installed packages, not `pip list`

### 2. Dataset Already Exists

**Symptom:**
```
Error: Dataset 'my_dataset' already exists
```

**Cause:** 
A dataset with the same name is already registered.

**Solutions:**

```bash
# Option 1: Use force flag to overwrite
mdm dataset register my_dataset /path/to/data --force

# Option 2: Choose a different name
mdm dataset register my_dataset_v2 /path/to/data

# Option 3: Remove the existing dataset first
mdm dataset remove my_dataset
mdm dataset register my_dataset /path/to/data
```

### 2. Column Case Sensitivity

**Symptom:**
```
Error: Column 'CustomerID' not found
```

**Cause:** 
MDM handles column names differently in source vs feature tables:
- **Source tables** (`train`, `test`): Preserve original column names
- **Feature tables** (`train_features`, `test_features`): All columns normalized to lowercase

**Solutions:**

```python
# Source tables preserve original case
df_train = dataset_manager.load_table("my_dataset", "train")
customer_id = df_train['CustomerID']  # Original case works

# Feature tables use lowercase for all columns
df_features = dataset_manager.load_table("my_dataset", "train_features")
customer_id = df_features['customerid']  # Must use lowercase

# Generated features also lowercase
age_binned = df_features['age_binned']  # Not 'Age_Binned'

# SQL queries must match the table's convention
# For source tables:
df1 = dataset_manager.query("SELECT CustomerID FROM train")

# For feature tables:
df2 = dataset_manager.query("SELECT customerid FROM train_features")
```

**Note**: This design allows backward compatibility with source data while ensuring consistency in ML pipelines.

### 3. Memory Issues with Large Datasets

**Symptom:**
```
Error: Out of memory
DuckDB Error: Out of Memory Error: failed to allocate memory
```

**Cause:** 
Dataset is too large for available memory.

**Solutions:**

```yaml
# Solution 1: Limit memory usage in config/mdm.yaml
database:
  duckdb:
    memory_limit: "8GB"  # Set appropriate limit
    temp_directory: "/path/to/fast/disk"  # Use SSD for temp files
```

```bash
# Solution 2: Reduce batch size for memory-constrained systems
export MDM_BATCH_SIZE=1000  # Use smaller batches
mdm dataset register large_data /path/to/data

# Solution 3: Skip analysis for faster registration
mdm dataset register large_data /path/to/data \
    --skip-analysis

# Solution 4: MDM automatically handles large files
# If still having issues, consider splitting the data files first
```

```python
# Solution 5: Use chunked processing in code with MDM's batch_size
from mdm.core.config import get_settings

def process_in_chunks(dataset_name, chunk_size=None):
    # Use MDM's configured batch_size
    if chunk_size is None:
        chunk_size = get_settings().batch_size
        
    conn = dataset_manager.get_dataset_connection(dataset_name)
    offset = 0
    
    while True:
        chunk = conn.execute(f"""
            SELECT * FROM train 
            LIMIT {chunk_size} 
            OFFSET {offset}
        """).fetch_df()
        
        if chunk.empty:
            break
            
        # Process chunk
        process_chunk(chunk)
        offset += chunk_size
```

**Batch Processing Tips:**
- Monitor the progress bar to see memory usage patterns
- If progress stalls, reduce batch_size
- For very large datasets (>10GB), use batch_size <= 5000

### 4. Slow Registration

**Symptom:**
Registration takes a very long time or appears to hang.

**Cause:** 
Large files or complex analysis during registration.

**Solutions:**

```bash
# Skip analysis for faster registration
mdm dataset register big_data /path/to/data --skip-analysis

# Register with minimal options, analyze later
mdm dataset register big_data /path/to/data \
    --skip-analysis \
    --no-auto
```

### 5. File Not Found Errors

**Symptom:**
```
Error: File not found: /path/to/train.csv
```

**Cause:**
Incorrect file paths or permissions issues.

**Solutions:**

```bash
# Check file exists and permissions
ls -la /path/to/train.csv

# Use absolute paths
mdm dataset register my_data /absolute/path/to/data

# Check current directory
pwd
mdm dataset register my_data ./relative/path/to/data

# With --no-auto, must specify exact paths
mdm dataset register my_data \
    --no-auto \
    --train /exact/path/to/train.csv \
    --test /exact/path/to/test.csv
```

### 6. Target Column Not Found

**Symptom:**
```
Error: Target column not specified and could not be auto-detected
```

**Cause:**
No sample_submission.csv for auto-detection or ambiguous target.

**Solutions:**

```bash
# Specify target explicitly
mdm dataset register my_data /path/to/data --target "price"

# Check available columns first
head -1 /path/to/data/train.csv
```

### 7. Database Connection Errors

**Symptom:**
```
Error: Unable to connect to database
```

**Cause:**
Database backend misconfiguration or connection issues.

**Solutions:**

```yaml
# Check PostgreSQL configuration in mdm.yaml
database:
  postgresql:
    host: localhost  # Verify host
    port: 5432      # Verify port
    user: mdm_user  # Check credentials
    password: mdm_pass
```

```bash
# Test PostgreSQL connection
psql -h localhost -U mdm_user -d postgres

# For DuckDB/SQLite, check file permissions
ls -la ./datasets/my_dataset/dataset.duckdb
chmod 664 ./datasets/my_dataset/dataset.duckdb
```

### 8. Export Failures

**Symptom:**
```
Error: Failed to export dataset package
```

**Cause:**
Corrupted data or insufficient permissions.

**Solutions:**

```bash
# Verify package integrity
file my_dataset.mdm
tar -tzf my_dataset.mdm  # If it's a tar archive

# Re-export with explicit format
mdm dataset export my_dataset --format csv

# Try manual extraction and registration
tar -xzf my_dataset.mdm
mdm dataset register my_dataset_new ./extracted_data/
```

## Debug Mode

Enable detailed logging to diagnose issues:

### Environment Variable

```bash
# Set log level to DEBUG (most verbose)
export MDM_LOG_LEVEL=DEBUG

# Run command with debug output
mdm dataset register test /path/to/data
```

### Command Line Flag

```bash
# Use debug flag
mdm dataset register test /path/to/data --debug

# Combine with verbose
mdm dataset register test /path/to/data --debug --verbose
```

### Python API Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure MDM logger specifically
mdm_logger = logging.getLogger('mdm')
mdm_logger.setLevel(logging.DEBUG)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
mdm_logger.addHandler(console_handler)
```

## Performance Issues

### Slow Queries

**Problem:** Queries take too long to execute.

**Solutions:**

```python
# Add indexes
with dataset_manager.get_dataset_connection("my_dataset") as conn:
    # Check existing indexes
    indexes = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type = 'index'
    """).fetchall()
    
    # Create indexes on frequently queried columns
    conn.execute("CREATE INDEX idx_date ON train(date_column)")
    conn.execute("CREATE INDEX idx_category ON train(category_column)")
    
    # For composite queries
    conn.execute("CREATE INDEX idx_date_category ON train(date_column, category_column)")
```

### High Memory Usage

**Problem:** MDM uses too much memory.

**Solutions:**

```python
# Use iterative processing
def process_iteratively(dataset_name):
    conn = dataset_manager.get_dataset_connection(dataset_name)
    
    # Use server-side cursor for PostgreSQL
    cursor = conn.cursor()
    cursor.itersize = 10000
    
    cursor.execute("SELECT * FROM train")
    for row in cursor:
        process_row(row)
```

### Disk Space Issues

**Problem:** Running out of disk space.

**Solutions:**

```bash
# Check disk usage
df -h
du -sh ./datasets/*

# Archive old datasets
for dataset in $(mdm dataset list --filter "last_updated_at<180days"); do
    mdm dataset export $dataset --output-dir /archive/$dataset/
    mdm dataset remove $dataset --force
done

# Register with automatic format detection
mdm dataset register my_data /path/to/data
```

## Platform-Specific Issues

### Windows

**Problem:** Path separator issues.

```python
# Use pathlib for cross-platform paths
from pathlib import Path

dataset_path = Path("C:/data/datasets") / "my_dataset"
mdm dataset register my_dataset str(dataset_path)
```

### Linux/Mac

**Problem:** Permission denied errors.

```bash
# Check permissions
ls -la ./datasets/

# Fix permissions
chmod -R 755 ./datasets/
chmod -R 664 ./datasets/*/*.duckdb

# Run with appropriate user
sudo -u mdm_user mdm dataset register my_data /path/to/data
```

### 10. YAML Configuration Issues

**Symptom:**
```
Error: Failed to parse dataset configuration
Dataset 'my_dataset' found in directory but not in configs
```

**Cause:** 
YAML configuration file is missing, corrupted, or has syntax errors.

**Solutions:**

```bash
# Solution 1: Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('~/.mdm/config/datasets/my_dataset.yaml'))"

# Solution 2: Manually create/fix YAML config
cat > ~/.mdm/config/datasets/my_dataset.yaml << EOF
name: my_dataset
database:
  path: ~/.mdm/datasets/my_dataset/dataset.duckdb
tables:
  train: train
  test: test
problem_type: classification
target_column: target
EOF

# Solution 3: List orphaned datasets (have directory but no config)
ls ~/.mdm/datasets/ | while read dataset; do
  if [ ! -f ~/.mdm/config/datasets/${dataset}.yaml ]; then
    echo "Missing config for: $dataset"
  fi
done

# Solution 4: Re-create from existing database
# If database exists but YAML is lost, manually create minimal YAML
```

**Prevention:**
- Always backup YAML configs along with datasets
- Use version control for configuration files
- Validate YAML syntax before committing

### 11. Batch Command Issues

**Symptom:**
```
Error: No such option: --datasets
```

**Cause:** 
Incorrect batch command syntax. Batch commands expect dataset names as positional arguments, not options.

**Solutions:**

```bash
# Correct syntax - dataset names as positional arguments
mdm batch export titanic house_prices --output ./exports/

# Incorrect syntax - using options
mdm batch export --datasets titanic --output ./exports/  # WRONG

# To export with filtering
mdm batch export --filter "problem_type=regression" --output ./exports/
```

**Common Batch Command Patterns:**
```bash
# Multiple specific datasets
mdm batch COMMAND dataset1 dataset2 dataset3 [OPTIONS]

# All datasets
mdm batch COMMAND --all [OPTIONS]

# Filtered datasets
mdm batch COMMAND --filter "CONDITION" [OPTIONS]
```

### 12. Backend Not Changing

**Symptom:**
```
Dataset registered with DuckDB even though I want SQLite
```

**Cause:** 
Backend selection is now controlled exclusively through the mdm.yaml configuration file. CLI parameters for backend selection have been removed.

**Solutions:**

```bash
# Solution 1: Check current backend setting
cat ~/.mdm/mdm.yaml | grep default_backend

# Solution 2: Change backend before registration
vim ~/.mdm/mdm.yaml
# Edit: default_backend: sqlite

# Solution 3: Verify the change
mdm info  # Should show "Default Backend: sqlite"

# Solution 4: Register new dataset (will use SQLite)
mdm dataset register my_dataset /path/to/data
```

**Important Notes:**
- The `--backend` CLI parameter no longer exists
- Backend must be set in mdm.yaml before registration
- Once a dataset is registered, its backend cannot be changed
- Each dataset stores its backend in its own configuration file

### 13. SQLite "Database is Locked" Error

**Symptom:**
```
sqlite3.OperationalError: database is locked
```

**Cause:** 
SQLite doesn't handle concurrent connections well, especially mixing synchronous and asynchronous connections.

**Solutions:**

```yaml
# Solution 1: Enable WAL mode in mdm.yaml
database:
  default_backend: sqlite
  sqlite:
    journal_mode: "WAL"  # Write-Ahead Logging
    synchronous: "NORMAL"
```

```python
# Solution 2: Ensure single connection in code
async with backend.transaction():
    # All operations within same transaction
    await backend.import_dataframe(df, "table_name")
    
# Avoid creating multiple connections
# Don't mix sync and async SQLite connections
```

**Prevention:**
- Use DuckDB for concurrent access scenarios
- Enable WAL mode for better SQLite concurrency
- Keep transactions short and focused

### 14. JSON Export Error with Delimiter

**Symptom:**
```
Error: JSON format does not support delimiters
```

**Cause:** 
JSON is a structured format and doesn't use delimiters.

**Solution:**
```bash
# For JSON, remove the delimiter option
mdm dataset export my_dataset ./output --format json

# If you need custom delimiters, use CSV
mdm dataset export my_dataset ./output --format csv --delimiter "|"
```

### 15. Batch Processing Issues

**Symptom:**
```
Progress bar shows: Processing: ━━━━━━━━━━━━━━━━ 0% [0 rows/s]
OR
Memory usage spikes during batch operations
OR
SQLite operations use different batch size than configured
```

**Cause:** 
Batch size configuration issues or backend limitations.

**Solutions:**

```bash
# Solution 1: Check current batch size setting
mdm info  # Look for batch_size in performance settings

# Solution 2: Override batch size via environment
export MDM_BATCH_SIZE=5000  # Smaller batches for limited memory
mdm dataset register large_data /path/to/data

# Solution 3: Configure permanently in mdm.yaml
echo "performance:
  batch_size: 25000" >> ~/.mdm/mdm.yaml
```

**Backend-Specific Issues:**
- **SQLite**: Currently uses hardcoded batch size of 1,000 (ignores configuration)
- **DuckDB**: Respects configured batch_size
- **PostgreSQL**: Will respect batch_size when implemented

**Progress Bar Interpretation:**
```
# Healthy progress:
Processing: ━━━━━━━━━━━━━━━━ 45% Batch 45/100 [12,500 rows/s, ETA: 2m 15s]

# Stalled progress (reduce batch size):
Processing: ━━━━━━━━━━━━━━━━ 23% Batch 23/100 [0 rows/s, ETA: --:--]

# Memory pressure (very slow speed):
Processing: ━━━━━━━━━━━━━━━━ 67% Batch 67/100 [125 rows/s, ETA: 45m 00s]
```

**Optimization Tips:**
- Start with default batch_size (10,000)
- Increase for better performance if memory allows
- Decrease if you see memory issues or stalled progress
- Monitor first run to find optimal size for your system

## Understanding Log Levels

MDM uses different log levels to distinguish between operational messages and actual errors:

### Operational Messages (Logged as INFO)
These are normal operational conditions, not errors:
- **DatasetNotFoundError** - When a requested dataset doesn't exist
- **DatasetAlreadyExistsError** - When trying to register an existing dataset
- **ValidationWarning** - Data quality issues that don't prevent operation

These messages appear in log files but not on console by default, as they represent expected conditions rather than system failures.

### Actual Errors (Logged as ERROR)
True system errors that require attention:
- **DatabaseConnectionError** - Cannot connect to database backend
- **FilePermissionError** - Cannot read/write required files
- **ConfigurationError** - Invalid configuration preventing operation
- **SystemError** - Unexpected system failures

## Error Messages Reference

| Error | Cause | Solution | Log Level |
|-------|-------|----------|-----------|
| `Dataset not found` | Dataset not registered | Check name with `mdm dataset list` | INFO |
| `Dataset already exists` | Dataset name taken | Use `--force` or remove first | INFO |
| `Invalid configuration` | Malformed YAML | Validate YAML syntax | ERROR |
| `Permission denied` | File permissions | Check and fix permissions | ERROR |
| `Network error` | PostgreSQL connection | Check network and credentials | ERROR |
| `Disk full` | No space left | Clean up or add storage | ERROR |
| `Invalid column type` | Type inference failed | Specify column types manually | WARNING |

## Getting Help

### Check Logs

```bash
# Default log location
tail -f ./logs/mdm.log

# Check for errors
grep ERROR ./logs/mdm.log

# View recent warnings
grep -B 2 -A 2 WARNING ./logs/mdm.log | tail -50
```

### Validate Installation

```python
# Test basic functionality
from mdm import DatasetManager, load_config

try:
    config = load_config()
    dm = DatasetManager(config)
    datasets = dm.list_datasets()
    print(f"MDM is working. Found {len(datasets)} datasets.")
except Exception as e:
    print(f"MDM error: {e}")
```

### Common Recovery Steps

1. **Validate configuration**: Check `mdm.yaml` syntax
2. **Check permissions**: Ensure read/write access to all paths
3. **Verify dependencies**: Reinstall if necessary
4. **Clear temporary files**: Remove temporary files in `~/.mdm/tmp/`
5. **Repair YAML configs**: Fix or recreate dataset configuration files

## Validating Fixes

After resolving issues, validate your fixes using the end-to-end testing scripts:

```bash
# Run comprehensive test
./scripts/test_e2e_nocolor.sh test_dataset ./data/sample

# Quick validation
./scripts/test_e2e_quick.sh test_dataset ./data/sample
```

See [Testing and Validation](13_Testing_and_Validation.md) for detailed information about testing strategies.

## Next Steps

- Review [Best Practices](10_Best_Practices.md) to avoid issues
- See [Summary](12_Summary.md) for key concepts
- Check [Advanced Features](09_Advanced_Features.md) for complex scenarios
- Use [Testing Scripts](13_Testing_and_Validation.md) to validate functionality