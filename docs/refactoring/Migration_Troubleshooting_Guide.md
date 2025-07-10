# Migration Troubleshooting Guide

## Overview
This guide provides solutions for common issues encountered during MDM refactoring migration. Each issue includes symptoms, root causes, diagnostic steps, and resolution procedures.

## Quick Diagnosis Flowchart

```
Migration Issue?
├─> Configuration Error? → Section 2
├─> Storage/Database Issue? → Section 3
├─> Performance Problem? → Section 4
├─> Data Integrity Issue? → Section 5
├─> Feature Engineering Error? → Section 6
├─> Registration Failure? → Section 7
└─> Unknown Issue? → Section 8 (General Diagnostics)
```

## 1. Pre-Migration Issues

### Issue: Backup Failures
**Symptoms**:
- `backup_mdm_state.sh` returns non-zero exit code
- "Permission denied" errors
- "Insufficient space" warnings

**Diagnosis**:
```bash
# Check disk space
df -h ~/.mdm/
df -h /backup/

# Check permissions
ls -la ~/.mdm/
ls -la /backup/

# Test backup process
./scripts/backup_mdm_state.sh --dry-run --verbose
```

**Solutions**:
```bash
# Fix permissions
sudo chown -R $(whoami):$(whoami) ~/.mdm/
chmod -R 755 ~/.mdm/

# Free up space
# Remove old backups
find /backup -name "mdm_backup_*.tar.gz" -mtime +30 -delete

# Use compression
./scripts/backup_mdm_state.sh --compress --level 9

# Use external storage
./scripts/backup_mdm_state.sh --output /mnt/external/backup/
```

### Issue: Incompatible Dependencies
**Symptoms**:
- Import errors after updating requirements
- Version conflict messages
- `pip install` failures

**Diagnosis**:
```bash
# Check current versions
pip list | grep -E "(sqlalchemy|pydantic|typer)"

# Check for conflicts
pip check

# Verify Python version
python --version
```

**Solutions**:
```bash
# Create clean environment
python -m venv .venv_migration
source .venv_migration/bin/activate

# Install with exact versions
pip install -r requirements_migration.txt --no-cache-dir

# If using uv
uv venv --python 3.10
uv pip install -r requirements_migration.txt
```

## 2. Configuration Migration Issues

### Issue: Configuration Not Loading
**Symptoms**:
- `MDMSettingsError: Failed to load configuration`
- Settings reverting to defaults
- Environment variables not recognized

**Diagnosis**:
```python
# Test configuration loading
from mdm.config import get_settings

# Check what's being loaded
settings = get_settings()
print(settings.model_dump_json(indent=2))

# Check environment variables
import os
mdm_vars = {k: v for k, v in os.environ.items() if k.startswith('MDM_')}
print(mdm_vars)
```

**Solutions**:
```bash
# Fix YAML syntax
yamllint ~/.mdm/mdm.yaml

# Validate configuration
mdm config validate

# Reset to defaults
cp ~/.mdm/backup/mdm.yaml ~/.mdm/mdm.yaml

# Debug environment variables
export MDM_DEBUG_CONFIG=true
mdm config show --sources
```

### Issue: Legacy Config Still Active
**Symptoms**:
- Old configuration values being used
- Changes to mdm.yaml not taking effect
- Mixed behavior between old and new systems

**Diagnosis**:
```bash
# Check for legacy config files
find ~/.mdm -name "*.conf" -o -name "*.ini"

# Check which config system is active
python -c "from mdm.config import CONFIG_TYPE; print(CONFIG_TYPE)"

# Look for hardcoded values
grep -r "CONFIG\[" src/mdm/
```

**Solutions**:
```python
# Force new configuration system
export MDM_FORCE_NEW_CONFIG=true

# Clear configuration cache
rm -rf ~/.mdm/cache/config/

# Update code references
# Old way:
# from mdm.config import CONFIG
# value = CONFIG['section']['key']

# New way:
from mdm.config import get_settings
settings = get_settings()
value = settings.section.key
```

## 3. Storage Backend Issues

### Issue: Connection Pool Exhaustion
**Symptoms**:
- `TimeoutError: QueuePool limit exceeded`
- Hanging queries
- "Too many connections" database errors

**Diagnosis**:
```python
# Check pool status
from mdm.storage import get_backend
backend = get_backend('dataset_name')
print(f"Pool size: {backend.pool.size()}")
print(f"Checked out: {backend.pool.checkedout()}")
print(f"Overflow: {backend.pool.overflow()}")

# Monitor connections
watch -n 1 'psql -c "SELECT count(*) FROM pg_stat_activity WHERE application_name = '\''mdm'\''"'
```

**Solutions**:
```python
# Increase pool size
export MDM_DATABASE_POOL_SIZE=30
export MDM_DATABASE_MAX_OVERFLOW=20

# Fix connection leaks
# Always use context managers
with backend.get_connection() as conn:
    result = conn.execute(query)
# Connection automatically returned to pool

# Force connection cleanup
backend.dispose_all_connections()
```

### Issue: SQLite Locking Errors
**Symptoms**:
- `sqlite3.OperationalError: database is locked`
- Slow write operations
- Intermittent failures under load

**Diagnosis**:
```bash
# Check for long-running transactions
lsof | grep "\.db$" | grep mdm

# Test concurrent access
python -c "
import sqlite3
import threading
db_path = '~/.mdm/datasets/test/test.db'
def access_db():
    conn = sqlite3.connect(db_path)
    conn.execute('SELECT 1')
    conn.close()
threads = [threading.Thread(target=access_db) for _ in range(10)]
for t in threads: t.start()
for t in threads: t.join()
"
```

**Solutions**:
```python
# Enable WAL mode
conn.execute("PRAGMA journal_mode=WAL")

# Increase busy timeout
conn.execute("PRAGMA busy_timeout=30000")  # 30 seconds

# Use read-only connections where possible
conn = sqlite3.connect(db_path, uri=True, mode='ro')

# Implement retry logic
import time
import sqlite3

def execute_with_retry(conn, query, max_retries=3):
    for i in range(max_retries):
        try:
            return conn.execute(query)
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and i < max_retries - 1:
                time.sleep(0.1 * (2 ** i))  # Exponential backoff
            else:
                raise
```

### Issue: PostgreSQL Performance Degradation
**Symptoms**:
- Queries taking longer than before migration
- High CPU/memory usage on database server
- Slow connection establishment

**Diagnosis**:
```sql
-- Check slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE mean_exec_time > 1000
ORDER BY mean_exec_time DESC;

-- Check connection count
SELECT count(*), state
FROM pg_stat_activity
WHERE application_name = 'mdm'
GROUP BY state;

-- Check for lock contention
SELECT * FROM pg_locks WHERE NOT granted;
```

**Solutions**:
```bash
# Update PostgreSQL statistics
psql -c "ANALYZE;"

# Tune PostgreSQL for MDM workload
psql -c "ALTER SYSTEM SET work_mem = '64MB';"
psql -c "ALTER SYSTEM SET effective_cache_size = '4GB';"
psql -c "SELECT pg_reload_conf();"

# Add missing indexes
python scripts/analyze_query_performance.py --suggest-indexes

# Enable connection pooling with pgbouncer
apt-get install pgbouncer
# Configure pgbouncer for transaction pooling
```

## 4. Performance Issues

### Issue: Slow Dataset Registration
**Symptoms**:
- Registration taking hours instead of minutes
- Progress bar stuck
- High memory usage during registration

**Diagnosis**:
```python
# Profile registration process
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run registration
mdm.dataset.register("test", "/path/to/data")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

**Solutions**:
```python
# Adjust batch size
export MDM_PERFORMANCE_BATCH_SIZE=5000  # Reduce if memory constrained

# Disable feature generation during registration
export MDM_FEATURES_GENERATE_ON_REGISTER=false

# Use faster backend for large datasets
export MDM_DATABASE_DEFAULT_BACKEND=duckdb

# Optimize data loading
# Add in dataset_registrar.py
def _load_data_optimized(self, file_path: str):
    # Use pyarrow for faster parquet reading
    if file_path.endswith('.parquet'):
        import pyarrow.parquet as pq
        return pq.read_table(file_path).to_pandas()
    
    # Use chunking for large CSV
    if file_path.endswith('.csv'):
        return pd.read_csv(
            file_path,
            chunksize=10000,
            low_memory=False,
            dtype_backend='pyarrow'  # Faster datatypes
        )
```

### Issue: Memory Exhaustion
**Symptoms**:
- `MemoryError` during operations
- System becomes unresponsive
- OOM killer terminates MDM process

**Diagnosis**:
```bash
# Monitor memory usage
htop -p $(pgrep -f mdm)

# Check memory limits
ulimit -a | grep memory

# Profile memory usage
python -m memory_profiler scripts/memory_test.py
```

**Solutions**:
```python
# Implement memory-aware batch processing
class MemoryAwareBatchProcessor:
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        
    def process_in_batches(self, data, batch_size=None):
        import psutil
        
        if batch_size is None:
            # Dynamically adjust batch size based on available memory
            available_mb = psutil.virtual_memory().available / 1024 / 1024
            batch_size = int(available_mb / 10)  # Use 10% of available
            
        for i in range(0, len(data), batch_size):
            # Check memory before processing
            if psutil.virtual_memory().percent > 90:
                import gc
                gc.collect()
                time.sleep(1)
                
            yield data[i:i + batch_size]

# Configure DuckDB memory limits
conn.execute("SET memory_limit='2GB'")
conn.execute("SET temp_directory='/tmp/duckdb_temp'")
```

## 5. Data Integrity Issues

### Issue: Data Corruption After Migration
**Symptoms**:
- Row counts don't match
- Checksum failures
- Missing or altered values

**Diagnosis**:
```python
# Compare data before/after migration
def verify_data_integrity(dataset_name: str):
    # Load from backup
    backup_data = pd.read_parquet(f"/backup/{dataset_name}.parquet")
    
    # Load from new system
    from mdm.api import MDMClient
    client = MDMClient()
    new_data = client.get_dataset(dataset_name).to_pandas()
    
    # Compare
    print(f"Backup rows: {len(backup_data)}")
    print(f"New rows: {len(new_data)}")
    
    # Check for missing columns
    missing_cols = set(backup_data.columns) - set(new_data.columns)
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
    
    # Sample comparison
    sample_size = min(1000, len(backup_data))
    backup_sample = backup_data.sample(sample_size, random_state=42)
    new_sample = new_data.loc[backup_sample.index]
    
    differences = backup_sample.compare(new_sample)
    if not differences.empty:
        print(f"Data differences found:\n{differences}")
```

**Solutions**:
```bash
# Restore from backup
./scripts/restore_dataset.sh dataset_name

# Rerun migration with validation
./scripts/migrate_dataset.sh dataset_name --validate --checksum

# Fix specific data issues
python scripts/fix_data_corruption.py \
    --dataset dataset_name \
    --issue-type encoding \
    --backup-path /backup/dataset_name.parquet
```

### Issue: Type Mismatches
**Symptoms**:
- `TypeError` when accessing data
- Unexpected data types in columns
- String columns that should be numeric

**Diagnosis**:
```python
# Check data types
dataset.dtypes

# Identify type issues
for col in dataset.columns:
    try:
        # Try to convert to expected type
        if should_be_numeric(col):
            pd.to_numeric(dataset[col])
    except ValueError as e:
        print(f"Column {col} has type issues: {e}")
        
# Check for mixed types
def check_mixed_types(df):
    for col in df.columns:
        types = df[col].apply(type).value_counts()
        if len(types) > 1:
            print(f"Column {col} has mixed types: {types.to_dict()}")
```

**Solutions**:
```python
# Fix type inference
type_hints = {
    'user_id': 'string',  # Don't convert IDs to numbers
    'timestamp': 'datetime64[ns]',
    'amount': 'float64',
    'count': 'int64'
}

# Apply during registration
df = pd.read_csv(file_path, dtype=type_hints, parse_dates=['timestamp'])

# Fix existing data
def fix_column_types(dataset_name: str, type_mapping: dict):
    backend = get_backend(dataset_name)
    
    with backend.get_connection() as conn:
        for col, dtype in type_mapping.items():
            if dtype == 'datetime':
                query = f"ALTER TABLE data ALTER COLUMN {col} TYPE TIMESTAMP USING {col}::timestamp"
            elif dtype == 'float':
                query = f"ALTER TABLE data ALTER COLUMN {col} TYPE FLOAT USING {col}::float"
            else:
                query = f"ALTER TABLE data ALTER COLUMN {col} TYPE {dtype}"
                
            conn.execute(query)
```

## 6. Feature Engineering Issues

### Issue: Feature Generation Failures
**Symptoms**:
- `FeatureGenerationError`
- Missing expected features
- Incompatible feature values

**Diagnosis**:
```python
# Test feature generation
from mdm.features import FeatureGenerator

generator = FeatureGenerator()
generator.set_log_level('DEBUG')

# Test on sample
sample_data = dataset.head(100)
features = generator.generate(sample_data)

# Check for errors
for error in generator.get_errors():
    print(f"Feature: {error['feature']}, Error: {error['message']}")
```

**Solutions**:
```python
# Skip problematic features
export MDM_FEATURES_SKIP_ON_ERROR=true
export MDM_FEATURES_SKIP_LIST="feature1,feature2"

# Fix custom feature compatibility
# Old feature format
def old_feature(df):
    return df['col1'] * df['col2']

# New feature format
from mdm.features import custom_feature

@custom_feature(
    name='my_feature',
    input_columns=['col1', 'col2'],
    output_type='float64'
)
def new_feature(df):
    return df['col1'] * df['col2']

# Regenerate features
mdm features regenerate dataset_name --force
```

### Issue: Custom Features Not Loading
**Symptoms**:
- Custom features not appearing in dataset
- "Feature not found" errors
- Features work in test but not production

**Diagnosis**:
```bash
# Check feature discovery
ls -la ~/.mdm/config/custom_features/

# Test feature loading
python -c "
from mdm.features import discover_custom_features
features = discover_custom_features()
print(f'Found {len(features)} custom features')
for f in features:
    print(f'  - {f.name}: {f.module}')
"
```

**Solutions**:
```python
# Fix feature registration
# In ~/.mdm/config/custom_features/__init__.py
from . import my_features

__all__ = ['my_features']

# Register features explicitly
from mdm.features import register_feature
from my_module import my_custom_feature

register_feature(my_custom_feature)

# Debug feature loading
export MDM_FEATURES_DEBUG=true
mdm features list --show-paths
```

## 7. Dataset Registration Issues

### Issue: Registration Pipeline Hangs
**Symptoms**:
- Progress bar stops moving
- No CPU/disk activity
- Process appears deadlocked

**Diagnosis**:
```bash
# Check process state
strace -p $(pgrep -f "mdm dataset register")

# Check for locks
lsof -p $(pgrep -f "mdm dataset register")

# Enable debug logging
export MDM_LOGGING_LEVEL=DEBUG
export MDM_LOGGING_FILE=/tmp/mdm_debug.log
tail -f /tmp/mdm_debug.log
```

**Solutions**:
```python
# Add timeout to steps
from concurrent.futures import TimeoutError
import signal

class TimeoutStep:
    def __init__(self, step, timeout_seconds=300):
        self.step = step
        self.timeout = timeout_seconds
        
    def execute(self, context):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Step {self.step.name} timed out")
            
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            result = self.step.execute(context)
        finally:
            signal.alarm(0)  # Cancel alarm
            
        return result

# Use smaller batches
export MDM_PERFORMANCE_BATCH_SIZE=1000
export MDM_PERFORMANCE_MAX_WORKERS=2
```

### Issue: Metadata Corruption
**Symptoms**:
- `KeyError` when accessing dataset properties
- Inconsistent dataset information
- Registration succeeds but dataset unusable

**Diagnosis**:
```python
# Check metadata integrity
import yaml

meta_path = "~/.mdm/config/datasets/dataset_name.yaml"
with open(meta_path) as f:
    metadata = yaml.safe_load(f)
    
required_fields = ['name', 'path', 'created_at', 'backend', 'version']
missing = [f for f in required_fields if f not in metadata]
print(f"Missing fields: {missing}")

# Validate against schema
from mdm.models import DatasetMetadata
try:
    DatasetMetadata(**metadata)
    print("Metadata is valid")
except Exception as e:
    print(f"Metadata validation failed: {e}")
```

**Solutions**:
```bash
# Rebuild metadata from database
python scripts/rebuild_metadata.py dataset_name

# Fix corrupted YAML
python -c "
import yaml
import json

# Read potentially corrupted YAML
with open('metadata.yaml', 'rb') as f:
    content = f.read().decode('utf-8', errors='ignore')
    
# Try to parse and fix
try:
    data = yaml.safe_load(content)
except yaml.YAMLError:
    # Fall back to more lenient parsing
    import re
    # Extract key-value pairs
    data = {}
    for match in re.finditer(r'^(\w+):\s*(.+)$', content, re.MULTILINE):
        data[match.group(1)] = match.group(2).strip()

# Save clean version
with open('metadata_fixed.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
"
```

## 8. General Diagnostic Tools

### Comprehensive System Check
```bash
#!/bin/bash
# mdm_health_check.sh

echo "=== MDM System Health Check ==="

# 1. Configuration
echo -n "Configuration: "
if mdm config validate &>/dev/null; then
    echo "✓ OK"
else
    echo "✗ FAILED"
    mdm config validate
fi

# 2. Storage backends
echo -n "Storage backends: "
if mdm storage test &>/dev/null; then
    echo "✓ OK"
else
    echo "✗ FAILED"
    mdm storage test --verbose
fi

# 3. Datasets
echo -n "Datasets: "
dataset_count=$(mdm dataset list --format json | jq length)
echo "✓ Found $dataset_count datasets"

# 4. Features
echo -n "Feature system: "
if mdm features test &>/dev/null; then
    echo "✓ OK"
else
    echo "✗ FAILED"
fi

# 5. Performance
echo "Performance metrics:"
mdm benchmark --quick --format json | jq '.summary'

echo "=== Health Check Complete ==="
```

### Debug Information Collection
```python
# collect_debug_info.py
import platform
import sys
import subprocess
import json
from pathlib import Path

def collect_debug_info():
    info = {
        'system': {
            'platform': platform.platform(),
            'python': sys.version,
            'mdm_version': subprocess.getoutput('mdm version'),
        },
        'environment': {
            k: v for k, v in os.environ.items() 
            if k.startswith('MDM_')
        },
        'configuration': subprocess.getoutput('mdm config show --format json'),
        'datasets': subprocess.getoutput('mdm dataset list --format json'),
        'disk_usage': subprocess.getoutput('df -h ~/.mdm'),
        'process_info': subprocess.getoutput('ps aux | grep mdm'),
        'recent_logs': subprocess.getoutput('tail -n 100 ~/.mdm/logs/mdm.log'),
    }
    
    with open('mdm_debug_info.json', 'w') as f:
        json.dump(info, f, indent=2)
        
    print("Debug information saved to mdm_debug_info.json")

if __name__ == '__main__':
    collect_debug_info()
```

## Common Error Messages Reference

| Error Message | Likely Cause | Quick Fix |
|--------------|--------------|-----------|
| `QueuePool limit exceeded` | Connection pool exhaustion | Increase pool_size |
| `database is locked` | SQLite concurrency | Enable WAL mode |
| `OperationalError: no such table` | Migration incomplete | Run migration script |
| `KeyError: 'backend'` | Metadata corruption | Rebuild metadata |
| `MemoryError` | Large dataset processing | Reduce batch_size |
| `TimeoutError: Operation timed out` | Slow queries/network | Check indexes, network |
| `PermissionError: [Errno 13]` | File permissions | Fix ownership/permissions |
| `ModuleNotFoundError: mdm.*` | Installation issue | Reinstall with pip -e |

## Getting Help

If issues persist after trying these solutions:

1. **Collect diagnostic information**:
   ```bash
   python scripts/collect_debug_info.py
   mdm diagnose --comprehensive > diagnosis.txt
   ```

2. **Check known issues**:
   - Review ISSUES.md in the repository
   - Search GitHub issues
   - Check migration logs

3. **Report the issue**:
   - Include diagnostic files
   - Describe steps to reproduce
   - Mention migration stage
   - Include error messages

4. **Emergency support**:
   - Slack: #mdm-migration-support
   - Email: mdm-support@company.com
   - On-call: See Emergency_Rollback_Procedures.md