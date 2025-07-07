# Troubleshooting

This guide helps you resolve common issues when using MDM.

## Installation Issues

### ModuleNotFoundError: No module named 'mdm'

**Problem**: Python cannot find the MDM module.

**Solutions**:
1. Ensure MDM is installed:
   ```bash
   pip install -e .
   ```
2. Check your Python path:
   ```python
   import sys
   print(sys.path)
   ```
3. Verify installation:
   ```bash
   pip list | grep mdm
   ```

### Permission Denied Errors

**Problem**: Cannot write to MDM directories.

**Solutions**:
1. Check directory permissions:
   ```bash
   ls -la ~/.mdm/
   ```
2. Fix permissions:
   ```bash
   chmod -R u+rw ~/.mdm/
   ```
3. Use a different config location:
   ```bash
   export MDM_CONFIG_PATH=/path/to/writable/location
   ```

## Configuration Issues

### Config File Not Found

**Problem**: MDM cannot find or load configuration.

**Solutions**:
1. Create default config:
   ```bash
   mkdir -p ~/.mdm
   mdm info  # This creates default config
   ```
2. Check config location:
   ```python
   from mdm.config import get_config_path
   print(get_config_path())
   ```
3. Specify custom config:
   ```bash
   export MDM_CONFIG_PATH=/path/to/mdm.yaml
   ```

### Invalid Configuration

**Problem**: Configuration file has syntax errors.

**Solutions**:
1. Validate YAML syntax:
   ```bash
   python -c "import yaml; yaml.safe_load(open('~/.mdm/mdm.yaml'))"
   ```
2. Reset to defaults:
   ```bash
   mv ~/.mdm/mdm.yaml ~/.mdm/mdm.yaml.backup
   mdm info  # Creates new default config
   ```

## Dataset Registration Issues

### Dataset Already Exists

**Problem**: Cannot register dataset with existing name.

**Solutions**:
1. Use force flag:
   ```bash
   mdm dataset register my_dataset /path/to/data --force
   ```
2. Choose different name:
   ```bash
   mdm dataset register my_dataset_v2 /path/to/data
   ```
3. Remove existing dataset:
   ```bash
   mdm dataset remove my_dataset --force
   mdm dataset register my_dataset /path/to/data
   ```

### Auto-detection Fails

**Problem**: MDM cannot detect data structure automatically.

**Solutions**:
1. Specify target column:
   ```bash
   mdm dataset register my_dataset /path/to/data --target "target_column"
   ```
2. Check file format:
   ```bash
   file /path/to/data/*.csv
   head -n 5 /path/to/data/train.csv
   ```
3. Use specific file paths:
   ```bash
   mdm dataset register my_dataset /path/to/data \
     --train /path/to/data/training.csv \
     --test /path/to/data/testing.csv
   ```

### Large Dataset Registration Slow

**Problem**: Registration takes too long for large datasets.

**Solutions**:
1. Skip feature generation:
   ```bash
   mdm dataset register large_dataset /path/to/data --no-features
   ```
2. Increase batch size:
   ```bash
   export MDM_BATCH_SIZE=50000
   mdm dataset register large_dataset /path/to/data
   ```
3. Use Parquet format:
   ```bash
   # Convert to Parquet first
   python -c "import pandas as pd; pd.read_csv('data.csv').to_parquet('data.parquet')"
   mdm dataset register large_dataset /path/to/parquet/files
   ```

## Database Backend Issues

### DuckDB Memory Errors

**Problem**: Out of memory errors with DuckDB.

**Solutions**:
1. Increase memory limit in config:
   ```yaml
   database:
     duckdb:
       memory_limit: "8GB"  # Increase from default
   ```
2. Use disk-based operations:
   ```yaml
   database:
     duckdb:
       temp_directory: "/path/to/large/disk"
   ```
3. Switch to SQLite for smaller memory footprint:
   ```yaml
   database:
     default_backend: sqlite
   ```

### SQLite Database Locked

**Problem**: "database is locked" errors.

**Solutions**:
1. Enable WAL mode:
   ```yaml
   database:
     sqlite:
       journal_mode: "WAL"
   ```
2. Close other connections:
   ```python
   # Ensure connections are closed
   backend.close()
   ```
3. Use timeout:
   ```yaml
   database:
     sqlite:
       timeout: 30.0  # seconds
   ```

### PostgreSQL Connection Failed

**Problem**: Cannot connect to PostgreSQL.

**Solutions**:
1. Check connection parameters:
   ```yaml
   database:
     postgresql:
       host: localhost
       port: 5432
       database: mdm
       user: your_user
       password: your_password
   ```
2. Test connection:
   ```bash
   psql -h localhost -U your_user -d mdm
   ```
3. Check PostgreSQL service:
   ```bash
   sudo systemctl status postgresql
   ```

## Export/Import Issues

### Export Fails with Large Datasets

**Problem**: Memory or timeout issues during export.

**Solutions**:
1. Export in chunks:
   ```python
   from mdm.api import MDMClient
   client = MDMClient()
   
   # Process in chunks
   for chunk in client.process_in_chunks("large_dataset", lambda x: x):
       chunk.to_csv(f"export_chunk_{i}.csv", index=False)
   ```
2. Use Parquet format (more efficient):
   ```bash
   mdm dataset export large_dataset --format parquet
   ```
3. Export specific tables:
   ```bash
   mdm dataset export large_dataset --table train --format csv
   ```

### Import Data Type Mismatches

**Problem**: Data types change after import.

**Solutions**:
1. Specify dtypes during registration:
   ```python
   dataset_info = registrar.register(
       name="my_dataset",
       path="/path/to/data",
       dtype_hints={
           'id': 'int64',
           'price': 'float64',
           'category': 'category'
       }
   )
   ```
2. Use Parquet to preserve types:
   ```bash
   # Parquet maintains data types better than CSV
   mdm dataset export my_dataset --format parquet
   ```

## Feature Engineering Issues

### Feature Generation Fails

**Problem**: Error during feature generation.

**Solutions**:
1. Check logs for specific error:
   ```bash
   tail -n 100 ~/.mdm/logs/mdm.log | grep ERROR
   ```
2. Disable problematic transformers:
   ```yaml
   feature_engineering:
     generic_features:
       text:
         enabled: false  # Disable text features
   ```
3. Skip feature generation:
   ```bash
   mdm dataset register my_dataset /path/to/data --no-features
   ```

### Custom Transformer Not Found

**Problem**: Custom feature transformer not loaded.

**Solutions**:
1. Check file location:
   ```bash
   ls ~/.mdm/custom_features/
   # Should see: dataset_name.py
   ```
2. Verify Python syntax:
   ```bash
   python -m py_compile ~/.mdm/custom_features/my_dataset.py
   ```
3. Force regeneration:
   ```bash
   mdm dataset register my_dataset /path/to/data --force
   ```

## Performance Issues

### Slow Query Performance

**Problem**: Queries take too long to execute.

**Solutions**:
1. Create indexes:
   ```python
   with client.get_dataset_connection("my_dataset") as conn:
       conn.execute("CREATE INDEX idx_date ON train(date_column)")
   ```
2. Optimize queries:
   ```python
   # Bad: Loading all data then filtering
   df = pd.read_sql("SELECT * FROM train", conn)
   df_filtered = df[df['category'] == 'A']
   
   # Good: Filter in database
   df_filtered = pd.read_sql(
       "SELECT * FROM train WHERE category = 'A'", 
       conn
   )
   ```
3. Use appropriate backend:
   - DuckDB: Best for analytical queries
   - SQLite: Good for small datasets
   - PostgreSQL: Best for concurrent access

### High Memory Usage

**Problem**: MDM uses too much memory.

**Solutions**:
1. Reduce batch size:
   ```bash
   export MDM_BATCH_SIZE=1000
   ```
2. Process in chunks:
   ```python
   client = MDMClient()
   results = client.process_in_chunks(
       "large_dataset",
       process_func=your_function,
       chunk_size=5000
   )
   ```
3. Use disk-based backends:
   ```yaml
   database:
     default_backend: sqlite  # Lower memory usage
   ```

## CLI Issues

### Command Not Found

**Problem**: `mdm` command not recognized.

**Solutions**:
1. Check installation:
   ```bash
   pip show mdm
   ```
2. Add to PATH:
   ```bash
   export PATH=$PATH:~/.local/bin
   ```
3. Use Python module:
   ```bash
   python -m mdm.cli dataset list
   ```

### Unicode/Encoding Errors

**Problem**: Errors with special characters in output.

**Solutions**:
1. Set encoding:
   ```bash
   export PYTHONIOENCODING=utf-8
   export LC_ALL=en_US.UTF-8
   ```
2. Use ASCII-only output:
   ```bash
   mdm dataset list --no-unicode
   ```

## Debugging Tips

### Enable Debug Logging

```bash
# For single command
MDM_LOG_LEVEL=DEBUG mdm dataset info my_dataset

# For session
export MDM_LOG_LEVEL=DEBUG
```

### Check System Status

```bash
# MDM configuration and status
mdm info

# Python environment
python -c "import mdm; print(mdm.__version__)"
python -c "import sys; print(sys.version)"

# Disk space
df -h ~/.mdm/
```

### Common Log Locations

- MDM logs: `~/.mdm/logs/mdm.log`
- Database logs: Check backend-specific locations
- System logs: `/var/log/syslog` or `journalctl`

## Getting Help

If you cannot resolve an issue:

1. Check the [documentation](https://github.com/your-org/mdm/docs)
2. Search [existing issues](https://github.com/your-org/mdm/issues)
3. Create a new issue with:
   - MDM version: `mdm version`
   - Python version: `python --version`
   - Error message and traceback
   - Steps to reproduce
   - Configuration (sanitized)

## Next Steps

- Review [Best Practices](10_Best_Practices.md) to avoid common issues
- See [Summary](12_Summary.md) for key points about MDM