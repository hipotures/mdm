# Best Practices

Follow these best practices to get the most out of MDM while maintaining a clean, efficient, and scalable dataset management system.

## 0. Language Standards

**All MDM code and documentation must be in English**:

### Code Standards
- **Variable names**: Use descriptive English names (`user_count`, not `liczba_uzytkownikow`)
- **Function names**: Clear English verbs (`calculate_average`, not `oblicz_srednia`)
- **Class names**: English nouns (`DatasetManager`, not `MenedzerDanych`)
- **Comments**: All code comments in English
- **Docstrings**: All function/class documentation in English

### Documentation Standards
- **User messages**: All CLI output, prompts, and messages in English
- **Error messages**: All error descriptions and hints in English
- **Log messages**: All log entries in English for easier debugging
- **Configuration**: All config keys, values, and comments in English

### Development Standards
- **Git commits**: Write commit messages in English
- **Pull requests**: Titles and descriptions in English
- **Code reviews**: Comments and feedback in English
- **Issue tracking**: Bug reports and feature requests in English

This ensures global accessibility and maintainability of the codebase.

## 1. Dataset Naming Conventions

Consistent naming conventions make dataset discovery and management easier.

### Versioning Strategy

Since MDM treats datasets as immutable, use versioning in names:

```bash
# Date-based versioning
sales_20240101
sales_20240201
sales_20240301

# Sequential versioning
customer_data_v1
customer_data_v2
customer_data_v3

# Experiment tracking
titanic_baseline
titanic_cleaned
titanic_final
```

**Benefits**:
- Clear data lineage
- Easy comparison between versions
- Preserves reproducibility
- Avoids accidental data loss

### Recommended Naming Patterns

- **Use lowercase with underscores**: `house_prices_2024` not `HousePrices2024`
- **Include version in name**: `customer_data_v2` or `customer_data_20240315`
- **Prefix with source**: `kaggle_titanic_dataset`, `internal_sales_data`
- **Be descriptive but concise**: `retail_transactions_q4_2023` not `data` or `retail_customer_transaction_data_for_q4_analysis_2023`

### Case-Insensitive Lookups

MDM provides case-insensitive dataset name lookups for convenience:

- **Register with any case**: `mdm dataset register MyDataset /path/to/data`
- **Access with any case**: `mdm dataset info mydataset`, `mdm dataset info MYDATASET`
- **Original case preserved**: Configuration files keep the exact name you provided
- **No duplicates**: Cannot register both `titanic` and `TITANIC` - they're the same dataset

This feature makes MDM more user-friendly, especially when working with datasets created by different team members.

### Examples

```bash
# Good naming examples
mdm dataset register kaggle_titanic_v1 /data/competitions/titanic
mdm dataset register sales_forecast_2024_03 /data/sales/march
mdm dataset register customer_churn_experiment_01 /data/experiments/churn

# Avoid these patterns
mdm dataset register Data1 /data/misc  # Too generic
mdm dataset register CUSTOMER-DATA /data/customers  # Inconsistent case and separators
mdm dataset register final_final_v2_updated /data/processed  # Confusing versioning
```

## 2. File Organization

Organize your MDM files following a clear structure for easy maintenance and backup.

### Recommended Directory Structure

Based on paths configured in `config/mdm.yaml`:

```
project_root/
├── config/
│   ├── mdm.yaml              # Main MDM configuration
│   └── datasets/             # Dataset configuration files (configs_path)
│       ├── titanic.yaml
│       ├── house_prices.yaml
│       └── playground-s5e7.yaml
├── datasets/                 # Dataset databases (datasets_path)
│   ├── titanic/
│   │   └── dataset.duckdb
│   ├── house_prices/
│   │   └── dataset.duckdb
│   ├── playground-s5e7/
│   │   └── dataset.duckdb
│   └── customer_churn/
│       └── dataset.sqlite
└── logs/                     # Log files
    └── mdm.log              # MDM operation logs
```

### Path Configuration

You can customize these paths in `config/mdm.yaml`:

```yaml
storage:
  datasets_path: /data/ml/datasets/    # Absolute path for production
  configs_path: ./configs/ml/          # Relative path for portability
```

### Organization Tips

1. **Group related datasets** in subdirectories
2. **Separate environments**: dev/, staging/, production/
3. **Archive old datasets** periodically
4. **Keep raw data separate** from MDM-managed datasets

## 3. Column Naming

Good column naming practices improve code readability and reduce errors.

### Column Naming Guidelines

- **Avoid spaces and special characters**: Use `customer_age` not `Customer Age`
- **Use consistent case**: Prefer lowercase (all columns will be normalized to lowercase)
- **Be descriptive**: `order_total_usd` not just `total`
- **Indicate units**: `weight_kg`, `distance_miles`, `duration_seconds`
- **Use standard prefixes/suffixes**:
  - `is_` for boolean: `is_active`, `is_verified`
  - `_id` for identifiers: `customer_id`, `transaction_id`
  - `_at` for timestamps: `created_at`, `updated_at`
  - `_count` for counts: `item_count`, `login_count`

### Examples

```python
# Good column names
good_columns = [
    'customer_id',
    'order_date',
    'total_amount_usd',
    'is_premium_member',
    'last_login_at',
    'product_count'
]

# Avoid these
bad_columns = [
    'ID',           # Inconsistent case
    'Date',         # Too generic
    'amt',          # Unclear abbreviation
    'Premium?',     # Special character
    'Last Login',   # Space in name
    '#Products'     # Special character
]
```

## 4. Performance Tips

Optimize MDM performance for your specific use case.

### Choose the Right Backend

Backend selection is configured in `~/.mdm/mdm.yaml` and applies to all new datasets:

```yaml
# For analytical workloads (recommended default)
database:
  default_backend: duckdb
  duckdb:
    memory_limit: "16GB"  # Adjust based on available RAM
    threads: 8            # Match CPU cores

# For small datasets or maximum compatibility
database:
  default_backend: sqlite
  sqlite:
    journal_mode: "WAL"   # Better concurrency

# For enterprise/multi-user environments
database:
  default_backend: postgresql
  postgresql:
    host: db.company.com
    pool_size: 20
```

**Important**: The backend is set at registration time and cannot be changed later. To use different backends for different datasets, update mdm.yaml before each registration.

### Use Parquet Files

```bash
# DuckDB has native Parquet support - very fast
mdm dataset register my_data /path/to/parquet_files

# MDM automatically detects file formats, encodings, and compression
# No need to specify format parameters - just point to your data
mdm dataset register compressed_data /path/to/data.csv.gz
mdm dataset register excel_data /path/to/data.parquet.snappy
```

### Optimize Large Datasets

```bash
# MDM automatically handles large datasets efficiently
mdm dataset register huge_dataset /path/to/data \
    --skip-analysis

# For extremely large datasets, consider splitting into smaller files
# before registration for better performance
```

### Batch Processing Configuration

Optimize batch processing for your system and dataset characteristics:

```bash
# For systems with limited memory (< 8GB RAM)
export MDM_BATCH_SIZE=5000

# For high-memory systems processing large datasets
export MDM_BATCH_SIZE=50000

# Configure in mdm.yaml for permanent settings
performance:
  batch_size: 25000  # Adjust based on your typical dataset size
```

**Batch Size Guidelines:**
- **Small datasets (< 100k rows)**: Use default 10,000
- **Medium datasets (100k - 1M rows)**: Use 25,000-50,000
- **Large datasets (> 1M rows)**: Use 50,000-100,000
- **Limited memory**: Use 1,000-5,000

**Progress Monitoring:**
- MDM shows real-time progress during batch operations
- Monitor memory usage during first runs to find optimal batch size
- Adjust if you see memory pressure or slow processing

### Query Optimization

```python
# Create indexes for frequently queried columns
with dataset_manager.get_dataset_connection("my_dataset") as conn:
    conn.execute("CREATE INDEX idx_date ON train(date_column)")
    conn.execute("CREATE INDEX idx_category ON train(category_column)")
    
# Use efficient queries
# Good: Push filtering to database
df = conn.execute("""
    SELECT * FROM train 
    WHERE category = 'A' AND date >= '2023-01-01'
""").fetch_df()

# Avoid: Loading everything then filtering
df = pd.read_sql("SELECT * FROM train", conn)
df = df[(df['category'] == 'A') & (df['date'] >= '2023-01-01')]
```

## 5. Security Considerations

Protect sensitive data and control access appropriately.

### Data Protection

1. **Store sensitive datasets separately**:
   ```yaml
   # Separate configuration for sensitive data
   storage:
     datasets_path: /secure/ml/datasets/  # Restricted directory
     configs_path: /secure/ml/configs/
   ```

2. **Use PostgreSQL for access control**:
   ```yaml
   database:
     default_backend: postgresql
     postgresql:
       sslmode: require
       sslcert: /path/to/client-cert.pem
   ```

3. **Encrypt database files at rest**:
   ```bash
   # Encrypt DuckDB files
   gpg --encrypt --recipient your-key dataset.duckdb
   ```

4. **Implement access logging**:
   ```python
   import logging
   
   # Log dataset access
   def log_dataset_access(dataset_name, user, operation):
       logging.info(f"Dataset access: {user} {operation} {dataset_name}")
   ```

### Best Practices for Sensitive Data

- Never commit real credentials to version control
- Use environment variables for passwords
- Anonymize PII before registration
- Implement row-level security in PostgreSQL
- Regular audit of dataset access

## 6. Maintenance

Regular maintenance keeps your MDM installation running smoothly.

### Feature Regeneration

Regenerate features when needed:

```bash
# After updating custom transformer
vim ~/.mdm/custom_features/titanic.py
mdm dataset register titanic /path/to/kaggle/titanic --force

# After changing feature configuration
vim ~/.mdm/mdm.yaml  # Update feature_engineering settings
mdm dataset register house_prices /path/to/data --force

# Batch regeneration for multiple datasets
for dataset in titanic house_prices customer_churn; do
    echo "Regenerating features for $dataset..."
    mdm dataset register $dataset /original/path/$dataset --force
done
```

**Important**: Always test custom transformers before regenerating:
```python
# Test transformer independently
python -m pytest ~/.mdm/custom_features/titanic.py
```

### Regular Tasks

```bash
# Quarterly: Archive and remove old datasets
for dataset in $(mdm dataset list --filter "last_updated_at<90days"); do
    mdm dataset export $dataset --output-dir archives/$dataset/
    mdm dataset remove $dataset --force
done
```

### Backup Procedures

```bash
# Backup configurations
cp -r ./config/datasets/ ./backups/configs_$(date +%Y%m%d)/

# Backup datasets
rsync -av ./datasets/ ./backups/datasets_$(date +%Y%m%d)/

# Export critical datasets
mdm dataset export important_dataset --output-dir backups/important_dataset/
```

### Monitoring

```python
# Monitor dataset sizes
def check_dataset_sizes():
    """Alert if datasets grow too large"""
    for dataset in dataset_manager.list_datasets():
        size_gb = dataset.database_size / (1024**3)
        if size_gb > 100:
            alert(f"Dataset {dataset.name} is {size_gb:.1f}GB")
```

## Workflow Best Practices

### Dataset Registration Workflow

1. **Check source data** before registration
2. **Use meaningful names** following conventions
3. **Document the dataset** with good descriptions
4. **Verify registration** with `mdm dataset info`
### Development Workflow

```bash
# 1. Register development dataset
mdm dataset register dev_experiment_01 /data/dev \
    --description "Testing new feature engineering"

# 2. Iterate and experiment
python experiments/feature_engineering.py

# 3. Register successful result
mdm dataset register features_v1 /data/processed \
    --description "Engineered features from experiment_01"

# 4. Clean up experiments
mdm dataset remove dev_experiment_01 --force
```

### Production Workflow

1. **Test thoroughly** before production
2. **Use version control** for configurations
3. **Document all changes**
4. **Test migrations** in staging first
5. **Keep audit trail** of all operations

## Common Pitfalls to Avoid

1. **Don't use spaces in names** - They cause issues in scripts
2. **Don't ignore disk space** - Remove unused datasets regularly
3. **Don't mix environments** - Keep dev/prod separate
4. **Don't forget backups** - Before major changes

## Integration Best Practices

### With Version Control

```bash
# Track configurations but not data
echo "datasets/" >> .gitignore
git add config/datasets/*.yaml
git commit -m "Add dataset configurations"
```

### With CI/CD

```yaml
# Example GitHub Actions workflow
- name: Generate Dataset Stats
  run: |
    mdm dataset stats --all --export stats.json
```

### With ML Pipelines

```python
# Standardize dataset loading
class DatasetLoader:
    @staticmethod
    def load_for_training(dataset_name):
        """Standard method for loading training data"""
        info = dataset_manager.get_dataset(dataset_name)
        train_df = dataset_manager.load_table(dataset_name, "train")
        
        # Standard preprocessing
        X = train_df.drop(columns=[info.target_column] + info.id_columns)
        y = train_df[info.target_column]
        
        return X, y, info
```

## Logging Best Practices

Understanding and configuring MDM's logging system helps with debugging and monitoring.

### Log Levels and Output

```bash
# Default behavior
- Console: Shows only WARNING and ERROR messages
- File: Logs all messages at INFO level and above to ~/.mdm/logs/mdm.log

# Enable verbose console output
export MDM_LOG_LEVEL=DEBUG
mdm dataset register my_data /path/to/data

# Or use the verbose flag
mdm dataset register my_data /path/to/data --verbose
```

### Understanding Log Messages

MDM distinguishes between operational conditions and actual errors:

```bash
# These are INFO level (normal operations):
- Dataset not found when trying to access
- Dataset already exists when registering
- Validation warnings during data import

# These are ERROR level (system failures):
- Database connection failures
- File permission errors
- Configuration syntax errors
```

### Log File Management

```bash
# Monitor logs in real-time
tail -f ~/.mdm/logs/mdm.log

# Search for specific operations
grep "dataset_register" ~/.mdm/logs/mdm.log

# Find errors only
grep "ERROR" ~/.mdm/logs/mdm.log

# Rotate logs periodically
# Add to crontab for automatic rotation
0 0 * * 0 mv ~/.mdm/logs/mdm.log ~/.mdm/logs/mdm.log.$(date +%Y%m%d) && touch ~/.mdm/logs/mdm.log
```

### Debug Mode for Troubleshooting

When encountering issues, enable debug mode for detailed information:

```bash
# Set debug mode for one command
MDM_LOG_LEVEL=DEBUG mdm dataset info titanic

# Enable debug mode for session
export MDM_LOG_LEVEL=DEBUG

# Check what MDM is doing
mdm dataset register complex_dataset /path/to/data --dry-run
```

## Testing and Validation

After implementing changes or completing documentation, always validate functionality:

```bash
# Full end-to-end test
./scripts/test_e2e_nocolor.sh validation_test ./data/sample

# Quick smoke test
./scripts/test_e2e_quick.sh quick_test ./data/sample

# Test with specific backend
echo "database:\n  default_backend: sqlite" > ~/.mdm/mdm.yaml
./scripts/test_e2e_nocolor.sh sqlite_test ./data/sample
```

See [Testing and Validation](13_Testing_and_Validation.md) for comprehensive testing guide.

## Next Steps

- Review [Troubleshooting](11_Troubleshooting.md) for common issues
- Read the [Summary](12_Summary.md) for key takeaways
- Explore [Advanced Features](09_Advanced_Features.md) for complex use cases
- Use [Testing Scripts](13_Testing_and_Validation.md) to validate your setup
