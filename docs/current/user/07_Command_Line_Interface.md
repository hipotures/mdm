# Command Line Interface

The MDM command line interface provides comprehensive dataset management capabilities through an intuitive and consistent command structure.

## Command Structure

```bash
# System information
mdm info

# Dataset commands
mdm dataset COMMAND [OPTIONS]

# Batch operations
mdm batch COMMAND [OPTIONS]
```

**Note**: Dataset names are case-insensitive. You can use `titanic`, `Titanic`, or `TITANIC` interchangeably.

## Available Commands

### System Commands

| Command | Description |
|---------|------------|
| `info` | Display system configuration and paths |

### Dataset Commands

| Command | Description |
|---------|------------|
| `register` | Register new dataset |
| `list` | List all datasets |
| `info` | Show dataset information |
| `search` | Search datasets |
| `stats` | Show statistics |
| `update` | Update dataset metadata |
| `remove` | Remove dataset |
| `export` | Export dataset tables to files |

### Batch Commands

| Command | Description |
|---------|------------|
| `export` | Export multiple datasets |

## Parameter Usage

### Important Distinction

- **`NAME`**: Used only with `register` command when creating a new dataset
- **`DATASET_NAME`**: Used with all other commands when referencing an existing dataset

```bash
# Creating a new dataset - uses NAME
mdm dataset register NAME /path/to/data

# All other commands - use DATASET_NAME
mdm dataset info DATASET_NAME
mdm dataset remove DATASET_NAME
mdm batch export DATASET_NAME
```

## System Information

The `mdm info` command displays comprehensive system configuration and status:

```bash
mdm info
```

### Information displayed:

- **MDM Version**: Current version of ML Data Manager
- **Configuration File**: Path to mdm.yaml (e.g., ~/.mdm/mdm.yaml)
- **Default Backend**: Currently configured database backend (sqlite by default, or duckdb, postgresql)
- **Storage Paths**:
  - Datasets directory: ~/.mdm/datasets/
  - Configuration directory: ~/.mdm/config/
- **Database Settings**: Backend-specific configuration from mdm.yaml
- **System Status**:
  - Number of registered datasets
  - Total storage used
  - Available disk space

### Example output:

```
ML Data Manager v1.0.0

Configuration:
  Config file: /home/user/.mdm/mdm.yaml
  Default backend: sqlite

Storage paths:
  Datasets: /home/user/.mdm/datasets/
  Configs: /home/user/.mdm/config/datasets/

Database settings:
  Backend: sqlite
  Memory limit: 8GB
  Threads: 8

System status:
  Registered datasets: 12
  Total storage: 2.4 GB
  Available space: 45.6 GB
```

## Registration Options

### Required Parameters

```bash
NAME                           # Unique dataset identifier (new dataset)
```

### Data Location Options

```bash
PATH                           # Directory path for automatic detection
--train PATH                   # Explicit training data path (with --no-auto)
--test PATH                    # Explicit test data path (with --no-auto)
--validation PATH              # Explicit validation data path (optional)
--submission PATH              # Explicit submission template path (optional)
```

### Column Specifications

```bash
-t, --target TEXT              # Target column for ML prediction (REQUIRED unless Kaggle dataset detected)
-i, --id-columns TEXT          # ID columns to exclude from features (optional, auto-detected if not provided)
--datetime-columns TEXT        # Date/time columns for parsing (optional, auto-detected if not provided)
--categorical-columns TEXT     # Categorical columns for encoding (optional, auto-detected if not provided)
--numeric-columns TEXT         # Force numeric interpretation (optional, auto-detected if not provided)
--text-columns TEXT            # Text columns for NLP (optional, auto-detected if not provided)
--ignore-columns TEXT          # Columns to exclude from feature building (still imported)
```

### ML-Specific Options

```bash
--problem-type TEXT            # Override auto-detected type: regression|classification|multiclass|multilabel|time-series
--stratify-column TEXT         # Column for stratified splitting
--group-column TEXT            # Column for grouped splitting (e.g., user_id)
--time-column TEXT             # Time column for temporal splits
--imbalanced                   # Dataset has class imbalance
```

### Metadata Options

```bash
-d, --description TEXT         # Dataset description
--source TEXT                  # Data source (kaggle, internal, etc.)
--tags TEXT                    # Comma-separated tags
```

**Note**: All metadata is stored locally within each dataset's database. There is no centralized metadata system - each dataset is self-contained with its own metadata. Only datasets matching the current backend configuration are accessible.

### Control Flags

```bash
-f, --force                    # Force overwrite existing (regenerates all features)
--no-auto                      # Disable auto-detection
--skip-analysis                # Skip initial analysis
--dry-run                      # Preview without saving
```

**Important**: The `--force` flag is essential for updating existing datasets:
- Regenerates all feature tables with current configuration
- Applies updated custom transformers from `~/.mdm/custom_features/`
- Preserves original data tables (train, test) unchanged
- See [Updating/Regenerating Features](04_Dataset_Registration.md#updatingregenerating-features) for details

## Export Options

**Default behavior**: Exports ALL tables as separate files, including:
- Source tables: `train`, `test`, `validation`, `submission` (if they exist)
- Feature tables: `train_features`, `test_features` (if feature engineering is enabled)
- Metadata: Dataset configuration and statistics

```bash
--table TABLE            # Export only specific table (default: all tables)
--format FORMAT          # Output format: csv (default), parquet, json
--output-dir PATH        # Output directory (default: current directory)
--compression TYPE       # Compression: zip (default), gzip, none; format-specific
--no-header             # Exclude header row (CSV only)
--metadata-only         # Export only metadata without data tables
--output PATH           # Output file path for single table export
```

## Common Options

Available across multiple commands:

```bash
--format FORMAT                # Output format: rich (default), text, or filename
                              # - rich: Rich formatting with colors and tables
                              # - text: Plain text without formatting
                              # - Any other value: Treated as filename, saves output in rich format
--limit N                      # Limit number of results
--sort-by FIELD                # Sort results by field
--filter EXPR                  # Filter expression
--verbose                      # Enable verbose output
--quiet                        # Minimal output (default behavior)
--debug                        # Enable debug mode (shows configuration values)
```

## Filter Syntax

The `--filter` parameter supports a simple but powerful expression syntax for filtering datasets:

### Basic Syntax

```bash
--filter "field=value"              # Exact match
--filter "field!=value"             # Not equal
--filter "field>value"              # Greater than
--filter "field<value"              # Less than
--filter "field>=value"             # Greater than or equal
--filter "field<=value"             # Less than or equal
--filter "field~pattern"            # Pattern match (supports wildcards)
--filter "field!~pattern"           # Not matching pattern
```

### Supported Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | string | Dataset name | `--filter "name~titanic*"` |
| `problem_type` | string | ML problem type | `--filter "problem_type=regression"` |
| `target_column` | string | Target column name | `--filter "target_column=price"` |
| `row_count` | number | Total number of rows | `--filter "row_count>10000"` |
| `column_count` | number | Number of columns | `--filter "column_count>=50"` |
| `size_mb` | number | Database size in MB | `--filter "size_mb<100"` |
| `registered_at` | date | Registration date | `--filter "registered_at>2024-01-01"` |
| `last_updated_at` | date | Last modification date | `--filter "last_updated_at<2024-12-31"` |
| `source` | string | Data source | `--filter "source=kaggle"` |
| `has_test` | boolean | Has test data | `--filter "has_test=true"` |
| `has_validation` | boolean | Has validation data | `--filter "has_validation=false"` |

**Note**: The `backend` filter is not available because MDM uses a single backend architecture. All visible datasets use the currently configured backend.

### Date Filtering

Date fields support multiple formats:
```bash
--filter "registered_at>2024-01-01"              # ISO date
--filter "registered_at>30days"                  # Relative to now
--filter "last_updated_at<1week"                 # Various units: days, weeks, months
--filter "registered_at>2024-01-01T10:30:00"    # Full timestamp
```

### Multiple Conditions

Combine multiple filters using logical operators:
```bash
--filter "problem_type=regression AND row_count>1000"
--filter "source=kaggle OR size_mb<50"
--filter "problem_type!=classification AND has_test=true"
```

### Pattern Matching

Use wildcards for flexible string matching:
```bash
--filter "name~house*"              # Starts with "house"
--filter "name~*prices"             # Ends with "prices"
--filter "name~*customer*"          # Contains "customer"
--filter "target_column!~*_id"      # Doesn't end with "_id"
```

### Examples

```bash
# Find all regression datasets larger than 100MB
mdm dataset list --filter "problem_type=regression AND size_mb>100"

# Find Kaggle datasets registered in the last 30 days
mdm dataset list --filter "source=kaggle AND registered_at>30days"

# Find datasets with "sales" in the name that have validation data
mdm dataset list --filter "name~*sales* AND has_validation=true"

# Export all classification datasets smaller than 50MB
mdm batch export --filter "problem_type=classification AND size_mb<50" --output ./small_datasets/

# Find datasets not updated in the last 6 months
mdm dataset list --filter "last_updated_at<6months" --sort-by last_updated_at
```

**Note**: All listed datasets use the current backend configured in `mdm.yaml`. Datasets with different backends are not visible.

### Performance Notes

- Filters are applied at the configuration file level before opening databases
- Complex filters may require opening each dataset's database for evaluation
- Use indexed fields (name, problem_type) for best performance
- Date comparisons are performed in UTC

## Example Usage Scenarios

### 1. Minimal Kaggle Registration (Auto-detection)

```bash
# Just provide name and path - MDM figures out the rest
mdm dataset register titanic /data/kaggle/titanic

# Output:
✓ Kaggle competition detected: titanic-disaster
✓ Target column: Survived (from sample_submission.csv)
✓ ID column: PassengerId (from sample_submission.csv)
✓ Problem type: binary-classification (auto-detected)
✓ Created: ~/.mdm/config/datasets/titanic.yaml
✓ Database: ~/.mdm/datasets/titanic/dataset.sqlite
```

### 2. Non-Kaggle Dataset (Target Required)

```bash
# For non-Kaggle datasets, --target is REQUIRED
mdm dataset register customer_churn /data/customers \
    --target "Churn"

# MDM auto-detects the rest:
# - Problem type from target values
# - ID columns from column patterns
# - Column types from data analysis
```

### 3. Manual Registration with No Auto-detection

```bash
# Must specify exact file paths when using --no-auto
mdm dataset register store_sales \
    --no-auto \
    --train /data/retail/train_2023.csv \
    --test /data/retail/test_2023.csv \
    --target "sales_amount" \
    --id-columns "transaction_id,store_id" \
    --datetime-columns "date" \
    --problem-type "regression"

# Error example - directory path not allowed with --no-auto
mdm dataset register bad_example /data/folder --no-auto

# Output:
✗ Error: Directory path not allowed with --no-auto flag.
  Please specify individual file paths:
  mdm dataset register bad_example --no-auto \
    --train /path/to/train.csv \
    --test /path/to/test.csv \
    --target "target_column"
```

### 4. Failed Registration Example

```bash
mdm dataset register unknown /data/mystery

# Output:
✗ Error: Target column is required for non-Kaggle datasets.
  
  Kaggle dataset not detected (missing train.csv, test.csv, or submission file).
  Unable to auto-detect target column.
  
Please specify target column:
  mdm dataset register unknown /data/mystery --target "column_name"
```

### 5. Export Example

```bash
# Export all tables to directory
mdm dataset export titanic --output-dir ./exports/

# Export specific table to parquet format
mdm dataset export house_prices --table train --format parquet --output train_data.parquet

# Dataset files are stored in:
# ~/.mdm/datasets/titanic/dataset.sqlite
# ~/.mdm/config/datasets/titanic.yaml
```

## Advanced Usage Examples

### Complex Registration

```bash
# Full control over registration
mdm dataset register financial_data \
    --train /data/finance/train_2024.parquet \
    --test /data/finance/test_2024.parquet \
    --validation /data/finance/val_2024.parquet \
    --target "default_risk" \
    --id-columns "loan_id,customer_id" \
    --datetime-columns "application_date,disbursement_date" \
    --categorical-columns "loan_type,purpose,region" \
    --numeric-columns "amount,interest_rate,term_months" \
    --problem-type "binary-classification" \
    --imbalanced \
    --stratify-column "region" \
    --description "Loan default prediction dataset" \
    --source "internal" \
    --tags "finance,risk,production"
```

### Dataset Management Workflow

```bash
# 1. Register dataset
mdm dataset register sales_forecast /data/sales \
    --target "next_month_sales" \
    --time-column "date"

# 2. Check registration
mdm dataset info sales_forecast --details

# 3. Generate statistics
mdm dataset stats sales_forecast --full --export stats.json

# 4. Export for sharing
mdm dataset export sales_forecast --output-dir ./export_for_sharing/

# 5. Update metadata
mdm dataset update sales_forecast \
    --description "Updated with Q4 2024 data"

# 6. List all datasets
mdm dataset list --filter "problem_type=regression"
```

### Batch Operations

MDM provides dedicated batch commands for operations on multiple datasets:

```bash
# Export specific datasets
mdm batch export titanic house_prices --output ./exports/ --format parquet

# Export all datasets with specific problem type
mdm batch export --filter "problem_type=regression" --output ./regression_exports/
```

#### Batch Command Syntax

```bash
# General structure
mdm batch COMMAND DATASET_NAMES... [OPTIONS]

# Examples
mdm batch export dataset1 dataset2 --output ./backups/ --format csv
```

#### Legacy Approach (still supported)

```bash
# Using shell loops for batch operations
for dataset in titanic house_prices customer_churn; do
    mdm dataset export $dataset --output-dir ./backups/
done

# Search and process
mdm dataset search "kaggle" --deep | while read dataset; do
    mdm dataset stats $dataset
done
```

## Output Formats

### Rich Format (Default)

```bash
mdm dataset list
# Displays rich formatted table with colors
```

### Text Format

```bash
mdm dataset list --format text
# Plain text output without formatting
```

### File Output

```bash
mdm dataset list --format output.txt
# Saves output to file with rich formatting

mdm dataset info titanic --format report.md
# Saves dataset info to report.md
```

## Error Handling

MDM provides clear error messages with suggested fixes:

```bash
# Missing required parameter
✗ Error: Target column not specified
  Suggestion: Use --target to specify the target column

# Invalid file path
✗ Error: File not found: /path/to/missing.csv
  Suggestion: Check file path and permissions

# Dataset already exists
✗ Error: Dataset 'titanic' already exists
  Suggestion: Use --force to overwrite or choose a different name
```

## Tips and Tricks

1. **Use auto-completion**: Install shell completion for faster command entry
2. **Combine with Unix tools**: Pipe output to `jq`, `grep`, `awk` for advanced filtering
3. **Use dry-run**: Always preview destructive operations with `--dry-run`
4. **Export before remove**: Create backups before deleting datasets
5. **Leverage auto-detection**: Let MDM figure out dataset structure when possible

## Next Steps

- Explore the [Programmatic API](08_Programmatic_API.md) for Python integration
- Learn about [Advanced Features](09_Advanced_Features.md)
- See [Best Practices](10_Best_Practices.md) for CLI usage tips