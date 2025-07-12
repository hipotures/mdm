# CLI Reference

Complete reference for all MDM command-line interface commands.

## Global Options

These options can be used with any MDM command:

```bash
mdm [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
|--------|-------------|
| `--help` | Show help message and exit |
| `--version` | Show MDM version and exit |
| `--config PATH` | Use custom config file |
| `--debug` | Enable debug logging |
| `--no-color` | Disable colored output |

## Core Commands

### mdm version

Display MDM version information.

```bash
mdm version
```

**Output:**
```
MDM version 1.0.0
```

### mdm info

Display system configuration and environment information.

```bash
mdm info
```

**Output:**
```
MDM System Information
═══════════════════════════════════════════════════════════
Version: 1.0.0
Python: 3.11.8
Platform: Linux-6.15.3-1-MANJARO-x86_64

Configuration:
  Config file: /home/user/.mdm/mdm.yaml
  Data directory: /home/user/.mdm
  Default backend: sqlite
  
Environment:
  MDM_DATABASE_DEFAULT_BACKEND: not set
  MDM_PERFORMANCE_BATCH_SIZE: not set
  MDM_LOGGING_LEVEL: not set
```

## Dataset Commands

### mdm dataset register

Register a new dataset with MDM.

```bash
mdm dataset register [OPTIONS] NAME PATH
```

**Arguments:**
- `NAME`: Unique name for the dataset
- `PATH`: Path to data file or directory

**Options:**
| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--target` | TEXT | Target column for ML | None |
| `--id-columns` | TEXT | Comma-separated ID columns | Auto-detect |
| `--problem-type` | CHOICE | ML problem type | Auto-detect |
| `--tags` | TEXT | Comma-separated tags | None |
| `--description` | TEXT | Dataset description | None |
| `--force` | FLAG | Overwrite if exists | False |
| `--no-features` | FLAG | Skip feature generation | False |
| `--categorical-columns` | TEXT | Force columns as categorical | None |
| `--datetime-columns` | TEXT | Force columns as datetime | None |
| `--numeric-columns` | TEXT | Force columns as numeric | None |
| `--text-columns` | TEXT | Force columns as text | None |

**Problem Types:**
- `classification`: Binary or multiclass classification
- `regression`: Continuous target prediction
- `clustering`: Unsupervised learning
- `time-series`: Temporal prediction
- `ranking`: Learning to rank

**Examples:**
```bash
# Basic registration
mdm dataset register iris iris.csv

# With ML configuration
mdm dataset register titanic ./titanic \
    --target Survived \
    --id-columns PassengerId \
    --problem-type classification

# Force column types
mdm dataset register sales data.csv \
    --categorical-columns region,product_type \
    --datetime-columns order_date,ship_date \
    --text-columns notes,description

# With metadata
mdm dataset register customers customers.parquet \
    --description "Customer data from CRM export" \
    --tags "production,customers,2024"
```

### mdm dataset list

List all registered datasets.

```bash
mdm dataset list [OPTIONS]
```

**Options:**
| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--limit` | INT | Maximum datasets to show | None (all) |
| `--sort-by` | CHOICE | Sort datasets by | registration_date |
| `--reverse` | FLAG | Reverse sort order | False |
| `--filter-backend` | TEXT | Show only specific backend | Current |
| `--all-backends` | FLAG | Show all backends | False |

**Sort Options:**
- `name`: Alphabetical by name
- `registration_date`: By creation date (default)
- `size`: By database size
- `rows`: By row count

**Examples:**
```bash
# List all datasets
mdm dataset list

# Show only 10 most recent
mdm dataset list --limit 10

# Sort by size, largest first
mdm dataset list --sort-by size --reverse

# Show datasets from all backends
mdm dataset list --all-backends
```

### mdm dataset info

Display detailed information about a dataset.

```bash
mdm dataset info [OPTIONS] NAME
```

**Arguments:**
- `NAME`: Dataset name

**Options:**
| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--json` | FLAG | Output as JSON | False |
| `--show-columns` | FLAG | List all columns | False |
| `--show-features` | FLAG | List generated features | False |

**Examples:**
```bash
# Basic info
mdm dataset info titanic

# Show all columns
mdm dataset info titanic --show-columns

# JSON output for scripting
mdm dataset info titanic --json | jq '.row_count'
```

### mdm dataset stats

Display statistical summary of a dataset.

```bash
mdm dataset stats [OPTIONS] NAME
```

**Arguments:**
- `NAME`: Dataset name

**Options:**
| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--table` | TEXT | Specific table to analyze | train |
| `--columns` | TEXT | Specific columns to analyze | All |
| `--extended` | FLAG | Show extended statistics | False |
| `--json` | FLAG | Output as JSON | False |

**Examples:**
```bash
# Basic statistics
mdm dataset stats sales

# Extended statistics with percentiles
mdm dataset stats sales --extended

# Specific columns only
mdm dataset stats sales --columns "price,quantity"

# Analyze test data
mdm dataset stats titanic --table test
```

### mdm dataset search

Search for datasets by name or tags.

```bash
mdm dataset search [OPTIONS] [PATTERN]
```

**Arguments:**
- `PATTERN`: Search pattern (optional)

**Options:**
| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--tag` | TEXT | Filter by tag | None |
| `--problem-type` | TEXT | Filter by problem type | None |
| `--min-rows` | INT | Minimum row count | None |
| `--max-rows` | INT | Maximum row count | None |

**Examples:**
```bash
# Search by name pattern
mdm dataset search sales

# Find all classification datasets
mdm dataset search --problem-type classification

# Find by tag
mdm dataset search --tag kaggle

# Complex search
mdm dataset search customer --tag production --min-rows 1000
```

### mdm dataset update

Update dataset metadata.

```bash
mdm dataset update [OPTIONS] NAME
```

**Arguments:**
- `NAME`: Dataset name

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--description` | TEXT | New description |
| `--target` | TEXT | New target column |
| `--problem-type` | CHOICE | New problem type |
| `--tags` | TEXT | New tags (replaces all) |
| `--add-tags` | TEXT | Add tags to existing |
| `--remove-tags` | TEXT | Remove specific tags |

**Examples:**
```bash
# Update description
mdm dataset update sales --description "2024 Q1 sales data"

# Change target column
mdm dataset update sales --target revenue

# Manage tags
mdm dataset update sales --add-tags "validated,q1-2024"
mdm dataset update sales --remove-tags "draft"
```

### mdm dataset export

Export dataset in various formats.

```bash
mdm dataset export [OPTIONS] NAME
```

**Arguments:**
- `NAME`: Dataset name

**Options:**
| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--format` | CHOICE | Export format | csv |
| `--output` | PATH | Output path | ./{name}_export.{ext} |
| `--compression` | CHOICE | Compression type | None |
| `--table` | TEXT | Table to export | All |
| `--include-features` | FLAG | Include generated features | True |
| `--columns` | TEXT | Specific columns to export | All |

**Formats:**
- `csv`: Comma-separated values
- `parquet`: Apache Parquet
- `json`: JSON lines
- `excel`: Excel workbook (requires openpyxl)

**Compression:**
- `gzip`: Gzip compression (.gz)
- `zip`: ZIP archive
- `bz2`: Bzip2 compression
- `xz`: XZ compression

**Examples:**
```bash
# Export as CSV
mdm dataset export titanic

# Export as compressed Parquet
mdm dataset export titanic --format parquet --compression gzip

# Export specific columns
mdm dataset export sales \
    --columns "date,product,price,quantity" \
    --output clean_sales.csv

# Export without features
mdm dataset export titanic --no-include-features
```

### mdm dataset remove

Remove a dataset from MDM.

```bash
mdm dataset remove [OPTIONS] NAME
```

**Arguments:**
- `NAME`: Dataset name

**Options:**
| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--force` | FLAG | Skip confirmation | False |
| `--dry-run` | FLAG | Show what would be removed | False |

**Examples:**
```bash
# Remove with confirmation
mdm dataset remove old_data

# Force removal
mdm dataset remove old_data --force

# Preview removal
mdm dataset remove large_dataset --dry-run
```

## Batch Commands

### mdm batch export

Export multiple datasets at once.

```bash
mdm batch export [OPTIONS] DATASETS
```

**Arguments:**
- `DATASETS`: Comma-separated dataset names or patterns

**Options:**
| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--format` | CHOICE | Export format | csv |
| `--output-dir` | PATH | Output directory | ./exports |
| `--compression` | CHOICE | Compression type | None |
| `--parallel` | FLAG | Export in parallel | False |

**Examples:**
```bash
# Export specific datasets
mdm batch export sales,customers,products

# Export with pattern
mdm batch export "sales_*" --format parquet

# Parallel export
mdm batch export "*" --parallel --output-dir /data/exports
```

### mdm batch stats

Generate statistics for multiple datasets.

```bash
mdm batch stats [OPTIONS] DATASETS
```

**Arguments:**
- `DATASETS`: Comma-separated dataset names or patterns

**Options:**
| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--output` | PATH | Save stats to file | None (stdout) |
| `--format` | CHOICE | Output format | table |

**Examples:**
```bash
# Stats for all datasets
mdm batch stats "*"

# Save to file
mdm batch stats "train_*" --output stats_report.json --format json
```

### mdm batch remove

Remove multiple datasets.

```bash
mdm batch remove [OPTIONS] DATASETS
```

**Arguments:**
- `DATASETS`: Comma-separated dataset names or patterns

**Options:**
| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--force` | FLAG | Skip confirmation | False |
| `--dry-run` | FLAG | Preview removal | False |

**Examples:**
```bash
# Remove test datasets
mdm batch remove "test_*" --dry-run

# Force remove old datasets
mdm batch remove "archive_*" --force
```

## Time Series Commands

### mdm timeseries

Time series specific operations.

```bash
mdm timeseries [COMMAND] [OPTIONS]
```

**Subcommands:**
- `split`: Create time-based train/test splits
- `validate`: Validate time series data
- `resample`: Resample time series data

**Examples:**
```bash
# Create time split
mdm timeseries split stock_data --test-days 30

# Validate time series
mdm timeseries validate sales --time-column date

# Resample to daily frequency
mdm timeseries resample sales --freq D --agg mean
```

## System Commands

### mdm stats

View system-wide statistics and monitoring data.

```bash
mdm stats [OPTIONS]
```

**Options:**
| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--system` | FLAG | Show system metrics | False |
| `--operations` | FLAG | Show operation history | False |
| `--performance` | FLAG | Show performance metrics | False |

**Examples:**
```bash
# System overview
mdm stats --system

# Recent operations
mdm stats --operations

# Performance metrics
mdm stats --performance
```

## Advanced Usage

### Environment Variables

MDM commands respect environment variables for configuration:

```bash
# Change default backend for session
export MDM_DATABASE_DEFAULT_BACKEND=duckdb
mdm dataset register large_data data.parquet

# Increase batch size for large dataset
export MDM_PERFORMANCE_BATCH_SIZE=50000
mdm dataset register huge_data massive.csv

# Enable debug logging
export MDM_LOGGING_LEVEL=DEBUG
mdm dataset info problematic_data
```

### Command Chaining

Combine MDM commands with shell features:

```bash
# Register all CSVs in directory
for f in data/*.csv; do
    mdm dataset register "$(basename $f .csv)" "$f"
done

# Export all datasets to parquet
mdm dataset list --json | \
    jq -r '.datasets[].name' | \
    xargs -I {} mdm dataset export {} --format parquet

# Find large datasets
mdm dataset list --json | \
    jq '.datasets[] | select(.size_mb > 100) | .name'
```

### Scripting with MDM

Example bash script using MDM:

```bash
#!/bin/bash
# prepare_ml_datasets.sh

# Configuration
BACKEND="duckdb"
EXPORT_FORMAT="parquet"

# Set backend
export MDM_DATABASE_DEFAULT_BACKEND=$BACKEND

# Register datasets
echo "Registering datasets..."
mdm dataset register train train.csv --target label
mdm dataset register test test.csv

# Generate statistics report
echo "Generating statistics..."
mdm batch stats "train,test" --output stats.json --format json

# Export for ML pipeline
echo "Exporting datasets..."
mdm batch export "train,test" \
    --format $EXPORT_FORMAT \
    --output-dir ./ml_ready \
    --compression gzip

echo "Datasets prepared successfully!"
```

## Error Codes

MDM uses standard exit codes:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Dataset not found |
| 4 | Dataset already exists |
| 5 | Permission denied |
| 6 | Backend error |
| 7 | Configuration error |

## Getting Help

```bash
# General help
mdm --help

# Command-specific help
mdm dataset register --help

# Subcommand help
mdm dataset --help
```