# Dataset Management Operations

MDM provides a comprehensive set of operations for managing datasets throughout their lifecycle. Each operation is designed to be intuitive, safe, and efficient.

## Overview of Operations

1. **Register** - Add new datasets
2. **List** - Browse available datasets
3. **Info** - Show detailed information
4. **Search** - Find datasets
5. **Export** - Export datasets
6. **Stats** - Display statistics
7. **Update** - Modify metadata
8. **Remove** - Delete dataset

## 1. Register - Adding New Datasets

The `register` command adds new datasets to MDM with automatic analysis and optimization.

```bash
# Basic registration (auto-detection)
mdm dataset register my_dataset /path/to/dataset/dir

# Registration with overrides
mdm dataset register sales_data /data/sales \
    --target "revenue" \
    --problem-type "regression"

# Manual registration with exact paths
mdm dataset register customer_data \
    --no-auto \
    --train /data/customers/train.parquet \
    --test /data/customers/test.parquet \
    --target "churn"
```

### What happens during registration:
- Validates all specified files exist and are readable
- Auto-detects target and ID columns
- Creates dataset directory: `~/.mdm/datasets/{name}/`
- Imports data into optimized database format
- Stores metadata within the database's own metadata tables
- Generates YAML configuration file in `~/.mdm/config/datasets/`
- No central registry - dataset is now discoverable via directory scanning

### Options:
- `--force`: Overwrite existing dataset
- `--skip-analysis`: Skip statistical analysis for faster registration
- `--dry-run`: Preview what would be registered without saving

## 2. List - Browse Available Datasets

Display all registered datasets by scanning directories and configuration files.

```bash
# List all datasets (default: rich format with colors and tables)
mdm dataset list

# List with different output formats
mdm dataset list --format rich              # Rich format with colors (default)
mdm dataset list --format text              # Plain text output
mdm dataset list --format datasets.txt      # Save to file with rich formatting

# List with filtering (see CLI docs for full filter syntax)
mdm dataset list --filter "problem_type=classification"

# Sort by registration date (default: by name)
mdm dataset list --sort-by registration_date

# Available sort options: name (default), registration_date

# Limit results
mdm dataset list --limit 10
```

### How it works (Directory-based discovery):
1. **Scans YAML directory**: Reads all files in `~/.mdm/config/datasets/`
2. **Parses configurations**: Each YAML file contains dataset metadata
3. **Optional database check**: With `--full` flag, opens each dataset's database
4. **No central registry**: Discovery works purely through directory scanning
5. **Displays results**: Shows datasets found in the directory structure

### Performance Considerations:
The `list` command must parse YAML files and optionally connect to each database to retrieve metadata. For large numbers of datasets:

- **YAML parsing is optimized**: Files are loaded quickly, parsing uses fast C extensions
- **Default mode**: Only parses YAML configs without connecting to databases
- **Full mode** (`--full`): Connects to each database for complete metadata

### Performance optimization strategies:
1. **Fast YAML parsing**: Uses C-based parsers (libYAML) for speed
2. **Parallel processing**: Multiple YAML files parsed concurrently
3. **Minimal parsing**: Only extracts required fields from YAML
4. **OS file optimization**: Repeated list operations benefit from OS optimizations

Expected performance:
- 10 datasets: <10ms
- 100 datasets: <50ms  
- 1000 datasets: <200ms (with parallel parsing)

### Example output:

**Default mode (fast - YAML only):**
```
Scanning ./config/datasets/ for registered datasets...

┌─────────────────┬───────────────────────┬──────────┬──────────┬──────────────┬──────────────┐
│ Name            │ Problem Type          │ Target   │ Tables   │ Total Rows   │ DB Size      │
├─────────────────┼───────────────────────┼──────────┼──────────┼──────────────┼──────────────┤
│ titanic         │ binary-classification │ Survived │ 2        │ ?            │ ?            │
│ house_prices    │ regression           │ SalePrice│ 3        │ ?            │ ?            │
│ playground-s5e7 │ multiclass           │ label    │ 2        │ ?            │ ?            │
└─────────────────┴───────────────────────┴──────────┴──────────┴──────────────┴──────────────┘

Note: Row counts and sizes not available in fast mode. Use --full flag for complete information.
```

**Full mode (slower - queries databases):**
```
mdm dataset list --full

┌─────────────────┬───────────────────────┬──────────┬──────────┬──────────────┬──────────────┐
│ Name            │ Problem Type          │ Target   │ Tables   │ Total Rows   │ DB Size      │
├─────────────────┼───────────────────────┼──────────┼──────────┼──────────────┼──────────────┤
│ titanic         │ binary-classification │ Survived │ 2        │ 1,309        │ 10.3 MB      │
│ house_prices    │ regression           │ SalePrice│ 3        │ 2,919        │ 1.5 MB       │
│ playground-s5e7 │ multiclass           │ label    │ 2        │ 15,000       │ 9.3 MB       │
└─────────────────┴───────────────────────┴──────────┴──────────┴──────────────┴──────────────┘
```

## 3. Info - Detailed Dataset Information

Show comprehensive information about a specific dataset by reading its database.

```bash
# Basic info
mdm dataset info house_prices

# Detailed view with statistics
mdm dataset info house_prices --details
```

### Information displayed:
- Configuration details (from YAML)
- Database location and size
- Table information
- Row and column counts
- Data quality metrics (stored in database)
- Column statistics (computed on demand with --details)

### Example output:
```
Dataset: house_prices
Configuration: ./config/datasets/house_prices.yaml
Database: ./datasets/house_prices/dataset.duckdb (1.5 MB)

Tables:
├── train: 1,460 rows, 81 columns
├── test: 1,459 rows, 80 columns
└── submission: 1,459 rows, 2 columns

Metadata:
├── Problem Type: regression
├── Target: SalePrice
├── ID Columns: [Id]
├── Registered: 2024-03-15 10:30:00
└── Last Modified: 2024-03-15 10:31:45
```

## 4. Search - Find Datasets

Search for datasets by scanning all YAML configuration files and optionally their databases.

```bash
# Search by name (config filenames)
mdm dataset search "titanic"

# Search in descriptions and metadata
mdm dataset search "classification" --deep

# Search with pattern matching
mdm dataset search "playground-s*" --pattern
```

### How search works (without central registry):
1. **Quick search** (default): 
   - Scans all YAML files in `~/.mdm/config/datasets/`
   - Searches in: name, display_name, description fields
   - Performance: O(n) where n = number of datasets
   
2. **Deep search** (`--deep`):
   - Opens each dataset's database file
   - Queries the `_metadata` table for full metadata
   - Searches in all metadata fields
   - Performance: Much slower, requires opening n databases

### Search options:
- `--deep`: Search within dataset metadata (requires opening each database)
- `--pattern`: Use glob patterns for filename matching
- `--case-sensitive`: Enable case-sensitive search
- `--limit N`: Stop after finding N matches

### Performance trade-offs:
- No instant search like with indexed central database
- Search time increases linearly with number of datasets
- Deep search can be very slow with many datasets
- Benefit: No maintenance of search indexes

## 5. Export - Export Datasets

Export dataset tables for sharing or backup. By default, exports ALL tables (train, test, validation, submission, metadata, etc.) as separate files.

```bash
# Export ALL tables to separate CSV files in the specified directory
mdm dataset export titanic --output-dir ./exports/
# Creates: ./exports/titanic_train.csv, ./exports/titanic_test.csv, 
#          ./exports/titanic_metadata.json, etc.

# Export all tables with default CSV format and ZIP compression
mdm dataset export house_prices

# Export to Parquet format with snappy compression
mdm dataset export house_prices --format parquet --compression snappy

# Export ONLY the training data
mdm dataset export customer_churn --table train --format csv

# Export ONLY metadata (no data tables)
mdm dataset export titanic --metadata-only --output metadata.json
```

### Export behavior:
- **Default**: Exports ALL tables (train, test, validation, submission, metadata, etc.) as separate files
- **File naming**: `{dataset_name}_{table_name}.{format}` (e.g., `titanic_train.csv`, `titanic_metadata.json`)

### Export options:
- `--table TABLE`: Export only specific table (default: all tables)
- `--format FORMAT`: Output format: csv (default), parquet, json
- `--output-dir PATH`: Output directory (default: current directory)
- `--compression TYPE`: Compression method (default: zip for CSV)
  - CSV: none, zip (default), gzip
  - Parquet: none, snappy, gzip, lz4
  - JSON: none, zip (default), gzip
- `--no-header`: Exclude header row (CSV only)
- `--metadata-only`: Export only metadata without data tables

## 6. Stats - Dataset Statistics

Display statistical summaries computed from the dataset.

```bash
# Statistics for specific dataset
mdm dataset stats house_prices

# Full statistical analysis
mdm dataset stats house_prices --full

# Export statistics
mdm dataset stats house_prices --export stats.json
```

### Statistics modes:
- **Normal mode**: Basic statistics from pre-computed metadata
- **Full mode**: Detailed analysis including correlations (requires database connection)

### Information includes:
- Row and column counts per table
- Column type distribution
- Missing value analysis
- Basic statistics (mean, std, min, max)
- Correlation matrix (--full mode)
- Cardinality for categorical columns

**Note**: Computing statistics requires querying the actual data, which can be slow for large datasets.

## 7. Update - Modify Dataset Metadata

Update dataset configuration and metadata without re-importing data.

```bash
# Update basic metadata
mdm dataset update house_prices \
    --description "Updated description" \
    --target "NewTarget"

# Update problem type
mdm dataset update customer_churn \
    --problem-type "binary-classification"

# Update column specifications
mdm dataset update sales_data \
    --id-columns "transaction_id,customer_id"
```

### What can be updated:
- Description
- Target column (must exist in data)
- ID columns
- Problem type

### Update process:
1. Updates YAML configuration file
2. Connects to dataset database
3. Updates metadata table
4. Validates changes against actual data

**Note**: Some changes (like target column) may require re-analysis of the dataset.

## 8. Remove - Delete Datasets

Completely remove a dataset, its database, and configuration.

```bash
# Remove with confirmation prompt
mdm dataset remove obsolete_data

# Force removal without confirmation
mdm dataset remove obsolete_data --force

# Dry run to see what would be deleted
mdm dataset remove large_dataset --dry-run
```

### What gets removed:
- Configuration file: `{configs_path}/{name}.yaml`
- Dataset directory: `{datasets_path}/{name}/`
- All database files within the directory
- For PostgreSQL: drops database `{database_prefix}{name}`

### Atomicity:
The remove operation is designed to be atomic:
1. First removes the YAML configuration (prevents dataset from being listed)
2. Then removes the dataset directory and database
3. If step 2 fails, the dataset won't appear in listings (no inconsistent state)

### Example output:
```
Removing dataset: obsolete_data
- Config: ./config/datasets/obsolete_data.yaml
- Database: ./datasets/obsolete_data/dataset.duckdb (25.3 MB)

Are you sure? [y/N]: y
✓ Dataset removed successfully
```

### Safety features:
- Confirmation prompt by default
- Shows size of data to be deleted
- Dry-run mode for preview

## Best Practices

1. **Export Before Remove**: Always export important datasets before removal
2. **Document Updates**: Include meaningful descriptions when updating

## Next Steps

- Learn about [Database Backends](06_Database_Backends.md)
- See [Command Line Interface](07_Command_Line_Interface.md) for detailed options
- Explore [Best Practices](10_Best_Practices.md) for operational tips