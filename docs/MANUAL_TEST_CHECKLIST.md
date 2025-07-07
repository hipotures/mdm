# MDM Manual Test Checklist

This checklist is for manual verification of MDM functionality. Check off items as you test them.

## Prerequisites

- [ ] Install uv if not already installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] Create virtual environment: `uv venv`
- [ ] Activate virtual environment: `source .venv/bin/activate`
- [ ] Install MDM using `uv pip`: `uv pip install -e .`
- [ ] Verify uv.lock file is present for reproducible installs
- [ ] Test that regular pip fails in uv environment (expected behavior)
- [ ] Verify packages are visible: `uv pip list | grep mdm`

## 1. Configuration System

### 1.1 YAML Configuration File
- [ ] Create `~/.mdm/mdm.yaml` with custom settings
- [ ] Run MDM and verify if settings from YAML are applied
- [ ] Modify mdm.yaml and check if changes take effect
- [ ] Delete mdm.yaml and verify MDM still works with defaults
- [ ] Test with invalid YAML syntax
- [ ] Test with unknown configuration keys

### 1.2 Environment Variables
- [ ] Set `MDM_LOG_LEVEL=DEBUG` and verify debug output appears
- [ ] Set `MDM_DATABASE_DEFAULT_BACKEND=sqlite` and register a dataset
- [ ] Set `MDM_DATABASE_DEFAULT_BACKEND=duckdb` and register another dataset
- [ ] Set `MDM_BATCH_SIZE=5000` and verify it's used in operations
- [ ] Set `MDM_MAX_CONCURRENT_OPERATIONS=2` and verify parallelism limit
- [ ] Set `MDM_MDM_HOME=/custom/path` and verify directory creation
- [ ] Set `MDM_DATASETS_PATH=/custom/datasets` and verify usage
- [ ] Set `MDM_DEFAULT_BACKEND=postgresql` and test

### 1.3 Database Backend Configuration

#### 1.3.1 SQLite Configuration
- [ ] Register dataset with SQLite backend
- [ ] Verify `.sqlite` file is created
- [ ] Check if WAL mode is enabled: `journal_mode: "WAL"`
- [ ] Verify `synchronous: "NORMAL"` setting is applied
- [ ] Test `cache_size: -64000` configuration
- [ ] Test `temp_store: "MEMORY"` setting
- [ ] Test `mmap_size: 268435456` setting

#### 1.3.2 DuckDB Configuration
- [ ] Register dataset with DuckDB backend
- [ ] Verify `.duckdb` file is created
- [ ] Check if `memory_limit` from config is applied (not hardcoded 4GB)
- [ ] Verify `threads` setting is used
- [ ] Test `temp_directory` configuration
- [ ] Test `access_mode: "READ_WRITE"` setting

#### 1.3.3 PostgreSQL Configuration
- [ ] Configure PostgreSQL connection settings
- [ ] Attempt to register dataset with PostgreSQL
- [ ] Verify connection string is built correctly
- [ ] Test `database_prefix: mdm_` setting
- [ ] Test SSL settings (`sslmode`, `sslcert`, `sslkey`)
- [ ] Check connection pooling settings
- [ ] Verify database creation with prefix

#### 1.3.4 SQLAlchemy Configuration
- [ ] Test `sqlalchemy.echo: true` - verify SQL queries are logged
- [ ] Test `sqlalchemy.pool_size` setting
- [ ] Test `sqlalchemy.max_overflow` setting
- [ ] Test `sqlalchemy.pool_timeout` setting
- [ ] Test `sqlalchemy.pool_recycle` setting

### 1.4 Logging Configuration
- [ ] Set log level to DEBUG and verify verbosity
- [ ] Set log level to ERROR and verify only errors show
- [ ] Test `logging.format: json` vs `logging.format: text`
- [ ] Configure log file path and verify file is created
- [ ] Test log rotation with `max_bytes` setting
- [ ] Test `backup_count` for number of rotated files
- [ ] Verify console output vs file logging levels differ

### 1.5 Performance Configuration
- [ ] Test `performance.batch_size` setting
- [ ] Test `performance.max_concurrent_operations` limit
- [ ] Verify batch size is used in operations
- [ ] Test concurrent operation limits

### 1.6 Comprehensive Configuration Value Testing

#### 1.6.1 Debug Mode Configuration Display
- [ ] Set `MDM_LOG_LEVEL=DEBUG` and run `mdm info`
- [ ] Verify debug output shows "Loading configuration from: ~/.mdm/mdm.yaml"
- [ ] Verify debug output shows all loaded configuration values
- [ ] Verify debug output shows which values are from YAML vs defaults
- [ ] Run with `--debug` flag and verify same output

#### 1.6.2 Parameter-by-Parameter Testing
For each parameter in mdm.yaml.default, perform:

**Database Parameters:**
- [ ] Set `database.default_backend: duckdb`, run `mdm info --debug`, verify loaded value
- [ ] Change to `sqlite`, run again, verify change is reflected
- [ ] Test `database.connection_timeout: 60`, verify in debug output
- [ ] Test `database.sqlalchemy.echo: true`, verify SQL queries are printed
- [ ] Test `database.sqlalchemy.pool_size: 20`, verify in debug output

**SQLite Parameters:**
- [ ] Set `database.sqlite.journal_mode: DELETE`, run debug, verify loaded
- [ ] Test each journal_mode option: [DELETE, TRUNCATE, PERSIST, MEMORY, WAL, OFF]
- [ ] Set `database.sqlite.cache_size: -128000`, verify change (default: -64000)
- [ ] Set `database.sqlite.synchronous: FULL`, test all options: [OFF, NORMAL, FULL, EXTRA]
- [ ] Set `database.sqlite.temp_store: FILE`, test options: [DEFAULT, FILE, MEMORY]

**DuckDB Parameters:**
- [ ] Set `database.duckdb.memory_limit: "16GB"`, verify in debug output
- [ ] Set `database.duckdb.threads: 8`, verify change from default (4)
- [ ] Set `database.duckdb.access_mode: READ_ONLY`, verify in debug

**Performance Parameters:**
- [ ] Set `performance.batch_size: 50000`, run `mdm info --debug`, verify loaded
- [ ] Register dataset and verify batch operations use this value
- [ ] Set `performance.max_concurrent_operations: 10`, verify in debug

**Logging Parameters:**
- [ ] Set `logging.level: WARNING`, verify only warnings/errors appear
- [ ] Test all levels: [DEBUG, INFO, WARNING, ERROR, CRITICAL]
- [ ] Set `logging.format: console`, verify output format changes
- [ ] Set `logging.file: custom.log`, verify file is created
- [ ] Set `logging.max_bytes: 1048576` (1MB), verify rotation happens

**Export Parameters:**
- [ ] Set `export.default_format: parquet`, export dataset, verify format
- [ ] Test all formats: [csv, parquet, json]
- [ ] Set `export.compression: gzip`, verify compression is applied
- [ ] Test all compressions: [none, zip, gzip, snappy, lz4]

**Validation Parameters:**
- [ ] Set `validation.before_features.check_duplicates: false`, verify no duplicate check
- [ ] Set `validation.before_features.drop_duplicates: false`, verify duplicates kept
- [ ] Set `validation.before_features.signal_detection: true`, verify warnings
- [ ] Set `validation.after_features.signal_detection: false`, verify all features kept
- [ ] Change `validation.*.max_missing_ratio: 0.5`, verify stricter validation
- [ ] Change `validation.*.min_rows: 1000`, verify minimum enforced

**CLI Parameters:**
- [ ] Set `cli.default_output_format: json`, run `mdm list`, verify JSON output
- [ ] Test all formats: [rich, json, csv, table]
- [ ] Set `cli.confirm_destructive: false`, verify no confirmation prompts
- [ ] Set `cli.show_progress: false`, verify no progress bars
- [ ] Set `cli.page_size: 50`, verify pagination changes

**Feature Engineering Parameters:**
- [ ] Set `feature_engineering.enabled: false`, register dataset, verify no features
- [ ] Set `feature_engineering.generic.temporal.enabled: false`, verify no temporal features
- [ ] Set `feature_engineering.generic.categorical.max_cardinality: 10`, verify limit
- [ ] Set `feature_engineering.generic.statistical.outlier_threshold: 2.0`, verify change

#### 1.6.3 Configuration Precedence Testing
- [ ] Set value in mdm.yaml and environment variable, verify env var wins
- [ ] Example: Set `batch_size: 5000` in YAML, `MDM_BATCH_SIZE=10000` in env
- [ ] Run `mdm info --debug`, verify shows "10000 (overridden by environment)"
- [ ] Remove env var, verify falls back to YAML value
- [ ] Remove YAML value, verify falls back to default

#### 1.6.4 Configuration Validation
- [ ] Test invalid backend name, verify error message
- [ ] Test invalid log level, verify error message
- [ ] Test negative batch_size, verify validation error
- [ ] Test invalid paths, verify error or directory creation
- [ ] Test mutually exclusive options

#### 1.6.5 Live Configuration Reload Testing
- [ ] Start operation with one config, change mdm.yaml during execution
- [ ] Verify if changes take effect immediately or need restart
- [ ] Test which settings require restart vs immediate effect

## 2. Dataset Operations

### 2.1 Dataset Registration

#### 2.1.1 Basic Registration
- [ ] Register dataset with minimal command: `mdm dataset register name /path`
- [ ] Register with case-insensitive names (test, Test, TEST)
- [ ] Register with explicit backend selection
- [ ] Register with spaces in dataset name
- [ ] Register with special characters in name
- [ ] Register with Unicode characters in name

#### 2.1.2 Column Specifications
- [ ] Register with `--target` column specification
- [ ] Register with `--id-columns` (multiple IDs)
- [ ] Register with `--datetime-columns` specification
- [ ] Register with `--categorical-columns` specification
- [ ] Register with `--numeric-columns` specification
- [ ] Register with `--text-columns` specification
- [ ] Register with `--ignore-columns` specification

#### 2.1.3 ML-Specific Options
- [ ] Register with `--problem-type` override
- [ ] Register with `--stratify-column` for stratified splitting
- [ ] Register with `--group-column` for grouped splitting
- [ ] Register with `--time-column` for temporal splits
- [ ] Register with `--imbalanced` flag
- [ ] Register with `--validation-split` ratio

#### 2.1.4 Metadata Options
- [ ] Register with `--description` text
- [ ] Register with `--source` specification
- [ ] Register with `--tags` (comma-separated)

#### 2.1.5 Control Flags
- [ ] Test `--force` flag to overwrite existing
- [ ] Test `--no-auto` flag with manual settings
- [ ] Test `--skip-analysis` for faster registration
- [ ] Test `--dry-run` for preview without saving

#### 2.1.6 Multi-file Registration
- [ ] Register with separate train/test files
- [ ] Register with train/test/validation files
- [ ] Register with submission template file
- [ ] Register from directory with mixed formats
- [ ] Test with non-existent file paths

### 2.2 Dataset Listing and Filtering

#### 2.2.1 Basic Listing
- [ ] Run `mdm dataset list` and verify output
- [ ] Test `--limit N` parameter
- [ ] Test `--format text` output
- [ ] Test `--format rich` output (default)
- [ ] Test `--format output.txt` (save to file)

#### 2.2.2 Sorting Options
- [ ] Test `--sort-by name`
- [ ] Test `--sort-by registered_at`
- [ ] Test `--sort-by size_mb`
- [ ] Test `--sort-by row_count`
- [ ] Test `--sort-by last_updated_at`

#### 2.2.3 Simple Filters
- [ ] Test `--filter "name=titanic"` (exact match)
- [ ] Test `--filter "name!=titanic"` (not equal)
- [ ] Test `--filter "row_count>1000"`
- [ ] Test `--filter "row_count<10000"`
- [ ] Test `--filter "row_count>=5000"`
- [ ] Test `--filter "size_mb<=100"`

#### 2.2.4 Pattern Matching Filters
- [ ] Test `--filter "name~test*"` (starts with)
- [ ] Test `--filter "name~*test"` (ends with)
- [ ] Test `--filter "name~*test*"` (contains)
- [ ] Test `--filter "name!~*temp*"` (not matching)

#### 2.2.5 Date Filters
- [ ] Test `--filter "registered_at>2024-01-01"`
- [ ] Test `--filter "registered_at>30days"` (relative)
- [ ] Test `--filter "last_updated_at<1week"`
- [ ] Test `--filter "registered_at>2024-01-01T10:30:00"`

#### 2.2.6 Combined Filters
- [ ] Test with AND: `--filter "problem_type=regression AND row_count>1000"`
- [ ] Test with OR: `--filter "source=kaggle OR size_mb<50"`
- [ ] Test complex: `--filter "problem_type!=classification AND has_test=true"`

### 2.3 Dataset Information and Statistics

#### 2.3.1 Dataset Info
- [ ] Run `mdm dataset info <name>` for existing dataset
- [ ] Test with `--details` flag for extended information
- [ ] Test `--format` options (text, rich, filename)
- [ ] Verify all metadata is displayed
- [ ] Check file size calculation

#### 2.3.2 Dataset Statistics
- [ ] Run `mdm dataset stats <name>`
- [ ] Test `--full` flag for detailed statistics
- [ ] Test `--export stats.json` to save statistics
- [ ] Test `--export stats.csv` with different format
- [ ] Verify column statistics accuracy

### 2.4 Dataset Search
- [ ] Run `mdm dataset search <pattern>`
- [ ] Test `--pattern` flag for complex patterns
- [ ] Test `--case-sensitive` flag
- [ ] Test `--limit N` to stop after N matches
- [ ] Test `--deep` flag for content search

### 2.5 Dataset Update
- [ ] Update description: `mdm dataset update <name> --description "New text"`
- [ ] Update tags: `--tags "tag1,tag2,tag3"`
- [ ] Update problem type: `--problem-type regression`
- [ ] Update target column: `--target new_target`
- [ ] Verify changes persist
- [ ] Test updating non-existent dataset

### 2.6 Dataset Export

#### 2.6.1 Basic Export
- [ ] Export with default settings (all tables)
- [ ] Export specific table: `--table train`
- [ ] Export to specific directory: `--output-dir /path/`
- [ ] Export to specific file: `--output file.csv`
- [ ] Test `--metadata-only` flag

#### 2.6.2 Format Options
- [ ] Export to CSV (default): `mdm dataset export <name>`
- [ ] Export to CSV explicitly: `--format csv`
- [ ] Export to Parquet: `--format parquet`
- [ ] Export to JSON: `--format json`
- [ ] Test with `--no-header` for CSV

#### 2.6.3 Compression Options
- [ ] Test default compression (zip) for CSV
- [ ] Test `--compression none` for uncompressed
- [ ] Test `--compression zip` for CSV/JSON
- [ ] Test `--compression gzip` for CSV/JSON
- [ ] Test `--compression snappy` for Parquet
- [ ] Test `--compression lz4` for Parquet

#### 2.6.4 Export with Transformations
- [ ] Export with column selection
- [ ] Export with preprocessing
- [ ] Export with custom transformations

### 2.7 Dataset Deletion
- [ ] Run `mdm dataset remove <name>`
- [ ] Verify confirmation prompt
- [ ] Test with `--force` flag (no prompt)
- [ ] Verify complete cleanup (files, configs)

### 2.8 Batch Operations
- [ ] Batch export: `mdm batch export dataset1 dataset2 dataset3`
- [ ] Batch export with filter: `mdm batch export --filter "source=kaggle"`
- [ ] Test output directory structure
- [ ] Monitor performance with many datasets

## 3. Auto-detection and File Handling

### 3.1 Kaggle Dataset Detection
- [ ] Place train.csv, test.csv, sample_submission.csv in directory
- [ ] Register without `--target` and verify target is auto-detected
- [ ] Verify ID column is extracted from submission file
- [ ] Test with non-standard file names
- [ ] Test with missing submission file

### 3.2 Column Type Detection
- [ ] Test datetime column detection
- [ ] Test categorical column detection (low cardinality)
- [ ] Test numeric column detection
- [ ] Test text column detection (high cardinality strings)
- [ ] Test ID column pattern detection (*_id, *_key)
- [ ] Test mixed type columns

### 3.3 Problem Type Inference
- [ ] Binary classification (2 unique values)
- [ ] Multi-class (3-10 unique values)
- [ ] Regression (continuous values)
- [ ] Time series (datetime patterns)

### 3.4 File Format Support
- [ ] CSV files (various delimiters)
- [ ] Parquet files
- [ ] JSON files
- [ ] Excel files (.xlsx)
- [ ] Compressed files (.csv.gz, .parquet.snappy)
- [ ] Mixed formats in same directory

## 4. Feature Engineering

### 4.1 Configuration
- [ ] Test `feature_engineering.enabled` setting

### 4.2 Generic Transformers

#### 4.2.1 Temporal Features
- [ ] Verify year, month, day extraction
- [ ] Test weekday, hour, minute features
- [ ] Test `generic.temporal.include_cyclical` (sin/cos)
- [ ] Test `generic.temporal.include_lag` features
- [ ] Verify is_weekend, is_holiday features

#### 4.2.2 Categorical Features
- [ ] Test one-hot encoding for low cardinality
- [ ] Test `generic.categorical.max_cardinality` limit
- [ ] Test `generic.categorical.min_frequency` threshold
- [ ] Verify frequency encoding
- [ ] Test target encoding with cross-validation

#### 4.2.3 Statistical Features
- [ ] Test log transformation for positive values
- [ ] Test z-score normalization
- [ ] Test `generic.statistical.outlier_threshold`
- [ ] Test percentile rank features
- [ ] Verify outlier indicators

#### 4.2.4 Text Features
- [ ] Test text length, word count features
- [ ] Test `generic.text.min_text_length` threshold
- [ ] Verify special character indicators
- [ ] Test average word length calculation

#### 4.2.5 Binning Features
- [ ] Test equal-width binning
- [ ] Test quantile-based binning
- [ ] Test `generic.binning.n_bins` configuration

### 4.3 Custom Features
- [ ] Create Python file in `config/custom_features/`
- [ ] Register dataset and verify custom features load
- [ ] Test syntax error handling in custom file
- [ ] Modify custom file and re-register with `--force`
- [ ] Test feature name conflicts
- [ ] Test `custom_features.auto_discover` setting

### 4.4 Feature Tables and Signal Detection
- [ ] Verify `train_features` table creation
- [ ] Verify `test_features` table creation
- [ ] Check intermediate tables (`train_generic`, `train_custom`)
- [ ] Verify all columns are lowercase in feature tables
- [ ] Test signal detection removes low-variance features
- [ ] Monitor feature generation timing and logs

### 4.5 Data Validation

#### 4.5.1 Before Features Validation
- [ ] Test `validation.before_features.check_duplicates` detects duplicate rows
- [ ] Test `validation.before_features.drop_duplicates` removes duplicates
- [ ] Verify warning logs when duplicates are found and dropped
- [ ] Test `validation.before_features.signal_detection` with low-signal data
- [ ] Verify warning logs for low-signal columns
- [ ] Test `validation.before_features.max_missing_ratio` enforcement
- [ ] Test `validation.before_features.min_rows` requirement

#### 4.5.2 After Features Validation
- [ ] Test `validation.after_features.check_duplicates` on feature data
- [ ] Test `validation.after_features.signal_detection` filters features
- [ ] Verify features with no signal are discarded
- [ ] Test `validation.after_features.max_missing_ratio` on features
- [ ] Test `validation.after_features.min_rows` after processing

#### 4.5.3 Validation Configuration
- [ ] Test validation with all checks disabled
- [ ] Test validation with all checks enabled
- [ ] Test environment variable overrides for validation settings
- [ ] Verify validation runs automatically during dataset operations
- [ ] Test that warnings don't stop processing
- [ ] Test that failures (min_rows) raise errors

## 5. Advanced Features

### 5.1 Query Optimization
- [ ] Test query performance with indexes
- [ ] Monitor slow queries
- [ ] Test direct SQL access
- [ ] Verify query plans

### 5.2 Memory Management
- [ ] Test with datasets >1GB
- [ ] Test with datasets >10GB
- [ ] Monitor memory usage during operations
- [ ] Test chunked processing

### 5.3 Export/Import Advanced
- [ ] Export with DataFrame transformations
- [ ] Save predictions back to MDM
- [ ] Test multi-dataset merge operations

### 5.4 Time Series Features
- [ ] Register with `--time-column`
- [ ] Register with `--group-column`
- [ ] Test time-based splitting
- [ ] Verify lag features generation

## 6. Error Handling and Edge Cases

### 6.1 Registration Failures
- [ ] Interrupt registration mid-process
- [ ] Register with insufficient disk space
- [ ] Register with corrupted files
- [ ] Register with invalid column specifications
- [ ] Verify partial cleanup after failures

### 6.2 Data Edge Cases
- [ ] Empty dataset (0 rows)
- [ ] Single row dataset
- [ ] Single column dataset
- [ ] Dataset with all null values
- [ ] Dataset with duplicate column names
- [ ] Dataset with 1000+ columns
- [ ] Binary features with >99% one class

### 6.3 Path and File Handling
- [ ] Paths with spaces
- [ ] Paths with special characters
- [ ] Unicode in file paths
- [ ] Relative vs absolute paths
- [ ] Symlinks
- [ ] Network paths

### 6.4 Concurrent Access
- [ ] Multiple MDM commands simultaneously
- [ ] Same dataset access from multiple processes
- [ ] Test locking mechanisms
- [ ] Verify data integrity

## 7. Backend-Specific Behavior

### 7.1 Backend Switching
- [ ] Register datasets with backend A
- [ ] Change default_backend to B
- [ ] Verify backend A datasets are invisible
- [ ] Switch back and verify datasets reappear
- [ ] Test mixed backend warning messages

### 7.2 Backend-Specific Features
- [ ] SQLite: Verify WAL mode is active
- [ ] SQLite: Test database locking
- [ ] DuckDB: Verify parallel query execution
- [ ] DuckDB: Test Parquet import/export
- [ ] PostgreSQL: Test remote connections
- [ ] PostgreSQL: Verify SSL connections

## 8. Performance and Benchmarks

### 8.1 Scaling Tests
- [ ] Register 10 datasets
- [ ] Register 100 datasets
- [ ] Register 1000 datasets
- [ ] Test list command performance at each scale
- [ ] Test search performance at scale

### 8.2 Operation Performance
- [ ] Time dataset registration for various sizes
- [ ] Measure export performance
- [ ] Test feature generation speed
- [ ] Monitor async operation benefits

## 9. Platform and Environment

### 9.1 Operating Systems
- [ ] Linux: Test path handling
- [ ] macOS: Test with recent version
- [ ] Windows: Test backslash paths
- [ ] Windows: Test drive letters
- [ ] Test case-sensitive filesystems

### 9.2 Python Environments
- [ ] Test with minimum Python version
- [ ] Test in virtual environment
- [ ] Test with conda environment
- [ ] Verify all dependencies install correctly

### 9.3 Security and Permissions
- [ ] Read-only dataset files
- [ ] Read-only MDM directories
- [ ] Test with restricted user permissions
- [ ] Verify no sensitive data in logs
- [ ] Test SQL injection prevention

## 10. CLI Behavior and User Experience

### 10.1 Output Formatting
- [ ] Rich format with colors and tables
- [ ] Text format without ANSI codes
- [ ] File output with `--format filename.txt`
- [ ] JSON output for programmatic use
- [ ] Progress bars for long operations

### 10.2 Interactive Features
- [ ] Confirmation prompts
- [ ] Error message clarity
- [ ] Suggestion accuracy
- [ ] Help text completeness
- [ ] Tab completion (if available)

### 10.3 Logging and Debugging
- [ ] Verbose mode output
- [ ] Debug information availability
- [ ] Log file creation and rotation
- [ ] Error stack traces (when appropriate)

## 11. Documentation Verification

### 11.1 Configuration Examples
- [ ] Test each example from mdm.yaml.default
- [ ] Test mdm.yaml.sqlite-example
- [ ] Test mdm.yaml.duckdb-example
- [ ] Test mdm.yaml.production
- [ ] Verify minimal configuration works

### 11.2 Command Examples
- [ ] Run every example command from docs
- [ ] Verify example outputs match reality
- [ ] Test all documented flags
- [ ] Check for undocumented features

### 11.3 Tutorial Workflows
- [ ] Complete getting started guide
- [ ] Follow each workflow example
- [ ] Test troubleshooting procedures
- [ ] Verify best practices

## Test Execution Notes

- Mark items with `[x]` when tested successfully
- Mark items with `[!]` when they fail or don't work as documented
- Mark items with `[?]` when behavior is unclear or inconsistent
- Mark items with `[-]` when not applicable to your environment
- Add notes after items for specific findings

## Summary Section

After completing tests, summarize:

- [ ] Total items tested: ___
- [ ] Passed: ___
- [ ] Failed: ___
- [ ] Not applicable: ___
- [ ] Major issues found: ___
- [ ] Documentation mismatches: ___