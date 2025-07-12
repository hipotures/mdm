# MDM Issues and Findings

This document tracks all issues discovered during testing of MDM (ML Data Manager).

## Recently Fixed Issues (2025-07-08)

### Dataset Update Command Improvements
1. **Exit Code Behavior** (FIXED)
   - **Previous**: `mdm dataset update` returned exit code 1 when no updates specified
   - **Fixed**: Now returns exit code 0 with message "No updates specified"
   
2. **Input Validation** (FIXED)
   - **--id-columns validation**: Now rejects invalid formats like "," or ",," with helpful error message
   - **--problem-type validation**: Only accepts valid problem types (binary_classification, multiclass_classification, regression, time_series, clustering)
   
3. **Error Handling** (FIXED)
   - **Previous**: Generic exceptions showed raw error messages, potentially leaking sensitive info
   - **Fixed**: Generic errors now show user-friendly message, actual error logged for debugging
   
4. **Test Coverage** (ADDED)
   - Added 16 comprehensive unit tests for update command
   - Added 13 integration tests for update functionality
   - Added pre-commit hook to check test import paths

## Configuration Issues

### 1. SQLAlchemy Echo Configuration Not Working
- **Issue**: Setting `database.sqlalchemy.echo: true` in mdm.yaml does not print SQL queries
- **Tested**: Both YAML config and environment variable `MDM_DATABASE_SQLALCHEMY_ECHO=true`
- **Expected**: SQL queries should be printed to console when echo is enabled
- **Status**: Not implemented

### 2. Logging Format Configuration Ignored
- **Issue**: Setting `logging.format: json` in mdm.yaml has no effect on CLI output
- **Tested**: Both YAML config and environment variable `MDM_LOGGING_FORMAT=json`
- **Expected**: Output format should change between console and JSON formats
- **Status**: Not implemented for CLI output

## CLI Parameter Issues

### 3. --time-column Parameter Error
- **Issue**: Using `--time-column` causes "multiple values for keyword argument 'time_column'" error
- **Command**: `mdm dataset register name file.csv --time-column timestamp`
- **Expected**: Should register dataset with specified time column for time series analysis
- **Status**: Implementation bug

### 4. Missing Column Specification Options
- **Issue**: Manual test checklist references options that don't exist:
  - `--datetime-columns` (for multiple datetime columns)
  - `--categorical-columns`
  - `--numeric-columns`
  - `--text-columns`
  - `--ignore-columns`
  - `--stratify-column`
- **Available options**: Only `--target`, `--id-columns`, `--time-column`, `--group-column`
- **Status**: Not implemented

### 5. --id-columns Multiple Values Error
- **Issue**: Using `--id-columns` with multiple columns causes "multiple values" error
- **Command**: `mdm dataset register name file.csv --id-columns "col1,col2"`
- **Status**: Implementation bug (from previous testing)

## Type Detection Issues

### 6. No Automatic Datetime Detection
- **Issue**: Datetime columns are not automatically detected and remain as TEXT type
- **Current behavior**: 
  - Pandas reads datetime strings as `object` type
  - MDM stores them as TEXT in SQLite
  - No datetime features are generated (year, month, day, hour, etc.)
- **Expected**: Automatic detection of datetime patterns and appropriate handling
- **Workaround**: None currently available

### 7. Limited Column Type Detection
- **Issue**: MDM has limited automatic type detection:
  - ✅ Detects ID columns (by name patterns and unique values)
  - ✅ Detects numeric types (int, float)
  - ❌ No datetime detection
  - ❌ No categorical vs text differentiation
  - ❌ No boolean detection (stored as TEXT)
- **Impact**: Feature engineering doesn't generate appropriate features for undetected types

## Feature Engineering Issues

### 8. No Temporal Features Generated
- **Issue**: Even when datetime columns exist, no temporal features are generated
- **Expected features**: year, month, day, hour, minute, weekday, is_weekend, etc.
- **Current**: data_features table just copies original columns without transformation
- **Status**: Temporal feature engineering not implemented

### 9. Problem Type Update Not Persisting
- **Issue**: Using `mdm dataset update --problem-type regression` doesn't persist the change
- **Command**: `mdm dataset update <name> --problem-type regression`
- **Status**: Bug in update operation (from previous testing)

## Documentation Issues

### 10. Test Checklist Contains Non-Existent Options
- **Issue**: MANUAL_TEST_CHECKLIST.md lists many CLI options that don't exist
- **Examples**: --validation-split, --imbalanced (already removed), and column specification options
- **Impact**: Misleading for users trying to follow the documentation
- **Status**: Documentation needs update

## Database Backend Issues

### 11. DuckDB Backend Missing SQLAlchemy Dialect
- **Issue**: DuckDB backend requires `duckdb-engine` package which is not installed
- **Error**: Missing sqlalchemy dialect for duckdb
- **Workaround**: Use SQLite backend instead
- **Status**: Optional dependency not included

## File Format Issues

### 12. No CSV Header Option Not Working
- **Issue**: `--no-header` option for CSV export mentioned in tests but not implemented
- **Expected**: Export CSV without header row
- **Status**: Option doesn't exist

### 13. File Output Format Not Working
- **Issue**: `--format output.txt` doesn't save to file, only changes display format
- **Command**: `mdm dataset list --format output.txt`
- **Expected**: Save output to specified file
- **Status**: Misunderstood functionality or not implemented

## Performance and Logging Issues

### 14. Missing uv.lock File
- **Issue**: No uv.lock file present for reproducible installs
- **Impact**: Package versions may vary between installations
- **Status**: File should be generated and committed

### 15. Batch Size Configuration Not Verified in Operations
- **Issue**: Cannot verify if `performance.batch_size` setting is actually used
- **Note**: Configuration loads correctly but usage in operations unclear
- **Status**: Needs verification

## Environment Variable Issues

### 16. MDM_MDM_HOME Not Respected
- **Issue**: Setting `MDM_MDM_HOME=/tmp/custom_mdm_home` doesn't change the MDM home directory
- **Expected**: MDM should use the custom path for datasets and configuration
- **Actual**: Still uses default `~/.mdm` directory
- **Status**: Environment variable ignored

## Dataset Name Validation

### 17. Spaces Not Allowed in Dataset Names
- **Issue**: Dataset names cannot contain spaces
- **Error**: "Dataset name can only contain alphanumeric characters, underscores, and dashes"
- **Note**: This is actually good validation, working as designed
- **Allowed**: Letters, numbers, underscores (_), and dashes (-)

## CLI Options Issues

### 18. --no-auto Flag Not Implemented
- **Issue**: `--no-auto` flag exists but manual registration not implemented
- **Command**: `mdm dataset register name --no-auto --train file.csv --target col`
- **Error**: "Manual registration not yet implemented"
- **Status**: Feature exists in CLI but backend not implemented

## File Format Support

### 19. JSON Format Fully Supported ✅
- **Success**: JSON files can be registered and processed correctly
- **Note**: MDM correctly reads JSON arrays as dataframes
- **ID Detection**: Detected both 'id' and 'age' as ID columns (interesting behavior)

## Feature Engineering Issues

### 20. Custom Features Not Loaded
- **Issue**: Custom feature files in `~/.mdm/config/custom_features/` are not loaded
- **Tested**: Created `my_custom_features.py` with feature functions
- **Expected**: Custom features should be applied during feature generation
- **Actual**: No custom features generated, no logs about loading custom files
- **Status**: Custom features functionality not implemented

## Logging Issues

### 21. Log Level Configuration Not Working
- **Issue**: Setting `logging.level: ERROR` in mdm.yaml doesn't suppress INFO messages
- **Tested**: Both YAML config and environment variable `MDM_LOGGING_LEVEL=ERROR`
- **Expected**: Only ERROR messages should appear
- **Actual**: INFO messages still displayed
- **Status**: Log level configuration ignored

## Dataset Name Support

### 22. Unicode Characters Supported ✅
- **Success**: Dataset names with Unicode characters work correctly
- **Tested**: `test_données_测试` (French and Chinese characters)
- **Note**: Full Unicode support confirmed

## Missing CLI Options

### 23. --skip-analysis Flag Not Implemented
- **Issue**: `--skip-analysis` flag mentioned in test checklist doesn't exist
- **Command**: `mdm dataset register name file.csv --skip-analysis`
- **Status**: Option not implemented

### 24. --dry-run Flag Not Implemented
- **Issue**: `--dry-run` flag mentioned in test checklist doesn't exist
- **Command**: `mdm dataset register name file.csv --dry-run`
- **Status**: Option not implemented

## Directory Registration

### 25. Directory Registration Works ✅
- **Success**: Can register datasets from directories containing multiple files
- **Tested**: Directory with train.csv and test.csv
- **Note**: Automatically detects and processes all CSV files
- **Features**: Generates features for each table separately

## CLI Features

### 26. Dataset List --limit Works ✅
- **Success**: `--limit N` parameter correctly limits output
- **Command**: `mdm dataset list --limit 3`
- **Note**: Shows only first N datasets

### 27. Dataset Remove Works ✅
- **Success**: Dataset removal with confirmation prompt and --force flag
- **Command**: `mdm dataset remove name [--force]`
- **Note**: Properly cleans up all files and configuration

## File Format Support

### 28. Excel Files Not Supported
- **Issue**: Excel (.xlsx) files are not supported
- **Error**: "Unsupported file type"
- **Note**: Would require additional implementation
- **Workaround**: Convert to CSV/Parquet/JSON

### 29. CSV Delimiter Auto-Detection Works ✅
- **Success**: Automatically detects various CSV delimiters
- **Tested**: Semicolon (;) and tab (\t) delimiters
- **Files**: .csv with semicolons, .tsv with tabs
- **Note**: Delimiter detection is automatic and reliable

## Path Handling

### 30. Relative Paths Work ✅
- **Success**: Relative paths are accepted and converted to absolute
- **Command**: `mdm dataset register name relative/path.csv`
- **Note**: MDM automatically resolves to absolute path
- **Storage**: Stores absolute path in metadata

### 31. --source Option Missing
- **Issue**: The --source option referenced in the checklist doesn't exist in the CLI
- **Command**: `mdm dataset register name file.csv --source "kaggle-competition"`
- **Error**: "No such option: --source"
- **Expected**: Should allow specifying data source with --source flag
- **Status**: Option not implemented

### 32. Non-existent File Path Handling Works ✅
- **Success**: Non-existent file paths are properly handled with error
- **Command**: `mdm dataset register name /path/that/does/not/exist/data.csv`
- **Error**: "Error: Path does not exist: /path/that/does/not/exist/data.csv"
- **Note**: Proper error handling implemented

### 33. Sort Options Limited
- **Issue**: Only 'name' and 'registration_date' sort options available
- **Checklist mentions**: row_count, size_mb, last_updated_at
- **Command**: `mdm dataset list --sort-by row_count`
- **Status**: Limited implementation

### 34. Compressed Files Not Supported
- **Issue**: Compressed files (.csv.gz) not supported
- **Command**: `mdm dataset register name file.csv.gz`
- **Error**: "Unsupported file type: /path/to/file.csv.gz"
- **Expected**: Should decompress and read compressed CSV files
- **Status**: Feature not implemented

### 35. Mixed Format Directory Support Works ✅
- **Success**: Directories with mixed file formats handled correctly
- **Test**: Directory with .csv, .json, .parquet files
- **Result**: Only CSV files loaded, others ignored
- **Note**: Good selective loading behavior

### 36. Single Column Dataset Works ✅
- **Success**: Single column datasets can be registered
- **Test**: CSV with only one column "value"
- **Result**: Registered successfully, column detected as ID
- **Note**: Handles edge case properly

### 37. SQLite cache_size Configuration Not Applied
- **Issue**: cache_size setting in mdm.yaml not applied to SQLite databases
- **Config**: database.sqlite.cache_size: -128000
- **Actual**: PRAGMA cache_size shows -2000 (default)
- **Status**: Configuration not implemented

### 38. SQLite temp_store Configuration Not Applied
- **Issue**: temp_store setting in mdm.yaml not applied to SQLite databases
- **Config**: database.sqlite.temp_store: "MEMORY" or "FILE"
- **Actual**: PRAGMA temp_store shows 0 (DEFAULT)
- **Status**: Configuration not implemented

### 39. Export Default Format Configuration Ignored
- **Issue**: export.default_format in mdm.yaml has no effect
- **Config**: export.default_format: "parquet"
- **Actual**: Always exports as CSV.zip
- **Status**: Configuration not implemented

### 40. Export Compression Configuration Partially Working
- **Issue**: export.compression in mdm.yaml ignored
- **Config**: export.compression: "gzip"
- **Note**: CLI options (--compression) work correctly
- **Status**: Configuration ignored, CLI works

### 41. Filter Not Equal Operator Not Working
- **Issue**: Filter with != operator doesn't work
- **Command**: mdm dataset list --filter "name!=cache_test"
- **Result**: Shows "No datasets registered yet"
- **Status**: Not equal operator not implemented

### 42. Pattern Matching Not Specific
- **Issue**: Pattern matching with ~ doesn't distinguish starts/ends/contains
- **Example**: "name~test*" shows all with "test" anywhere
- **Expected**: Should match only names starting with "test"
- **Status**: Limited pattern matching implementation

### 43. Empty Dataset Support Works ✅
- **Success**: Datasets with 0 rows can be registered
- **Test**: CSV with headers only
- **Result**: Registered successfully, stats show 0 rows
- **Note**: Edge case handled properly

### 44. All Null Values Dataset Works ✅
- **Success**: Datasets with all null values (except ID) handled correctly
- **Test**: Dataset with ID column and all other values null
- **Result**: Shows correct completeness percentage (33.3%)
- **Note**: Null handling working properly

### 45. SQLite mmap_size Configuration Not Applied
- **Issue**: mmap_size setting in mdm.yaml not applied to SQLite databases
- **Config**: database.sqlite.mmap_size: 268435456
- **Actual**: PRAGMA mmap_size shows 0
- **Status**: Configuration not implemented

### 46. Log File Creation Not Working
- **Issue**: Log file configuration has no effect
- **Config**: logging.file: "/tmp/mdm.log"
- **Result**: No log file created even with DEBUG level
- **Status**: File logging not implemented

### 47. Date Filter Behavior Unclear
- **Issue**: Date filters show all datasets regardless of filter
- **Commands**: --filter "registered_at>2025-07-07" and "registered_at>30days"
- **Result**: All datasets shown, filtering may not be working
- **Status**: Implementation unclear or broken

### 48. Dataset Update Target Column Bug
- **Issue**: Target column update saves to description instead of target field
- **Command**: mdm dataset update name --target column
- **Result**: Shows in description but Target Column remains None
- **Status**: Update operation bug

### 49. Batch Export Works ✅
- **Success**: Batch export command working correctly
- **Command**: mdm batch export dataset1 dataset2 dataset3
- **Result**: Creates proper directory structure with exports
- **Note**: Efficient multi-dataset export

### 50. Kaggle Dataset Auto-Detection Works ✅
- **Success**: Automatic detection of Kaggle dataset structure
- **Test**: Directory with train.csv, test.csv, sample_submission.csv
- **Result**: Correctly detects target, problem type, and ID columns
- **Note**: Excellent auto-detection implementation

### 51. Binary Classification Detection Works ✅
- **Success**: Binary target correctly identified
- **Test**: Dataset with 0/1 target values
- **Result**: Problem type set to binary_classification
- **Note**: ML type detection working well

### 52. Single Row Dataset Works ✅
- **Success**: Single row datasets handled correctly
- **Test**: CSV with one data row
- **Result**: Registered successfully, stats show 1 row
- **Note**: Edge case handled properly

### 53. Duplicate Column Names Handled ✅
- **Success**: Duplicate columns automatically renamed
- **Test**: CSV with two "value" columns
- **Result**: Pandas renames to "value" and "value.1"
- **Note**: Graceful handling by pandas

## Date: 2025-07-08
## MDM Version: 0.1.0
## Test Environment: Linux/SQLite

### 54. Feature Engineering Enabled Setting Not Working ❌
- **Issue**: `feature_engineering.enabled: false` setting ignored
- **Steps**: Set enabled: false in mdm.yaml, register dataset
- **Expected**: No feature generation
- **Actual**: Features still generated
- **Impact**: Cannot disable feature engineering

### 55. Multiclass Classification Detection Works ✅
- **Success**: Multiclass targets correctly identified
- **Test**: Dataset with string target having 4 unique values
- **Result**: Problem type set to multiclass_classification
- **Note**: Excellent ML type detection

### 56. Regression Detection Works ✅
- **Success**: Continuous targets correctly identified
- **Test**: Dataset with float price values
- **Result**: Problem type set to regression
- **Note**: ML type detection working well

### 57. Large Column Datasets Supported ✅
- **Success**: 1000+ column datasets handled
- **Test**: Created CSV with 1001 columns
- **Result**: Registered successfully
- **Note**: Good scalability for wide datasets

### 58. Paths with Spaces Work ✅
- **Success**: Directories with spaces supported
- **Test**: "test dir with spaces/data.csv"
- **Result**: Dataset registered correctly
- **Note**: Proper path handling

### 59. Backend Switching Not Implemented ❌
- **Issue**: Changing backend doesn't hide other backend's datasets
- **Steps**: Switch from sqlite to duckdb in config
- **Expected**: SQLite datasets become invisible
- **Actual**: All datasets still visible, DuckDB plugin error
- **Impact**: Multi-backend support incomplete

### 60. JSON Output Format Missing ❌
- **Issue**: No JSON output option for CLI commands
- **Steps**: Try --format json or --output json
- **Expected**: JSON formatted output
- **Actual**: Option doesn't exist
- **Impact**: No programmatic output parsing

### 61. Confirmation Prompts Working ✅
- **Success**: Interactive confirmations work correctly
- **Test**: Dataset removal with y/n prompt
- **Result**: Properly cancels on 'n', proceeds on 'y'
- **Note**: Good UX for destructive operations

### 62. Help Text Complete ✅
- **Success**: Comprehensive help documentation
- **Test**: Check --help for all commands
- **Result**: Clear descriptions and options documented
- **Note**: Good user documentation

### 63. Minimal Configuration Supported ✅
- **Success**: MDM works with minimal or no config
- **Test**: Delete config, use minimal config
- **Result**: Works with defaults
- **Note**: Good zero-config experience

### 64. Temporal Feature Extraction Not Implemented ❌
- **Issue**: No temporal features generated from datetime columns
- **Steps**: Register dataset with datetime column
- **Expected**: Year, month, day, hour features
- **Actual**: No features generated, datetime kept as text
- **Impact**: No automatic time-based feature engineering

### 65. Categorical Feature Encoding Not Implemented ❌
- **Issue**: No categorical encoding features generated
- **Steps**: Register dataset with categorical columns
- **Expected**: One-hot encoding for low cardinality
- **Actual**: Categories kept as text, no encoding
- **Impact**: No automatic categorical feature engineering

### 66. Statistical Transformations Not Implemented ❌
- **Issue**: No statistical features generated
- **Steps**: Register dataset with numeric columns
- **Expected**: Log transform, z-score, percentile ranks
- **Actual**: No transformations applied
- **Impact**: No automatic statistical feature engineering

### 67. Text Feature Extraction Not Implemented ❌
- **Issue**: No text features generated
- **Steps**: Register dataset with text descriptions
- **Expected**: Text length, word count features
- **Actual**: Text kept as-is, no features
- **Impact**: No automatic text feature engineering

### 68. Custom Features Not Loaded ❌
- **Issue**: Custom Python features not applied
- **Steps**: Create custom_features/*.py file, register dataset
- **Expected**: Custom functions applied to generate features
- **Actual**: Custom features ignored
- **Impact**: Cannot extend feature engineering

### 69. Duplicate Detection Not Working ❌
- **Issue**: Duplicate rows not detected or removed
- **Steps**: Configure validation.before_features.check_duplicates
- **Expected**: Warning about duplicates, optional removal
- **Actual**: Duplicates kept, no warnings
- **Impact**: Data quality issues not caught

### 70. Signal Detection Not Working ❌
- **Issue**: Low signal columns not detected
- **Steps**: Configure signal_detection, use constant columns
- **Expected**: Warnings about low variance columns
- **Actual**: No warnings, all columns kept
- **Impact**: Useless features not filtered

### 71. Large Dataset Performance Good ✅
- **Success**: 5M rows (317MB) registered in 40 seconds
- **Test**: Create and register large dataset
- **Result**: Efficient processing
- **Note**: Good scalability

### 72. Time Series Registration Bug ❌
- **Issue**: --time-column parameter causes error
- **Steps**: Register with --time-column and --group-column
- **Expected**: Time series dataset configuration
- **Actual**: "multiple values for keyword argument" error
- **Impact**: Cannot properly register time series data

### 73. Concurrent Access Works ✅
- **Success**: Multiple commands on same dataset work
- **Test**: Run info/stats commands simultaneously
- **Result**: No errors or conflicts
- **Note**: Good concurrency handling

### 74. Log Level Configuration Not Applied ❌
- **Issue**: WARNING log level still shows INFO messages
- **Steps**: Set logging.level: WARNING in mdm.yaml
- **Expected**: Only warnings and errors shown
- **Actual**: INFO messages still displayed
- **Impact**: Cannot control log verbosity

### 75. CLI Output Format Not Applied ❌
- **Issue**: default_output_format: json ignored
- **Steps**: Set cli.default_output_format: json
- **Expected**: JSON output by default
- **Actual**: Rich table format still used
- **Impact**: Configuration setting has no effect

### 76. CLI Confirm Destructive Not Applied ❌
- **Issue**: confirm_destructive: false ignored
- **Steps**: Set cli.confirm_destructive: false
- **Expected**: No confirmation prompts
- **Actual**: Still prompts for confirmation
- **Impact**: Cannot disable confirmation prompts

### 77. ID Columns Parameter Bug ❌
- **Issue**: --id-columns causes parameter error
- **Steps**: Register with --id-columns user_id,order_id
- **Expected**: Multiple ID columns configured
- **Actual**: "multiple values for keyword argument" error
- **Impact**: Cannot specify multiple ID columns

### 78. Force Flag Not Implemented ❌
- **Issue**: --force flag doesn't overwrite existing dataset
- **Steps**: Register existing dataset with --force
- **Expected**: Overwrites existing dataset
- **Actual**: "already exists" error
- **Impact**: Cannot force overwrite datasets

### 79. Details Flag No Effect ❌
- **Issue**: --details flag shows same info
- **Steps**: Run dataset info --details
- **Expected**: Extended information
- **Actual**: Same as without flag
- **Impact**: No additional details available

### 80. Stats Full Flag Works ✅
- **Success**: --full flag shows detailed statistics
- **Test**: mdm dataset stats --full
- **Result**: Column-level statistics displayed
- **Note**: Proper implementation

### 81. Export Table Option Works ✅
- **Success**: --table exports specific table
- **Test**: Export with --table data
- **Result**: Only specified table exported
- **Note**: Good table filtering

### 82. Dataset Search Rendering Bug ❌
- **Issue**: Search command has render error
- **Steps**: Run mdm dataset search test
- **Expected**: List of matching datasets
- **Actual**: "unable to render dict" error
- **Impact**: Search feature unusable

### 83. SQLite WAL Mode Active ✅
- **Success**: WAL mode enabled for SQLite
- **Test**: Check PRAGMA journal_mode
- **Result**: Shows "wal"
- **Note**: Performance optimization working

### 84. Log Level ERROR Not Applied ❌
- **Issue**: ERROR log level still shows INFO messages
- **Steps**: Set logging.level: ERROR in mdm.yaml
- **Expected**: Only errors shown
- **Actual**: INFO messages still displayed
- **Impact**: Cannot set strictest log level

### 85. Dataset Names Validated ✅
- **Success**: Spaces in names properly rejected
- **Test**: Register with name "my test dataset"
- **Result**: Error about allowed characters
- **Note**: Good validation, but maybe too strict

### 86. Problem Type Override Works ✅
- **Success**: --problem-type flag overrides detection
- **Test**: Binary data registered as regression
- **Result**: Problem type set correctly
- **Note**: Useful for custom ML scenarios

### 87. List Limit Parameter Works ✅
- **Success**: --limit restricts output rows
- **Test**: mdm dataset list --limit 3
- **Result**: Only 3 datasets shown
- **Note**: Good for large dataset collections

### 88. Export Metadata Only Works ✅
- **Success**: --metadata-only exports just metadata
- **Test**: Export with flag
- **Result**: Only JSON metadata file created
- **Note**: Useful for documentation

### 89. Export Compression Control Works ✅
- **Success**: --compression none creates uncompressed files
- **Test**: Export with compression none
- **Result**: Plain CSV files
- **Note**: Good compression flexibility

### 90. Remove Force Flag Works ✅
- **Success**: --force skips confirmation for removal
- **Test**: mdm dataset remove --force
- **Result**: Immediate deletion
- **Note**: Good for automation

### 91. Datetime Columns Handled Correctly ✅
- **Success**: Date/timestamp columns work
- **Test**: Various datetime formats
- **Result**: Stored as TEXT in SQLite (expected)
- **Note**: Standard SQLite behavior

### 92. No Categorical Column Detection ❌
- **Issue**: Low cardinality columns not detected as categorical
- **Steps**: Register data with color, size, status columns
- **Expected**: Categorical type detection
- **Actual**: No detection in configuration
- **Impact**: No automatic categorical handling

### 93. CSV Delimiter Auto-Detection Works ✅
- **Success**: Various delimiters detected
- **Test**: Semicolon and tab delimited files
- **Result**: Both parsed correctly
- **Note**: Good CSV flexibility