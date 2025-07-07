# MDM Issues and Findings

This document tracks all issues discovered during testing of MDM (ML Data Manager).

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

## Date: 2025-07-07
## MDM Version: 0.1.0
## Test Environment: Linux/SQLite