# Dataset Registration

Dataset registration is the process of adding new datasets to MDM. The system provides flexible registration options with intelligent auto-detection capabilities.

## Important Notes

- **Dataset names are case-insensitive for lookups**: You can register as `MyDataset` but access it as `mydataset`, `MYDATASET`, or any other case variation
- **Original case is preserved**: The dataset is stored with the exact name you provide during registration
- **Unique names required**: `titanic` and `TITANIC` are considered the same dataset
- **Single Backend Architecture**: All datasets use the backend specified in `database.default_backend` in `~/.mdm/mdm.yaml`
- **Backend Switching**: If you change the backend in `mdm.yaml`, datasets registered with other backends won't be visible. You must re-register them with the new backend.

## Registration Approaches

MDM supports two registration approaches:

1. **Automatic Registration** - Default behavior with intelligent detection
2. **Manual Registration** - Explicit control using `--no-auto` flag

For a complete flowchart and detailed rules, see [Target & ID Detection Schema](14_Target_ID_Detection_Schema.md).

## 1. Automatic Registration (Default)

MDM automatically scans the directory and intelligently identifies dataset characteristics:

```bash
# Minimal command - relies entirely on auto-detection
mdm dataset register my_dataset /path/to/dataset/dir

# Auto-detection with user overrides
mdm dataset register house_prices /data/kaggle/house-prices \
    --problem-type "regression"  # Override auto-detected problem type
    --description "House prices with custom features"
```

### Auto-Detection Capabilities

- **Kaggle Competition**: Detects train.csv + test.csv + sample_submission.csv structure
- **Target & ID**: Automatically extracts from Kaggle datasets (no --target needed)
- **Problem Type**: Automatically infers from target column values:
  - Binary values → `classification`
  - Multiple categories → `multiclass`
  - Continuous values → `regression`
  - Date/time patterns → `time-series`
- **Column Types**: Automatically detects and categorizes columns:
  - ID columns: Patterns like `*_id`, `*_key`, unique identifiers
  - Datetime columns: Date/time formats and patterns
  - Categorical columns: Low cardinality text/numeric columns
  - Numeric columns: Integer and float values
  - Text columns: High cardinality text, long strings
- **File Formats**: Automatically detects csv, parquet, json, feather, orc files
- **Encoding**: Automatically detects file encoding (UTF-8, Latin-1, etc.)
- **Delimiters**: Automatically detects CSV delimiters (comma, semicolon, tab, etc.)
- **Compression**: Automatically handles gzip, snappy, lz4, zstd compressed files
- **Headers**: Automatically detects header rows and skip rows in CSV files

### Auto-Detection Patterns

MDM looks for standard file naming patterns:

- **Train**: `train.csv`, `training.csv`, `train_data.csv`, `X_train.csv`
- **Test**: `test.csv`, `testing.csv`, `test_data.csv`, `X_test.csv`
- **Validation**: `val.csv`, `valid.csv`, `validation.csv`, `X_val.csv`
- **Submission**: `sample_submission.csv`, `submission.csv`, `sample.csv`

## 2. Manual Registration (--no-auto)

When auto-detection is disabled with `--no-auto`, you must specify:
1. Exact file paths for data files
2. Target column name

```bash
# Single file dataset
mdm dataset register my_dataset \
    --no-auto \
    --train /path/to/data.csv \
    --target "price" \
    --id-columns "id" \
    --problem-type "regression"

# Multiple files - must specify each file path
mdm dataset register customer_churn \
    --no-auto \
    --train /data/processed/train_customers.parquet \
    --test /data/processed/test_customers.parquet \
    --validation /data/processed/val_customers.parquet \
    --target "Churn" \
    --id-columns "CustomerID,AccountNumber" \
    --problem-type "binary-classification"

# Full manual registration with all options
mdm dataset register sales_data \
    --no-auto \
    --train /data/sales/2023/train.csv \
    --test /data/sales/2023/test.csv \
    --submission /data/sales/2023/submission_template.csv \
    --target "Revenue" \
    --id-columns "TransactionID" \
    --datetime-columns "Date,OrderDate,ShipDate" \
    --categorical-columns "Region,ProductCategory,CustomerSegment" \
    --numeric-columns "Quantity,Discount,ShippingCost" \
    --problem-type "regression"
```

**Important**: With `--no-auto`, directory paths are not accepted. You must specify individual file paths using `--train`, `--test`, etc.

## 3. Registration Process

When registering a dataset, the system follows these steps:

1. **Validates Files**: Checks file existence and readability
2. **Auto-Detection** (default behavior):
   - Detects Kaggle competition structure (train.csv + sample_submission.csv)
   - Extracts target and ID columns from sample_submission.csv
   - Infers problem type from target column values
   - Identifies column types automatically
3. **Target Column**: MUST be specified via `-t/--target` parameter (required UNLESS Kaggle dataset is auto-detected)
4. **User Override**: Column types and other auto-detected values can be overridden with CLI parameters
5. **Validation**: 
   - Ensures target column exists in the dataset
   - Validates all specified columns exist in the dataset
6. **Creates Directory Structure**: 
   - Creates `~/.mdm/datasets/{dataset_name}/` directory
7. **Creates Database**: Dataset-specific database with both data and metadata
   - Backend type is determined by `database.default_backend` in `~/.mdm/mdm.yaml`
   - Database file: `~/.mdm/datasets/{dataset_name}/dataset.{sqlite|duckdb}` or PostgreSQL database
   - Contains data tables AND metadata tables (_metadata, _columns, etc.)
   - All datasets must use the same backend type
8. **Imports Source Tables**: Loads train, test, validation data into database
9. **Stores Local Metadata**: 
   - Creates `_metadata` table with dataset information
   - Creates `_columns` table with column statistics
   - Stores all metadata within the dataset's own database
10. **Generates Feature Tables**: 
   - Applies two-tier feature engineering system:
     * **Generic transformers**: Automatically applied to all datasets based on column types
     * **Custom transformers**: Optional dataset-specific features from `~/.mdm/config/custom_features/{dataset_name}.py`
   - Creates feature tables: `train_features`, `test_features`, `validation_features` (if applicable)
   - Each feature table contains:
     * All original columns (unchanged)
     * Generic engineered features based on column types
     * Custom features (if custom transformer exists)
   - **Signal detection**: Features with no variance are automatically discarded
   - Example generic transformations:
     * DateTime → year, month, day, weekday, hour, is_weekend, days_since_start
     * Text → length, word_count, avg_word_length, contains_digits
     * Categorical → one-hot encoding, frequency encoding, target encoding
     * Numeric → log transform, z-score, percentile rank, binned values
   - Process is logged with timing: `Generated feature 'age_binned' in 0.023s`
11. **Creates YAML Pointer**: Writes `~/.mdm/config/datasets/{dataset_name}.yaml`
    - Serves as a discovery pointer for MDM to find the dataset
    - Contains basic configuration and path to database file
12. **Displays Summary**: Shows registration results including generated feature tables

## Configuration File Structure

After successful registration, a YAML configuration file is created at `~/.mdm/config/datasets/{dataset_name}.yaml`:

### Standard Dataset Configuration

```yaml
# Example: ~/.mdm/config/datasets/house_prices.yaml
name: house_prices
description: Kaggle House Prices - Advanced Regression Techniques

# Database connection
database:
  path: ~/.mdm/datasets/house_prices/dataset.duckdb          # Full path to database
  # Note: The backend is determined by database.default_backend in mdm.yaml at registration time
  # Once a dataset is registered with a backend, it cannot be changed
  # For PostgreSQL datasets:
  # connection_string: postgresql://user:pass@localhost/mdm_house_prices
  
# Table names
tables:
  train: train
  test: test
  validation: validation  # if exists
  submission: submission  # if exists

# Essential metadata
target_column: SalePrice
id_columns: [Id]
problem_type: regression
```

### Unsupervised Learning Configuration

```yaml
# Example: ~/.mdm/config/datasets/customer_segments.yaml
name: customer_segments
description: Customer segmentation dataset

database:
  path: ~/.mdm/datasets/customer_segments/dataset.sqlite         # Full path
  # Note: File extension indicates which backend was used during registration
  
tables:
  data: data
  
# No target column for unsupervised learning
id_columns: [customer_id]
problem_type: clustering
```

**Note**: During registration, MDM creates both source tables (train, test) and feature tables (train_features, test_features) within each dataset's database. The feature tables contain all original columns plus engineered features based on column types. There is no central registry - the YAML configuration files serve only as pointers to the dataset databases.

## Auto-Detection Logic

MDM employs smart auto-detection with the following priority:

### 1. Kaggle Competition Detection

MDM uses three validation methods to confirm a Kaggle dataset:

```
If train.csv, test.csv, and *submission*.csv exist:
  Method 1: Extract from submission file
    - ID column: First column in submission file
    - Target column: Last column in submission file
  
  Method 2: Column difference analysis
    - Target: Column present in train.csv but NOT in test.csv
    - Validates against submission file
  
  Method 3: ID column pattern matching
    - Searches for columns matching ^id.* or .*id$ (case-insensitive)
    - Confirms ID column exists in all files
  
  If all three methods validate:
    - Set source = "kaggle"
    - Log: "Kaggle competition detected: {name}"
    - Target parameter NOT required
    - ID parameter NOT required
```

**Important**: When Kaggle dataset is detected, the `--target` parameter is NOT required. For all other datasets, `--target` is mandatory.

### 2. Problem Type Detection
```
If target column specified/detected:
  - If target has 2 unique values → binary-classification
  - If target has 3-10 unique values → multiclass
  - If target has >10 unique values and is numeric → regression
  - If target is float → regression
```

### 3. Column Type Detection and Feature Generation
```
For each column:
  - If column name ends with 'id' or starts with 'id' (case-insensitive) → excluded from features
  - If column contains dates → DateTimeFeatures transformer applied
  - If unique values < 0.5 * total_rows → CategoricalFeatures transformer applied
  - If all values are numeric → kept as-is, optionally normalized
  - If average length > 50 characters → TextFeatures transformer applied
```

### 4. Feature Engineering System

MDM implements a sophisticated two-tier feature engineering system:

#### Generic Feature Transformers (Automatic)

Applied to all datasets during registration based on detected column types:

```python
# Base class for all generic transformers
class GenericFeatureOperation(ABC):
    """Base class with signal detection and timing."""
    
    def generate_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate features with automatic signal checking."""
        features = {}
        for feature_name, feature_values in self._generate_features_impl(df, **kwargs).items():
            if self._check_feature_signal(feature_values):
                features[feature_name] = feature_values
            else:
                logger.info(f"Feature '{feature_name}' has no signal, discarded")
        return features
```

**Available Generic Transformers:**

1. **TemporalFeatures** (datetime columns):
   - Components: year, month, day, weekday, hour, minute
   - Cyclical: sin/cos transforms for month, day, hour
   - Derived: is_weekend, is_holiday, days_since_start
   - Time series: lag features, rolling statistics

2. **CategoricalFeatures** (low cardinality columns):
   - One-hot encoding (for <50 unique values)
   - Frequency encoding
   - Target encoding with cross-validation
   - Rare category grouping (threshold: 1%)

3. **StatisticalFeatures** (numeric columns):
   - Log transformation (for positive values)
   - Z-score normalization
   - Percentile ranks
   - Outlier indicators (>3 std devs)

4. **TextFeatures** (text columns):
   - Length metrics: char_count, word_count, avg_word_length
   - Content flags: has_digits, has_special_chars
   - Complexity: unique_word_ratio

5. **BinningFeatures** (continuous numeric):
   - Equal-width bins (5, 10 bins)
   - Quantile bins (quartiles, deciles)
   - Custom threshold bins

#### Custom Feature Transformers (Optional)

Dataset-specific features can be added by creating a Python file:
`~/.mdm/config/custom_features/{dataset_name}.py`

```python
# Example: ~/.mdm/config/custom_features/titanic.py
from mdm.features.custom.base import BaseDomainFeatures

class CustomFeatureOperations(BaseDomainFeatures):
    """Custom features for Titanic dataset."""
    
    def _register_operations(self):
        """Register all custom operations."""
        self._operation_registry = {
            'family_features': self.get_family_features,
            'title_features': self.get_title_features,
            'survival_indicators': self.get_survival_indicators,
        }
    
    def get_family_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Family-based features."""
        features = {}
        
        # Family size
        features['family_size'] = df['sibsp'] + df['parch'] + 1
        features['is_alone'] = (features['family_size'] == 1).astype(int)
        
        # Family type
        features['has_child'] = ((df['age'] < 18) | (df['parch'] > 0)).astype(int)
        features['family_survival_ratio'] = self._get_family_survival_ratio(df)
        
        return features
    
    def get_title_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract titles from names."""
        features = {}
        
        # Extract title
        features['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        features['is_rare_title'] = ~features['title'].isin(['Mr', 'Miss', 'Mrs', 'Master'])
        
        return features
```

#### Signal Detection in Validation

Signal detection is now part of the validation configuration and runs at two stages:

1. **Before feature generation** (if `validation.before_features.signal_detection: true`):
   - Checks raw input columns for signal
   - Logs warnings for low-signal columns but continues processing
   - Helps identify problematic input data early

2. **After feature generation** (if `validation.after_features.signal_detection: true`):
   - Validates that generated features contain meaningful signal
   - Discards features with no signal (all identical values, etc.)
   - Ensures only useful features are retained

```python
def check_signal(series: pd.Series, min_signal_ratio: float = 0.01) -> bool:
    """
    Check if data has signal (variance).
    
    Returns False if:
    - All values are identical
    - All values are null  
    - Only one unique non-null value
    - Less than min_signal_ratio unique values
    
    For large datasets (>2000 rows), uses sampling for performance.
    """
    non_null_count = series.count()
    if non_null_count <= 1:
        return False
        
    unique_ratio = series.nunique() / len(series)
    return unique_ratio >= min_signal_ratio
```

Configure in `mdm.yaml`:
```yaml
validation:
  after_features:
    signal_detection: true  # Default: true
```

#### Feature Generation Output

During registration, you'll see detailed logs:

```
Generating features for train (1460 rows)...
[TemporalFeatures] Processing 3 datetime columns...
  Generated feature 'date_year' in 0.012s
  Generated feature 'date_month' in 0.008s
  Generated feature 'date_month_sin' in 0.015s
  Generated feature 'date_is_weekend' in 0.007s [no signal, discarded]
[CategoricalFeatures] Processing 5 categorical columns...
  Generated feature 'embarked_S' in 0.023s
  Generated feature 'sex_male' in 0.019s
[CustomFeatures] Applying titanic-specific features...
  Generated feature 'family_size' in 0.011s
  Generated feature 'title_Mr' in 0.034s
Feature generation complete: 47 features created (3 discarded)
```

## Registration Examples

### Minimal Registration (Auto-detection)
```bash
# Kaggle dataset - NO --target needed (auto-detected from submission file)
mdm dataset register house_prices /data/kaggle/house-prices

# Output:
✓ Kaggle competition detected: house-prices
✓ Target column: SalePrice (from sample_submission.csv)
✓ ID column: Id (from sample_submission.csv)
✓ Dataset 'house_prices' registered successfully!

Created:
- Directory: ~/.mdm/datasets/house_prices/
- Database: ~/.mdm/datasets/house_prices/dataset.duckdb
- Config: ~/.mdm/config/datasets/house_prices.yaml
- Tables: train (1460 rows), test (1459 rows), submission (1459 rows)
- Target: SalePrice (regression) [auto-detected]
- ID: Id [auto-detected]
```

### Registration with Required Target (Non-Kaggle)
```bash
# Non-Kaggle dataset - --target is REQUIRED
mdm dataset register customer_data /data/customers \
    --target "churned" \
    --problem-type "binary-classification" \
    --id-columns "customer_id,account_id"

# Output:
✓ Dataset 'customer_data' registered successfully!

Configuration:
- Target: churned (binary-classification) [user-specified]
- ID Columns: customer_id, account_id [user-specified]
- Files: data.csv (50000 rows)
- Columns: 25 total (12 numeric, 10 categorical, 3 datetime)

Created:
- Directory: ~/.mdm/datasets/customer_data/
- Database: ~/.mdm/datasets/customer_data/dataset.duckdb
- Config: ~/.mdm/config/datasets/customer_data.yaml
```

### Failed Auto-Detection Examples

#### No Kaggle Structure Found
```bash
# Directory without recognized structure
mdm dataset register my_data /data/custom_format

# Output:
✗ Auto-detection failed: No recognized dataset structure found.

Found files: data.csv, metadata.json
Expected structure for Kaggle datasets:
  - train.csv (or similar training file)
  - test.csv (or similar test file)
  - sample_submission.csv (for target detection)

Options:
1. Use --no-auto and specify files explicitly:
   mdm dataset register my_data --no-auto \
     --train /data/custom_format/data.csv \
     --target "target_column"

2. Reorganize your data to match expected structure
```

#### No Target Column Detected
```bash
# Kaggle structure found but no sample_submission.csv
mdm dataset register sales /data/sales

# Output:
✗ Registration failed: Target column not specified and could not be auto-detected.

Found: train.csv, test.csv
Missing: sample_submission.csv (needed for target auto-detection)

Please specify the target column:
  mdm dataset register sales /data/sales --target "revenue"

Note: When auto-detection fails, you must explicitly specify the target column.
```

## Updating Source Data

**Important**: MDM treats datasets as immutable once registered. If your source data files (e.g., `train.csv`) have changed, you have two options:

### Option 1: Register as New Dataset (Recommended)

Register the updated data as a new dataset with a versioned name:

```bash
# Original dataset
mdm dataset register sales_v1 /data/sales_jan2024

# Updated dataset with February data
mdm dataset register sales_v2 /data/sales_feb2024

# Or use date suffix
mdm dataset register sales_20240201 /data/sales_current
```

**Benefits**:
- Preserves original dataset for reproducibility
- Clear versioning and lineage
- Can compare results between versions
- Follows data science best practices

### Option 2: Force Overwrite (Use with Caution)

If you must replace the existing dataset:

```bash
# Remove existing dataset
mdm dataset remove sales_data

# Register updated data with same name
mdm dataset register sales_data /data/updated_sales
```

**Warning**: This destroys the original dataset and all its history. Use only for corrections or during development.

### Limitations of --force Flag

The `--force` flag only regenerates features. Specifically:
- Re-imports source data: ❌ (original CSVs remain unchanged)
- Regenerates feature tables: ✅
- Updates metadata: ✅
- Preserves original data tables: ✅

If you use `--force` expecting updated CSVs to be imported, your data will remain unchanged!

## Updating/Regenerating Features

For existing datasets, you may need to regenerate features when:
- Custom transformer code is updated (`~/.mdm/config/custom_features/{dataset_name}.py`)
- Feature engineering configuration changes in `mdm.yaml`
- You want to add new feature types or change signal detection thresholds

### Force Re-registration

Use the `--force` flag to update an existing dataset and regenerate all features:

```bash
# Re-register dataset with updated features
mdm dataset register titanic /path/to/kaggle/titanic --force

# Output:
⚠️  Dataset 'titanic' already exists. Force re-registration...
✓ Removing existing feature tables...
✓ Regenerating features with updated configuration...
✓ Dataset 'titanic' updated successfully!

Feature Generation:
- Generic features: 45 created (3 discarded - no signal)
- Custom features: 12 created (from titanic.py)
- Total features: 57 (was: 42)
```

### What Happens During Force Re-registration

1. **Preserves source data**: Original tables (`train`, `test`) remain unchanged
2. **Drops feature tables**: Removes `train_features`, `test_features`, `train_generic`, `train_custom`
3. **Re-runs feature generation**: Applies current configuration and transformers
4. **Updates metadata**: Refreshes dataset statistics and feature information
5. **Maintains configuration**: Keeps existing dataset YAML settings

### Important Notes

- **Data safety**: Source data is never modified or deleted
- **Downtime**: Dataset is temporarily unavailable during regeneration
- **Custom code**: Ensure custom transformer has no syntax errors before running
- **Configuration**: Changes to `mdm.yaml` only affect new feature generation

## Registration Options

For a complete list of registration options, see:
- [Command Line Interface](07_Command_Line_Interface.md) for all available flags
- [Best Practices](10_Best_Practices.md) for naming conventions
- [Troubleshooting](11_Troubleshooting.md) for common registration issues

## Next Steps

- Explore [Dataset Management Operations](05_Dataset_Management_Operations.md)
- Learn about [Database Backends](06_Database_Backends.md)
- See [Example Usage Scenarios](07_Command_Line_Interface.md#example-usage-scenarios)