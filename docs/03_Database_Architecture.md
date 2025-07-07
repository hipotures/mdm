# Database Architecture

MDM employs a simple two-tier database architecture designed for maximum portability and ease of use, with dataset discovery handled through directory scanning.

## Two-Tier Database System

```
┌─────────────────────────────┐
│  Dataset-Specific Databases │  ← SQLite/DuckDB/PostgreSQL
│  (Data + Local Metadata)    │     via SQLAlchemy ORM
│                             │     ~/.mdm/datasets/{name}/
└─────────────┬───────────────┘
              │

Dataset Discovery: Directory scanning + YAML configs
```

## Backend Selection and Configuration (Authoritative)

**This section provides the definitive explanation of how MDM handles database backend selection.**

### Key Principles

1. **Single Backend Architecture**
   - MDM uses ONE backend type for all datasets at any given time
   - The backend is determined by the `database.default_backend` setting in `~/.mdm/mdm.yaml`
   - All datasets must use the same backend type
   - Changing the backend in configuration affects what datasets are visible

2. **Backend Selection at Registration**
   - Backend selection happens ONLY at registration time
   - There is NO CLI parameter to override this setting
   - You must edit `mdm.yaml` BEFORE registering a dataset to use a different backend

3. **Backend Switching Behavior**
   - When you change `default_backend` in `mdm.yaml`, MDM will only show datasets matching the current backend
   - Datasets registered with different backends become invisible (but are not deleted)
   - To use datasets with a new backend, you must re-register them
   - Example: If you switch from SQLite to DuckDB, all SQLite datasets disappear from listings

### Backend Selection Process

```mermaid
graph TD
    A[User runs: mdm dataset register NAME PATH] --> B{Read mdm.yaml}
    B --> C[Get database.default_backend]
    C --> D{Backend Type}
    D -->|sqlite| E[SQLAlchemy creates SQLite database]
    D -->|duckdb| F[SQLAlchemy creates DuckDB database]
    D -->|postgresql| G[SQLAlchemy creates PostgreSQL database]
    E --> H[Create ~/.mdm/datasets/NAME/dataset.sqlite]
    F --> I[Create ~/.mdm/datasets/NAME/dataset.duckdb]
    G --> J[Create PostgreSQL database: mdm_NAME]
    H --> K[Write path to dataset YAML]
    I --> K
    J --> L[Write connection_string to dataset YAML]
    K --> M[Dataset registered via SQLAlchemy]
    L --> M
```

### Configuration Examples

**In `~/.mdm/mdm.yaml`:**
```yaml
database:
  default_backend: sqlite  # Options: sqlite (default), duckdb, postgresql
  
  # SQLAlchemy configuration (applies to all backends)
  sqlalchemy:
    echo: false           # Log SQL queries
    pool_size: 5         # Connection pool size
```

**Resulting dataset YAML for DuckDB:**
```yaml
name: titanic
description: Titanic passenger survival prediction
registered_at: "2024-01-15T10:30:00Z"
database:
  path: ~/.mdm/datasets/titanic/dataset.duckdb
tables:
  train: train
  test: test
target_column: Survived
id_columns: [PassengerId]
problem_type: binary_classification
```

**Resulting dataset YAML for PostgreSQL:**
```yaml
name: large_dataset
description: Large scale customer data
registered_at: "2024-01-15T11:45:00Z"
database:
  connection_string: postgresql://user:pass@localhost/mdm_large_dataset
tables:
  train: train
  test: test
target_column: customer_value
id_columns: [customer_id]
problem_type: regression
```

### Common Misconceptions

❌ **WRONG**: "The backend type is inferred from the file extension"
✅ **RIGHT**: The backend is chosen at registration based on `mdm.yaml`, then the appropriate file/connection is created

❌ **WRONG**: "You can specify backend in the dataset YAML"
✅ **RIGHT**: Dataset YAML only stores the connection info, not the backend type

❌ **WRONG**: "You can change backend with a CLI parameter"
✅ **RIGHT**: You must edit `mdm.yaml` before registration to change backend

### Changing Backend for New Datasets

To register datasets with different backends:

```bash
# Step 1: Check current backend
mdm info  # Shows current default_backend

# Step 2: Edit configuration
vim ~/.mdm/mdm.yaml
# Change: default_backend: sqlite

# Step 3: Register dataset (will use SQLite)
mdm dataset register my_data /path/to/data

# Step 4: Change back if needed
vim ~/.mdm/mdm.yaml
# Change: default_backend: duckdb
```

### Important Notes

- **Single Backend**: MDM only works with one backend type at a time
- **Backend Switching**: Changing backend hides all datasets using different backends
- **Re-registration Required**: To use existing data with a new backend, you must export and re-register
- **No Automatic Migration**: There is no built-in migration between backends
- **Default**: If `mdm.yaml` doesn't exist or `default_backend` is not set, SQLite is used (via SQLAlchemy)

## Dataset Discovery Mechanism

Instead of a central registry, MDM uses a directory-based discovery approach:
- **Dataset directories**: `~/.mdm/datasets/{dataset_name}/`
- **Configuration files**: `~/.mdm/config/datasets/{dataset_name}.yaml`
- **Discovery**: Scanning directories and reading YAML files
- **Trade-off**: Slightly slower search operations, but simpler and more portable

### Tier 1: Dataset-Specific Databases

Each dataset gets its own dedicated database:
- **Purpose**: Store both data tables and all dataset metadata
- **Technology**: SQLite (default), DuckDB, or PostgreSQL - all via SQLAlchemy
- **Location**: `~/.mdm/datasets/{dataset_name}/dataset.{db|duckdb}`
- **Content**: 
  - Data tables: train, test, validation, submission
  - Metadata table: dataset statistics, column information, quality metrics
  - Usage tracking and update timestamps
- **Benefits**: 
  - Complete isolation between datasets
  - Self-contained - all information in one file
  - Easy backup, move, or share by copying the directory
  - No dependency on external registry

### Tier 2: Feature Engineering Cache

Temporary storage for ML operations:
- **Purpose**: Cache generated features and intermediate results
- **Technology**: Temporary processing
- **Content**: Engineered features, transformations, aggregations
- **Lifecycle**: Cleared periodically or on-demand

## Local Metadata Storage

Each dataset database contains its own metadata tables for complete self-containment:

### Within each dataset database:

```sql
-- Metadata table stored in each dataset's database
CREATE TABLE _metadata (
    key VARCHAR(255) PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Example metadata entries
INSERT INTO _metadata (key, value) VALUES
    ('dataset_name', 'titanic'),
    ('display_name', 'Titanic Survival Dataset'),
    ('description', 'Passenger survival prediction dataset'),
    ('dataset_type', 'tabular'),
    ('problem_type', 'binary_classification'),
    ('target_column', 'survived'),
    ('registered_at', '2024-01-15T10:30:00Z'),
    ('last_updated_at', '2024-01-15T10:30:00Z'),
    ('row_count', '891'),
    ('column_count', '12');

-- Note: registered_at and last_updated_at are required for sorting datasets by date

-- Column metadata table
CREATE TABLE _columns (
    column_name VARCHAR(255) PRIMARY KEY,
    data_type VARCHAR(50),
    is_target BOOLEAN DEFAULT FALSE,
    is_id BOOLEAN DEFAULT FALSE,
    null_percentage FLOAT,
    unique_count INTEGER,
    metadata JSON
);

-- Data quality metrics table
CREATE TABLE _quality_metrics (
    metric_name VARCHAR(255) PRIMARY KEY,
    metric_value FLOAT,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

```

## Column Name Handling

MDM uses different column naming conventions for different table types:

### Source Tables (`train`, `test`, `validation`)
- **Preserve original column names** exactly as they appear in the source files
- Example: `CustomerID`, `First_Name`, `purchaseDate`
- SQL queries must use the original case

### Feature Tables (`train_features`, `test_features`)
- **All column names normalized to lowercase**
- This includes:
  - Original columns from source tables
  - Generated feature columns
- Example: `customerid`, `first_name`, `purchasedate`, `age_binned`
- SQL queries must use lowercase

### Example:
```python
# Source table preserves original case
df_train = dataset_manager.load_table("sales", "train")
print(df_train.columns)  # ['CustomerID', 'ProductName', 'SalePrice']

# Feature table uses all lowercase
df_features = dataset_manager.load_table("sales", "train_features")
print(df_features.columns)  # ['customerid', 'productname', 'saleprice', 'saleprice_log', ...]

# SQL queries must match the convention
conn.execute("SELECT CustomerID FROM train")  # Original case for source
conn.execute("SELECT customerid FROM train_features")  # Lowercase for features
```

**Rationale**: This design ensures consistency in generated features while preserving original data fidelity.

## YAML Configuration Files

Each dataset has a lightweight YAML configuration file in `~/.mdm/config/datasets/{name}.yaml`:

```yaml
# Example: ~/.mdm/config/datasets/titanic.yaml
name: titanic
display_name: Titanic Survival Dataset
database:
  path: ~/.mdm/datasets/titanic/dataset.duckdb
tables:
  train: train
  test: test
problem_type: binary_classification
target_column: survived
registered_at: 2024-01-15T10:30:00Z
```

These YAML files serve as:
- **Discovery pointers**: MDM scans this directory to find datasets
- **Quick metadata**: Basic info without opening the database
- **Configuration**: Database path and table mappings (backend is determined from mdm.yaml)

## Directory Structure

```
~/.mdm/
├── datasets/
│   ├── titanic/
│   │   └── dataset.duckdb      # Contains data + metadata tables
│   └── house_prices/
│       └── dataset.duckdb
├── config/
│   └── datasets/
│       ├── titanic.yaml        # Configuration pointer
│       └── house_prices.yaml
└── tmp/                        # Temporary files
```

### Feature Engineering Tables

During registration, MDM applies a sophisticated two-tier feature engineering system:

1. **Generic Feature Transformers** - Automatically applied to all datasets
2. **Custom Feature Transformers** - Dataset-specific transformations (optional)

#### Feature Table Creation

MDM creates feature tables through a multi-step process:

1. **Source Tables** (unchanged): `train`, `test`, `validation`
   - Preserve original column names and data types
   - No modifications to source data

2. **Intermediate Tables** (temporary):
   - `train_generic`, `test_generic`: Generic features only (lowercase columns)
   - `train_custom`, `test_custom`: Custom features only (if custom transformer exists, lowercase)

3. **Final Feature Tables**: `train_features`, `test_features`
   - Combine source + generic + custom features
   - ALL columns normalized to lowercase
   - Features without signal automatically excluded

Example table progression:
```
train (original: CustomerID, Age, Name)
  ├→ train_generic (age_binned, age_zscore, name_length)
  ├→ train_custom (family_size, is_vip)
  └→ train_features (customerid, age, name, age_binned, age_zscore, name_length, family_size, is_vip)
```

#### Generic Feature Transformers

Built-in transformers that automatically detect and process column types:

```python
# Temporal Features - Applied to datetime columns
class TemporalFeatures(GenericFeatureOperation):
    """
    Extracts:
    - Date components: year, month, day, weekday, hour, minute
    - Cyclical encodings: sin/cos transforms for periodic features
    - Time differences: days_since_start, is_weekend, is_holiday
    - Lag features: previous values for time series
    """

# Categorical Features - Applied to low-cardinality columns
class CategoricalFeatures(GenericFeatureOperation):
    """
    Creates:
    - One-hot encoding for categories with <50 unique values
    - Target encoding with cross-validation
    - Frequency encoding
    - Rare category grouping
    """

# Statistical Features - Applied to numeric columns
class StatisticalFeatures(GenericFeatureOperation):
    """
    Generates:
    - Log transformations (for positive values)
    - Standardized values (z-score)
    - Percentile ranks
    - Outlier indicators
    """

# Text Features - Applied to text columns
class TextFeatures(GenericFeatureOperation):
    """
    Extracts:
    - Text length, word count, character count
    - Contains digit/special character indicators
    - Average word length
    - Text complexity metrics
    """

# Binning Features - Applied to continuous numeric columns
class BinningFeatures(GenericFeatureOperation):
    """
    Creates:
    - Equal-width bins
    - Quantile-based bins
    - Custom threshold bins
    """
```

#### Custom Feature Transformers

Dataset-specific transformers are automatically loaded from `~/.mdm/config/custom_features/{dataset_name}.py`:

```python
# Example: ~/.mdm/config/custom_features/titanic.py
class CustomFeatureOperations(BaseDomainFeatures):
    """Custom features for Titanic dataset."""
    
    def _register_operations(self):
        self._operation_registry = {
            'family_features': self.get_family_features,
            'title_features': self.get_title_features,
            'cabin_features': self.get_cabin_features,
        }
    
    def get_family_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Family-related features specific to Titanic."""
        features = {}
        features['family_size'] = df['sibsp'] + df['parch'] + 1
        features['is_alone'] = (features['family_size'] == 1).astype(int)
        features['has_child'] = ((df['age'] < 18) | (df['parch'] > 0)).astype(int)
        return features
```

#### Signal Detection

All generated features undergo automatic signal detection to ensure quality:

```python
class FeatureTimingMixin:
    def _check_feature_signal(self, feature: pd.Series) -> bool:
        """
        Checks if feature has signal (variance in values).
        
        Features are discarded if:
        - All values are identical
        - All values are null
        - Only one unique non-null value exists
        
        For large datasets (>2000 rows), uses sampling for efficiency.
        """
```

**Signal Detection Process:**
1. Each generated feature is checked for signal
2. Features with <1% unique values are flagged
3. Constant features are automatically removed
4. Process is logged: `[no signal, discarded]`

### Usage Tracking
- Feature generation history
- Best performing feature sets
- Session tracking
- Usage frequency

## Architecture Benefits

### 1. True Decentralization
- No central point of failure
- Each dataset completely self-contained
- Works offline or in isolated environments

### 2. Simplicity
- No registry service to maintain
- Simple directory structure
- Easy to understand and debug

### 3. Portability
- Copy a directory to move a dataset
- No database migrations needed
- Works across different systems

### 4. Flexibility
- Single backend ensures consistency
- Easy manual dataset addition
- Simple backup and restore

## Implementation Notes

### Dataset Discovery
MDM discovers datasets through directory scanning:
- Scans `~/.mdm/config/datasets/` for YAML files
- Each YAML points to a dataset database
- No network calls or service dependencies

### Performance Trade-offs
- **List operations**: Fast - just directory listing
- **Search operations**: Slower - must read YAML files
- **Metadata queries**: Fast when working with single dataset
- **Cross-dataset queries**: Slower - must open multiple databases

### Data Integrity
- Each database maintains its own integrity
- No cross-dataset foreign keys
- Complete isolation between datasets

## Next Steps

- Learn about [Dataset Registration](04_Dataset_Registration.md) process
- Explore [Database Backends](06_Database_Backends.md) options
- See [Best Practices](10_Best_Practices.md) for architecture decisions