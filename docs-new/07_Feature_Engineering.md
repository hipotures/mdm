# Feature Engineering

MDM includes a powerful two-tier feature engineering system that automatically generates features during dataset registration. This guide covers the architecture, usage, and customization of the feature system.

## Overview

### Two-Tier Architecture

```
Feature Engineering System
├── Tier 1: Generic Transformers (Automatic)
│   ├── Statistical Features
│   ├── Temporal Features
│   ├── Categorical Features
│   ├── Text Features
│   ├── Missing Data Features
│   └── Distribution Features
│
└── Tier 2: Custom Transformers (User-Defined)
    └── Dataset-specific features via plugins
```

### Key Concepts

1. **Automatic Generation**: Features are generated during dataset registration
2. **Type-Based Selection**: Transformers are applied based on column types
3. **Signal Detection**: Low-signal features are automatically removed
4. **Extensibility**: Add custom features via Python plugins
5. **Performance**: Batch processing with progress tracking

## Generic Feature Transformers

### 1. Statistical Features

Applied to numeric columns to capture distribution characteristics.

**Generated Features:**
- `{col}_zscore`: Standardized values (mean=0, std=1)
- `{col}_log`: Log transformation (for positive values)
- `{col}_sqrt`: Square root transformation
- `{col}_squared`: Squared values
- `{col}_is_outlier`: Binary outlier indicator (>3 std dev)
- `{col}_percentile_rank`: Percentile ranking (0-100)

**Example:**
```python
# Original column: price = [10, 20, 30, 100]
# Generated:
# price_zscore = [-0.84, -0.52, -0.21, 1.57]
# price_log = [2.30, 2.99, 3.40, 4.60]
# price_is_outlier = [0, 0, 0, 1]
```

**Configuration:**
```yaml
features:
  statistical:
    enable_log_transform: true
    outlier_threshold: 3.0  # Standard deviations
    min_variance: 0.01      # Skip if variance below this
```

### 2. Temporal Features

Extracts time components and patterns from datetime columns.

**Generated Features:**
- `{col}_year`: Year component
- `{col}_month`: Month (1-12)
- `{col}_day`: Day of month
- `{col}_dayofweek`: Day of week (0=Monday)
- `{col}_hour`: Hour (for timestamps)
- `{col}_quarter`: Quarter (1-4)
- `{col}_is_weekend`: Binary weekend indicator
- `{col}_is_month_start`: First day of month
- `{col}_is_month_end`: Last day of month
- `{col}_days_since_epoch`: Days since 1970-01-01

**Cyclical Encoding (Optional):**
- `{col}_month_sin`, `{col}_month_cos`: Cyclical month encoding
- `{col}_day_sin`, `{col}_day_cos`: Cyclical day encoding
- `{col}_hour_sin`, `{col}_hour_cos`: Cyclical hour encoding

**Example:**
```python
# Original: order_date = '2024-03-15 14:30:00'
# Generated:
# order_date_year = 2024
# order_date_month = 3
# order_date_day = 15
# order_date_dayofweek = 4  # Friday
# order_date_hour = 14
# order_date_is_weekend = 0
# order_date_quarter = 1
```

**Configuration:**
```yaml
features:
  temporal:
    enable_cyclical: true    # Sin/cos transformations
    enable_lag: false        # Time series lags (experimental)
    components: [year, month, day, dayofweek, hour, quarter]
```

### 3. Categorical Features

Encodes categorical variables for ML algorithms.

**Generated Features:**
- **One-Hot Encoding**: `{col}_{value}` binary columns (if cardinality < threshold)
- **Frequency Encoding**: `{col}_frequency` - occurrence count
- **Target Encoding**: `{col}_target_mean` - mean target per category (if target specified)
- **Count Encoding**: `{col}_count` - number of occurrences

**Example:**
```python
# Original: color = ['red', 'blue', 'red', 'green']
# Generated (one-hot):
# color_red = [1, 0, 1, 0]
# color_blue = [0, 1, 0, 0]
# color_green = [0, 0, 0, 1]
# color_frequency = [2, 1, 2, 1]
```

**Configuration:**
```yaml
features:
  categorical:
    max_cardinality: 50      # Max unique values for one-hot
    min_frequency: 0.01      # Min frequency to create feature
    enable_target_encoding: true
    handle_unknown: ignore   # How to handle new categories
```

### 4. Text Features

Extracts characteristics from text columns.

**Generated Features:**
- `{col}_length`: Character count
- `{col}_word_count`: Number of words
- `{col}_unique_words`: Unique word count
- `{col}_avg_word_length`: Average word length
- `{col}_special_char_count`: Special characters
- `{col}_digit_count`: Number of digits
- `{col}_uppercase_ratio`: Ratio of uppercase letters
- `{col}_has_url`: Contains URL (binary)
- `{col}_has_email`: Contains email (binary)
- `{col}_sentiment_*`: Sentiment scores (if enabled)

**Example:**
```python
# Original: description = "Great product! Worth $99.99"
# Generated:
# description_length = 27
# description_word_count = 4
# description_special_char_count = 3
# description_digit_count = 4
# description_has_url = 0
```

**Configuration:**
```yaml
features:
  text:
    min_text_length: 20      # Minimum length to process
    enable_sentiment: false  # Sentiment analysis (slower)
    max_features: 100       # For TF-IDF/BoW (if enabled)
```

### 5. Missing Data Features

Captures patterns in missing data.

**Generated Features:**
- `{col}_is_missing`: Binary indicator for null values
- `missing_count`: Total missing values per row
- `missing_ratio`: Ratio of missing values per row

**Example:**
```python
# Original row: [10, None, 'A', None, 5]
# Generated:
# col2_is_missing = 1
# col4_is_missing = 1
# missing_count = 2
# missing_ratio = 0.4
```

### 6. Distribution Features

Bins continuous variables and creates distribution-based features.

**Generated Features:**
- `{col}_binned`: Discretized into equal-width bins
- `{col}_quantile_binned`: Discretized by quantiles
- `{col}_decile`: Decile ranking (1-10)

**Configuration:**
```yaml
features:
  distribution:
    n_bins: 10              # Number of bins
    strategy: quantile      # 'uniform' or 'quantile'
```

### 7. Interaction Features

Creates feature interactions (experimental).

**Generated Features:**
- `{col1}_x_{col2}`: Multiplication of numeric features
- `{col1}_div_{col2}`: Ratios (with zero handling)
- `{cat1}_{cat2}`: Categorical combinations

### 8. Sequential Features

For ordered/time series data (experimental).

**Generated Features:**
- `{col}_lag_{n}`: Previous n values
- `{col}_rolling_mean_{n}`: Rolling average
- `{col}_rolling_std_{n}`: Rolling standard deviation
- `{col}_diff`: First difference

## Custom Feature Transformers

### Creating Custom Features

Custom features allow domain-specific transformations. Create a Python file in `~/.mdm/config/custom_features/{dataset_name}.py`:

```python
from mdm.features.custom.base import BaseDomainFeatures
from typing import Dict
import pandas as pd
import numpy as np

class CustomFeatureOperations(BaseDomainFeatures):
    """Custom features for sales dataset."""
    
    def __init__(self):
        # Must match your dataset name
        super().__init__('sales_dataset')
    
    def _register_operations(self):
        """Register all feature operations."""
        self._operation_registry = {
            'revenue_features': self.calculate_revenue_features,
            'customer_features': self.calculate_customer_features,
            'time_features': self.calculate_time_features,
            'product_features': self.calculate_product_features
        }
    
    def calculate_revenue_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Revenue-based features."""
        features = {}
        
        # Basic revenue
        features['revenue'] = df['price'] * df['quantity']
        
        # Revenue tiers
        revenue = features['revenue']
        features['revenue_tier'] = pd.cut(
            revenue,
            bins=[0, 100, 500, 1000, np.inf],
            labels=['small', 'medium', 'large', 'enterprise']
        ).astype(str)
        
        # High-value indicator
        features['is_high_value'] = (revenue > revenue.quantile(0.9)).astype(int)
        
        # Discount features
        if 'list_price' in df and 'price' in df:
            features['discount_amount'] = df['list_price'] - df['price']
            features['discount_percent'] = (
                features['discount_amount'] / df['list_price']
            ).fillna(0) * 100
            
        return features
    
    def calculate_customer_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Customer-based features."""
        features = {}
        
        if 'customer_id' not in df:
            return features
        
        # Customer purchase patterns
        customer_stats = df.groupby('customer_id').agg({
            'price': ['sum', 'mean', 'count']
        }).reset_index()
        
        # Merge back
        df_with_stats = df.merge(
            customer_stats,
            on='customer_id',
            how='left'
        )
        
        features['customer_total_spent'] = df_with_stats[('price', 'sum')]
        features['customer_avg_order'] = df_with_stats[('price', 'mean')]
        features['customer_order_count'] = df_with_stats[('price', 'count')]
        
        # VIP customer flag
        features['is_vip_customer'] = (
            features['customer_total_spent'] > 
            features['customer_total_spent'].quantile(0.8)
        ).astype(int)
        
        return features
    
    def calculate_time_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Time-based features."""
        features = {}
        
        if 'date' not in df:
            return features
        
        # Ensure datetime
        date_col = pd.to_datetime(df['date'])
        
        # Shopping patterns
        features['is_weekend_purchase'] = date_col.dt.dayofweek.isin([5, 6]).astype(int)
        features['is_holiday_season'] = date_col.dt.month.isin([11, 12]).astype(int)
        features['days_since_year_start'] = date_col.dt.dayofyear
        
        # Time-based aggregations
        if 'customer_id' in df:
            # Days since last purchase per customer
            df_sorted = df.sort_values(['customer_id', 'date'])
            features['days_since_last_purchase'] = (
                df_sorted.groupby('customer_id')['date']
                .diff()
                .dt.days
                .fillna(0)
            )
        
        return features
    
    def calculate_product_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Product-based features."""
        features = {}
        
        if 'product' not in df or 'category' not in df:
            return features
        
        # Product popularity
        product_counts = df['product'].value_counts()
        features['product_popularity'] = df['product'].map(product_counts)
        
        # Category statistics
        category_stats = df.groupby('category')['price'].agg(['mean', 'std'])
        df_with_cat_stats = df.merge(
            category_stats,
            left_on='category',
            right_index=True,
            how='left'
        )
        
        features['category_avg_price'] = df_with_cat_stats['mean']
        features['price_vs_category_avg'] = (
            df['price'] - features['category_avg_price']
        )
        
        # Cross-features
        features['product_category_pair'] = (
            df['product'].astype(str) + '_' + df['category'].astype(str)
        )
        
        return features
```

### Best Practices for Custom Features

1. **Handle Missing Values**
```python
def safe_divide(a, b, fill_value=0):
    """Safe division with zero handling."""
    return np.where(b != 0, a / b, fill_value)

features['ratio'] = safe_divide(df['numerator'], df['denominator'])
```

2. **Avoid Data Leakage**
```python
# Bad: Uses future information
features['next_day_sales'] = df['sales'].shift(-1)

# Good: Uses only past information
features['prev_day_sales'] = df['sales'].shift(1)
```

3. **Maintain Consistency**
```python
# Ensure consistent encoding
category_mapping = {'A': 1, 'B': 2, 'C': 3}
features['category_encoded'] = df['category'].map(category_mapping).fillna(0)
```

4. **Document Features**
```python
def calculate_complex_feature(self, df):
    """
    Calculate customer lifetime value proxy.
    
    Formula: total_spent * frequency * recency_score
    Where recency_score = 1 / (days_since_last_purchase + 1)
    """
    # Implementation
```

## Feature Generation Process

### Registration Flow

```
1. Column Type Detection
   ├── Numeric → Statistical, Distribution
   ├── Datetime → Temporal
   ├── Categorical → Categorical Encoding
   └── Text → Text Features

2. Feature Generation
   ├── Apply Generic Transformers
   ├── Apply Custom Transformers (if exists)
   └── Combine Results

3. Signal Detection
   ├── Remove Zero Variance
   ├── Remove High Correlation (>0.95)
   └── Remove Sparse Features (<1% non-zero)

4. Storage
   ├── Create Feature Tables
   └── Update Metadata
```

### Feature Tables

MDM creates structured feature tables:

```sql
-- Original table: train
CREATE TABLE train (
    order_id INTEGER,
    date TEXT,
    product TEXT,
    price REAL,
    quantity INTEGER
);

-- Generic features table: train_generic
CREATE TABLE train_generic (
    price_zscore REAL,
    price_log REAL,
    date_year INTEGER,
    date_month INTEGER,
    product_electronics INTEGER,  -- one-hot
    product_furniture INTEGER,    -- one-hot
    ...
);

-- Custom features table: train_custom
CREATE TABLE train_custom (
    revenue REAL,
    is_high_value INTEGER,
    customer_total_spent REAL,
    ...
);

-- Combined features table: train_features
CREATE TABLE train_features AS
SELECT * FROM train
NATURAL JOIN train_generic
NATURAL JOIN train_custom;
```

## Configuration Options

### Global Feature Configuration

```yaml
# ~/.mdm/mdm.yaml
features:
  # Master switch
  enable_at_registration: true
  
  # Performance
  batch_size: 10000
  n_jobs: -1  # Use all CPU cores
  
  # Signal detection
  min_variance: 0.01
  max_correlation: 0.95
  min_non_zero_ratio: 0.01
  
  # Feature types
  statistical:
    enabled: true
    features: [zscore, log, sqrt, outlier]
    outlier_method: iqr  # or 'zscore'
    
  temporal:
    enabled: true
    extract_components: true
    cyclical_encoding: true
    
  categorical:
    enabled: true
    max_cardinality: 50
    encoding_method: onehot  # or 'target', 'frequency'
    
  text:
    enabled: true
    min_length: 20
    extract_entities: false  # NER (slower)
    
  custom:
    enabled: true
    auto_discover: true
    paths: ["~/.mdm/config/custom_features"]
```

### Per-Dataset Configuration

Override global settings for specific datasets:

```yaml
# ~/.mdm/config/datasets/sales.yaml
features:
  statistical:
    outlier_threshold: 2.5  # More sensitive
    
  categorical:
    max_cardinality: 100  # Allow more categories
    
  custom:
    module: "sales_advanced"  # Use different module
```

## Monitoring Feature Generation

### During Registration

```
[INFO] Starting feature generation for 'sales_data'...
[INFO] Detected column types:
  - Numeric: ['price', 'quantity', 'discount']
  - Categorical: ['product', 'category', 'region']
  - Datetime: ['order_date', 'ship_date']
  - Text: ['description', 'notes']

Processing features: ━━━━━━━━━━━━━━━━ 100% 
  Statistical: 15 features (2 removed - no signal)
  Temporal: 20 features
  Categorical: 25 features (5 removed - low frequency)
  Text: 16 features
  Custom: 12 features

[INFO] Feature generation complete:
  - Original columns: 10
  - Generated features: 88
  - Total features: 98
  - Processing time: 3.4s
```

### Feature Quality Report

```python
# Get feature statistics
from mdm import MDMClient

client = MDMClient()
info = client.get_dataset_info("sales")

# Analyze feature quality
features_df = client.load_dataset("sales", limit=1000)
feature_cols = [col for col in features_df.columns if col not in info.columns["original"]]

# Check distributions
for col in feature_cols[:5]:
    print(f"\n{col}:")
    print(f"  Non-null: {features_df[col].notna().sum()}")
    print(f"  Unique: {features_df[col].nunique()}")
    print(f"  Variance: {features_df[col].var():.4f}")
```

## Performance Optimization

### 1. Batch Processing

```python
# Configure batch size based on memory
export MDM_FEATURES_BATCH_SIZE=50000  # Larger batches
```

### 2. Selective Features

```python
# Register with only specific features
client.register_dataset(
    "large_dataset",
    "/data/large.csv",
    feature_config={
        "statistical": {"enabled": True},
        "temporal": {"enabled": False},  # Skip
        "categorical": {"enabled": True, "max_cardinality": 20},
        "text": {"enabled": False}  # Skip
    }
)
```

### 3. Parallel Processing

```yaml
features:
  n_jobs: 8  # Use 8 CPU cores
  backend: threading  # or 'multiprocessing'
```

### 4. Caching

```python
# Features are cached in database
# To regenerate features:
client.register_dataset("sales", "/data/sales.csv", force=True)
```

## Advanced Topics

### Feature Selection

```python
# After generation, select best features
from sklearn.feature_selection import SelectKBest, f_classif

# Load features
X = client.load_dataset("sales", table="train_features")
y = X[info.target_column]
X = X.drop(columns=[info.target_column])

# Select top 50 features
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
```

### Feature Validation

```python
def validate_features(dataset_name):
    """Check feature quality."""
    client = MDMClient()
    df = client.load_dataset(dataset_name)
    
    issues = []
    
    # Check for constant features
    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append(f"{col}: constant value")
    
    # Check for high correlation
    corr = df.corr().abs()
    high_corr = np.where(np.triu(corr > 0.95, 1))
    for i, j in zip(*high_corr):
        issues.append(
            f"{df.columns[i]} - {df.columns[j]}: "
            f"correlation = {corr.iloc[i, j]:.3f}"
        )
    
    return issues
```

### Custom Feature Debugging

```python
# Test custom features before registration
from mdm.features.custom.base import BaseDomainFeatures

# Load your custom features
import sys
sys.path.append("~/.mdm/config/custom_features")
from sales import CustomFeatureOperations

# Test on sample data
ops = CustomFeatureOperations()
sample_df = pd.read_csv("/data/sales.csv", nrows=100)

# Test each operation
for name, func in ops._operation_registry.items():
    print(f"\nTesting {name}:")
    try:
        features = func(sample_df)
        print(f"  Generated: {list(features.keys())}")
    except Exception as e:
        print(f"  ERROR: {e}")
```

## FAQ

### Q: How do I disable feature generation?
```bash
mdm dataset register mydata data.csv --no-features
```

### Q: Can I add features after registration?
Currently, you need to re-register with `--force`:
```bash
# Add custom features file
# Then re-register
mdm dataset register mydata data.csv --force
```

### Q: How do I see what features were generated?
```python
info = client.get_dataset_info("mydata")
print("Original columns:", info.columns["original"])
print("Feature columns:", info.columns["features"])
```

### Q: Why were some features removed?
Features are removed if they have:
- Zero variance (all same value)
- Very low variance (<0.01)
- High correlation with existing features (>0.95)
- Too sparse (<1% non-zero values)