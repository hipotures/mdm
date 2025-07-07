# Advanced Features

MDM provides several advanced features that enhance its capabilities beyond basic dataset management. These features enable sophisticated workflows and integration with complex ML pipelines.

## 1. Feature Engineering System

MDM includes a comprehensive two-tier feature engineering system that automatically generates features during dataset registration.

### Architecture Overview

```
Feature Engineering Pipeline
├── Generic Transformers (Automatic)
│   ├── TemporalFeatures      → DateTime columns
│   ├── CategoricalFeatures   → Low cardinality columns
│   ├── StatisticalFeatures   → Numeric columns
│   ├── TextFeatures          → Text columns
│   └── BinningFeatures       → Continuous numeric
└── Custom Transformers (Optional)
    └── {dataset_name}.py     → Dataset-specific features
```

### Configuring Feature Generation

Feature generation can be configured in `mdm.yaml`:

```yaml
feature_engineering:
  enabled: true
  generic_features:
    temporal:
      enable_cyclical: true     # sin/cos transforms
      enable_lag: false         # Time series lags
    categorical:
      max_cardinality: 50       # Max unique values for one-hot
      enable_target_encoding: true
    statistical:
      enable_log_transform: true
      outlier_threshold: 3.0    # Standard deviations
    text:
      min_text_length: 20       # Minimum length to process
  custom_features:
    auto_discover: true         # Auto-load from custom_features/
```

### Creating Custom Transformers

To add dataset-specific features, create `~/.mdm/custom_features/{dataset_name}.py`:

**Note**: After creating or updating custom transformers, use `mdm dataset register {dataset_name} /path --force` to regenerate features.

```python
from mdm.features.custom.base import BaseDomainFeatures
from typing import Dict
import pandas as pd
import numpy as np

class CustomFeatureOperations(BaseDomainFeatures):
    """Custom features for your dataset."""
    
    def __init__(self):
        super().__init__('your_dataset_name')
        
    def _register_operations(self):
        """Register all feature operations."""
        self._operation_registry = {
            'domain_specific_features': self.get_domain_features,
            'interaction_features': self.get_interactions,
            'advanced_features': self.get_advanced_features,
        }
    
    def get_domain_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create domain-specific features."""
        features = {}
        
        # Example: Ratio features
        if 'column_a' in df and 'column_b' in df:
            features['a_b_ratio'] = df['column_a'] / (df['column_b'] + 1)
            
        # Example: Conditional features
        features['is_high_value'] = (df['value'] > df['value'].quantile(0.9)).astype(int)
        
        return features
    
    def get_interactions(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create interaction features."""
        features = {}
        
        # Example: Multiplicative interactions
        if 'feature_1' in df and 'feature_2' in df:
            features['f1_x_f2'] = df['feature_1'] * df['feature_2']
            
        return features
```

### Signal Detection in Detail

Signal detection ensures only informative features are retained:

```python
class SignalDetector:
    """Detects and removes features without signal."""
    
    def has_signal(self, feature: pd.Series, threshold: float = 0.01) -> bool:
        """
        Check if feature has sufficient signal.
        
        Returns False if:
        - All values identical (no variance)
        - All values are null
        - Unique ratio < threshold (default 1%)
        - Binary feature with extreme imbalance (>99% one class)
        """
        # Null check
        if feature.isna().all():
            return False
            
        # Variance check
        if feature.nunique() <= 1:
            return False
            
        # Unique ratio check
        unique_ratio = feature.nunique() / len(feature)
        if unique_ratio < threshold:
            return False
            
        # Binary imbalance check
        if feature.nunique() == 2:
            value_counts = feature.value_counts(normalize=True)
            if value_counts.max() > 0.99:
                return False
                
        return True
```

### Feature Tables Structure

MDM creates feature tables through intermediate steps for better organization:

#### Table Creation Pipeline

```
Source Tables (Original Case)     Intermediate Tables (Lowercase)     Final Tables (All Lowercase)
─────────────────────────────    ─────────────────────────────────    ───────────────────────────
train                        →   train_generic                    ↘
  CustomerID                       age_binned                       train_features
  Age                             age_zscore                         customerid
  Name                            name_length                        age
                                                                     name
                            →   train_custom (if exists)       ↗    age_binned
                                  family_size                        age_zscore
                                  is_vip_customer                    name_length
                                                                     family_size
                                                                     is_vip_customer
```

#### Column Naming Convention

```sql
-- Original table: train (1000 rows, 10 columns) 
-- Columns: CustomerID, Age, PurchaseDate, ProductCategory, Description

-- Intermediate generic features: train_generic
-- All lowercase, only generated features:
age_binned, age_zscore, purchasedate_year, purchasedate_month,
productcategory_electronics, productcategory_clothing,
description_length, description_word_count

-- Intermediate custom features: train_custom (if custom transformer exists)
-- All lowercase, only custom features:
customer_lifetime_value, days_since_last_purchase, is_frequent_buyer

-- Final feature table: train_features (1000 rows, 25 columns)
-- ALL columns lowercase, combines everything:
customerid, age, purchasedate, productcategory, description,  -- original (lowercase)
age_binned, age_zscore, purchasedate_year, purchasedate_month,  -- generic
productcategory_electronics, productcategory_clothing,  -- generic
description_length, description_word_count,  -- generic
customer_lifetime_value, days_since_last_purchase, is_frequent_buyer  -- custom
```

**Important**: The intermediate tables (`train_generic`, `train_custom`) are temporary and may be optionally persisted for debugging. The final `train_features` table is always created and contains all features with lowercase column names.

### Performance Optimization

Feature generation is optimized for large datasets:

1. **Parallel Processing**: Features generated in parallel when possible
2. **Chunking**: Large datasets processed in chunks based on `performance.batch_size`
3. **Batch Processing**: 
   - Configurable batch size (default: 10,000 rows)
   - Progress indicators show current batch and estimated completion
   - Memory-efficient processing for datasets larger than RAM
4. **Optimization**: Intermediate results optimized for reuse
5. **Early Exit**: Signal detection stops processing early for constant features

**Batch Processing Details:**
```python
# MDM automatically batches large operations
# Progress shown during processing:
# Processing features: ━━━━━━━━━━━━━━━━ 60% Batch 6/10 [12,500 rows/s]

# Configure batch size via environment:
export MDM_BATCH_SIZE=25000  # Larger batches for more memory/speed
```

### Monitoring Feature Generation

During registration, detailed logs show the feature generation process:

```
[INFO] Starting feature generation for dataset 'sales_data'...
[INFO] Detected column types:
  - Datetime: ['order_date', 'ship_date'] 
  - Categorical: ['region', 'category', 'segment']
  - Numeric: ['sales', 'quantity', 'discount']
  - Text: ['product_name', 'customer_notes']

[INFO] Applying generic transformers...
[TemporalFeatures] Generated 14 features in 0.34s (2 discarded - no signal)
[CategoricalFeatures] Generated 23 features in 0.67s 
[StatisticalFeatures] Generated 12 features in 0.23s
[TextFeatures] Generated 8 features in 0.45s (1 discarded - no signal)

[INFO] Applying custom transformers...
[CustomFeatures] Found sales_data.py
[CustomFeatures] Generated 15 features in 0.89s

[INFO] Feature generation complete:
  - Total features: 72 (original: 10, generated: 62)
  - Features discarded: 3 (no signal)
  - Total time: 2.58s
  - Feature table size: 145.3 MB
```

## 2. Performance Optimization

MDM provides several optimization strategies for different use cases.

### Database Backend Selection

Backend selection is configured through the `~/.mdm/mdm.yaml` configuration file. The backend is determined at dataset registration time and cannot be changed later.

```yaml
# In ~/.mdm/mdm.yaml
database:
  default_backend: duckdb  # Options: duckdb, sqlite, postgresql
```

Choose the right backend based on your needs:

- **DuckDB**: Best for analytical queries on medium to large datasets
  - Columnar storage optimized for aggregations
  - Excellent compression ratios
  - Native Parquet support
  
- **SQLite**: Good for small datasets or when portability is key
  - Minimal overhead
  - Zero configuration
  - Wide compatibility
  
- **PostgreSQL**: When you need multi-user access or advanced features
  - Concurrent access control
  - Advanced indexing options
  - Enterprise features

**Note**: The BackendFactory automatically handles both string and enum backend types for compatibility.

### Query Optimization

```python
# Optimize queries with proper indexing
with dataset_manager.get_dataset_connection("large_dataset") as conn:
    # Create indexes for frequently queried columns
    conn.execute("CREATE INDEX idx_date ON train(date_column)")
    conn.execute("CREATE INDEX idx_category ON train(category_column)")
    
    # Use optimized queries
    result = conn.execute("""
        SELECT category_column, AVG(target) as avg_target
        FROM train
        WHERE date_column >= '2023-01-01'
        GROUP BY category_column
        HAVING COUNT(*) > 100
    """).fetch_df()
```

### Memory Management

```python
# Process large datasets in chunks
def process_large_dataset(dataset_name, chunk_size=None):
    """Process dataset in memory-efficient chunks"""
    # Use MDM's configured batch_size if not specified
    if chunk_size is None:
        from mdm.core.config import get_settings
        chunk_size = get_settings().batch_size
    
    conn = dataset_manager.get_dataset_connection(dataset_name)
    
    # Get total rows
    total_rows = conn.execute("SELECT COUNT(*) FROM train").fetchone()[0]
    total_batches = (total_rows + chunk_size - 1) // chunk_size
    
    results = []
    # MDM would show progress like this:
    # Processing: ━━━━━━━━━━━━━━━━ 40% Batch 4/10 [8,234 rows/s, ETA: 45s]
    for batch_num, offset in enumerate(range(0, total_rows, chunk_size)):
        chunk = conn.execute(f"""
            SELECT * FROM train 
            LIMIT {chunk_size} 
            OFFSET {offset}
        """).fetch_df()
        
        # Process chunk
        result = process_chunk(chunk)
        results.append(result)
        
        # Progress would be shown automatically by MDM
        print(f"Processed batch {batch_num + 1}/{total_batches}")
    
    return pd.concat(results)
```

**Batch Size Configuration:**
- Default: 10,000 rows per batch
- Configure via `~/.mdm/mdm.yaml` or `MDM_BATCH_SIZE` environment variable
- Larger batches = faster processing but more memory usage
- Smaller batches = safer for limited memory systems

### Compression Strategies

- **Automatic compression** in DuckDB reduces storage by 3-10x
- **Parquet format** provides excellent compression for columnar data
- **Table partitioning** for very large datasets

## 3. Integration Features

MDM integrates seamlessly with popular ML frameworks and tools.

### Direct SQL Access

```python
# Execute complex analytical queries
with dataset_manager.get_dataset_connection("sales_data") as conn:
    # Window functions
    result = conn.execute("""
        SELECT 
            date,
            revenue,
            AVG(revenue) OVER (
                ORDER BY date 
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) as moving_avg_7d
        FROM train
        ORDER BY date
    """).fetch_df()
```

### DataFrame Loading

```python
# Load with automatic type inference
train_df = dataset_manager.load_table("titanic", "train")

# Load with specific dtypes
dtypes = {
    'PassengerId': 'int32',
    'Survived': 'int8',
    'Pclass': 'category',
    'Name': 'string',
    'Age': 'float32'
}
train_df = dataset_manager.load_table("titanic", "train", dtype=dtypes)
```

### Export Capabilities

```python
# Export to various formats
dataset_service.export_dataset(
    "processed_data",
    output_dir="./exports",
    format="csv",  # or parquet, json, feather (default: csv)
    compression="zip"  # or gzip, snappy, none (default: zip)
)

# Export with transformations
def export_with_preprocessing(dataset_name):
    df = dataset_manager.load_table(dataset_name, "train")
    
    # Apply transformations
    df['log_target'] = np.log1p(df['target'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Export
    df.to_parquet(
        f"./processed/{dataset_name}_processed.parquet",
        engine='pyarrow',
        compression='snappy',
        index=False
    )
```

### API Access

MDM provides programmatic access for ML pipelines:

```python
# Integration with ML pipelines
class MLPipeline:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dm = dataset_manager
        
    def prepare_data(self):
        """Prepare data for training"""
        # Load data
        train_df = self.dm.load_table(self.dataset_name, "train")
        test_df = self.dm.load_table(self.dataset_name, "test")
        
        # Get dataset info
        info = self.dm.get_dataset(self.dataset_name)
        
        # Separate features and target
        X_train = train_df.drop(columns=[info.target_column] + info.id_columns)
        y_train = train_df[info.target_column]
        X_test = test_df.drop(columns=info.id_columns)
        
        return X_train, y_train, X_test
        
    def save_predictions(self, predictions, model_name):
        """Save predictions back to MDM"""
        # Create submission DataFrame
        test_df = self.dm.load_table(self.dataset_name, "test")
        submission = pd.DataFrame({
            'Id': test_df['Id'],
            'Prediction': predictions
        })
        
        # Save to database
        with self.dm.get_dataset_connection(self.dataset_name) as conn:
            submission.to_sql(
                f'predictions_{model_name}',
                conn,
                if_exists='replace',
                index=False
            )
```

## 4. Data Type Handling in Metadata

MDM automatically handles serialization of various data types, including NumPy types, when storing metadata and configurations.

### Supported Data Types

MDM seamlessly converts between Python/NumPy types and storage formats:

```python
# Automatic type handling
- numpy.int64 → int
- numpy.float64 → float
- numpy.bool_ → bool
- numpy.ndarray → list
- pandas.Timestamp → ISO format string
- datetime → ISO format string
```

### Type Preservation in Metadata

When MDM analyzes datasets, it preserves type information in metadata:

```python
# Example: Column statistics with mixed types
{
    "column_stats": {
        "age": {
            "mean": 29.7,  # numpy.float64 → float
            "min": 0,      # numpy.int64 → int
            "max": 80,     # numpy.int64 → int
            "nulls": 177   # numpy.int64 → int
        },
        "survived": {
            "unique_values": [0, 1],  # numpy.ndarray → list
            "mode": 0                 # numpy.int64 → int
        }
    }
}
```

### Custom Type Handling

For advanced use cases, you can extend type serialization:

```python
# Example: Registering datasets with custom metadata
import numpy as np
from datetime import datetime

metadata = {
    "created_at": datetime.now(),  # Automatically converted to ISO string
    "shape": np.array([1000, 50]),  # numpy array → list [1000, 50]
    "metrics": {
        "accuracy": np.float64(0.95),  # numpy float → 0.95
        "samples": np.int32(1000)      # numpy int → 1000
    }
}

# MDM handles serialization transparently
dataset_manager.register_dataset(
    name="experiment_results",
    train_path="/path/to/data.csv",
    metadata=metadata
)
```

### Working with Serialized Data

When retrieving metadata, types are preserved appropriately:

```python
# Load dataset info
info = dataset_manager.get_dataset_info("my_dataset")

# Numeric values are standard Python types
print(type(info.row_count))  # <class 'int'>
print(type(info.column_stats['age']['mean']))  # <class 'float'>

# Dates are returned as strings in ISO format
# Parse them as needed
from datetime import datetime
created = datetime.fromisoformat(info.metadata['created_at'])
```

### Best Practices for Type Handling

1. **Let MDM handle conversions**: Don't pre-convert NumPy types
2. **Use standard types in configs**: When manually creating YAML configs, use standard Python types
3. **Parse dates as needed**: Dates are stored as ISO strings for compatibility
4. **Arrays become lists**: NumPy arrays are serialized as lists in metadata

## 5. Advanced Use Cases

### Time Series Data Management

```python
# Register time series dataset with special handling
mdm dataset register stock_prices /data/stocks \
    --time-column "date" \
    --group-column "symbol" \
    --problem-type "time-series"

# Time-based data splitting
def create_time_splits(dataset_name, test_days=30):
    df = dataset_manager.load_table(dataset_name, "train")
    info = dataset_manager.get_dataset(dataset_name)
    
    # Sort by time
    df = df.sort_values(info.time_column)
    
    # Calculate split point
    split_date = df[info.time_column].max() - pd.Timedelta(days=test_days)
    
    # Split data
    train = df[df[info.time_column] <= split_date]
    test = df[df[info.time_column] > split_date]
    
    return train, test
```

### Multi-Dataset Operations

```python
# Combine multiple datasets
def merge_datasets(dataset_names, join_key):
    """Merge multiple datasets on a common key"""
    dfs = []
    
    for name in dataset_names:
        df = dataset_manager.load_table(name, "train")
        df['source_dataset'] = name
        dfs.append(df)
    
    # Merge all datasets
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=join_key, how='outer', suffixes=('', f'_{df.iloc[0]["source_dataset"]}'))
    
    return merged
```

## Next Steps

- Review [Best Practices](10_Best_Practices.md) for optimal MDM usage
- Check [Troubleshooting](11_Troubleshooting.md) for solutions to common issues
- See the [Summary](12_Summary.md) for key takeaways