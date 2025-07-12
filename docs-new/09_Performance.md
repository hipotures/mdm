# Performance Optimization Guide

MDM is designed for high performance with datasets ranging from megabytes to gigabytes. This guide covers optimization strategies, benchmarks, and best practices for maximum performance.

## Performance Overview

### Key Performance Features

1. **Batch Processing**: Configurable batch sizes for memory efficiency
2. **Parallel Execution**: Multi-threaded feature generation and data processing
3. **Lazy Loading**: Fast CLI startup and on-demand module loading
4. **Optimized Storage**: Backend-specific optimizations
5. **Progress Tracking**: Real-time progress without performance overhead
6. **Memory Management**: Streaming processing for large datasets

### Performance Benchmarks

| Operation | 100K rows | 1M rows | 10M rows | 100M rows |
|-----------|-----------|----------|-----------|-----------|
| Registration (SQLite) | 2s | 15s | 3m | 35m |
| Registration (DuckDB) | 1s | 8s | 1.5m | 18m |
| Registration (PostgreSQL) | 3s | 20s | 4m | 45m |
| Feature Generation | 5s | 45s | 8m | 85m |
| Query (aggregate) | 0.1s | 0.5s | 3s | 20s |
| Export (CSV) | 1s | 8s | 80s | 15m |
| Export (Parquet) | 0.5s | 4s | 35s | 6m |

*Benchmarks on: Intel i7-10700K, 32GB RAM, NVMe SSD*

## Configuration for Performance

### Basic Performance Settings

```yaml
# ~/.mdm/mdm.yaml
performance:
  batch_size: 50000         # Larger batches for more RAM
  max_workers: 8            # Match CPU cores
  chunk_size: 50000         # Data loading chunks
  enable_progress: true     # Progress bars (minimal overhead)
  memory_limit: "8GB"       # Prevent OOM
  
  cache:
    enabled: true
    ttl: 7200              # 2 hours
    max_size: 10000        # Cache entries
```

### Backend-Specific Optimization

#### SQLite Performance

```yaml
database:
  sqlite:
    # Optimize for speed
    journal_mode: WAL       # Better concurrency
    synchronous: NORMAL     # Faster than FULL
    cache_size: -256000     # 256MB cache
    temp_store: MEMORY      # RAM for temp tables
    mmap_size: 1073741824   # 1GB memory mapping
    
    # Pragmas for speed
    pragmas:
      - "PRAGMA optimize"
      - "PRAGMA analysis_limit=1000"
```

#### DuckDB Performance

```yaml
database:
  duckdb:
    memory_limit: 16GB      # Use available RAM
    threads: 16             # All CPU cores
    temp_directory: /fast/ssd/temp  # SSD for spills
    
    # DuckDB specific
    settings:
      preserve_insertion_order: false  # Faster inserts
      checkpoint_threshold: "1GB"      # Less frequent checkpoints
```

#### PostgreSQL Performance

```yaml
database:
  postgresql:
    # Connection pooling
    pool_size: 20
    max_overflow: 40
    pool_pre_ping: true     # Verify connections
    
    # Server settings (via connection)
    connect_args:
      options: "-c work_mem=256MB -c maintenance_work_mem=1GB"
```

## Optimization Strategies

### 1. Dataset Registration

#### Batch Size Optimization

```python
# Determine optimal batch size based on available memory
import psutil

def get_optimal_batch_size():
    """Calculate batch size based on available memory."""
    available_mb = psutil.virtual_memory().available / 1024 / 1024
    
    # Use 25% of available memory for batches
    batch_memory_mb = available_mb * 0.25
    
    # Assume ~100 bytes per row average
    batch_size = int(batch_memory_mb * 1024 * 1024 / 100)
    
    # Clamp to reasonable range
    return max(1000, min(batch_size, 100000))

# Set dynamically
import os
os.environ['MDM_PERFORMANCE_BATCH_SIZE'] = str(get_optimal_batch_size())
```

#### Disable Features for Speed

```bash
# Skip feature generation for faster registration
mdm dataset register large_data /path/to/data.csv --no-features

# Generate features later if needed
mdm dataset update large_data --regenerate-features
```

#### Parallel File Loading

```python
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def load_multiple_files(file_paths, n_workers=4):
    """Load multiple files in parallel."""
    def load_file(path):
        return pd.read_csv(path, low_memory=False)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        dfs = list(executor.map(load_file, file_paths))
    
    return pd.concat(dfs, ignore_index=True)

# Use for multi-file datasets
files = [f"data_{i}.csv" for i in range(10)]
combined_df = load_multiple_files(files)
```

### 2. Query Optimization

#### Index Strategy

```python
def optimize_dataset_indexes(dataset_name):
    """Create optimal indexes for common queries."""
    client = MDMClient()
    
    with client.get_connection(dataset_name) as conn:
        # Get dataset info
        info = client.get_dataset_info(dataset_name)
        
        # Index ID columns
        for id_col in info.id_columns:
            conn.execute(f"CREATE INDEX idx_{id_col} ON train({id_col})")
        
        # Index target column if exists
        if info.target_column:
            conn.execute(f"CREATE INDEX idx_target ON train({info.target_column})")
        
        # Composite indexes for common patterns
        if 'date' in info.columns["original"]:
            conn.execute("CREATE INDEX idx_date_target ON train(date, target)")
        
        # Analyze for query planner
        conn.execute("ANALYZE")
```

#### Query Patterns

```python
# Bad: Loading all data then filtering
df = client.load_dataset("large_data")
filtered = df[df['category'] == 'A']

# Good: Filter in database
query = """
    SELECT * FROM train 
    WHERE category = 'A'
"""
filtered = pd.read_sql(query, client.get_connection("large_data"))

# Better: Use query builder
filtered = client.query.query(
    dataset="large_data",
    filters={"category": "A"},
    columns=["id", "value", "date"]  # Only needed columns
)
```

#### Aggregation Performance

```python
# Use database aggregation instead of pandas
def get_category_stats(dataset_name):
    """Efficient aggregation in database."""
    query = """
        SELECT 
            category,
            COUNT(*) as count,
            AVG(value) as avg_value,
            STDDEV(value) as std_value,
            MIN(value) as min_value,
            MAX(value) as max_value
        FROM train
        GROUP BY category
        HAVING COUNT(*) > 100
    """
    
    with client.get_connection(dataset_name) as conn:
        return pd.read_sql(query, conn)
```

### 3. Feature Engineering Performance

#### Selective Feature Generation

```python
# Configure features for performance
feature_config = {
    "statistical": {
        "enabled": True,
        "features": ["zscore", "log"]  # Only essential features
    },
    "temporal": {
        "enabled": True,
        "enable_cyclical": False  # Skip expensive cyclical encoding
    },
    "categorical": {
        "enabled": True,
        "max_cardinality": 20  # Limit one-hot encoding
    },
    "text": {
        "enabled": False  # Skip text features if not needed
    }
}

client.register_dataset(
    "optimized_data",
    "/path/to/data.csv",
    feature_config=feature_config
)
```

#### Parallel Feature Generation

```yaml
# Maximum parallel processing
features:
  n_jobs: -1               # Use all CPU cores
  backend: multiprocessing # or 'threading' for I/O bound
  batch_size: 100000       # Larger batches for parallel work
```

#### Custom Feature Optimization

```python
class OptimizedFeatures(BaseDomainFeatures):
    """Performance-optimized custom features."""
    
    def calculate_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        features = {}
        
        # Vectorized operations only
        features['revenue'] = df['price'] * df['quantity']
        
        # Avoid loops
        # Bad:
        # for idx, row in df.iterrows():
        #     features[idx] = complex_calculation(row)
        
        # Good: Use numpy/pandas vectorization
        features['discount_rate'] = np.where(
            df['list_price'] > 0,
            (df['list_price'] - df['price']) / df['list_price'],
            0
        )
        
        # Use numba for complex calculations
        @numba.jit(nopython=True)
        def fast_calculation(arr1, arr2):
            result = np.empty_like(arr1)
            for i in range(len(arr1)):
                result[i] = complex_math(arr1[i], arr2[i])
            return result
        
        features['complex_feature'] = fast_calculation(
            df['col1'].values,
            df['col2'].values
        )
        
        return features
```

### 4. Export Optimization

#### Format Selection

```python
# Performance comparison
def benchmark_export_formats(dataset_name):
    """Compare export performance."""
    import time
    
    formats = ['csv', 'parquet', 'json']
    results = {}
    
    for fmt in formats:
        start = time.time()
        client.export_dataset(
            dataset_name,
            format=fmt,
            compression='gzip' if fmt != 'parquet' else 'snappy'
        )
        results[fmt] = time.time() - start
    
    return results

# Parquet is typically fastest for large datasets
```

#### Chunked Export

```python
def export_large_dataset_chunked(dataset_name, chunk_size=1000000):
    """Export large dataset in chunks."""
    info = client.get_dataset_info(dataset_name)
    total_rows = info.row_count
    
    # Create Parquet writer
    writer = None
    
    for offset in range(0, total_rows, chunk_size):
        # Load chunk
        chunk = client.query.query(
            dataset=dataset_name,
            offset=offset,
            limit=chunk_size
        )
        
        # Write chunk
        if writer is None:
            writer = pd.io.parquet.FastParquetWriter(
                f"{dataset_name}_export.parquet",
                chunk
            )
        else:
            writer.write(chunk)
    
    if writer:
        writer.close()
```

### 5. Memory Optimization

#### Data Type Optimization

```python
def optimize_dataframe_dtypes(df):
    """Reduce DataFrame memory usage."""
    initial_memory = df.memory_usage().sum() / 1024**2
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert strings to category if low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:  # Less than 50% unique
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage().sum() / 1024**2
    print(f"Memory reduced from {initial_memory:.1f}MB to {final_memory:.1f}MB")
    
    return df
```

#### Streaming Processing

```python
def process_dataset_streaming(dataset_name, process_func, chunk_size=50000):
    """Process dataset without loading all into memory."""
    info = client.get_dataset_info(dataset_name)
    
    results = []
    for offset in range(0, info.row_count, chunk_size):
        # Load chunk
        chunk = client.query.query(
            dataset=dataset_name,
            offset=offset,
            limit=chunk_size
        )
        
        # Process chunk
        result = process_func(chunk)
        results.append(result)
        
        # Free memory
        del chunk
        
    return combine_results(results)
```

## Backend-Specific Performance

### SQLite Performance Tuning

```python
def tune_sqlite_for_performance(dataset_name):
    """Apply SQLite performance optimizations."""
    with client.get_connection(dataset_name) as conn:
        # Performance pragmas
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -256000")  # 256MB
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 1073741824")  # 1GB
        
        # Optimize database
        conn.execute("PRAGMA optimize")
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
```

### DuckDB Performance Tuning

```python
def tune_duckdb_for_analytics(dataset_name):
    """Optimize DuckDB for analytical queries."""
    with client.get_connection(dataset_name) as conn:
        # Set memory limit
        conn.execute("SET memory_limit='16GB'")
        conn.execute("SET threads=16")
        
        # Optimize for analytics
        conn.execute("SET preserve_insertion_order=false")
        conn.execute("SET enable_progress_bar=false")
        
        # Create statistics
        conn.execute("ANALYZE")
```

### PostgreSQL Performance Tuning

```python
def tune_postgresql_for_performance(dataset_name):
    """PostgreSQL performance settings."""
    with client.get_connection(dataset_name) as conn:
        # Session settings
        conn.execute("SET work_mem = '256MB'")
        conn.execute("SET maintenance_work_mem = '1GB'")
        conn.execute("SET effective_cache_size = '4GB'")
        
        # Parallel query
        conn.execute("SET max_parallel_workers_per_gather = 4")
        conn.execute("SET parallel_tuple_cost = 0.1")
        
        # Update statistics
        conn.execute(f"ANALYZE {dataset_name}.train")
```

## Monitoring Performance

### Built-in Monitoring

```python
# Enable performance monitoring
from mdm.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.track("dataset_registration"):
    client.register_dataset("test_data", "/path/to/data.csv")

# Get metrics
metrics = monitor.get_metrics()
print(f"Registration took: {metrics['dataset_registration']['duration']:.2f}s")
```

### Query Performance Analysis

```python
def analyze_query_performance(dataset_name, query):
    """Analyze query performance."""
    with client.get_connection(dataset_name) as conn:
        # SQLite
        if client.get_backend() == "sqlite":
            plan = pd.read_sql(f"EXPLAIN QUERY PLAN {query}", conn)
            
        # PostgreSQL
        elif client.get_backend() == "postgresql":
            plan = pd.read_sql(f"EXPLAIN ANALYZE {query}", conn)
            
        # DuckDB
        elif client.get_backend() == "duckdb":
            conn.execute("PRAGMA enable_profiling")
            conn.execute(query)
            plan = pd.read_sql("SELECT * FROM duckdb_profiling_output()", conn)
    
    return plan
```

### Memory Profiling

```python
import tracemalloc
import gc

def profile_memory_usage(func, *args, **kwargs):
    """Profile memory usage of a function."""
    gc.collect()
    tracemalloc.start()
    
    # Get initial memory
    initial = tracemalloc.get_traced_memory()[0]
    
    # Run function
    result = func(*args, **kwargs)
    
    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Memory usage: {(peak - initial) / 1024 / 1024:.1f} MB")
    
    return result

# Profile dataset loading
df = profile_memory_usage(client.load_dataset, "large_data")
```

## Performance Checklist

### Before Registration

- [ ] Choose appropriate backend for workload
- [ ] Configure batch size based on available memory
- [ ] Disable unnecessary features
- [ ] Set up fast storage (SSD) for temp files
- [ ] Increase system limits if needed

### During Operation

- [ ] Use database queries instead of loading full dataset
- [ ] Create indexes for frequently queried columns
- [ ] Use appropriate data types
- [ ] Enable query result caching
- [ ] Monitor memory usage

### Optimization Opportunities

- [ ] Profile slow operations
- [ ] Analyze query plans
- [ ] Review generated features
- [ ] Consider data partitioning
- [ ] Implement incremental updates

## Performance Anti-Patterns

### 1. Loading Everything into Memory

```python
# Bad
df = client.load_dataset("huge_dataset")  # 50GB dataset!
result = df[df['category'] == 'A']['value'].mean()

# Good
result = client.query.aggregate(
    dataset="huge_dataset",
    filters={"category": "A"},
    aggregations={"avg_value": ("value", "mean")}
)
```

### 2. Inefficient Iterations

```python
# Bad
for idx, row in df.iterrows():
    df.loc[idx, 'new_col'] = complex_function(row)

# Good
df['new_col'] = df.apply(complex_function, axis=1)

# Better (vectorized)
df['new_col'] = vectorized_function(df['col1'], df['col2'])
```

### 3. Redundant Features

```python
# Bad: Generating all features
client.register_dataset("data", path)  # 500 features generated!

# Good: Only needed features
client.register_dataset(
    "data",
    path,
    feature_config={"statistical": {"features": ["mean", "std"]}}
)
```

## Advanced Performance Techniques

### 1. Query Result Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def cached_query(dataset_name, query_hash):
    """Cache query results."""
    with client.get_connection(dataset_name) as conn:
        return pd.read_sql(query_map[query_hash], conn)

def run_cached_query(dataset_name, query):
    """Run query with caching."""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    query_map[query_hash] = query
    return cached_query(dataset_name, query_hash)
```

### 2. Approximate Queries

```python
# For large datasets, use approximations
def get_approximate_stats(dataset_name, sample_rate=0.01):
    """Get approximate statistics using sampling."""
    query = f"""
        SELECT 
            AVG(value) as avg_value,
            STDDEV(value) as std_value,
            COUNT(*) / {sample_rate} as estimated_count
        FROM train
        TABLESAMPLE BERNOULLI({sample_rate * 100})
    """
    
    with client.get_connection(dataset_name) as conn:
        return pd.read_sql(query, conn)
```

### 3. Materialized Views

```python
def create_materialized_summary(dataset_name):
    """Create pre-computed summaries."""
    with client.get_connection(dataset_name) as conn:
        # Create summary table
        conn.execute("""
            CREATE TABLE daily_summary AS
            SELECT 
                DATE(date) as day,
                category,
                COUNT(*) as count,
                SUM(value) as total_value,
                AVG(value) as avg_value
            FROM train
            GROUP BY DATE(date), category
        """)
        
        # Index for fast lookups
        conn.execute("CREATE INDEX idx_summary_day ON daily_summary(day)")
```

## Hardware Recommendations

### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: SSD with 2x dataset size free

### Recommended Setup
- CPU: 8+ cores (benefits parallel processing)
- RAM: 32GB+ (larger batch sizes)
- Storage: NVMe SSD (4x faster than SATA)
- Network: 1Gbps+ for PostgreSQL

### Scaling Considerations
- **Vertical**: More RAM = larger batches = faster processing
- **Horizontal**: PostgreSQL supports read replicas
- **Storage**: Consider RAID 0 for temp files
- **Cloud**: Use compute-optimized instances