# Storage Backends

MDM supports multiple storage backends to accommodate different use cases and performance requirements. This guide covers backend selection, configuration, and optimization.

## Overview

MDM uses a **single backend principle**: all datasets in an MDM instance use the same storage backend. This is configured via `database.default_backend` in your configuration file.

### Available Backends

| Backend | Use Case | Pros | Cons |
|---------|----------|------|------|
| **SQLite** | Small-medium datasets, single user | Zero config, portable, embedded | Limited concurrency, size limits |
| **DuckDB** | Analytics, medium-large datasets | Fast analytics, columnar storage | Single writer, memory usage |
| **PostgreSQL** | Enterprise, multi-user, production | Full ACID, scalability, concurrent | Requires server, more complex |

## Backend Selection Guide

### SQLite (Default)

**Best for:**
- Datasets under 1GB
- Single-user environments
- Development and prototyping
- Portable datasets
- Quick experiments

**Characteristics:**
- File-based storage
- No server required
- Excellent for read-heavy workloads
- Limited write concurrency
- Maximum database size: 281 TB (theoretical)

**Configuration:**
```yaml
database:
  default_backend: sqlite
  sqlite:
    journal_mode: WAL        # Write-Ahead Logging for better concurrency
    synchronous: NORMAL      # Balance between safety and speed
    cache_size: -64000       # 64MB cache (negative = KB)
    temp_store: MEMORY       # Use memory for temporary tables
    mmap_size: 268435456     # 256MB memory-mapped I/O
```

### DuckDB

**Best for:**
- Analytical queries (OLAP)
- Datasets 1GB-100GB
- Complex aggregations
- Time series analysis
- Columnar data operations

**Characteristics:**
- Columnar storage format
- Vectorized query execution
- Excellent compression
- Native Parquet support
- In-process analytical database

**Configuration:**
```yaml
database:
  default_backend: duckdb
  duckdb:
    memory_limit: 4GB        # Maximum memory usage
    threads: 4               # Number of threads
    temp_directory: /tmp     # Temporary file location
    enable_progress_bar: false  # Disable for MDM's progress bars
```

### PostgreSQL

**Best for:**
- Production deployments
- Multi-user access
- Datasets over 100GB
- Complex transactions
- Enterprise requirements

**Characteristics:**
- Client-server architecture
- Full ACID compliance
- Advanced indexing options
- Concurrent read/write
- Extensible with plugins

**Configuration:**
```yaml
database:
  default_backend: postgresql
  postgresql:
    host: localhost
    port: 5432
    user: mdm_user
    password: ${MDM_PG_PASSWORD}  # Environment variable
    database: mdm_db
    pool_size: 10
    max_overflow: 20
    pool_timeout: 30
    echo: false  # SQL query logging
```

## Backend-Specific Features

### SQLite Features

#### 1. Write-Ahead Logging (WAL)
```python
# MDM automatically enables WAL mode for better concurrency
# This allows multiple readers while writing
```

#### 2. Full-Text Search
```python
# Create FTS index for text search
with client.get_connection("dataset") as conn:
    conn.execute("""
        CREATE VIRTUAL TABLE train_fts USING fts5(
            description, notes, 
            content='train', 
            content_rowid='rowid'
        )
    """)
    
    # Search
    results = pd.read_sql("""
        SELECT * FROM train_fts 
        WHERE train_fts MATCH 'machine learning'
    """, conn)
```

#### 3. JSON Support
```python
# SQLite has native JSON functions
df = pd.read_sql("""
    SELECT 
        json_extract(metadata, '$.category') as category,
        json_extract(metadata, '$.tags[0]') as first_tag
    FROM train
    WHERE json_valid(metadata)
""", conn)
```

### DuckDB Features

#### 1. Native File Format Support
```python
# DuckDB can query files directly
with client.get_connection("dataset") as conn:
    # Query Parquet files directly
    df = pd.read_sql("""
        SELECT * FROM '/data/archive/*.parquet'
        WHERE year = 2024
    """, conn)
    
    # Query CSV files
    df = pd.read_sql("""
        SELECT * FROM read_csv_auto('/data/raw/*.csv')
    """, conn)
```

#### 2. Advanced Analytics
```python
# Window functions and analytics
df = pd.read_sql("""
    SELECT 
        date,
        revenue,
        AVG(revenue) OVER (
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as moving_avg_7d,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY revenue) 
            OVER (PARTITION BY category) as median_by_category
    FROM train
""", conn)
```

#### 3. Automatic Compression
```python
# DuckDB automatically compresses data
# Check compression stats
stats = pd.read_sql("""
    SELECT 
        table_name,
        estimated_size,
        total_compressed_size,
        total_compressed_size::FLOAT / estimated_size as compression_ratio
    FROM duckdb_tables()
""", conn)
```

### PostgreSQL Features

#### 1. Advanced Indexing
```python
# Create specialized indexes
with client.get_connection("dataset") as conn:
    # B-tree for exact matches
    conn.execute("CREATE INDEX idx_customer ON train(customer_id)")
    
    # GIN for array/JSON columns
    conn.execute("CREATE INDEX idx_tags ON train USING gin(tags)")
    
    # BRIN for time-series data
    conn.execute("CREATE INDEX idx_date ON train USING brin(date)")
    
    # Partial index for common queries
    conn.execute("""
        CREATE INDEX idx_active_users ON train(user_id) 
        WHERE status = 'active'
    """)
```

#### 2. Partitioning
```python
# Create partitioned tables for large datasets
conn.execute("""
    CREATE TABLE train_partitioned (LIKE train) 
    PARTITION BY RANGE (date);
    
    CREATE TABLE train_2023 PARTITION OF train_partitioned
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
    
    CREATE TABLE train_2024 PARTITION OF train_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
""")
```

#### 3. Extensions
```python
# Enable useful extensions
conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
conn.execute("CREATE EXTENSION IF NOT EXISTS tablefunc")
```

## Performance Comparison

### Benchmark Results

| Operation | SQLite | DuckDB | PostgreSQL |
|-----------|--------|--------|------------|
| Load 1M rows CSV | 15s | 8s | 20s |
| Aggregate query | 2.5s | 0.3s | 1.2s |
| Join 2 tables | 5s | 1s | 3s |
| Write 100k rows | 3s | 5s | 2s |
| Concurrent reads | Limited | Good | Excellent |
| Memory usage | Low | Medium | High |

### Query Performance Tips

#### SQLite Optimization
```python
# 1. Use indexes wisely
conn.execute("CREATE INDEX idx_composite ON train(category, date)")

# 2. Analyze tables for query planner
conn.execute("ANALYZE train")

# 3. Use covering indexes
conn.execute("""
    CREATE INDEX idx_covering ON train(customer_id) 
    INCLUDE (name, email)
""")
```

#### DuckDB Optimization
```python
# 1. Use table statistics
conn.execute("ANALYZE")

# 2. Optimize join order
df = pd.read_sql("""
    SELECT /*+ LEADING(small_table) */ *
    FROM large_table
    JOIN small_table USING (id)
""", conn)

# 3. Use columnar projections
df = pd.read_sql("""
    SELECT col1, col2  -- Only needed columns
    FROM train
    WHERE condition
""", conn)
```

#### PostgreSQL Optimization
```python
# 1. Tune query planner
conn.execute("SET random_page_cost = 1.1")  # For SSD
conn.execute("SET effective_cache_size = '4GB'")

# 2. Use EXPLAIN ANALYZE
plan = pd.read_sql("""
    EXPLAIN ANALYZE
    SELECT * FROM train WHERE category = 'Electronics'
""", conn)

# 3. Parallel queries
conn.execute("SET max_parallel_workers_per_gather = 4")
```

## Migration Between Backends

### Important Notes

1. **Backend Lock-in**: Once a dataset is registered with a backend, it cannot be changed
2. **Backend Visibility**: Changing `default_backend` makes datasets from other backends invisible
3. **No Automatic Migration**: Must export and re-import to change backends

### Migration Process

```python
# Step 1: Export from current backend
export_path = client.export_dataset(
    "my_dataset",
    format="parquet",
    compression="gzip"
)

# Step 2: Change backend in config
# Edit ~/.mdm/mdm.yaml:
# database:
#   default_backend: duckdb  # Changed from sqlite

# Step 3: Re-register with new backend
client = MDMClient()  # Reload with new config
client.register_dataset(
    "my_dataset",
    export_path,
    force=True  # Overwrite
)
```

### Batch Migration Script

```python
import os
import tempfile
from mdm import MDMClient

def migrate_all_datasets(from_backend, to_backend):
    """Migrate all datasets between backends."""
    
    # Step 1: Export all datasets
    os.environ['MDM_DATABASE_DEFAULT_BACKEND'] = from_backend
    client_from = MDMClient()
    
    datasets = client_from.list_datasets()
    export_dir = tempfile.mkdtemp()
    exports = {}
    
    for dataset in datasets:
        print(f"Exporting {dataset.name}...")
        path = client_from.export_dataset(
            dataset.name,
            format="parquet",
            output_path=f"{export_dir}/{dataset.name}.parquet"
        )
        exports[dataset.name] = {
            'path': path,
            'info': dataset
        }
    
    # Step 2: Import to new backend
    os.environ['MDM_DATABASE_DEFAULT_BACKEND'] = to_backend
    client_to = MDMClient()
    
    for name, data in exports.items():
        print(f"Importing {name}...")
        client_to.register_dataset(
            name=name,
            path=data['path'],
            target_column=data['info'].target_column,
            id_columns=data['info'].id_columns,
            problem_type=data['info'].problem_type,
            tags=data['info'].tags,
            description=data['info'].description
        )
    
    print(f"Migrated {len(exports)} datasets from {from_backend} to {to_backend}")

# Usage
migrate_all_datasets("sqlite", "duckdb")
```

## Backend Selection Decision Tree

```
Start
│
├─ Dataset size?
│  ├─ < 1GB → SQLite
│  ├─ 1-100GB → Analytical queries?
│  │  ├─ Yes → DuckDB
│  │  └─ No → PostgreSQL
│  └─ > 100GB → PostgreSQL
│
├─ Concurrent users?
│  ├─ Single → SQLite or DuckDB
│  └─ Multiple → PostgreSQL
│
├─ Query patterns?
│  ├─ Simple lookups → SQLite
│  ├─ Complex analytics → DuckDB
│  └─ Mixed workload → PostgreSQL
│
└─ Environment?
   ├─ Development → SQLite
   ├─ Analytics/Research → DuckDB
   └─ Production → PostgreSQL
```

## Backend-Specific Configuration

### SQLite Advanced Config

```yaml
database:
  default_backend: sqlite
  sqlite:
    # Performance
    journal_mode: WAL
    synchronous: NORMAL
    cache_size: -64000  # 64MB
    temp_store: MEMORY
    mmap_size: 268435456  # 256MB
    
    # Reliability
    foreign_keys: true
    recursive_triggers: true
    
    # Limits
    max_page_count: 1073741823  # ~4TB with 4KB pages
    max_sql_length: 1000000
```

### DuckDB Advanced Config

```yaml
database:
  default_backend: duckdb
  duckdb:
    # Memory management
    memory_limit: 4GB
    max_memory: 8GB
    temp_directory: /tmp/duckdb
    
    # Performance
    threads: 8
    enable_progress_bar: false
    enable_profiling: false
    
    # I/O
    max_open_files: 1000
    enable_external_access: true
    
    # Extensions
    autoload_extensions: ["parquet", "json"]
```

### PostgreSQL Advanced Config

```yaml
database:
  default_backend: postgresql
  postgresql:
    # Connection
    host: ${MDM_PG_HOST:-localhost}
    port: ${MDM_PG_PORT:-5432}
    user: ${MDM_PG_USER}
    password: ${MDM_PG_PASSWORD}
    database: ${MDM_PG_DATABASE:-mdm}
    
    # Connection pool
    pool_size: 10
    max_overflow: 20
    pool_timeout: 30
    pool_recycle: 3600
    
    # Performance
    connect_args:
      connect_timeout: 10
      options: "-c statement_timeout=300000"  # 5 minutes
    
    # SSL
    sslmode: require
    sslcert: /path/to/client-cert.pem
    sslkey: /path/to/client-key.pem
    sslrootcert: /path/to/ca-cert.pem
```

## Monitoring Backend Performance

### SQLite Monitoring

```python
# Check database statistics
stats = pd.read_sql("""
    SELECT * FROM pragma_database_list()
""", conn)

# Page cache statistics
cache_stats = pd.read_sql("""
    SELECT * FROM pragma_cache_stats()
""", conn)

# Table sizes
sizes = pd.read_sql("""
    SELECT 
        name,
        SUM(pgsize) as size_bytes
    FROM dbstat
    GROUP BY name
""", conn)
```

### DuckDB Monitoring

```python
# Memory usage
memory = pd.read_sql("SELECT * FROM duckdb_memory()", conn)

# Table statistics
stats = pd.read_sql("SELECT * FROM duckdb_tables()", conn)

# Query profiling
conn.execute("PRAGMA enable_profiling")
# Run query
conn.execute("PRAGMA disable_profiling")
profile = pd.read_sql("SELECT * FROM duckdb_profiling_output()", conn)
```

### PostgreSQL Monitoring

```python
# Database statistics
stats = pd.read_sql("""
    SELECT * FROM pg_stat_database 
    WHERE datname = current_database()
""", conn)

# Table statistics
table_stats = pd.read_sql("""
    SELECT 
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
        n_live_tup as row_count
    FROM pg_stat_user_tables
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
""", conn)

# Slow queries
slow_queries = pd.read_sql("""
    SELECT 
        query,
        calls,
        total_time,
        mean_time,
        max_time
    FROM pg_stat_statements
    ORDER BY mean_time DESC
    LIMIT 10
""", conn)
```

## Best Practices

### 1. Choose the Right Backend Early
- Backend selection affects performance and capabilities
- Migration is possible but requires effort
- Consider future growth when selecting

### 2. Configure for Your Hardware
- SSD vs HDD affects configuration
- Memory availability determines cache sizes
- CPU cores affect parallelism settings

### 3. Monitor and Tune
- Regular ANALYZE/VACUUM for PostgreSQL
- Monitor query performance
- Adjust configuration based on workload

### 4. Backup Strategies
- SQLite: Simple file copy
- DuckDB: Export to Parquet
- PostgreSQL: pg_dump or streaming replication

### 5. Security Considerations
- SQLite: File system permissions
- DuckDB: Process isolation
- PostgreSQL: User roles and SSL