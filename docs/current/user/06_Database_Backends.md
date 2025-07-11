# Database Backends

MDM uses SQLAlchemy as a unified database abstraction layer, supporting multiple backends to accommodate different use cases, performance requirements, and infrastructure constraints. Each backend offers unique advantages while maintaining full compatibility with MDM's features through the SQLAlchemy ORM.

## Supported Backends

1. **SQLite** - Default backend, lightweight and portable
2. **DuckDB** - High-performance analytical workloads
3. **PostgreSQL** - Enterprise solution for multi-user environments

## Backend Architecture

All database operations go through SQLAlchemy, providing:
- Unified API across all backends
- Automatic SQL dialect translation
- Connection pooling and management
- ORM-based table definitions
- Migration support

## Backend Configuration

Database backends are configured in `mdm.yaml` at the system level. The default backend for new installations is SQLite.

**Important**: MDM uses a single-backend architecture. All datasets must use the same backend type. Changing the backend in `mdm.yaml` will make datasets with different backends invisible until you re-register them with the new backend.

## 1. SQLite (Default)

SQLite is the default backend for MDM, providing excellent portability and ease of use for most datasets. It's accessed through SQLAlchemy's SQLite dialect.

### Configuration

```yaml
# In mdm.yaml
database:
  default_backend: sqlite    # Default backend
  
  # SQLAlchemy settings (applies to all backends)
  sqlalchemy:
    echo: false              # Log SQL queries
    pool_size: 5            # Connection pool size
    
  # SQLite-specific settings
  sqlite:
    journal_mode: "WAL"      # Write-Ahead Logging for better concurrency
    synchronous: "NORMAL"    # Balance between safety and speed
```

### Advantages

- **Zero configuration** - works out of the box
- **Excellent portability** - single file database
- **Ubiquitous support** - included with Python
- **Good performance** for datasets up to several GB
- **Full ACID compliance** with transaction support
- **SQLAlchemy integration** - full ORM capabilities

### Dataset Storage

- File location: `{datasets_path}/{dataset_name}/dataset.sqlite`
- Self-contained database file
- WAL mode for better concurrency
- Compatible with all SQLite tools

### Best For

- Small to medium datasets (up to 10GB)
- Maximum portability requirements
- Single-user scenarios
- Development and prototyping
- Default choice for new users

### Example Usage

```bash
# Register with SQLite (default)
mdm dataset register titanic /path/to/data

# Query directly with SQLite CLI
sqlite3 ~/.mdm/datasets/titanic/dataset.sqlite

# Or access via SQLAlchemy in Python
from sqlalchemy import create_engine
engine = create_engine('sqlite:///~/.mdm/datasets/titanic/dataset.sqlite')
```

## 2. DuckDB

DuckDB provides exceptional analytical performance through columnar storage, accessed via SQLAlchemy's DuckDB dialect.

### Configuration

```yaml
# In mdm.yaml
database:
  default_backend: duckdb
  
  # DuckDB-specific settings
  duckdb:
    memory_limit: "8GB"      # Maximum memory usage
    threads: 4               # Number of threads
    temp_directory: "/tmp"   # Temporary file location
```

### Advantages

- **Columnar storage** optimized for analytics
- **Excellent query performance** for aggregations and joins
- **Built-in Parquet support** for efficient data exchange
- **OLAP optimized** - designed for analytical workloads
- **SQLAlchemy integration** - full ORM support

### Dataset Storage

- File location: `{datasets_path}/{dataset_name}/dataset.duckdb`
- Columnar storage format
- Efficient compression
- Native Parquet import/export

### Best For

- Analytical workloads
- Medium to large datasets (1GB - 100GB)
- Complex aggregations and joins
- Kaggle competitions and research

### Example Usage

```bash
# To use DuckDB, first configure it in mdm.yaml:
# database:
#   default_backend: duckdb

# Then register dataset
mdm dataset register analytics_data /path/to/data

# Access via SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine('duckdb:///~/.mdm/datasets/analytics_data/dataset.duckdb')
```

## 3. PostgreSQL

PostgreSQL provides an enterprise-grade solution with advanced features for production environments, accessed via SQLAlchemy's PostgreSQL dialect with psycopg2 driver.

### Configuration

```yaml
# In mdm.yaml
database:
  default_backend: postgresql
  
  postgresql:
    host: localhost
    port: 5432
    user: mdm_user
    password: mdm_pass
    database_prefix: mdm_    # Prefix for dataset databases
    
    # Connection pool settings
    pool_size: 10
    max_overflow: 20
    pool_timeout: 30
    
    # SSL settings (optional)
    sslmode: require
    sslcert: /path/to/client-cert.pem
    sslkey: /path/to/client-key.pem
```

### Advantages

- **Enterprise features** - transactions, replication, backups
- **Multi-user support** - concurrent access control
- **Advanced indexing** - B-tree, Hash, GiST, SP-GiST
- **Scalability** - handles very large datasets
- **Rich ecosystem** - extensions, tools, monitoring
- **SQLAlchemy integration** - full ORM with migrations support

### Dataset Storage

- Creates separate database per dataset: `mdm_titanic`, `mdm_house_prices`
- Tables within each database: `train`, `test`, `validation`, `submission`, `_metadata`
- Leverages PostgreSQL's advanced features
- Supports partitioning for very large tables

### Best For

- Production environments
- Multi-user access requirements
- Very large datasets (> 100GB)
- Enterprise compliance needs
- Advanced query requirements

### Example Usage

```bash
# To use PostgreSQL, first configure it in mdm.yaml:
# database:
#   default_backend: postgresql
#   postgresql:
#     host: localhost
#     user: mdm_user
#     password: mdm_pass

# Then register dataset (will use PostgreSQL)
mdm dataset register production_data /path/to/data

# Connect with psql
psql -h localhost -U mdm_user -d mdm_production_data

# Or via SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine('postgresql://mdm_user:mdm_pass@localhost/mdm_production_data')
```

## Backend Selection

### Setting the Backend

The database backend is configured exclusively through the `mdm.yaml` configuration file. There is no CLI parameter to override this setting.

```yaml
# In ~/.mdm/mdm.yaml
database:
  default_backend: sqlite  # Default. Options: sqlite, duckdb, postgresql
```

### How Backend Selection Works

For the complete, authoritative explanation of backend selection, see [Backend Selection and Configuration](03_Database_Architecture.md#backend-selection-and-configuration-authoritative).

**Key points**:
- MDM uses ONE backend type for all datasets at any given time
- Backend is chosen at registration time based on `mdm.yaml`
- Changing backend in `mdm.yaml` hides datasets with different backends
- To use datasets with a new backend, you must re-register them
- No CLI parameter to override backend selection

### Changing Backend

When you change the backend in `mdm.yaml`, MDM's behavior changes:

```bash
# Step 1: Check current backend and datasets
mdm info
mdm dataset list  # Shows all SQLite datasets

# Step 2: Export important datasets before switching
mdm dataset export my_dataset --output-dir ./backup/

# Step 3: Change backend in mdm.yaml
vim ~/.mdm/mdm.yaml
# Change: default_backend: postgresql

# Step 4: List datasets again
mdm dataset list  # SQLite datasets are now invisible!

# Step 5: Re-register datasets with new backend
mdm dataset register my_dataset ./backup/
```

**Warning**: Datasets registered with different backends become invisible when you switch backends. They are not deleted, but MDM will not show or access them until you switch back to their backend.

### Backend in Configuration

Dataset configuration files store the database path based on the backend used during registration:

```yaml
# config/datasets/my_dataset.yaml
name: my_dataset
database:
  # For PostgreSQL datasets:
  connection_string: postgresql://user:pass@host/mdm_my_dataset
  # For DuckDB/SQLite datasets:
  # path: ~/.mdm/datasets/my_dataset/dataset.duckdb
```

## Performance Comparison

| Feature | SQLite | DuckDB | PostgreSQL |
|---------|---------|---------|------------|
| Query Speed | Good | Excellent | Very Good |
| Write Speed | Excellent | Good | Good |
| Concurrent Users | Limited | Limited | Excellent |
| Dataset Size | <10GB | 1-100GB | Unlimited |
| Memory Usage | Low | Configurable | Configurable |
| Setup Complexity | None | None | Medium |
| Default Backend | ✓ | | |

## Moving Datasets Between Backends

Since MDM uses a single-backend architecture, moving datasets requires exporting and re-registering:

```bash
# Step 1: While using the old backend, export datasets
mdm dataset list  # Note which datasets you want to keep
mdm dataset export dataset1 --output-dir ./exports/dataset1/
mdm dataset export dataset2 --output-dir ./exports/dataset2/

# Step 2: Change backend in mdm.yaml
vim ~/.mdm/mdm.yaml
# Change: default_backend: postgresql

# Step 3: Verify backend switch
mdm info  # Shows new backend
mdm dataset list  # Old datasets are invisible now

# Step 4: Re-register datasets with new backend
mdm dataset register dataset1 ./exports/dataset1/
mdm dataset register dataset2 ./exports/dataset2/
```

**Important**: This is not a migration - it's a complete re-registration. The old dataset files remain in `~/.mdm/datasets/` but are inaccessible until you switch back to their backend.

## SQLAlchemy Integration

MDM uses SQLAlchemy as the unified database abstraction layer, providing:

### Core Benefits

- **Unified API**: Same code works across all backends
- **ORM Mapping**: Tables are defined using SQLAlchemy declarative models
- **Connection Management**: Automatic pooling and connection handling
- **Query Building**: Type-safe query construction
- **Migration Support**: Schema evolution using Alembic (future feature)

### Architecture

```python
# MDM's SQLAlchemy architecture
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

# Dataset tables inherit from Base
class TrainTable(Base):
    __tablename__ = 'train'
    # Column definitions
    
# Engine creation based on backend
if backend == 'sqlite':
    engine = create_engine(f'sqlite:///{dataset_path}')
elif backend == 'duckdb':
    engine = create_engine(f'duckdb:///{dataset_path}')
elif backend == 'postgresql':
    engine = create_engine(f'postgresql://{connection_string}')
```

### Backend-Specific Dialects

- **SQLite**: Uses `sqlite3` driver (included with Python)
- **DuckDB**: Uses `duckdb-engine` SQLAlchemy dialect
- **PostgreSQL**: Uses `psycopg2` driver

### Future Enhancements

- Schema migrations with Alembic
- Custom SQLAlchemy types for ML-specific data
- Query optimization hints per backend
- Distributed query support

## Choosing the Right Backend

### Decision Matrix

1. **Start with DuckDB if**:
   - You're doing analytical queries
   - Dataset size is 1GB - 100GB
   - Single-user or read-heavy access
   - You want best default performance

2. **Use SQLite if**:
   - Dataset is small (< 1GB)
   - Maximum portability is required
   - Minimal dependencies are important
   - Development/prototyping phase

3. **Choose PostgreSQL if**:
   - Multi-user access is required
   - Dataset is very large (> 100GB)
   - Enterprise features are needed
   - Production deployment

## Backend-Specific Features

### DuckDB Extensions

```sql
-- Enable Parquet extension
INSTALL parquet;
LOAD parquet;

-- Direct Parquet queries
SELECT * FROM 'data.parquet';
```

### SQLite Optimizations

```sql
-- Enable memory-mapped I/O
PRAGMA mmap_size = 268435456;

-- Optimize for read-heavy workloads
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
```

### PostgreSQL Extensions

```sql
-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Partitioning for large tables
CREATE TABLE train PARTITION BY RANGE (date_column);
```

## Best Practices

1. **DuckDB Best Practices**:
   - Set appropriate memory limits
   - Use Parquet for data exchange
   - Leverage columnar storage benefits

2. **SQLite Best Practices**:
   - Enable WAL mode for better concurrency
   - Regular VACUUM for space reclamation
   - Keep datasets under 1GB

3. **PostgreSQL Best Practices**:
   - Configure connection pooling
   - Set up regular backups
   - Monitor performance metrics
   - Use appropriate indexes

## Next Steps

- Configure your preferred backend in [Configuration](02_Configuration.md)
- Learn about the [Command Line Interface](07_Command_Line_Interface.md)
- See [Best Practices](10_Best_Practices.md) for backend-specific tips