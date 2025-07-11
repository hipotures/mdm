# Storage Backend Usage Analysis

This document provides detailed analysis of how storage backend methods are used throughout the MDM codebase.

## Usage Summary

Total method calls found: **62 calls** across **14 unique methods**

## Detailed Usage Breakdown

### 1. `get_engine()` - 11 calls (17.7%)

Most frequently used method. Critical for obtaining database connections.

**Usage Locations**:
- `src/mdm/api.py:195` - Getting engine for dataset operations
- `src/mdm/api.py:244` - Engine for metadata operations
- `src/mdm/api.py:575` - Engine for query execution
- `src/mdm/api.py:616` - Engine for column analysis
- `src/mdm/cli/batch.py:69` - Batch operations
- `src/mdm/cli/batch.py:117` - Batch export
- `src/mdm/cli/dataset.py:111` - Dataset info display
- `src/mdm/dataset/exporter.py:96` - Export operations
- `src/mdm/dataset/exporter.py:113` - Export validation
- `src/mdm/dataset/registrar.py:412` - Registration process
- `src/mdm/features/generator.py:239` - Feature generation

**Pattern**: Always called with database path to get engine for subsequent operations.

### 2. `create_table_from_dataframe()` - 10 calls (16.1%)

Second most used method. Essential for data loading and table creation.

**Usage Locations**:
- `src/mdm/dataset/registrar.py:441` - Initial table creation with first chunk
- `src/mdm/dataset/registrar.py:447` - Appending subsequent chunks
- `src/mdm/dataset/registrar.py:481` - Creating table from full DataFrame
- `src/mdm/dataset/registrar.py:551` - Metadata table creation
- `src/mdm/dataset/registrar.py:560` - Features table creation
- `src/mdm/dataset/registrar.py:798` - Sample data table
- `src/mdm/dataset/registrar.py:814` - Statistics table
- `src/mdm/dataset/registrar.py:841` - Schema table
- `src/mdm/dataset/registrar.py:867` - Manifest table
- `src/mdm/utils/integration.py:99` - Test data creation

**Pattern**: Used with different `if_exists` strategies:
- `'replace'` for initial creation
- `'append'` for adding chunks
- Always includes engine parameter

### 3. `query()` - 9 calls (14.5%)

Third most used. Primary method for SQL execution returning DataFrames.

**Usage Locations**:
- `src/mdm/api.py:577` - General query execution
- `src/mdm/cli/dataset.py:57` - Row count queries
- `src/mdm/cli/dataset.py:87` - Table statistics
- `src/mdm/dataset/utils.py:147` - Data validation queries
- `src/mdm/dataset/utils.py:171` - Schema queries
- `src/mdm/dataset/utils.py:190` - Metadata queries
- `src/mdm/services/dataset_service.py:192` - Service layer queries
- `src/mdm/services/dataset_service.py:238` - Data retrieval
- `src/mdm/utils/integration.py:118` - Integration testing

**Common Queries**:
```sql
SELECT COUNT(*) as count FROM {table}
SELECT * FROM {table_name}
SELECT DISTINCT {column} FROM {table}
SELECT {columns} FROM {table} WHERE {conditions}
```

### 4. `read_table_to_dataframe()` - 7 calls (11.3%)

Convenient method for reading entire tables.

**Usage Locations**:
- `src/mdm/api.py:196` - Read main data table
- `src/mdm/api.py:206` - Read with row limit
- `src/mdm/api.py:245` - Read metadata table
- `src/mdm/api.py:252` - Read features table
- `src/mdm/api.py:549` - Read for statistics
- `src/mdm/cli/batch.py:85` - Batch processing
- `src/mdm/dataset/utils.py:43` - Utility functions

**Pattern**: Often used with `limit` parameter for large datasets.

### 5. `close_connections()` - 7 calls (11.3%)

Critical for resource cleanup.

**Usage Locations**:
- `src/mdm/cli/dataset.py:116` - CLI cleanup
- `src/mdm/dataset/exporter.py:122` - After export
- `src/mdm/dataset/registrar.py:666` - Registration cleanup (in finally block)
- `src/mdm/features/generator.py:282` - After feature generation
- `src/mdm/services/dataset_service.py:259` - Service cleanup
- `src/mdm/storage/sqlite.py:147` - Before dropping database
- `src/mdm/utils/integration.py:132` - Test cleanup

**Pattern**: Always in finally blocks or cleanup methods.

### 6. `read_table()` - 7 calls (11.3%)

Backend-specific table reading (not in base class).

**Usage Locations**:
- `src/mdm/features/engine.py:126` - Read for processing
- `src/mdm/services/dataset_service.py:220` - Service layer
- `src/mdm/services/dataset_service.py:228` - With column selection
- Additional service layer usage

**Note**: Method signature unclear - needs investigation.

### 7. `write_table()` - 3 calls (4.8%)

Backend-specific table writing (not in base class).

**Usage Locations**:
- `src/mdm/features/engine.py:163` - Write processed features
- `src/mdm/features/engine.py:168` - Write intermediate results
- `src/mdm/features/engine.py:172` - Write final output

**Note**: Appears to be feature engineering specific.

### 8. `get_table_info()` - 2 calls (3.2%)

Get table metadata and statistics.

**Usage Locations**:
- `src/mdm/cli/dataset.py:61` - Display table info
- `src/mdm/dataset/registrar.py:861` - Registration validation

**Returns**: Dictionary with:
- Table name
- Column information
- Row count
- Column count

### 9-14. Single-Use Methods (1.6% each)

#### `execute_query()`
- `src/mdm/api.py:261` - Execute SQL without DataFrame return

#### `get_connection()`
- `src/mdm/api.py:279` - Get raw connection object

#### `get_columns()`
- `src/mdm/api.py:619` - Get column names

#### `analyze_column()`
- `src/mdm/api.py:623` - Column statistics

#### `database_exists()`
- `src/mdm/dataset/registrar.py:367` - Check before creation

#### `create_database()`
- `src/mdm/dataset/registrar.py:368` - Create new database

## Usage Patterns

### 1. **Engine Lifecycle Pattern**
```python
engine = backend.get_engine(db_path)
try:
    # Use engine for operations
    backend.create_table_from_dataframe(df, table, engine)
    result = backend.query("SELECT ...")
finally:
    backend.close_connections()
```

### 2. **Chunked Data Loading Pattern**
```python
first_chunk = True
for chunk in pd.read_csv(file, chunksize=10000):
    if first_chunk:
        backend.create_table_from_dataframe(chunk, table, engine, if_exists='replace')
        first_chunk = False
    else:
        backend.create_table_from_dataframe(chunk, table, engine, if_exists='append')
```

### 3. **Query Pattern**
```python
# Simple queries use query() method
df = backend.query(f"SELECT * FROM {table}")

# Complex operations use engine directly
engine = backend.get_engine(db_path)
result = backend.execute_query(complex_sql, engine)
```

## Critical Paths

### Dataset Registration
1. `database_exists()` - Check if exists
2. `create_database()` - Create if needed
3. `get_engine()` - Get connection
4. `create_table_from_dataframe()` - Load data (multiple calls)
5. `close_connections()` - Cleanup

### Data Export
1. `get_engine()` - Get connection
2. `read_table_to_dataframe()` or `query()` - Read data
3. Process/transform data
4. `close_connections()` - Cleanup

### Feature Generation
1. `get_engine()` - Get connection
2. `read_table()` - Read input data
3. Process features
4. `write_table()` - Save results
5. `close_connections()` - Cleanup

## Missing Method Analysis

Methods called but not defined in base class:
- `read_table()` - 7 calls
- `write_table()` - 3 calls  
- `get_connection()` - 1 call
- `get_columns()` - 1 call
- `analyze_column()` - 1 call

These appear to be:
1. Backend-specific implementations
2. Methods added ad-hoc without updating base class
3. Methods that should be in the interface but were missed

## Recommendations

1. **Standardize Missing Methods**: Add the 5 missing methods to base class
2. **Document Method Signatures**: Especially for `read_table()` and `write_table()`
3. **Create Integration Tests**: Cover all 14 methods in use
4. **Add Type Hints**: Improve IDE support and catch errors
5. **Consider Deprecation**: Some methods might be redundant
   - `query()` vs `execute_query()`
   - `read_table()` vs `read_table_to_dataframe()`