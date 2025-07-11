# Missing Methods Specification

This document specifies the 11 methods that are missing from the new stateless backend implementations but are required for compatibility with existing MDM code.

## Summary of Missing Methods

| Method | Usage Count | Priority | Complexity |
|--------|-------------|----------|------------|
| `create_table_from_dataframe()` | 10 | Critical | Medium |
| `query()` | 9 | Critical | Low |
| `read_table_to_dataframe()` | 7 | High | Low |
| `close_connections()` | 7 | High | Low |
| `read_table()` | 7 | High | Medium |
| `write_table()` | 3 | Medium | Medium |
| `get_table_info()` | 2 | Medium | Low |
| `execute_query()` | 1 | Low | Low |
| `get_connection()` | 1 | Low | Medium |
| `get_columns()` | 1 | Low | Low |
| `analyze_column()` | 1 | Low | Medium |

## Detailed Specifications

### 1. `create_table_from_dataframe()`

**Priority**: Critical (10 uses)  
**Purpose**: Create or append to a table from a pandas DataFrame

```python
def create_table_from_dataframe(
    self,
    df: pd.DataFrame,
    table_name: str,
    engine: Engine,
    if_exists: str = "fail"
) -> None:
    """Create table from pandas DataFrame.
    
    Args:
        df: Pandas DataFrame to save
        table_name: Name of the table to create
        engine: SQLAlchemy engine instance
        if_exists: Behavior if table exists ('fail', 'replace', 'append')
        
    Raises:
        StorageError: If table creation fails
    """
```

**Implementation Notes**:
- Must support chunked writing for large datasets
- Must handle all pandas dtypes correctly
- SQLite: Use `df.to_sql()` with appropriate parameters
- DuckDB: Can use native DataFrame support
- PostgreSQL: Consider using COPY for performance

**Example Implementation**:
```python
def create_table_from_dataframe(self, df, table_name, engine, if_exists="fail"):
    try:
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    except Exception as e:
        raise StorageError(f"Failed to create table {table_name}: {e}")
```

### 2. `query()`

**Priority**: Critical (9 uses)  
**Purpose**: Execute SQL query and return results as DataFrame

```python
def query(self, query: str) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame.
    
    Args:
        query: SQL query string
        
    Returns:
        Query results as pandas DataFrame
        
    Raises:
        StorageError: If query execution fails
        
    Note:
        This method assumes an engine is already available via get_engine()
    """
```

**Implementation Notes**:
- Must handle the singleton pattern (use cached engine)
- Should support parameterized queries (future enhancement)
- Must return empty DataFrame for queries with no results

**Example Implementation**:
```python
def query(self, query: str) -> pd.DataFrame:
    if self._engine is None:
        raise StorageError("No engine available. Call get_engine() first.")
    try:
        return pd.read_sql_query(query, self._engine)
    except Exception as e:
        raise StorageError(f"Failed to execute query: {e}")
```

### 3. `read_table_to_dataframe()`

**Priority**: High (7 uses)  
**Purpose**: Read entire table into DataFrame with optional row limit

```python
def read_table_to_dataframe(
    self,
    table_name: str,
    engine: Engine,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """Read table into pandas DataFrame.
    
    Args:
        table_name: Name of the table to read
        engine: SQLAlchemy engine instance
        limit: Optional row limit for large tables
        
    Returns:
        Table contents as pandas DataFrame
        
    Raises:
        StorageError: If table doesn't exist or read fails
    """
```

**Implementation Notes**:
- Must validate table exists before reading
- Should handle empty tables gracefully
- Limit parameter is crucial for large datasets

### 4. `close_connections()`

**Priority**: High (7 uses)  
**Purpose**: Close all database connections and cleanup resources

```python
def close_connections(self) -> None:
    """Close all database connections and cleanup resources.
    
    Note:
        - Must be idempotent (safe to call multiple times)
        - Should dispose of connection pools
        - Must clear any cached engines
    """
```

**Implementation Notes**:
- For stateless backends, might need to close connection pools
- Must set `_engine = None` for compatibility
- Should be called in finally blocks

### 5. `read_table()`

**Priority**: High (7 uses)  
**Purpose**: Read table data (signature unclear from usage)

```python
def read_table(
    self,
    table_name: str,
    columns: Optional[List[str]] = None,
    where: Optional[str] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """Read table with optional filtering.
    
    Args:
        table_name: Name of the table
        columns: Optional list of columns to select
        where: Optional WHERE clause
        limit: Optional row limit
        
    Returns:
        Filtered table data as DataFrame
        
    Note:
        This method's signature is inferred from usage patterns
    """
```

**Implementation Notes**:
- Appears to be used primarily in feature engineering
- May need to support additional parameters
- Should build SQL dynamically based on parameters

### 6. `write_table()`

**Priority**: Medium (3 uses)  
**Purpose**: Write DataFrame to table

```python
def write_table(
    self,
    table_name: str,
    df: pd.DataFrame,
    if_exists: str = "replace"
) -> None:
    """Write DataFrame to table.
    
    Args:
        table_name: Target table name
        df: DataFrame to write
        if_exists: Behavior if table exists
        
    Note:
        Appears to be a simpler version of create_table_from_dataframe
    """
```

**Implementation Notes**:
- Used in feature engineering pipeline
- May be redundant with `create_table_from_dataframe()`
- Consider making it an alias

### 7. `get_table_info()`

**Priority**: Medium (2 uses)  
**Purpose**: Get table metadata and statistics

```python
def get_table_info(
    self,
    table_name: str,
    engine: Engine
) -> Dict[str, Any]:
    """Get table information.
    
    Args:
        table_name: Name of the table
        engine: SQLAlchemy engine
        
    Returns:
        Dictionary containing:
        - name: Table name
        - columns: List of column info dicts
        - row_count: Number of rows
        - column_count: Number of columns
    """
```

**Implementation Notes**:
- Use SQLAlchemy inspector for schema info
- Execute COUNT(*) for row count
- Format should match existing implementation

### 8. `execute_query()`

**Priority**: Low (1 use)  
**Purpose**: Execute SQL without returning DataFrame

```python
def execute_query(
    self,
    query: str,
    engine: Engine
) -> Any:
    """Execute arbitrary SQL query.
    
    Args:
        query: SQL query string
        engine: SQLAlchemy engine
        
    Returns:
        Query result (backend-specific)
        
    Note:
        Used for DDL or queries that don't return data
    """
```

### 9. `get_connection()`

**Priority**: Low (1 use)  
**Purpose**: Get raw database connection

```python
def get_connection(self) -> Any:
    """Get raw database connection.
    
    Returns:
        Backend-specific connection object
        
    Note:
        For connection pooling backends, this should
        return a connection from the pool
    """
```

### 10. `get_columns()`

**Priority**: Low (1 use)  
**Purpose**: Get column names for a table

```python
def get_columns(
    self,
    table_name: str
) -> List[str]:
    """Get column names for table.
    
    Args:
        table_name: Name of the table
        
    Returns:
        List of column names
    """
```

### 11. `analyze_column()`

**Priority**: Low (1 use)  
**Purpose**: Get column statistics

```python
def analyze_column(
    self,
    table_name: str,
    column_name: str
) -> Dict[str, Any]:
    """Analyze column statistics.
    
    Args:
        table_name: Name of the table
        column_name: Name of the column
        
    Returns:
        Dictionary with statistics:
        - count: Non-null count
        - unique: Unique value count
        - min: Minimum value
        - max: Maximum value
        - mean: Average (for numeric)
        - std: Std deviation (for numeric)
    """
```

## Implementation Strategy

### Phase 1: Critical Methods (Week 1)
1. Implement `create_table_from_dataframe()`
2. Implement `query()`
3. Add comprehensive tests

### Phase 2: High Priority Methods (Week 1-2)
1. Implement `read_table_to_dataframe()`
2. Implement `close_connections()`
3. Investigate and implement `read_table()` signature

### Phase 3: Remaining Methods (Week 2)
1. Implement remaining 6 methods
2. Ensure all signatures match usage
3. Add integration tests

## Testing Requirements

Each method must have:
1. **Unit tests** verifying correct behavior
2. **Integration tests** with real databases
3. **Compatibility tests** comparing old vs new
4. **Error handling tests** for edge cases
5. **Performance tests** for large datasets

## Backward Compatibility

To ensure smooth migration:
1. Methods must have **exact same signatures**
2. Must handle **singleton pattern** transparently
3. Error messages should **match existing format**
4. Return types must be **identical**
5. Side effects must be **preserved** (e.g., connection caching)