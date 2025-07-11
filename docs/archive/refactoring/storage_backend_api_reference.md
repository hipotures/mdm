# Storage Backend API Reference

This document provides the complete API reference for MDM Storage Backends based on actual usage analysis of the codebase.

## Overview

The Storage Backend system in MDM provides an abstraction layer over different database backends (SQLite, DuckDB, PostgreSQL). Based on code analysis, there are **14 unique methods** actively used across the codebase.

## Usage Statistics

| Method | Usage Count | Status in New Backend |
|--------|-------------|----------------------|
| `get_engine()` | 11 | ✅ Implemented |
| `create_table_from_dataframe()` | 10 | ❌ Missing |
| `query()` | 9 | ❌ Missing |
| `read_table_to_dataframe()` | 7 | ❌ Missing |
| `close_connections()` | 7 | ❌ Missing |
| `read_table()` | 7 | ❌ Missing |
| `write_table()` | 3 | ❌ Missing |
| `get_table_info()` | 2 | ❌ Missing |
| `execute_query()` | 1 | ❌ Missing |
| `get_connection()` | 1 | ❌ Missing |
| `get_columns()` | 1 | ❌ Missing |
| `analyze_column()` | 1 | ❌ Missing |
| `database_exists()` | 1 | ✅ Implemented |
| `create_database()` | 1 | ✅ Implemented |

## Complete API Specification

### Core Methods

#### 1. `get_engine(database_path: str) -> Engine`
**Usage**: 11 times  
**Purpose**: Get or create SQLAlchemy engine for database  
**Implementation**: Present in base class and all backends

```python
def get_engine(self, database_path: str) -> Engine:
    """Get or create engine for database.
    
    Args:
        database_path: Path or connection string to database
        
    Returns:
        SQLAlchemy Engine instance
    """
```

**Example Usage**:
```python
# From api.py:195
engine = backend.get_engine(db_path)
```

#### 2. `create_table_from_dataframe(df: pd.DataFrame, table_name: str, engine: Engine, if_exists: str = "fail") -> None`
**Usage**: 10 times  
**Purpose**: Create table from pandas DataFrame  
**Implementation**: Present in base class

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
        df: Pandas DataFrame
        table_name: Name of the table to create
        engine: SQLAlchemy engine
        if_exists: What to do if table exists ('fail', 'replace', 'append')
    """
```

**Example Usage**:
```python
# From dataset/registrar.py:441
backend.create_table_from_dataframe(
    chunk_df, table_name, engine, if_exists='replace'
)
```

#### 3. `query(query: str) -> pd.DataFrame`
**Usage**: 9 times  
**Purpose**: Execute SQL query and return results as DataFrame  
**Implementation**: Present in base class

```python
def query(self, query: str) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame.
    
    Args:
        query: SQL query string
        
    Returns:
        Query results as pandas DataFrame
    """
```

**Example Usage**:
```python
# From api.py:577
df = backend.query(f"SELECT * FROM {table_name}")

# From cli/dataset.py:57
rows = backend.query(f"SELECT COUNT(*) as count FROM {table}")
```

#### 4. `read_table_to_dataframe(table_name: str, engine: Engine, limit: Optional[int] = None) -> pd.DataFrame`
**Usage**: 7 times  
**Purpose**: Read entire table into DataFrame  
**Implementation**: Present in base class

```python
def read_table_to_dataframe(
    self, 
    table_name: str, 
    engine: Engine, 
    limit: Optional[int] = None
) -> pd.DataFrame:
    """Read table into pandas DataFrame.
    
    Args:
        table_name: Name of the table
        engine: SQLAlchemy engine
        limit: Optional row limit
        
    Returns:
        Pandas DataFrame
    """
```

**Example Usage**:
```python
# From api.py:196
df = backend.read_table_to_dataframe("data", engine)
```

#### 5. `close_connections() -> None`
**Usage**: 7 times  
**Purpose**: Close all database connections  
**Implementation**: Present in base class

```python
def close_connections(self) -> None:
    """Close all database connections."""
```

**Example Usage**:
```python
# From dataset/registrar.py:666
finally:
    backend.close_connections()
```

### Data I/O Methods

#### 6. `read_table()` 
**Usage**: 7 times  
**Purpose**: Read table data (method signature varies in usage)  
**Note**: This appears to be called but not defined in base class - likely backend-specific

**Example Usage**:
```python
# From features/engine.py:126
data = backend.read_table(table_name)
```

#### 7. `write_table()`
**Usage**: 3 times  
**Purpose**: Write data to table (method signature varies in usage)  
**Note**: This appears to be called but not defined in base class - likely backend-specific

**Example Usage**:
```python
# From features/engine.py:163
backend.write_table(table_name, processed_data)
```

### Information Methods

#### 8. `get_table_info(table_name: str, engine: Engine) -> dict[str, Any]`
**Usage**: 2 times  
**Purpose**: Get table schema and metadata  
**Implementation**: Present in base class

```python
def get_table_info(self, table_name: str, engine: Engine) -> dict[str, Any]:
    """Get table information.
    
    Args:
        table_name: Name of the table
        engine: SQLAlchemy engine
        
    Returns:
        Dictionary with table information including:
        - name: Table name
        - columns: Column information
        - row_count: Number of rows
        - column_count: Number of columns
    """
```

### Utility Methods

#### 9. `execute_query(query: str, engine: Engine) -> Any`
**Usage**: 1 time  
**Purpose**: Execute arbitrary SQL without returning DataFrame  
**Implementation**: Present in base class

```python
def execute_query(self, query: str, engine: Engine) -> Any:
    """Execute arbitrary SQL query.
    
    Args:
        query: SQL query string
        engine: SQLAlchemy engine
        
    Returns:
        Query result
    """
```

#### 10. `get_connection()`
**Usage**: 1 time  
**Purpose**: Get raw database connection  
**Note**: Usage found but method not defined in base class

**Example Usage**:
```python
# From api.py:279
conn = backend.get_connection()
```

#### 11. `get_columns()`
**Usage**: 1 time  
**Purpose**: Get column names for a table  
**Note**: Usage found but method not defined in base class

**Example Usage**:
```python
# From api.py:619
columns = backend.get_columns(table_name)
```

#### 12. `analyze_column()`
**Usage**: 1 time  
**Purpose**: Analyze column statistics  
**Note**: Usage found but method not defined in base class

**Example Usage**:
```python
# From api.py:623
stats = backend.analyze_column(table_name, column_name)
```

### Database Management Methods

#### 13. `database_exists(database_path: str) -> bool`
**Usage**: 1 time  
**Purpose**: Check if database exists  
**Implementation**: Abstract method, implemented in each backend

```python
def database_exists(self, database_path: str) -> bool:
    """Check if database exists.
    
    Args:
        database_path: Path or connection string to database
        
    Returns:
        True if database exists
    """
```

#### 14. `create_database(database_path: str) -> None`
**Usage**: 1 time  
**Purpose**: Create a new database  
**Implementation**: Abstract method, implemented in each backend

```python
def create_database(self, database_path: str) -> None:
    """Create a new database.
    
    Args:
        database_path: Path or connection string to database
    """
```

## Additional Base Class Methods (Not Found in Usage)

These methods are defined in the base class but were not found in the usage analysis:

- `drop_database(database_path: str) -> None`
- `initialize_database(engine: Engine) -> None`
- `get_database_path(dataset_name: str, base_path: Path) -> str`
- `table_exists(engine: Engine, table_name: str) -> bool`
- `get_table_names(engine: Engine) -> list[str]`
- `session(database_path: str) -> Generator[Session, None, None]`

## Backend-Specific Implementations

### SQLiteBackend
- Inherits from `StorageBackend`
- Implements all abstract methods
- Uses file-based storage
- Sets SQLite-specific pragmas (WAL mode, cache size, etc.)

### DuckDBBackend
- Would implement same interface
- In-memory or file-based analytical database
- Optimized for OLAP workloads

### PostgreSQLBackend
- Would implement same interface
- Network-based relational database
- Requires connection parameters (host, port, user, password)

## Migration Requirements

For the new "stateless" backends to work with existing code, they MUST implement:

1. **All 14 methods** found in usage analysis
2. **Maintain exact method signatures** for compatibility
3. **Handle the singleton pattern** transparently (even if internally stateless)
4. **Provide backward compatibility** during migration period

## Testing Requirements

Each backend implementation should be tested for:

1. **Method presence**: All 14 methods must exist
2. **Signature compatibility**: Parameters and return types must match
3. **Behavioral compatibility**: Same inputs produce equivalent outputs
4. **Error handling**: Same exceptions for error conditions
5. **Performance**: No significant regression vs current implementation