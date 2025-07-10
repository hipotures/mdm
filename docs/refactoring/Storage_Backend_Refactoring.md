# Storage Backend Refactoring Guide

## Overview

The storage backend system is currently implemented with singleton anti-patterns and stateful base classes. This guide details the refactoring to a clean, stateless, and testable architecture.

## Current Problems

### 1. Singleton Pattern in Base Class
```python
# CURRENT - Problematic
class StorageBackend(ABC):
    def __init__(self):
        self._engine = None
        self._session_factory = None
    
    def get_engine(self, database_path: str) -> Engine:
        if self._engine is None:  # Singleton pattern!
            self._engine = self.create_engine(database_path)
        return self._engine
```

### 2. Inconsistent Configuration
```python
# SQLiteBackend special handling
if isinstance(connection_params, dict) and 'database' in connection_params:
    nested_config = connection_params['database']
    if 'path' in nested_config:
        database_path = nested_config['path']
```

### 3. Poor Connection Management
- No connection pooling strategy
- No proper cleanup
- State management issues

## Target Architecture

### 1. Stateless Backend Interface
```python
# NEW - Clean interface
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
import pandas as pd

class StorageBackend(Protocol):
    """Protocol defining storage backend interface."""
    
    def create_table(
        self,
        connection: Connection,
        table_name: str,
        df: pd.DataFrame,
        if_exists: str = 'replace'
    ) -> None:
        """Create table from DataFrame."""
        ...
    
    def read_table(
        self,
        connection: Connection,
        table_name: str,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Read table into DataFrame."""
        ...
    
    def execute_query(
        self,
        connection: Connection,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute SQL query."""
        ...
```

### 2. Connection Manager
```python
# NEW - Separate connection management
from contextlib import contextmanager
from typing import Iterator

class ConnectionManager:
    """Manages database connections and lifecycle."""
    
    def __init__(self, backend_type: str, config: ConnectionConfig):
        self.backend_type = backend_type
        self.config = config
        self._pool = self._create_pool()
    
    @contextmanager
    def get_connection(self) -> Iterator[Connection]:
        """Get connection from pool."""
        conn = self._pool.get_connection()
        try:
            yield conn
        finally:
            self._pool.return_connection(conn)
    
    def _create_pool(self) -> ConnectionPool:
        """Create connection pool based on backend type."""
        if self.backend_type == "sqlite":
            return SQLitePool(self.config)
        elif self.backend_type == "postgresql":
            return PostgreSQLPool(self.config)
        # ... other backends
```

### 3. Backend Implementations
```python
# NEW - Stateless backend implementation
class SQLiteBackend:
    """SQLite storage backend implementation."""
    
    def create_table(
        self,
        connection: Connection,
        table_name: str,
        df: pd.DataFrame,
        if_exists: str = 'replace'
    ) -> None:
        """Create table from DataFrame."""
        # Use connection directly, no state
        df.to_sql(
            name=table_name,
            con=connection,
            if_exists=if_exists,
            index=False
        )
    
    def read_table(
        self,
        connection: Connection,
        table_name: str,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Read table into DataFrame."""
        query = self._build_query(table_name, columns, limit)
        return pd.read_sql_query(query, connection)
    
    def _build_query(
        self,
        table_name: str,
        columns: Optional[List[str]],
        limit: Optional[int]
    ) -> str:
        """Build SQL query."""
        col_str = "*" if not columns else ", ".join(columns)
        query = f"SELECT {col_str} FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        return query
```

### 4. Backend Factory with DI
```python
# NEW - Factory with dependency injection
from typing import Protocol

class BackendFactory:
    """Factory for creating storage backends."""
    
    def __init__(self, registry: BackendRegistry):
        self.registry = registry
    
    def create(self, backend_type: str) -> StorageBackend:
        """Create backend instance."""
        backend_class = self.registry.get(backend_type)
        if not backend_class:
            raise ValueError(f"Unknown backend type: {backend_type}")
        return backend_class()

class BackendRegistry:
    """Registry for backend implementations."""
    
    def __init__(self):
        self._backends: Dict[str, Type[StorageBackend]] = {}
    
    def register(self, name: str, backend_class: Type[StorageBackend]) -> None:
        """Register backend implementation."""
        self._backends[name] = backend_class
    
    def get(self, name: str) -> Optional[Type[StorageBackend]]:
        """Get backend implementation."""
        return self._backends.get(name)
```

## Migration Steps

### Step 1: Create New Interfaces
```python
# storage/interfaces.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class StorageBackend(Protocol):
    """Storage backend protocol."""
    # Define interface methods
    
class Connection(Protocol):
    """Database connection protocol."""
    # Define connection interface

class ConnectionPool(Protocol):
    """Connection pool protocol."""
    # Define pool interface
```

### Step 2: Implement Connection Management
```python
# storage/connection.py
class ConnectionConfig:
    """Connection configuration."""
    def __init__(self, **kwargs):
        self.config = kwargs

class ConnectionManager:
    """Manages connections."""
    # Implementation as shown above
```

### Step 3: Refactor Backends
```python
# storage/backends/sqlite.py
class SQLiteBackend:
    """Stateless SQLite backend."""
    # Implementation without state

# storage/backends/postgresql.py  
class PostgreSQLBackend:
    """Stateless PostgreSQL backend."""
    # Implementation without state
```

### Step 4: Update Dataset Operations
```python
# OLD - Direct backend usage
class DatasetManager:
    def load_dataset(self, name: str):
        backend = self._get_backend()
        engine = backend.get_engine(db_path)
        df = backend.read_table_to_dataframe("train", engine)

# NEW - With connection manager
class DatasetManager:
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    def load_dataset(self, name: str):
        backend = self._get_backend()
        with self.connection_manager.get_connection() as conn:
            df = backend.read_table(conn, "train")
```

## Testing Strategy

### 1. Unit Tests
```python
# tests/unit/storage/test_backends.py
class TestSQLiteBackend:
    def test_create_table(self):
        # Test with mock connection
        mock_conn = Mock()
        backend = SQLiteBackend()
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        backend.create_table(mock_conn, 'test', df)
        
        # Verify to_sql was called correctly
        df.to_sql.assert_called_once_with(
            name='test',
            con=mock_conn,
            if_exists='replace',
            index=False
        )
```

### 2. Integration Tests
```python
# tests/integration/storage/test_connection_manager.py
class TestConnectionManager:
    def test_connection_lifecycle(self, tmp_path):
        config = ConnectionConfig(path=tmp_path / "test.db")
        manager = ConnectionManager("sqlite", config)
        
        with manager.get_connection() as conn:
            # Test connection is valid
            assert conn is not None
            # Perform operations
        
        # Verify connection returned to pool
```

### 3. Migration Tests
```python
# tests/migration/test_backend_compatibility.py
class TestBackendMigration:
    def test_old_vs_new_behavior(self):
        # Compare old and new implementations
        old_result = old_backend.read_table(...)
        new_result = new_backend.read_table(...)
        
        assert old_result.equals(new_result)
```

## Rollout Plan

### Phase 1: Parallel Implementation
1. Implement new interfaces alongside old code
2. Add feature flag for backend selection
3. Test new implementation in development

### Phase 2: Gradual Migration
1. Migrate read operations first
2. Then migrate write operations
3. Update tests incrementally

### Phase 3: Cleanup
1. Remove old backend code
2. Remove feature flags
3. Update documentation

## Performance Considerations

### Connection Pooling
```python
# Configuration for optimal performance
pool_config = {
    "pool_size": 5,
    "max_overflow": 10,
    "pool_pre_ping": True,
    "pool_recycle": 3600
}
```

### Query Optimization
- Use prepared statements
- Implement query caching
- Batch operations where possible

## Backward Compatibility

### Adapter Pattern
```python
# Temporary adapter for old code
class BackendAdapter:
    """Adapts new backend to old interface."""
    
    def __init__(self, backend: StorageBackend, conn_manager: ConnectionManager):
        self.backend = backend
        self.conn_manager = conn_manager
    
    def get_engine(self, database_path: str):
        # Adapt to old interface
        warnings.warn("get_engine is deprecated", DeprecationWarning)
        return self.conn_manager._pool._engine
    
    def read_table_to_dataframe(self, table_name: str, engine):
        # Adapt to old interface
        with self.conn_manager.get_connection() as conn:
            return self.backend.read_table(conn, table_name)
```

## Configuration Migration

### Old Configuration
```yaml
database:
  default_backend: sqlite
  sqlite:
    path: ~/.mdm/datasets/{name}/dataset.sqlite
```

### New Configuration
```yaml
storage:
  default_backend: sqlite
  connection_pool:
    size: 5
    max_overflow: 10
  backends:
    sqlite:
      path_template: "~/.mdm/datasets/{name}/dataset.sqlite"
      options:
        journal_mode: WAL
        synchronous: NORMAL
```

## Success Criteria

1. **No Singletons**: All state removed from base classes
2. **Testability**: 100% unit test coverage
3. **Performance**: No degradation in benchmarks
4. **Compatibility**: All existing functionality works
5. **Extensibility**: Easy to add new backends