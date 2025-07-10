# Backend Migration Compatibility Guide

This guide explains how to add missing methods to the new stateless backends to ensure compatibility with existing MDM code.

## Overview

The new stateless backends in `mdm-refactor-2025` are missing 11 critical methods that are actively used throughout the codebase. This guide provides strategies for adding these methods while maintaining the benefits of the new architecture.

## Compatibility Strategies

### Strategy 1: Compatibility Mixin (Recommended)

Create a mixin class that adds all missing methods to stateless backends.

**File**: `src/mdm/storage/backends/compatibility_mixin.py`

```python
"""
Compatibility mixin to add missing methods to stateless backends.
This is a TEMPORARY solution until the codebase is updated.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
from pathlib import Path
from sqlalchemy import text, inspect
import logging

logger = logging.getLogger(__name__)


class BackendCompatibilityMixin:
    """
    Mixin that adds backward compatibility methods to stateless backends.
    
    This mixin provides methods that exist in the old backends but are
    missing from the new stateless implementations. Each method logs a
    deprecation warning to encourage migration to new patterns.
    """
    
    def query(self, query: str, params: Optional[Dict[str, Any]] = None, 
              database_path: Optional[str] = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame (compatibility method).
        
        NOTE: This method is deprecated. Use execute_query() instead.
        """
        logger.warning("Using deprecated method 'query'. Please update to 'execute_query'.")
        
        # Try to infer database path if not provided
        if not database_path:
            if hasattr(self, '_current_dataset'):
                dataset_name = self._current_dataset
                database_path = self.datasets_path / dataset_name / f"{dataset_name}.db"
            elif hasattr(self, '_engine') and self._engine:
                # Use existing engine if available (singleton compatibility)
                return pd.read_sql_query(query, self._engine, params=params)
            else:
                raise ValueError("database_path required for query in stateless backend")
        
        # Get dataset name from path
        dataset_name = Path(database_path).parent.name
        
        with self.get_engine_context(dataset_name) as engine:
            return pd.read_sql_query(query, engine, params=params)
    
    def create_table_from_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str,
        engine: Optional[Engine] = None,
        database_path: Optional[str] = None,
        if_exists: str = "fail"
    ) -> None:
        """Create table from pandas DataFrame (compatibility method).
        
        NOTE: This method is deprecated. Use save_data() instead.
        """
        logger.warning("Using deprecated method 'create_table_from_dataframe'. Please update to 'save_data'.")
        
        # Handle different calling patterns
        if engine is not None:
            # Old pattern: engine provided directly
            df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        elif database_path:
            # Path provided
            dataset_name = Path(database_path).parent.name
            self.save_data(dataset_name, df, table_name, if_exists)
        elif hasattr(self, '_current_dataset'):
            # Use current dataset context
            self.save_data(self._current_dataset, df, table_name, if_exists)
        else:
            raise ValueError("Either engine or database_path required")
    
    def close_connections(self) -> None:
        """Close all connections (compatibility method).
        
        NOTE: This method is deprecated. Use close() instead.
        """
        logger.warning("Using deprecated method 'close_connections'. Please update to 'close'.")
        
        # Close connection pool if exists
        if hasattr(self, 'pool'):
            self.pool.close_all()
            
        # Clear singleton engine for compatibility
        if hasattr(self, '_engine'):
            if self._engine:
                self._engine.dispose()
            self._engine = None
            self._session_factory = None
            
        # Call new close method if exists
        if hasattr(self, 'close'):
            self.close()
    
    def read_table_to_dataframe(
        self, 
        table_name: str, 
        engine: Optional[Engine] = None,
        database_path: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Read entire table to DataFrame (compatibility method).
        
        NOTE: This method is deprecated. Use load_data() instead.
        """
        logger.warning("Using deprecated method 'read_table_to_dataframe'. Please update to 'load_data'.")
        
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
            
        if engine:
            return pd.read_sql_query(query, engine)
        else:
            return self.query(query, database_path=database_path)
    
    def read_table(
        self, 
        table_name: str, 
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        database_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Read table with optional filtering."""
        # Build query
        if columns:
            cols = ", ".join(columns)
        else:
            cols = "*"
            
        query = f"SELECT {cols} FROM {table_name}"
        
        if where:
            query += f" WHERE {where}"
            
        if limit:
            query += f" LIMIT {limit}"
            
        return self.query(query, database_path=database_path)
    
    def write_table(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        if_exists: str = "replace",
        database_path: Optional[str] = None
    ) -> None:
        """Write DataFrame to table."""
        self.create_table_from_dataframe(
            df, table_name, 
            database_path=database_path,
            if_exists=if_exists
        )
    
    def get_table_info(
        self, 
        table_name: str, 
        engine: Optional[Engine] = None,
        database_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get table schema information."""
        if not engine and not database_path:
            raise ValueError("Either engine or database_path required")
            
        if database_path and not engine:
            dataset_name = Path(database_path).parent.name
            with self.get_engine_context(dataset_name) as eng:
                return self._get_table_info_impl(table_name, eng)
        else:
            return self._get_table_info_impl(table_name, engine)
    
    def _get_table_info_impl(self, table_name: str, engine: Engine) -> Dict[str, Any]:
        """Implementation of get_table_info."""
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        
        # Get row count
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()
        
        return {
            "name": table_name,
            "columns": columns,
            "row_count": row_count,
            "column_count": len(columns),
        }
    
    def execute_query(
        self, 
        query: str, 
        engine: Optional[Engine] = None,
        database_path: Optional[str] = None
    ) -> Any:
        """Execute query without returning results."""
        if not engine and not database_path:
            raise ValueError("Either engine or database_path required")
            
        if database_path and not engine:
            dataset_name = Path(database_path).parent.name
            with self.get_engine_context(dataset_name) as eng:
                with eng.begin() as conn:
                    return conn.execute(text(query))
        else:
            with engine.begin() as conn:
                return conn.execute(text(query))
    
    def get_connection(self, database_path: Optional[str] = None) -> Any:
        """Get raw connection (compatibility method)."""
        if not database_path:
            raise ValueError("database_path required for stateless backend")
            
        dataset_name = Path(database_path).parent.name
        return self.pool.get_connection(dataset_name)
    
    def get_columns(
        self, 
        table_name: str, 
        database_path: Optional[str] = None
    ) -> List[str]:
        """Get column names for table."""
        info = self.get_table_info(table_name, database_path=database_path)
        return [col['name'] for col in info['columns']]
    
    def analyze_column(
        self, 
        table_name: str, 
        column_name: str,
        database_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze column statistics."""
        query = f"""
        SELECT 
            COUNT(*) as count,
            COUNT(DISTINCT {column_name}) as unique_count,
            MIN({column_name}) as min_value,
            MAX({column_name}) as max_value
        FROM {table_name}
        """
        
        stats_df = self.query(query, database_path=database_path)
        stats = stats_df.iloc[0].to_dict()
        
        # Try to get numeric statistics
        try:
            numeric_query = f"""
            SELECT 
                AVG(CAST({column_name} AS FLOAT)) as mean,
                STDEV(CAST({column_name} AS FLOAT)) as std
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            """
            numeric_df = self.query(numeric_query, database_path=database_path)
            stats.update(numeric_df.iloc[0].to_dict())
        except:
            # Not a numeric column
            pass
            
        return stats
    
    def set_current_dataset(self, dataset_name: str):
        """Set current dataset for operations (helper for compatibility)."""
        self._current_dataset = dataset_name
        
    # Singleton compatibility support
    _engine: Optional[Engine] = None
    _session_factory: Optional[Any] = None
```

### Strategy 2: Gradual Migration

Update the codebase gradually to use new patterns while maintaining compatibility.

#### Step 1: Update Import Statements

```python
# Old import
from mdm.storage.factory import BackendFactory

# New import with compatibility
try:
    from mdm.storage.factory import BackendFactory
except ImportError:
    from mdm.storage import BackendFactory
```

#### Step 2: Update Method Calls

```python
# Old pattern
df = backend.query("SELECT * FROM data")

# New pattern with fallback
if hasattr(backend, 'execute_query'):
    df = pd.read_sql_query("SELECT * FROM data", backend.get_engine())
else:
    df = backend.query("SELECT * FROM data")
```

#### Step 3: Feature Flags

```python
# Use feature flags to control behavior
if feature_flags.get("use_new_storage_api", False):
    # New API
    with backend.get_engine_context(dataset_name) as engine:
        df = pd.read_sql_query(query, engine)
else:
    # Old API
    df = backend.query(query)
```

## Implementation Steps

### 1. Add Mixin to Stateless Backends

Update each stateless backend to include the mixin:

```python
# In stateless_sqlite.py
from .compatibility_mixin import BackendCompatibilityMixin

class StatelessSQLiteBackend(BackendCompatibilityMixin, IStorageBackend):
    """Stateless SQLite backend with compatibility layer"""
    # ... existing implementation
```

### 2. Add Singleton Support

For full compatibility, support the singleton pattern transparently:

```python
class StatelessSQLiteBackend(BackendCompatibilityMixin, IStorageBackend):
    def __init__(self):
        super().__init__()
        self._engine = None  # For singleton compatibility
        self._session_factory = None
        
    def get_engine(self, database_path: Optional[str] = None) -> Engine:
        """Get SQLAlchemy engine with singleton support."""
        if database_path:
            # New behavior - always create fresh engine
            return self._create_engine(database_path)
        elif self._engine:
            # Singleton compatibility
            return self._engine
        else:
            raise ValueError("No database_path provided and no cached engine")
```

### 3. Testing Strategy

Create compatibility tests that verify both old and new APIs work:

```python
import pytest
from mdm.storage.backends.stateless_sqlite import StatelessSQLiteBackend
from mdm.storage.sqlite import SQLiteBackend

class TestBackendCompatibility:
    """Test that new backends are compatible with old API"""
    
    @pytest.mark.parametrize("backend_class", [
        SQLiteBackend,
        StatelessSQLiteBackend
    ])
    def test_query_method(self, backend_class, tmp_path):
        """Test query() method works on both backends"""
        backend = backend_class()
        
        # Setup
        db_path = tmp_path / "test.db"
        backend.create_database(str(db_path))
        engine = backend.get_engine(str(db_path))
        
        # Create test data
        df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        backend.create_table_from_dataframe(df, "test_table", engine)
        
        # Test query
        result = backend.query("SELECT * FROM test_table")
        assert len(result) == 3
        assert list(result.columns) == ["id", "value"]
        
        # Cleanup
        backend.close_connections()
```

## Migration Timeline

### Week 1: Immediate Compatibility
1. Add compatibility mixin to all stateless backends
2. Ensure all 14 methods are present
3. Run existing test suite

### Week 2: Gradual Refactoring
1. Update high-frequency code paths to use new API
2. Add deprecation warnings
3. Update documentation

### Week 3-4: Complete Migration
1. Remove compatibility layer from refactored code
2. Update remaining code to new API
3. Remove deprecated methods

## Monitoring and Validation

### Deprecation Tracking

Log all deprecated method usage:

```python
import logging
from collections import defaultdict

deprecation_counter = defaultdict(int)

def log_deprecation(method_name: str):
    deprecation_counter[method_name] += 1
    if deprecation_counter[method_name] % 100 == 0:
        logger.warning(
            f"Method '{method_name}' called {deprecation_counter[method_name]} times. "
            "Please update to new API."
        )
```

### Performance Monitoring

Compare performance between old and new implementations:

```python
import time

def benchmark_compatibility():
    old_time = time.time()
    old_result = old_backend.query("SELECT * FROM large_table")
    old_duration = time.time() - old_time
    
    new_time = time.time()
    new_result = new_backend.query("SELECT * FROM large_table")
    new_duration = time.time() - new_time
    
    print(f"Old: {old_duration:.2f}s, New: {new_duration:.2f}s")
    assert old_result.equals(new_result)
```

## Common Issues and Solutions

### Issue 1: Missing Engine Context
**Problem**: Methods called without proper engine setup  
**Solution**: Add engine caching for singleton compatibility

### Issue 2: Different Parameter Orders
**Problem**: Old code passes parameters in different order  
**Solution**: Use **kwargs and parameter detection

### Issue 3: Connection Pool Exhaustion
**Problem**: Not closing connections properly  
**Solution**: Implement automatic cleanup in compatibility layer

### Issue 4: Transaction Handling
**Problem**: Different transaction semantics  
**Solution**: Wrap operations in appropriate transaction contexts

## Success Criteria

The migration is complete when:
1. ✅ All existing tests pass with new backends
2. ✅ No deprecated method calls in codebase
3. ✅ Performance is equal or better
4. ✅ Zero runtime errors in production
5. ✅ Compatibility layer can be removed

## References

- [Storage Backend API Reference](storage_backend_api_reference.md)
- [Usage Analysis](storage_backend_usage_analysis.md)
- [Missing Methods Specification](missing_methods_specification.md)
- Original implementation: `src/mdm/storage/base.py`