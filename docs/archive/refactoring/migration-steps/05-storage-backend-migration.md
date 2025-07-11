# Step 5: Storage Backend Migration

## Overview

Migrate from singleton-based storage backends to stateless implementations with connection pooling. This is one of the most critical migrations as it affects all data operations.

## Duration

3 weeks (Weeks 8-10)

## Objectives

1. Implement stateless storage backends with connection pooling
2. Create connection management infrastructure
3. Migrate all three backends (SQLite, DuckDB, PostgreSQL)
4. Ensure zero data loss and backward compatibility
5. Improve performance with connection reuse

## Current State Analysis

Current issues:
- Singleton pattern prevents multiple connections
- No connection pooling (performance bottleneck)
- State stored in backend instances
- Difficult to test in isolation
- No connection lifecycle management

## Detailed Steps

### Week 8: Connection Pool Infrastructure

#### Day 1-2: Connection Pool Design

##### 1.1 Create Connection Pool Interface
```python
# Create: src/mdm/storage/pooling.py
from typing import Protocol, Any, Dict, Optional, List, ContextManager
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import queue
import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a pooled connection"""
    connection: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    dataset_name: str = ""
    
    def is_stale(self, max_age: timedelta) -> bool:
        """Check if connection is too old"""
        return datetime.now() - self.created_at > max_age
    
    def is_idle(self, max_idle: timedelta) -> bool:
        """Check if connection has been idle too long"""
        return datetime.now() - self.last_used > max_idle


class ConnectionPool(ABC):
    """Abstract base class for connection pools"""
    
    def __init__(self, 
                 min_size: int = 1,
                 max_size: int = 10,
                 max_age: timedelta = timedelta(hours=1),
                 max_idle: timedelta = timedelta(minutes=15)):
        self.min_size = min_size
        self.max_size = max_size
        self.max_age = max_age
        self.max_idle = max_idle
        
        self._pool: queue.Queue[ConnectionInfo] = queue.Queue(maxsize=max_size)
        self._all_connections: List[ConnectionInfo] = []
        self._lock = threading.RLock()
        self._created_count = 0
        self._closed = False
        
        # Initialize minimum connections
        self._initialize_pool()
    
    @abstractmethod
    def create_connection(self, dataset_name: str) -> Any:
        """Create a new connection"""
        pass
    
    @abstractmethod
    def validate_connection(self, connection: Any) -> bool:
        """Validate that connection is still alive"""
        pass
    
    @abstractmethod
    def close_connection(self, connection: Any) -> None:
        """Close a connection"""
        pass
    
    def _initialize_pool(self):
        """Initialize pool with minimum connections"""
        for _ in range(self.min_size):
            try:
                conn_info = self._create_new_connection("")
                self._pool.put(conn_info)
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")
    
    def _create_new_connection(self, dataset_name: str) -> ConnectionInfo:
        """Create a new connection and track it"""
        with self._lock:
            if self._created_count >= self.max_size:
                raise RuntimeError(f"Connection pool exhausted (max={self.max_size})")
            
            connection = self.create_connection(dataset_name)
            conn_info = ConnectionInfo(
                connection=connection,
                dataset_name=dataset_name
            )
            
            self._all_connections.append(conn_info)
            self._created_count += 1
            
            logger.debug(f"Created new connection for {dataset_name}, total: {self._created_count}")
            return conn_info
    
    @contextmanager
    def get_connection(self, dataset_name: str) -> ContextManager[Any]:
        """Get a connection from the pool"""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn_info = None
        try:
            # Try to get from pool
            try:
                conn_info = self._pool.get(timeout=1.0)
                
                # Validate connection
                if (conn_info.is_stale(self.max_age) or 
                    not self.validate_connection(conn_info.connection)):
                    # Connection is bad, close it and create new one
                    self.close_connection(conn_info.connection)
                    with self._lock:
                        self._all_connections.remove(conn_info)
                        self._created_count -= 1
                    conn_info = self._create_new_connection(dataset_name)
                
            except queue.Empty:
                # No connections available, create new one if possible
                conn_info = self._create_new_connection(dataset_name)
            
            # Update usage info
            conn_info.last_used = datetime.now()
            conn_info.use_count += 1
            
            yield conn_info.connection
            
        finally:
            # Return to pool
            if conn_info and not self._closed:
                self._pool.put(conn_info)
    
    def cleanup_idle_connections(self):
        """Remove idle connections"""
        with self._lock:
            removed = 0
            remaining = []
            
            for conn_info in self._all_connections:
                if (conn_info.is_idle(self.max_idle) and 
                    self._created_count > self.min_size):
                    try:
                        self.close_connection(conn_info.connection)
                        self._created_count -= 1
                        removed += 1
                    except Exception as e:
                        logger.error(f"Error closing idle connection: {e}")
                        remaining.append(conn_info)
                else:
                    remaining.append(conn_info)
            
            self._all_connections = remaining
            if removed > 0:
                logger.info(f"Cleaned up {removed} idle connections")
    
    def close_all(self):
        """Close all connections in the pool"""
        with self._lock:
            self._closed = True
            
            # Close all connections
            for conn_info in self._all_connections:
                try:
                    self.close_connection(conn_info.connection)
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            self._all_connections.clear()
            self._created_count = 0
            
            # Clear the queue
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                except queue.Empty:
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            available = self._pool.qsize()
            in_use = self._created_count - available
            
            return {
                "total_connections": self._created_count,
                "available": available,
                "in_use": in_use,
                "min_size": self.min_size,
                "max_size": self.max_size,
                "closed": self._closed
            }
```

##### 1.2 Implement Backend-Specific Pools
```python
# Create: src/mdm/storage/pools/sqlite_pool.py
import sqlite3
from pathlib import Path
from typing import Any
import logging

from ..pooling import ConnectionPool
from ...config import get_config

logger = logging.getLogger(__name__)


class SQLiteConnectionPool(ConnectionPool):
    """Connection pool for SQLite databases"""
    
    def __init__(self, **kwargs):
        # SQLite doesn't benefit from many connections
        kwargs["min_size"] = kwargs.get("min_size", 1)
        kwargs["max_size"] = kwargs.get("max_size", 5)
        super().__init__(**kwargs)
        
        self.config = get_config()
        self.datasets_path = self.config.paths.datasets_path
    
    def create_connection(self, dataset_name: str) -> sqlite3.Connection:
        """Create a new SQLite connection"""
        if dataset_name:
            db_path = self.datasets_path / dataset_name / f"{dataset_name}.db"
        else:
            # For pool initialization, use in-memory database
            db_path = ":memory:"
        
        # Create connection with optimized settings
        conn = sqlite3.connect(
            str(db_path),
            timeout=30.0,
            isolation_level=None,  # Autocommit mode
            check_same_thread=False  # Allow sharing between threads
        )
        
        # Apply performance optimizations
        conn.execute(f"PRAGMA synchronous = {self.config.database.sqlite_synchronous}")
        conn.execute(f"PRAGMA journal_mode = {self.config.database.sqlite_journal_mode}")
        conn.execute(f"PRAGMA cache_size = {self.config.database.sqlite_cache_size}")
        conn.execute("PRAGMA temp_store = MEMORY")
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        logger.debug(f"Created SQLite connection for {dataset_name}")
        return conn
    
    def validate_connection(self, connection: sqlite3.Connection) -> bool:
        """Check if SQLite connection is valid"""
        try:
            connection.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def close_connection(self, connection: sqlite3.Connection) -> None:
        """Close SQLite connection"""
        try:
            connection.close()
        except Exception as e:
            logger.error(f"Error closing SQLite connection: {e}")


# Create: src/mdm/storage/pools/duckdb_pool.py
import duckdb
from pathlib import Path
from typing import Any
import logging

from ..pooling import ConnectionPool
from ...config import get_config

logger = logging.getLogger(__name__)


class DuckDBConnectionPool(ConnectionPool):
    """Connection pool for DuckDB databases"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = get_config()
        self.datasets_path = self.config.paths.datasets_path
    
    def create_connection(self, dataset_name: str) -> duckdb.DuckDBPyConnection:
        """Create a new DuckDB connection"""
        if dataset_name:
            db_path = self.datasets_path / dataset_name / f"{dataset_name}.duckdb"
            db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            db_path = ":memory:"
        
        # Create connection with configuration
        config = {
            'threads': self.config.database.duckdb_threads or 0,  # 0 = auto
            'memory_limit': self.config.database.duckdb_memory_limit,
            'max_memory': self.config.database.duckdb_memory_limit,
        }
        
        if self.config.database.duckdb_temp_directory:
            config['temp_directory'] = str(self.config.database.duckdb_temp_directory)
        
        conn = duckdb.connect(str(db_path), config=config)
        
        logger.debug(f"Created DuckDB connection for {dataset_name}")
        return conn
    
    def validate_connection(self, connection: duckdb.DuckDBPyConnection) -> bool:
        """Check if DuckDB connection is valid"""
        try:
            connection.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False
    
    def close_connection(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Close DuckDB connection"""
        try:
            connection.close()
        except Exception as e:
            logger.error(f"Error closing DuckDB connection: {e}")


# Create: src/mdm/storage/pools/postgresql_pool.py
import psycopg2
from psycopg2 import pool
from typing import Any, Optional
import logging

from ..pooling import ConnectionPool
from ...config import get_config

logger = logging.getLogger(__name__)


class PostgreSQLConnectionPool(ConnectionPool):
    """Connection pool for PostgreSQL databases"""
    
    def __init__(self, **kwargs):
        self.config = get_config()
        
        # Use psycopg2's built-in pool
        self._pg_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=kwargs.get("min_size", 2),
            maxconn=kwargs.get("max_size", 20),
            host=self.config.database.postgresql_host,
            port=self.config.database.postgresql_port,
            user=self.config.database.postgresql_user,
            password=self.config.database.postgresql_password,
            database=self.config.database.postgresql_database
        )
        
        # Don't call parent __init__ as we're using psycopg2's pool
        self.closed = False
    
    def create_connection(self, dataset_name: str) -> Any:
        """Get connection from psycopg2 pool"""
        return self._pg_pool.getconn()
    
    def validate_connection(self, connection: Any) -> bool:
        """Check if PostgreSQL connection is valid"""
        try:
            with connection.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def close_connection(self, connection: Any) -> None:
        """Return connection to psycopg2 pool"""
        if not self.closed:
            self._pg_pool.putconn(connection)
    
    @contextmanager
    def get_connection(self, dataset_name: str) -> ContextManager[Any]:
        """Get connection from pool"""
        conn = None
        try:
            conn = self._pg_pool.getconn()
            
            # Set schema for dataset
            if dataset_name:
                with conn.cursor() as cur:
                    cur.execute(f"SET search_path TO {dataset_name}, public")
            
            yield conn
            
        finally:
            if conn:
                self._pg_pool.putconn(conn)
    
    def close_all(self):
        """Close all connections"""
        self.closed = True
        self._pg_pool.closeall()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        # Note: psycopg2 pool doesn't expose detailed stats
        return {
            "type": "PostgreSQL",
            "closed": self.closed
        }
```

#### Day 3-4: Stateless Backend Implementation

##### 1.3 Create New Stateless Backends
```python
# Create: src/mdm/storage/backends/stateless_sqlite.py
from typing import Any, Dict, Optional, List
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.engine import Engine
from contextlib import contextmanager
import logging

from ...interfaces.storage import IStorageBackend
from ..pools.sqlite_pool import SQLiteConnectionPool
from ...config import get_config

logger = logging.getLogger(__name__)


class StatelessSQLiteBackend(IStorageBackend):
    """Stateless SQLite backend with connection pooling"""
    
    def __init__(self):
        self.config = get_config()
        self.pool = SQLiteConnectionPool(
            min_size=1,
            max_size=self.config.performance.max_workers
        )
        self.datasets_path = self.config.paths.datasets_path
    
    @contextmanager
    def get_engine(self, dataset_name: str) -> Engine:
        """Get SQLAlchemy engine for dataset"""
        with self.pool.get_connection(dataset_name) as conn:
            # Create engine from existing connection
            engine = create_engine(
                "sqlite://",
                creator=lambda: conn,
                echo=self.config.database.echo_sql
            )
            yield engine
    
    def create_dataset(self, dataset_name: str, config: Dict[str, Any]) -> None:
        """Create a new dataset database"""
        # Create directory
        dataset_path = self.datasets_path / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create database file
        db_path = dataset_path / f"{dataset_name}.db"
        
        # Initialize schema
        with self.get_engine(dataset_name) as engine:
            # Create metadata table
            engine.execute(text("""
                CREATE TABLE IF NOT EXISTS _metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create data table (will be modified later based on actual data)
            engine.execute(text("""
                CREATE TABLE IF NOT EXISTS data (
                    _row_id INTEGER PRIMARY KEY AUTOINCREMENT
                )
            """))
            
            # Store initial metadata
            for key, value in config.items():
                engine.execute(
                    text("INSERT OR REPLACE INTO _metadata (key, value) VALUES (:key, :value)"),
                    {"key": key, "value": str(value)}
                )
        
        logger.info(f"Created SQLite dataset: {dataset_name}")
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset exists"""
        db_path = self.datasets_path / dataset_name / f"{dataset_name}.db"
        return db_path.exists()
    
    def drop_dataset(self, dataset_name: str) -> None:
        """Remove dataset"""
        dataset_path = self.datasets_path / dataset_name
        
        if dataset_path.exists():
            # Close any connections
            # Note: In stateless design, connections are not held
            
            # Remove directory
            import shutil
            shutil.rmtree(dataset_path)
            
        logger.info(f"Dropped SQLite dataset: {dataset_name}")
    
    def load_data(self, dataset_name: str, table_name: str = "data") -> pd.DataFrame:
        """Load data from dataset"""
        with self.get_engine(dataset_name) as engine:
            return pd.read_sql_table(table_name, engine)
    
    def save_data(self, dataset_name: str, data: pd.DataFrame,
                  table_name: str = "data", if_exists: str = "replace") -> None:
        """Save data to dataset"""
        with self.get_engine(dataset_name) as engine:
            data.to_sql(
                table_name,
                engine,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=self.config.performance.chunk_size
            )
    
    def get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset metadata"""
        with self.get_engine(dataset_name) as engine:
            result = engine.execute(text("SELECT key, value FROM _metadata"))
            return {row[0]: row[1] for row in result}
    
    def update_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        """Update dataset metadata"""
        with self.get_engine(dataset_name) as engine:
            for key, value in metadata.items():
                engine.execute(
                    text("""
                        INSERT OR REPLACE INTO _metadata (key, value, updated_at) 
                        VALUES (:key, :value, CURRENT_TIMESTAMP)
                    """),
                    {"key": key, "value": str(value)}
                )
    
    def close(self) -> None:
        """Close connection pool"""
        self.pool.close_all()


# Similar implementations for DuckDB and PostgreSQL...
```

##### 1.4 Create Backend Factory
```python
# Create: src/mdm/storage/factory.py
from typing import Dict, Type, Optional
import logging

from ..interfaces.storage import IStorageBackend
from .backends.stateless_sqlite import StatelessSQLiteBackend
from .backends.stateless_duckdb import StatelessDuckDBBackend
from .backends.stateless_postgresql import StatelessPostgreSQLBackend
from .adapters import SQLiteAdapter, DuckDBAdapter, PostgreSQLAdapter
from ..config import get_config
from ..core.feature_flags import feature_flags

logger = logging.getLogger(__name__)


class StorageBackendFactory:
    """Factory for creating storage backends"""
    
    # Registry of backend implementations
    _backends: Dict[str, Type[IStorageBackend]] = {
        # New stateless backends
        "sqlite_new": StatelessSQLiteBackend,
        "duckdb_new": StatelessDuckDBBackend,
        "postgresql_new": StatelessPostgreSQLBackend,
        
        # Legacy backends (via adapters)
        "sqlite_legacy": SQLiteAdapter,
        "duckdb_legacy": DuckDBAdapter,
        "postgresql_legacy": PostgreSQLAdapter,
    }
    
    @classmethod
    def create(cls, backend_type: Optional[str] = None) -> IStorageBackend:
        """Create a storage backend instance"""
        config = get_config()
        
        # Determine backend type
        if backend_type is None:
            backend_type = config.database.default_backend
        
        # Check feature flag for new vs legacy
        use_new = feature_flags.get("use_new_backend", False)
        
        # Construct registry key
        suffix = "new" if use_new else "legacy"
        registry_key = f"{backend_type}_{suffix}"
        
        # Create backend
        if registry_key not in cls._backends:
            raise ValueError(f"Unknown backend type: {backend_type} ({suffix})")
        
        backend_class = cls._backends[registry_key]
        backend = backend_class()
        
        logger.info(f"Created {registry_key} storage backend")
        return backend
    
    @classmethod
    def register_backend(cls, name: str, backend_class: Type[IStorageBackend]):
        """Register a custom backend"""
        cls._backends[name] = backend_class


# Convenience function
def get_storage_backend(backend_type: Optional[str] = None) -> IStorageBackend:
    """Get a storage backend instance"""
    return StorageBackendFactory.create(backend_type)
```

### Week 9: Migration Implementation

#### Day 5-6: Gradual Migration Strategy

##### 2.1 Create Migration Coordinator
```python
# Create: src/mdm/storage/migration/coordinator.py
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

from ...testing.comparison import ComparisonTester
from ...core.metrics import metrics_collector
from ..factory import get_storage_backend

logger = logging.getLogger(__name__)


class StorageBackendMigrator:
    """Coordinates storage backend migration"""
    
    def __init__(self):
        self.comparison_tester = ComparisonTester()
        self.migration_log: List[Dict[str, Any]] = []
    
    def validate_backend_compatibility(self, dataset_name: str) -> Dict[str, Any]:
        """Validate that new backend produces same results as legacy"""
        results = {
            "dataset": dataset_name,
            "timestamp": datetime.now(),
            "tests": [],
            "passed": True
        }
        
        # Get both backends
        legacy_backend = get_storage_backend()  # Will use legacy
        
        # Temporarily enable new backend
        from ...core.feature_flags import feature_flags
        original_flag = feature_flags.get("use_new_backend")
        feature_flags.set("use_new_backend", True)
        new_backend = get_storage_backend()
        feature_flags.set("use_new_backend", original_flag)
        
        try:
            # Test 1: Dataset existence
            test_result = self._test_dataset_exists(
                dataset_name, legacy_backend, new_backend
            )
            results["tests"].append(test_result)
            if not test_result["passed"]:
                results["passed"] = False
            
            # Test 2: Data loading
            test_result = self._test_data_loading(
                dataset_name, legacy_backend, new_backend
            )
            results["tests"].append(test_result)
            if not test_result["passed"]:
                results["passed"] = False
            
            # Test 3: Metadata access
            test_result = self._test_metadata_access(
                dataset_name, legacy_backend, new_backend
            )
            results["tests"].append(test_result)
            if not test_result["passed"]:
                results["passed"] = False
            
            # Test 4: Performance comparison
            test_result = self._test_performance(
                dataset_name, legacy_backend, new_backend
            )
            results["tests"].append(test_result)
            
        finally:
            # Cleanup
            if hasattr(legacy_backend, 'close'):
                legacy_backend.close()
            if hasattr(new_backend, 'close'):
                new_backend.close()
        
        return results
    
    def _test_dataset_exists(self, dataset_name: str, 
                            legacy: IStorageBackend, 
                            new: IStorageBackend) -> Dict[str, Any]:
        """Test dataset existence check"""
        result = self.comparison_tester.compare(
            test_name=f"dataset_exists_{dataset_name}",
            old_impl=lambda: legacy.dataset_exists(dataset_name),
            new_impl=lambda: new.dataset_exists(dataset_name)
        )
        
        return {
            "test": "dataset_exists",
            "passed": result.passed,
            "performance_delta": result.performance_delta,
            "details": result.differences
        }
    
    def _test_data_loading(self, dataset_name: str,
                          legacy: IStorageBackend,
                          new: IStorageBackend) -> Dict[str, Any]:
        """Test data loading compatibility"""
        def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
            try:
                pd.testing.assert_frame_equal(
                    df1.sort_index(axis=1),
                    df2.sort_index(axis=1),
                    check_dtype=False,
                    check_index_type=False
                )
                return True
            except AssertionError:
                return False
        
        result = self.comparison_tester.compare(
            test_name=f"load_data_{dataset_name}",
            old_impl=lambda: legacy.load_data(dataset_name),
            new_impl=lambda: new.load_data(dataset_name),
            compare_func=compare_dataframes
        )
        
        return {
            "test": "load_data",
            "passed": result.passed,
            "performance_delta": result.performance_delta,
            "memory_delta": result.memory_delta,
            "details": result.differences
        }
    
    def _test_metadata_access(self, dataset_name: str,
                             legacy: IStorageBackend,
                             new: IStorageBackend) -> Dict[str, Any]:
        """Test metadata operations"""
        result = self.comparison_tester.compare(
            test_name=f"metadata_{dataset_name}",
            old_impl=lambda: legacy.get_metadata(dataset_name),
            new_impl=lambda: new.get_metadata(dataset_name)
        )
        
        return {
            "test": "metadata_access",
            "passed": result.passed,
            "performance_delta": result.performance_delta,
            "details": result.differences
        }
    
    def _test_performance(self, dataset_name: str,
                         legacy: IStorageBackend,
                         new: IStorageBackend) -> Dict[str, Any]:
        """Run performance benchmarks"""
        # Create test data
        test_data = pd.DataFrame({
            'id': range(1000),
            'value': range(1000, 2000),
            'category': ['A', 'B', 'C'] * 333 + ['A']
        })
        
        # Test write performance
        def write_test(backend: IStorageBackend):
            backend.save_data(
                f"{dataset_name}_perftest",
                test_data,
                "perftest",
                "replace"
            )
        
        write_result = self.comparison_tester.compare(
            test_name=f"write_performance_{dataset_name}",
            old_impl=lambda: write_test(legacy),
            new_impl=lambda: write_test(new)
        )
        
        # Cleanup test data
        try:
            legacy.drop_dataset(f"{dataset_name}_perftest")
            new.drop_dataset(f"{dataset_name}_perftest")
        except:
            pass
        
        return {
            "test": "performance",
            "passed": abs(write_result.performance_delta) < 20,  # Allow 20% variance
            "write_performance_delta": write_result.performance_delta,
            "write_memory_delta": write_result.memory_delta
        }
    
    def migrate_dataset(self, dataset_name: str, validate: bool = True) -> Dict[str, Any]:
        """Migrate a single dataset to new backend"""
        result = {
            "dataset": dataset_name,
            "success": False,
            "validation": None,
            "error": None,
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            # Validation phase
            if validate:
                validation = self.validate_backend_compatibility(dataset_name)
                result["validation"] = validation
                
                if not validation["passed"]:
                    result["error"] = "Validation failed"
                    return result
            
            # Migration is automatic with feature flag
            # Just need to ensure data is accessible
            
            # Verify access with new backend
            from ...core.feature_flags import feature_flags
            feature_flags.set("use_new_backend", True)
            
            backend = get_storage_backend()
            if not backend.dataset_exists(dataset_name):
                result["error"] = "Dataset not accessible with new backend"
                return result
            
            # Quick data verification
            data = backend.load_data(dataset_name)
            if data.empty:
                logger.warning(f"Dataset {dataset_name} appears empty")
            
            result["success"] = True
            result["duration"] = time.time() - start_time
            
            # Track metrics
            metrics_collector.increment("storage.migration.success")
            
        except Exception as e:
            result["error"] = str(e)
            metrics_collector.increment("storage.migration.failure")
            logger.error(f"Migration failed for {dataset_name}: {e}")
        
        self.migration_log.append(result)
        return result
    
    def migrate_all_datasets(self, 
                           batch_size: int = 5,
                           validate: bool = True) -> Dict[str, Any]:
        """Migrate all datasets in batches"""
        from ...dataset import list_datasets
        
        all_datasets = list_datasets()
        total = len(all_datasets)
        
        results = {
            "total": total,
            "successful": 0,
            "failed": 0,
            "datasets": []
        }
        
        for i in range(0, total, batch_size):
            batch = all_datasets[i:i + batch_size]
            
            for dataset_info in batch:
                dataset_name = dataset_info["name"]
                logger.info(f"Migrating dataset {dataset_name} ({i+1}/{total})")
                
                result = self.migrate_dataset(dataset_name, validate=validate)
                results["datasets"].append(result)
                
                if result["success"]:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                
                # Brief pause between datasets
                time.sleep(0.1)
        
        return results


# Create migration CLI commands
def create_migration_cli():
    """Create CLI for storage migration"""
    import typer
    from rich.console import Console
    from rich.progress import Progress
    
    app = typer.Typer()
    console = Console()
    
    @app.command()
    def validate(dataset_name: str):
        """Validate backend compatibility for a dataset"""
        migrator = StorageBackendMigrator()
        result = migrator.validate_backend_compatibility(dataset_name)
        
        if result["passed"]:
            console.print(f"[green]✓ Validation passed for {dataset_name}[/green]")
        else:
            console.print(f"[red]✗ Validation failed for {dataset_name}[/red]")
        
        # Show test results
        for test in result["tests"]:
            status = "✓" if test["passed"] else "✗"
            console.print(f"  {status} {test['test']}: {test.get('performance_delta', 'N/A'):.1f}%")
    
    @app.command()
    def migrate(
        dataset_name: Optional[str] = None,
        all: bool = False,
        validate: bool = True,
        batch_size: int = 5
    ):
        """Migrate datasets to new storage backend"""
        migrator = StorageBackendMigrator()
        
        if all:
            with Progress() as progress:
                task = progress.add_task("Migrating datasets...", total=100)
                
                results = migrator.migrate_all_datasets(
                    batch_size=batch_size,
                    validate=validate
                )
                
                progress.update(task, completed=100)
            
            console.print(f"\nMigration Results:")
            console.print(f"  Total: {results['total']}")
            console.print(f"  Successful: {results['successful']}")
            console.print(f"  Failed: {results['failed']}")
            
        elif dataset_name:
            result = migrator.migrate_dataset(dataset_name, validate=validate)
            
            if result["success"]:
                console.print(f"[green]✓ Successfully migrated {dataset_name}[/green]")
            else:
                console.print(f"[red]✗ Failed to migrate {dataset_name}: {result['error']}[/red]")
        else:
            console.print("[yellow]Specify --all or provide a dataset name[/yellow]")
    
    return app
```

#### Day 7-8: Performance Optimization

##### 2.2 Connection Pool Monitoring
```python
# Create: src/mdm/storage/monitoring.py
import threading
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

from .pooling import ConnectionPool
from ..core.metrics import metrics_collector

logger = logging.getLogger(__name__)


class PoolMonitor:
    """Monitor connection pools and collect metrics"""
    
    def __init__(self, check_interval: int = 60):
        self.pools: Dict[str, ConnectionPool] = {}
        self.check_interval = check_interval
        self._stop = False
        self._thread = None
    
    def register_pool(self, name: str, pool: ConnectionPool):
        """Register a pool for monitoring"""
        self.pools[name] = pool
        logger.info(f"Registered pool for monitoring: {name}")
    
    def start(self):
        """Start monitoring thread"""
        if self._thread is None:
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info("Pool monitor started")
    
    def stop(self):
        """Stop monitoring"""
        self._stop = True
        if self._thread:
            self._thread.join()
            self._thread = None
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop:
            try:
                for name, pool in self.pools.items():
                    stats = pool.get_stats()
                    
                    # Record metrics
                    metrics_collector.gauge(
                        f"storage.pool.{name}.total_connections",
                        stats.get("total_connections", 0)
                    )
                    metrics_collector.gauge(
                        f"storage.pool.{name}.available",
                        stats.get("available", 0)
                    )
                    metrics_collector.gauge(
                        f"storage.pool.{name}.in_use",
                        stats.get("in_use", 0)
                    )
                    
                    # Check pool health
                    self._check_pool_health(name, pool, stats)
                    
                    # Cleanup idle connections
                    pool.cleanup_idle_connections()
                
            except Exception as e:
                logger.error(f"Error in pool monitor: {e}")
            
            time.sleep(self.check_interval)
    
    def _check_pool_health(self, name: str, pool: ConnectionPool, stats: Dict[str, Any]):
        """Check pool health and alert on issues"""
        # Check if pool is exhausted
        if stats.get("in_use", 0) >= stats.get("max_size", 0) * 0.9:
            logger.warning(f"Pool {name} is near capacity: {stats}")
            metrics_collector.increment(f"storage.pool.{name}.near_capacity")
        
        # Check for connection leaks (connections in use for too long)
        # This would require tracking connection checkout times


# Global pool monitor
pool_monitor = PoolMonitor()


# Create performance benchmark suite
class StoragePerformanceBenchmark:
    """Benchmark storage backend performance"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def run_benchmarks(self, backend_type: str = None) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks"""
        from ..factory import get_storage_backend
        
        backend = get_storage_backend(backend_type)
        dataset_name = f"benchmark_{int(time.time())}"
        
        results = {
            "backend_type": backend_type or "default",
            "timestamp": datetime.now(),
            "benchmarks": {}
        }
        
        try:
            # Setup
            backend.create_dataset(dataset_name, {"type": "benchmark"})
            
            # Benchmark 1: Sequential writes
            results["benchmarks"]["sequential_writes"] = self._benchmark_sequential_writes(
                backend, dataset_name
            )
            
            # Benchmark 2: Batch writes
            results["benchmarks"]["batch_writes"] = self._benchmark_batch_writes(
                backend, dataset_name
            )
            
            # Benchmark 3: Random reads
            results["benchmarks"]["random_reads"] = self._benchmark_random_reads(
                backend, dataset_name
            )
            
            # Benchmark 4: Concurrent operations
            results["benchmarks"]["concurrent_ops"] = self._benchmark_concurrent_ops(
                backend, dataset_name
            )
            
        finally:
            # Cleanup
            try:
                backend.drop_dataset(dataset_name)
            except:
                pass
            
            if hasattr(backend, 'close'):
                backend.close()
        
        self.results.append(results)
        return results
    
    def _benchmark_sequential_writes(self, backend: IStorageBackend, 
                                   dataset_name: str) -> Dict[str, Any]:
        """Benchmark sequential write performance"""
        import pandas as pd
        
        row_counts = [100, 1000, 10000]
        results = {}
        
        for rows in row_counts:
            data = pd.DataFrame({
                'id': range(rows),
                'value': range(rows),
                'text': [f'text_{i}' for i in range(rows)]
            })
            
            start = time.perf_counter()
            backend.save_data(dataset_name, data, f"test_{rows}", "replace")
            duration = time.perf_counter() - start
            
            results[f"{rows}_rows"] = {
                "duration": duration,
                "rows_per_second": rows / duration
            }
        
        return results
    
    def _benchmark_batch_writes(self, backend: IStorageBackend,
                               dataset_name: str) -> Dict[str, Any]:
        """Benchmark batch write performance"""
        # Implementation similar to above but with batched writes
        pass
    
    def _benchmark_random_reads(self, backend: IStorageBackend,
                               dataset_name: str) -> Dict[str, Any]:
        """Benchmark random read performance"""
        # Implementation
        pass
    
    def _benchmark_concurrent_ops(self, backend: IStorageBackend,
                                 dataset_name: str) -> Dict[str, Any]:
        """Benchmark concurrent operation performance"""
        import concurrent.futures
        
        def worker(n):
            data = pd.DataFrame({'n': [n] * 100})
            backend.save_data(dataset_name, data, f"concurrent_{n}", "replace")
            return backend.load_data(dataset_name, f"concurrent_{n}")
        
        start = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(20)]
            results = [f.result() for f in futures]
        
        duration = time.perf_counter() - start
        
        return {
            "total_duration": duration,
            "operations": 20,
            "ops_per_second": 20 / duration
        }
```

### Week 10: Validation and Rollout

#### Day 9: Comprehensive Testing

##### 3.1 Create Integration Tests
```python
# Create: tests/integration/test_stateless_backends.py
import pytest
import pandas as pd
import concurrent.futures
from typing import List
import tempfile
from pathlib import Path

from mdm.storage.factory import get_storage_backend
from mdm.storage.backends.stateless_sqlite import StatelessSQLiteBackend
from mdm.storage.backends.stateless_duckdb import StatelessDuckDBBackend
from mdm.storage.monitoring import StoragePerformanceBenchmark


class TestStatelessBackends:
    @pytest.fixture
    def temp_mdm_home(self, tmp_path):
        """Create temporary MDM home"""
        import os
        original = os.environ.get("MDM_PATHS_HOME")
        os.environ["MDM_PATHS_HOME"] = str(tmp_path)
        yield tmp_path
        if original:
            os.environ["MDM_PATHS_HOME"] = original
        else:
            os.environ.pop("MDM_PATHS_HOME", None)
    
    @pytest.mark.parametrize("backend_class", [
        StatelessSQLiteBackend,
        StatelessDuckDBBackend
    ])
    def test_concurrent_operations(self, backend_class, temp_mdm_home):
        """Test concurrent operations with stateless backends"""
        backend = backend_class()
        dataset_name = "test_concurrent"
        
        # Create dataset
        backend.create_dataset(dataset_name, {})
        
        # Concurrent writes
        def write_data(n):
            data = pd.DataFrame({
                'thread_id': [n] * 100,
                'value': range(100)
            })
            backend.save_data(dataset_name, data, f"table_{n}", "replace")
            return n
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_data, i) for i in range(20)]
            results = [f.result() for f in futures]
        
        assert len(results) == 20
        
        # Verify all data was written
        for i in range(20):
            data = backend.load_data(dataset_name, f"table_{i}")
            assert len(data) == 100
            assert data['thread_id'].iloc[0] == i
        
        # Cleanup
        backend.drop_dataset(dataset_name)
        backend.close()
    
    def test_connection_pool_limits(self, temp_mdm_home):
        """Test connection pool respects limits"""
        backend = StatelessSQLiteBackend()
        dataset_name = "test_pool_limits"
        
        backend.create_dataset(dataset_name, {})
        
        # Get pool stats
        stats = backend.pool.get_stats()
        max_size = stats["max_size"]
        
        # Try to exceed pool size
        connections = []
        try:
            for i in range(max_size + 5):
                cm = backend.pool.get_connection(dataset_name)
                conn = cm.__enter__()
                connections.append((cm, conn))
        except RuntimeError as e:
            assert "exhausted" in str(e)
        
        # Release connections
        for cm, conn in connections:
            cm.__exit__(None, None, None)
        
        backend.drop_dataset(dataset_name)
        backend.close()
    
    def test_performance_comparison(self, temp_mdm_home):
        """Compare performance between legacy and stateless backends"""
        benchmark = StoragePerformanceBenchmark()
        
        # Run benchmarks for both implementations
        from mdm.core.feature_flags import feature_flags
        
        # Legacy benchmark
        feature_flags.set("use_new_backend", False)
        legacy_results = benchmark.run_benchmarks("sqlite")
        
        # Stateless benchmark
        feature_flags.set("use_new_backend", True)
        stateless_results = benchmark.run_benchmarks("sqlite")
        
        # Compare results
        legacy_seq = legacy_results["benchmarks"]["sequential_writes"]["1000_rows"]
        stateless_seq = stateless_results["benchmarks"]["sequential_writes"]["1000_rows"]
        
        # Stateless should not be significantly slower
        performance_ratio = stateless_seq["duration"] / legacy_seq["duration"]
        assert performance_ratio < 1.1, f"Stateless is {performance_ratio:.2f}x slower"
    
    def test_data_integrity(self, temp_mdm_home):
        """Test data integrity with concurrent access"""
        backend = StatelessSQLiteBackend()
        dataset_name = "test_integrity"
        
        backend.create_dataset(dataset_name, {})
        
        # Initial data
        initial_data = pd.DataFrame({
            'id': range(1000),
            'value': range(1000)
        })
        backend.save_data(dataset_name, initial_data)
        
        # Concurrent reads and writes
        def reader():
            for _ in range(10):
                data = backend.load_data(dataset_name)
                assert len(data) >= 1000
        
        def writer(n):
            new_data = pd.DataFrame({
                'id': [1000 + n],
                'value': [1000 + n]
            })
            backend.save_data(dataset_name, new_data, if_exists="append")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # Mix of readers and writers
            futures = []
            for i in range(10):
                futures.append(executor.submit(reader))
                futures.append(executor.submit(writer, i))
            
            # Wait for completion
            for f in futures:
                f.result()
        
        # Verify final state
        final_data = backend.load_data(dataset_name)
        assert len(final_data) == 1010
        
        backend.drop_dataset(dataset_name)
        backend.close()
```

#### Day 10: Rollout Plan

##### 3.2 Create Rollout Documentation
```markdown
# Create: docs/storage_backend_rollout.md

# Storage Backend Migration Rollout Plan

## Overview

This document outlines the rollout plan for migrating from singleton storage backends to stateless implementations with connection pooling.

## Rollout Phases

### Phase 1: Internal Testing (Week 1)
- Enable for development team only
- Monitor performance and stability
- Fix any issues discovered

### Phase 2: Beta Users (Week 2)
- Enable for 10% of users
- Monitor metrics closely
- Gather feedback

### Phase 3: Gradual Rollout (Weeks 3-4)
- Increase to 25%, 50%, 75% progressively
- Monitor for performance regression
- Be ready to rollback if needed

### Phase 4: Full Rollout (Week 5)
- Enable for all users
- Keep legacy code for 1 month
- Remove legacy code after stability confirmed

## Monitoring Checklist

### Performance Metrics
- [ ] Connection pool utilization
- [ ] Query execution time
- [ ] Memory usage
- [ ] Concurrent operation throughput

### Error Metrics
- [ ] Connection failures
- [ ] Pool exhaustion events
- [ ] Timeout errors
- [ ] Data integrity issues

### User Impact Metrics
- [ ] Dataset registration time
- [ ] Data loading latency
- [ ] Export performance
- [ ] API response times

## Rollback Procedures

### Immediate Rollback
```bash
# Set feature flag
mdm feature-flags set use_new_backend false
```

### Connection Pool Issues
```python
# Increase pool size temporarily
from mdm.config import config_manager
config_manager.set("performance.max_workers", 20)
```

### Performance Issues
1. Check pool statistics
2. Identify bottlenecks
3. Adjust pool parameters
4. Consider increasing resources

## Success Criteria

- No increase in error rates
- Performance within 5% of legacy
- Connection pool utilization < 80%
- Zero data integrity issues
- Positive user feedback

## Communication Plan

### For Users
- Announcement of migration benefits
- No action required
- Performance improvements expected

### For Developers
- New API documentation
- Migration guide for custom code
- Best practices for connection usage

## Post-Migration Tasks

1. Remove legacy backend code
2. Update all documentation
3. Performance optimization based on metrics
4. Plan for future enhancements
```

## Validation Checklist

### Week 8 Complete
- [ ] Connection pool infrastructure implemented
- [ ] All three backend pools tested
- [ ] Pool monitoring active
- [ ] Performance benchmarks established

### Week 9 Complete
- [ ] Stateless backends implemented
- [ ] Factory pattern working
- [ ] Migration coordinator tested
- [ ] A/B testing configured

### Week 10 Complete
- [ ] Integration tests passing
- [ ] Performance validated
- [ ] Rollout plan approved
- [ ] Documentation updated

## Success Criteria

- **Zero data loss** during migration
- **Performance improvement** for concurrent operations
- **Connection pool efficiency** > 80%
- **All backends migrated** successfully
- **Rollback tested** and working

## Next Steps

With storage backends migrated, proceed to [06-feature-engineering-migration.md](06-feature-engineering-migration.md).

## Notes

- Monitor connection pool metrics continuously
- Be prepared for quick rollback if issues arise
- Document any performance tuning discoveries
- Consider backend-specific optimizations