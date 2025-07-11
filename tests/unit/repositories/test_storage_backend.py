"""Unit tests for storage backend repository operations."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

from mdm.storage.base import StorageBackend
from mdm.storage.backends.stateless_sqlite import StatelessSQLiteBackend as SQLiteBackend
from mdm.storage.backends.stateless_duckdb import StatelessDuckDBBackend as DuckDBBackend
from mdm.storage.factory import BackendFactory
from mdm.core.exceptions import StorageError, BackendError


class StorageBackendTestHelper:
    """Helper for testing storage backends with engine management."""
    
    def __init__(self, backend, dataset_name="test_dataset", base_path=None):
        self.backend = backend
        self.dataset_name = dataset_name
        self.base_path = base_path or Path("/tmp/test")
        self._setup_engine()
    
    def _setup_engine(self):
        """Set up database and engine."""
        self.db_path = self.backend.get_database_path(self.dataset_name, self.base_path)
        if not self.backend.database_exists(self.db_path):
            self.backend.create_database(self.db_path)
        self.engine = self.backend.get_engine(self.db_path)
    
    def create_table(self, table_name, df):
        """Create table from dataframe."""
        return self.backend.create_table_from_dataframe(df, table_name, self.engine)
    
    def read_table(self, table_name, limit=None):
        """Read table to dataframe."""
        return self.backend.read_table_to_dataframe(table_name, self.engine, limit)
    
    def table_exists(self, table_name):
        """Check if table exists."""
        return self.backend.table_exists(self.engine, table_name)
    
    def list_tables(self):
        """List all table names."""
        return self.backend.get_table_names(self.engine)
    
    def get_table_info(self, table_name):
        """Get table information."""
        return self.backend.get_table_info(table_name, self.engine)
    
    def execute_query(self, query):
        """Execute SQL query."""
        return self.backend.execute_query(query, self.engine)
    
    def get_row_count(self, table_name):
        """Get row count for a table."""
        df = self.read_table(table_name)
        return len(df)
    
    def close(self):
        """Close connections."""
        self.backend.close_connections()


class TestStorageBackendBase:
    """Test cases for base storage backend functionality."""

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            StorageBackend({})

    def test_backend_requires_implementation(self):
        """Test that subclasses must implement required methods."""
        # Create a partial implementation
        class IncompleteBackend(StorageBackend):
            @property
            def backend_type(self):
                return "incomplete"
            
            # Missing other required methods
        
        # Should fail to instantiate because abstract methods not implemented
        with pytest.raises(TypeError):
            IncompleteBackend({})


class TestSQLiteBackend:
    """Test cases for SQLite backend."""

    @pytest.fixture
    def sqlite_config(self):
        """SQLite configuration."""
        return {
            'type': 'sqlite',
            'dataset_name': 'test_dataset'
        }

    @pytest.fixture
    def backend_helper(self, sqlite_config, tmp_path):
        """Create SQLite backend with helper."""
        # Add SQLite-specific configuration
        sqlite_config.update({
            'synchronous': 'NORMAL',
            'journal_mode': 'WAL',
            'cache_size': -64000,
            'temp_store': 'MEMORY',
            'mmap_size': 268435456,
            'sqlalchemy': {
                'echo': False,
                'pool_size': 5,
                'max_overflow': 10
            }
        })
        
        backend = SQLiteBackend(sqlite_config)
        return StorageBackendTestHelper(backend, base_path=tmp_path)

    def test_init_creates_database(self, backend_helper):
        """Test that initialization creates database file."""
        # The helper should have created the database
        assert Path(backend_helper.db_path).exists()
        assert Path(backend_helper.db_path).suffix == '.sqlite'

    def test_create_and_read_table(self, backend_helper):
        """Test creating and reading a table."""
        # Create table
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [90, 85, 95]
        })
        
        backend_helper.create_table("test_table", df)
        
        # Read table
        result = backend_helper.read_table("test_table")
        
        # Assert
        assert len(result) == 3
        assert list(result.columns) == ['id', 'name', 'score']
        assert result['name'].tolist() == ['Alice', 'Bob', 'Charlie']

    def test_table_exists(self, backend_helper):
        """Test checking table existence."""
        # Before creation
        assert not backend_helper.table_exists("test_table")
        
        # Create table
        df = pd.DataFrame({'id': [1]})
        backend_helper.create_table("test_table", df)
        
        # After creation
        assert backend_helper.table_exists("test_table")

    def test_get_table_info(self, backend_helper):
        """Test getting table information."""
        # Create table
        df = pd.DataFrame({
            'id': range(100),
            'value': range(100)
        })
        backend_helper.create_table("test_table", df)
        
        # Get info
        info = backend_helper.get_table_info("test_table")
        
        # Assert
        assert info['row_count'] == 100
        assert 'id' in [col['name'] for col in info['columns']]
        assert 'value' in [col['name'] for col in info['columns']]

    def test_list_tables(self, backend_helper):
        """Test listing all tables."""
        # Create multiple tables
        backend_helper.create_table("table1", pd.DataFrame({'a': [1]}))
        backend_helper.create_table("table2", pd.DataFrame({'b': [2]}))
        
        # List tables
        tables = backend_helper.list_tables()
        
        # Assert
        assert len(tables) == 2
        assert "table1" in tables
        assert "table2" in tables

    def test_execute_query(self, backend_helper):
        """Test executing custom SQL query."""
        # Create table
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        backend_helper.create_table("test_table", df)
        
        # Use the query method which returns DataFrame
        result = backend_helper.backend.query(
            "SELECT * FROM test_table WHERE value > 15"
        )
        
        # Assert
        assert len(result) == 2
        assert sorted(result['value'].tolist()) == [20, 30]

    def test_get_row_count(self, backend_helper):
        """Test getting row count."""
        # Create table
        df = pd.DataFrame({'id': range(50)})
        backend_helper.create_table("test_table", df)
        
        # Get count
        count = backend_helper.get_row_count("test_table")
        
        # Assert
        assert count == 50

    def test_close(self, backend_helper):
        """Test closing the backend."""
        # Should not raise
        backend_helper.close()


class TestDuckDBBackend:
    """Test cases for DuckDB backend."""

    @pytest.fixture
    def duckdb_config(self):
        """DuckDB configuration."""
        return {
            'type': 'duckdb',
            'dataset_name': 'test_dataset'
        }

    @pytest.fixture
    def backend_helper(self, duckdb_config, tmp_path):
        """Create DuckDB backend with helper."""
        # Add DuckDB-specific configuration
        duckdb_config.update({
            'threads': 4,
            'memory_limit': '1GB',
            'access_mode': 'READ_WRITE',
            'temp_directory': str(tmp_path / 'temp'),
            'enable_object_cache': True,
            'sqlalchemy': {
                'echo': False,
                'pool_size': 5,
                'max_overflow': 10
            }
        })
        
        backend = DuckDBBackend(duckdb_config)
        return StorageBackendTestHelper(backend, base_path=tmp_path)

    def test_init_creates_database(self, backend_helper):
        """Test that initialization creates database file."""
        # The helper should have created the database
        assert Path(backend_helper.db_path).exists()
        assert Path(backend_helper.db_path).suffix == '.duckdb'

    def test_create_and_read_table(self, backend_helper):
        """Test creating and reading a table."""
        # Create table
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['hello', 'world', 'test'],
            'number': [1.5, 2.5, 3.5]
        })
        
        backend_helper.create_table("test_table", df)
        
        # Read table
        result = backend_helper.read_table("test_table")
        
        # Assert
        assert len(result) == 3
        assert result['text'].tolist() == ['hello', 'world', 'test']

    def test_read_table_with_limit(self, backend_helper):
        """Test reading table with row limit."""
        # Create large table
        df = pd.DataFrame({'id': range(1000), 'value': range(1000)})
        backend_helper.create_table("test_table", df)
        
        # Read with limit
        result = backend_helper.read_table("test_table", limit=10)
        
        # Assert
        assert len(result) == 10


class TestBackendFactory:
    """Test cases for backend factory."""

    def test_create_sqlite_backend(self):
        """Test creating SQLite backend."""
        mock_backend = Mock(spec=StorageBackend)
        with patch.dict('mdm.storage.factory.BackendFactory._backends', {'sqlite': Mock(return_value=mock_backend)}):
            config = {'type': 'sqlite'}
            result = BackendFactory.create('sqlite', config)
            assert result == mock_backend

    def test_create_duckdb_backend(self):
        """Test creating DuckDB backend."""
        mock_backend = Mock(spec=StorageBackend)
        with patch.dict('mdm.storage.factory.BackendFactory._backends', {'duckdb': Mock(return_value=mock_backend)}):
            config = {'type': 'duckdb'}
            result = BackendFactory.create('duckdb', config)
            assert result == mock_backend

    def test_create_postgresql_backend(self):
        """Test creating PostgreSQL backend."""
        mock_backend = Mock(spec=StorageBackend)
        with patch.dict('mdm.storage.factory.BackendFactory._backends', {'postgresql': Mock(return_value=mock_backend)}):
            config = {'type': 'postgresql'}
            result = BackendFactory.create('postgresql', config)
            assert result == mock_backend

    def test_create_invalid_backend(self):
        """Test error with invalid backend type."""
        from mdm.core.exceptions import BackendError
        with pytest.raises(BackendError, match="Unsupported backend type: invalid"):
            BackendFactory.create('invalid', {})

    def test_get_backend_class(self):
        """Test getting backend class - method doesn't exist, test backends dict instead."""
        from mdm.storage.sqlite import SQLiteBackend
        from mdm.storage.duckdb import DuckDBBackend
        
        # Access the private _backends dict for testing
        assert BackendFactory._backends['sqlite'] == SQLiteBackend
        assert BackendFactory._backends['duckdb'] == DuckDBBackend

    def test_get_supported_backends(self):
        """Test getting list of supported backends."""
        backends = BackendFactory.get_supported_backends()
        assert 'sqlite' in backends
        assert 'duckdb' in backends
        assert 'postgresql' in backends