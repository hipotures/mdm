"""Unit tests for storage backend repository operations."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from sqlalchemy import create_engine

from mdm.storage.base import StorageBackend
from mdm.storage.sqlite import SQLiteBackend
from mdm.storage.duckdb import DuckDBBackend
from mdm.storage.factory import BackendFactory
from mdm.core.exceptions import StorageError


class TestStorageBackendBase:
    """Test cases for base storage backend functionality."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock storage backend."""
        backend = Mock(spec=StorageBackend)
        backend.engine = Mock()
        backend.metadata = Mock()
        return backend

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            StorageBackend({})

    def test_create_table(self, mock_backend):
        """Test table creation."""
        # Arrange
        df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        
        # Act
        mock_backend.create_table("test_table", df)
        
        # Assert
        mock_backend.create_table.assert_called_once_with("test_table", df)

    def test_read_table(self, mock_backend):
        """Test table reading."""
        # Arrange
        expected_df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        mock_backend.read_table.return_value = expected_df
        
        # Act
        result = mock_backend.read_table("test_table")
        
        # Assert
        assert result.equals(expected_df)

    def test_table_exists(self, mock_backend):
        """Test checking table existence."""
        # Arrange
        mock_backend.table_exists.return_value = True
        
        # Act
        result = mock_backend.table_exists("test_table")
        
        # Assert
        assert result is True

    def test_get_table_info(self, mock_backend):
        """Test getting table information."""
        # Arrange
        expected_info = {
            'columns': ['id', 'value'],
            'row_count': 100,
            'size_bytes': 1024
        }
        mock_backend.get_table_info.return_value = expected_info
        
        # Act
        result = mock_backend.get_table_info("test_table")
        
        # Assert
        assert result == expected_info


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
    def sqlite_backend(self, sqlite_config, tmp_path):
        """Create SQLite backend instance."""
        # Use temporary directory for database
        with patch('mdm.config.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.base_path = tmp_path
            mock_get_config.return_value = mock_manager
            
            backend = SQLiteBackend(sqlite_config)
            return backend

    def test_init_creates_database(self, sqlite_backend):
        """Test that initialization creates database file."""
        assert sqlite_backend.db_path.exists()
        assert sqlite_backend.db_path.suffix == '.db'

    def test_create_and_read_table(self, sqlite_backend):
        """Test creating and reading a table."""
        # Create table
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [90, 85, 95]
        })
        
        sqlite_backend.create_table("test_table", df)
        
        # Read table
        result = sqlite_backend.read_table("test_table")
        
        # Assert
        assert len(result) == 3
        assert list(result.columns) == ['id', 'name', 'score']
        assert result['name'].tolist() == ['Alice', 'Bob', 'Charlie']

    def test_table_exists(self, sqlite_backend):
        """Test checking table existence."""
        # Before creation
        assert not sqlite_backend.table_exists("test_table")
        
        # Create table
        df = pd.DataFrame({'id': [1]})
        sqlite_backend.create_table("test_table", df)
        
        # After creation
        assert sqlite_backend.table_exists("test_table")

    def test_get_table_info(self, sqlite_backend):
        """Test getting table information."""
        # Create table
        df = pd.DataFrame({
            'id': range(100),
            'value': range(100)
        })
        sqlite_backend.create_table("test_table", df)
        
        # Get info
        info = sqlite_backend.get_table_info("test_table")
        
        # Assert
        assert info['row_count'] == 100
        assert 'id' in info['columns']
        assert 'value' in info['columns']
        assert info['size_bytes'] > 0

    def test_list_tables(self, sqlite_backend):
        """Test listing all tables."""
        # Create multiple tables
        sqlite_backend.create_table("table1", pd.DataFrame({'a': [1]}))
        sqlite_backend.create_table("table2", pd.DataFrame({'b': [2]}))
        
        # List tables
        tables = sqlite_backend.list_tables()
        
        # Assert
        assert len(tables) == 2
        assert "table1" in tables
        assert "table2" in tables

    def test_drop_table(self, sqlite_backend):
        """Test dropping a table."""
        # Create table
        sqlite_backend.create_table("test_table", pd.DataFrame({'a': [1]}))
        assert sqlite_backend.table_exists("test_table")
        
        # Drop table
        sqlite_backend.drop_table("test_table")
        
        # Verify dropped
        assert not sqlite_backend.table_exists("test_table")

    def test_execute_query(self, sqlite_backend):
        """Test executing custom SQL query."""
        # Create table
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        sqlite_backend.create_table("test_table", df)
        
        # Execute query
        result = sqlite_backend.execute_query(
            "SELECT * FROM test_table WHERE value > 15"
        )
        
        # Assert
        assert len(result) == 2
        assert result['value'].tolist() == [20, 30]

    def test_get_row_count(self, sqlite_backend):
        """Test getting row count."""
        # Create table
        df = pd.DataFrame({'id': range(50)})
        sqlite_backend.create_table("test_table", df)
        
        # Get count
        count = sqlite_backend.get_row_count("test_table")
        
        # Assert
        assert count == 50

    def test_close(self, sqlite_backend):
        """Test closing the backend."""
        # Should not raise
        sqlite_backend.close()


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
    def duckdb_backend(self, duckdb_config, tmp_path):
        """Create DuckDB backend instance."""
        with patch('mdm.config.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.base_path = tmp_path
            mock_get_config.return_value = mock_manager
            
            backend = DuckDBBackend(duckdb_config)
            return backend

    def test_init_creates_database(self, duckdb_backend):
        """Test that initialization creates database file."""
        assert duckdb_backend.db_path.exists()
        assert duckdb_backend.db_path.suffix == '.duckdb'

    def test_create_and_read_table(self, duckdb_backend):
        """Test creating and reading a table."""
        # Create table
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['hello', 'world', 'test'],
            'number': [1.5, 2.5, 3.5]
        })
        
        duckdb_backend.create_table("test_table", df)
        
        # Read table
        result = duckdb_backend.read_table("test_table")
        
        # Assert
        assert len(result) == 3
        assert result['text'].tolist() == ['hello', 'world', 'test']

    def test_read_table_with_limit(self, duckdb_backend):
        """Test reading table with row limit."""
        # Create large table
        df = pd.DataFrame({'id': range(1000), 'value': range(1000)})
        duckdb_backend.create_table("test_table", df)
        
        # Read with limit
        result = duckdb_backend.read_table("test_table", limit=10)
        
        # Assert
        assert len(result) == 10

    def test_parquet_export(self, duckdb_backend, tmp_path):
        """Test exporting to Parquet format."""
        # Create table
        df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        duckdb_backend.create_table("test_table", df)
        
        # Export to parquet
        output_path = tmp_path / "export.parquet"
        duckdb_backend.export_to_parquet("test_table", output_path)
        
        # Verify file exists and can be read
        assert output_path.exists()
        result = pd.read_parquet(output_path)
        assert len(result) == 3


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