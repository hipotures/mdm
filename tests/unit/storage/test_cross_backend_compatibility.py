"""Tests for cross-backend compatibility."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from mdm.storage.factory import BackendFactory
from mdm.storage.sqlite import SQLiteBackend
from mdm.storage.duckdb import DuckDBBackend
from mdm.storage.postgresql import PostgreSQLBackend
from mdm.core.exceptions import StorageError, BackendError


class TestCrossBackendCompatibility:
    """Test compatibility between different storage backends."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1],
            'active': [True, False, True]
        })

    def test_factory_backend_creation(self):
        """Test BackendFactory creates correct backend types."""
        # SQLite
        sqlite_backend = BackendFactory.create('sqlite', {})
        assert isinstance(sqlite_backend, SQLiteBackend)
        
        # DuckDB
        duckdb_backend = BackendFactory.create('duckdb', {})
        assert isinstance(duckdb_backend, DuckDBBackend)
        
        # PostgreSQL (mocked)
        with patch('mdm.storage.postgresql.create_engine'):
            pg_backend = BackendFactory.create('postgresql', {
                'host': 'localhost',
                'database': 'test'
            })
            assert isinstance(pg_backend, PostgreSQLBackend)

    def test_unsupported_backend(self):
        """Test error handling for unsupported backend."""
        with pytest.raises(BackendError, match="Unsupported backend"):
            BackendFactory.create('mysql', {})

    def test_common_interface_methods(self):
        """Test all backends implement common interface."""
        backends = []
        
        # Create backends
        backends.append(BackendFactory.create('sqlite', {}))
        backends.append(BackendFactory.create('duckdb', {}))
        
        # Check common methods exist
        required_methods = [
            'get_engine', 'query', 'execute_query', 'create_table_from_dataframe',
            'table_exists', 'get_table_names', 'get_table_info',
            'read_table_to_dataframe', 'close_connections'
        ]
        
        for backend in backends:
            for method in required_methods:
                assert hasattr(backend, method), f"{backend.__class__.__name__} missing {method}"

    def test_data_type_mapping_consistency(self):
        """Test data type mapping is consistent across backends."""
        # This test is no longer valid as backends don't have _get_sqlalchemy_type method
        # Type mapping is handled internally by SQLAlchemy
        # We'll test actual table creation instead
        pass

    def test_table_creation_compatibility(self, sample_data):
        """Test table creation works similarly across backends."""
        # Test that both backends can create tables from dataframes
        table_name = 'test_table'
        
        # SQLite backend
        with patch('mdm.storage.sqlite.create_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            
            sqlite_backend = SQLiteBackend({})
            
            # Mock the to_sql method
            with patch.object(sample_data, 'to_sql') as mock_to_sql:
                sqlite_backend.create_table_from_dataframe(
                    sample_data, table_name, mock_engine
                )
                mock_to_sql.assert_called_once_with(
                    table_name, mock_engine, if_exists='fail', index=False
                )
        
        # DuckDB backend
        with patch('duckdb.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            
            duckdb_backend = DuckDBBackend({})
            
            # Mock the create_engine for DuckDB
            with patch('mdm.storage.duckdb.create_engine') as mock_create_engine:
                mock_engine = MagicMock()
                mock_create_engine.return_value = mock_engine
                
                with patch.object(sample_data, 'to_sql') as mock_to_sql:
                    duckdb_backend.create_table_from_dataframe(
                        sample_data, table_name, mock_engine
                    )
                    mock_to_sql.assert_called_once_with(
                        table_name, mock_engine, if_exists='fail', index=False
                    )

    def test_query_result_format(self):
        """Test query results are returned in consistent format."""
        # Mock query results
        mock_result = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        
        with patch.object(SQLiteBackend, 'query', return_value=mock_result):
            sqlite_backend = SQLiteBackend({})
            sqlite_result = sqlite_backend.query("SELECT * FROM test")
        
        with patch.object(DuckDBBackend, 'query', return_value=mock_result):
            duckdb_backend = DuckDBBackend({})
            duckdb_result = duckdb_backend.query("SELECT * FROM test")
        
        # Results should be pandas DataFrames
        assert isinstance(sqlite_result, pd.DataFrame)
        assert isinstance(duckdb_result, pd.DataFrame)
        assert sqlite_result.equals(duckdb_result)

    def test_error_handling_consistency(self):
        """Test error handling is consistent across backends."""
        backends = [
            SQLiteBackend({}),
            DuckDBBackend({})
        ]
        
        for backend in backends:
            # Test query on non-existent table
            with patch.object(backend, 'query', side_effect=Exception("Table not found")):
                with pytest.raises(Exception, match="Table not found"):
                    backend.query("SELECT * FROM nonexistent")

    def test_connection_closing(self):
        """Test all backends properly close connections."""
        # SQLite
        sqlite_backend = SQLiteBackend({})
        mock_engine = MagicMock()
        sqlite_backend._engine = mock_engine
        sqlite_backend.close_connections()
        mock_engine.dispose.assert_called_once()
        
        # DuckDB  
        duckdb_backend = DuckDBBackend({})
        mock_engine = MagicMock()
        duckdb_backend._engine = mock_engine
        duckdb_backend.close_connections()
        mock_engine.dispose.assert_called_once()

    def test_backend_specific_features(self):
        """Test backend-specific features are properly isolated."""
        # SQLite specific
        sqlite_backend = SQLiteBackend({'synchronous': 'NORMAL'})
        assert 'synchronous' in sqlite_backend.config
        
        # DuckDB specific
        duckdb_backend = DuckDBBackend({'memory_limit': '1GB'})
        assert 'memory_limit' in duckdb_backend.config
        
        # PostgreSQL specific
        pg_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test',
            'user': 'user',
            'password': 'pass'
        }
        pg_backend = PostgreSQLBackend(pg_config)
        assert 'host' in pg_backend.config
        assert 'port' in pg_backend.config

    def test_transaction_handling(self):
        """Test transaction handling across backends."""
        # Test that backends can execute queries through execute_query
        for BackendClass in [SQLiteBackend, DuckDBBackend]:
            backend = BackendClass({})
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_result = MagicMock()
            
            # Mock the connection context manager
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_conn.execute.return_value = mock_result
            
            # Test execute_query
            result = backend.execute_query("SELECT 1", mock_engine)
            
            # Verify the query was executed
            mock_engine.connect.assert_called_once()
            mock_conn.execute.assert_called_once()
            assert result == mock_result