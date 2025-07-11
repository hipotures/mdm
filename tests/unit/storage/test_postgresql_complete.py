"""Comprehensive unit tests for PostgreSQL storage backend.

NOTE: These tests are for the legacy backend API. PostgreSQL backend has not been
migrated to the new stateless architecture yet. Tests are marked as skip.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
import psycopg2

from mdm.core.exceptions import StorageError
from mdm.storage.postgresql import PostgreSQLBackend


@pytest.mark.skip(reason="PostgreSQL backend not yet migrated to stateless architecture")
class TestPostgreSQLBackendComplete:
    """Comprehensive test coverage for PostgreSQL backend."""

    @pytest.fixture
    def config(self):
        """Create backend configuration."""
        return {
            "postgresql": {
                "host": "localhost",
                "port": 5432,
                "user": "test_user",
                "password": "test_pass",
                "database_prefix": "mdm_",
                "pool_size": 10,
                "sslmode": "prefer"
            },
            "sqlalchemy": {
                "echo": False,
                "max_overflow": 20,
                "pool_timeout": 30,
                "pool_recycle": 3600
            }
        }

    @pytest.fixture
    def mock_engine(self):
        """Create mock SQLAlchemy engine."""
        engine = Mock()
        engine.dispose = Mock()
        engine.url = "postgresql://test_user:***@localhost:5432/test_db"
        engine.connect = Mock()
        return engine

    @pytest.fixture
    def mock_connection(self):
        """Create mock database connection."""
        conn = Mock()
        conn.execute = Mock()
        conn.commit = Mock()
        conn.rollback = Mock()
        conn.close = Mock()
        conn.__enter__ = Mock(return_value=conn)
        conn.__exit__ = Mock(return_value=None)
        return conn

    @pytest.fixture
    def backend(self, config):
        """Create PostgreSQL backend instance."""
        return PostgreSQLBackend(config)

    def test_init_success(self, config):
        """Test successful initialization."""
        backend = PostgreSQLBackend(config)
        assert backend.config == config
        assert backend.backend_type == "postgresql"

    def test_init_missing_user(self):
        """Test initialization without user."""
        config = {"postgresql": {"host": "localhost"}}
        backend = PostgreSQLBackend(config)
        
        # Error should happen when building connection string
        with pytest.raises(StorageError, match="PostgreSQL user not configured"):
            backend._build_connection_string()

    def test_build_connection_string_basic(self, backend):
        """Test building basic connection string."""
        conn_str = backend._build_connection_string("test_db")
        
        assert conn_str == "postgresql://test_user:test_pass@localhost:5432/test_db?sslmode=prefer"

    def test_build_connection_string_admin(self, backend):
        """Test building admin connection string."""
        conn_str = backend._build_connection_string(admin=True)
        
        assert conn_str == "postgresql://test_user:test_pass@localhost:5432/postgres?sslmode=prefer"

    def test_build_connection_string_no_password(self, config):
        """Test building connection string without password."""
        config["postgresql"].pop("password")
        backend = PostgreSQLBackend(config)
        
        conn_str = backend._build_connection_string("test_db")
        assert conn_str == "postgresql://test_user@localhost:5432/test_db?sslmode=prefer"

    def test_build_connection_string_with_ssl_params(self, config):
        """Test building connection string with SSL parameters."""
        config["postgresql"]["sslcert"] = "/path/to/cert"
        config["postgresql"]["sslkey"] = "/path/to/key"
        backend = PostgreSQLBackend(config)
        
        conn_str = backend._build_connection_string("test_db")
        assert "sslmode=prefer" in conn_str
        assert "sslcert=/path/to/cert" in conn_str
        assert "sslkey=/path/to/key" in conn_str

    def test_create_engine_with_connection_string(self, backend, mock_engine):
        """Test creating engine with connection string."""
        with patch('mdm.storage.postgresql.create_engine', return_value=mock_engine) as mock_create:
            engine = backend.create_engine("postgresql://user:pass@host/db")
            
            assert engine == mock_engine
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args[0][0] == "postgresql://user:pass@host/db"

    def test_create_engine_with_database_name(self, backend, mock_engine):
        """Test creating engine with database name."""
        with patch('mdm.storage.postgresql.create_engine', return_value=mock_engine) as mock_create:
            engine = backend.create_engine("test_db")
            
            assert engine == mock_engine
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert "test_db" in call_args[0][0]
            
            # Check engine options
            call_kwargs = call_args[1]
            assert call_kwargs['pool_size'] == 10
            assert call_kwargs['max_overflow'] == 20
            assert call_kwargs['pool_timeout'] == 30
            assert call_kwargs['pool_recycle'] == 3600

    def test_initialize_database(self, backend, mock_engine, mock_connection):
        """Test database initialization."""
        mock_engine.connect.return_value = mock_connection
        
        with patch('mdm.models.base.Base.metadata') as mock_metadata:
            backend.initialize_database(mock_engine)
            
            # Check tables were created
            mock_metadata.create_all.assert_called_once_with(mock_engine)
            
            # Check extension was created
            mock_connection.execute.assert_called_once()
            call_args = mock_connection.execute.call_args[0]
            assert 'uuid-ossp' in str(call_args[0])
            mock_connection.commit.assert_called_once()

    def test_get_database_path(self, backend):
        """Test getting database path."""
        path = backend.get_database_path("MyDataset", Path("/some/path"))
        
        assert path == "mdm_mydataset"  # Lowercase with prefix

    def test_get_database_path_custom_prefix(self, config):
        """Test getting database path with custom prefix."""
        config["postgresql"]["database_prefix"] = "custom_"
        backend = PostgreSQLBackend(config)
        
        path = backend.get_database_path("TestData", Path("/ignored"))
        assert path == "custom_testdata"

    def test_database_exists_true(self, backend, mock_engine, mock_connection):
        """Test checking if database exists."""
        with patch('mdm.storage.postgresql.create_engine', return_value=mock_engine):
            mock_engine.connect.return_value = mock_connection
            
            # Mock query result
            result = Mock()
            result.scalar.return_value = 1
            mock_connection.execute.return_value = result
            
            exists = backend.database_exists("test_db")
            
            assert exists is True
            mock_connection.execute.assert_called_once()
            call_args = mock_connection.execute.call_args
            assert "pg_database" in str(call_args[0][0])

    def test_database_exists_false(self, backend, mock_engine, mock_connection):
        """Test checking if database doesn't exist."""
        with patch('mdm.storage.postgresql.create_engine', return_value=mock_engine):
            mock_engine.connect.return_value = mock_connection
            
            # Mock query result
            result = Mock()
            result.scalar.return_value = None
            mock_connection.execute.return_value = result
            
            exists = backend.database_exists("nonexistent_db")
            
            assert exists is False

    def test_database_exists_with_connection_string(self, backend, mock_engine, mock_connection):
        """Test checking database exists with connection string."""
        with patch('mdm.storage.postgresql.create_engine', return_value=mock_engine):
            mock_engine.connect.return_value = mock_connection
            
            result = Mock()
            result.scalar.return_value = 1
            mock_connection.execute.return_value = result
            
            exists = backend.database_exists("postgresql://user:pass@host/mydb")
            
            assert exists is True
            # Check that "mydb" was extracted
            call_args = mock_connection.execute.call_args
            assert "mydb" in str(call_args[0][1])

    def test_database_exists_error(self, backend):
        """Test database existence check error."""
        with patch('mdm.storage.postgresql.create_engine', side_effect=Exception("Connection failed")):
            with pytest.raises(StorageError, match="Failed to check database existence"):
                backend.database_exists("test_db")

    def test_create_database_success(self, backend):
        """Test successful database creation."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch('psycopg2.connect', return_value=mock_conn):
            backend.create_database("test_dataset")
            
            # Check connection parameters
            psycopg2.connect.assert_called_once()
            call_kwargs = psycopg2.connect.call_args[1]
            assert call_kwargs['host'] == 'localhost'
            assert call_kwargs['port'] == 5432
            assert call_kwargs['user'] == 'test_user'
            assert call_kwargs['database'] == 'postgres'
            
            # Check database creation
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert "CREATE DATABASE" in call_args[0]
            assert "test_dataset" in call_args[0]
            assert "UTF8" in call_args[0]

    def test_create_database_already_exists(self, backend):
        """Test creating database that already exists."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        # Simulate duplicate database error
        mock_cursor.execute.side_effect = psycopg2.errors.DuplicateDatabase("Database exists")
        
        with patch('psycopg2.connect', return_value=mock_conn):
            # Should not raise
            backend.create_database("existing_db")

    def test_create_database_error(self, backend):
        """Test database creation error."""
        with patch('psycopg2.connect', side_effect=Exception("Connection failed")):
            with pytest.raises(StorageError, match="Failed to create PostgreSQL database"):
                backend.create_database("test_db")

    def test_drop_database_success(self, backend):
        """Test successful database drop."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch('psycopg2.connect', return_value=mock_conn):
            with patch.object(backend, 'close_connections'):
                backend.drop_database("test_dataset")
                
                backend.close_connections.assert_called_once()
                
                # Check termination query
                assert mock_cursor.execute.call_count == 2
                terminate_call = mock_cursor.execute.call_args_list[0][0][0]
                assert "pg_terminate_backend" in terminate_call
                assert "test_dataset" in terminate_call
                
                # Check drop query
                drop_call = mock_cursor.execute.call_args_list[1][0][0]
                assert "DROP DATABASE" in drop_call
                assert "test_dataset" in drop_call

    def test_drop_database_with_connection_string(self, backend):
        """Test dropping database with connection string."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch('psycopg2.connect', return_value=mock_conn):
            with patch.object(backend, 'close_connections'):
                backend.drop_database("postgresql://user:pass@host/mydb")
                
                # Check that "mydb" was extracted
                drop_call = mock_cursor.execute.call_args_list[1][0][0]
                assert "mydb" in drop_call

    def test_drop_database_error(self, backend):
        """Test database drop error."""
        with patch('psycopg2.connect', side_effect=Exception("Connection failed")):
            with pytest.raises(StorageError, match="Failed to drop PostgreSQL database"):
                backend.drop_database("test_db")

    def test_get_engine_caching(self, config):
        """Test engine caching."""
        # Disable connection pooling for this test
        config['use_connection_pool'] = False
        backend = PostgreSQLBackend(config)
        
        with patch.object(backend, 'create_engine') as mock_create:
            mock_engine = Mock()
            mock_create.return_value = mock_engine
            
            # First call
            engine1 = backend.get_engine("test_db")
            assert engine1 == mock_engine
            mock_create.assert_called_once_with("test_db")
            
            # Second call - should use cached engine
            engine2 = backend.get_engine("test_db")
            assert engine2 == engine1
            assert mock_create.call_count == 1  # Not called again

    def test_query_method(self, backend, mock_engine):
        """Test query method."""
        backend._engine = mock_engine
        expected_df = pd.DataFrame({'col': [1, 2, 3]})
        
        with patch('pandas.read_sql_query', return_value=expected_df) as mock_read:
            result = backend.query("SELECT * FROM table")
            
            assert result.equals(expected_df)
            mock_read.assert_called_once_with("SELECT * FROM table", mock_engine, params=None)

    def test_query_no_engine(self, backend):
        """Test query without engine."""
        with pytest.raises(StorageError, match="No engine available"):
            backend.query("SELECT 1")

    def test_close_connections(self, backend, mock_engine):
        """Test closing connections."""
        backend._engine = mock_engine
        backend._session_factory = Mock()
        
        backend.close_connections()
        
        mock_engine.dispose.assert_called_once()
        assert backend._engine is None
        assert backend._session_factory is None

    def test_table_exists(self, backend, mock_engine):
        """Test checking table existence."""
        # Need to patch the inspect function in the base module
        with patch('mdm.storage.base.inspect') as mock_inspect:
            mock_inspector = Mock()
            # Make mock_inspector.get_table_names() return a list
            mock_inspector.get_table_names.return_value = ['table1', 'table2', 'table3']
            mock_inspect.return_value = mock_inspector
            
            # The base class table_exists method expects two arguments: engine and table_name
            assert backend.table_exists(mock_engine, 'table2') is True
            assert backend.table_exists(mock_engine, 'table4') is False

    def test_create_table_from_dataframe(self, backend, mock_engine):
        """Test creating table from DataFrame."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        
        with patch.object(df, 'to_sql') as mock_to_sql:
            backend.create_table_from_dataframe(df, 'test_table', mock_engine)
            
            mock_to_sql.assert_called_once_with(
                'test_table',
                mock_engine,
                if_exists='fail',
                index=False,
                method='multi',
                chunksize=10000
            )

    def test_read_table_to_dataframe(self, backend, mock_engine):
        """Test reading table to DataFrame."""
        expected_df = pd.DataFrame({'col': [1, 2, 3]})
        
        with patch('pandas.read_sql_query', return_value=expected_df) as mock_read:
            result = backend.read_table_to_dataframe('test_table', mock_engine)
            
            assert result.equals(expected_df)
            mock_read.assert_called_once_with('SELECT * FROM test_table', mock_engine)

    def test_execute_query(self, backend, mock_engine):
        """Test executing query."""
        mock_connection = Mock()
        expected_result = Mock()
        mock_connection.execute.return_value = expected_result
        
        # Mock the begin context manager
        mock_begin = MagicMock()
        mock_begin.__enter__.return_value = mock_connection
        mock_begin.__exit__.return_value = None
        mock_engine.begin.return_value = mock_begin
        
        with patch('mdm.storage.backends.compatibility_mixin.text') as mock_text:
            mock_text.return_value = "SELECT 1"
            
            result = backend.execute_query("SELECT 1", mock_engine)
            
            # Backend mixin returns the result proxy, not the fetched result
            assert result == expected_result
            mock_connection.execute.assert_called_once()

    def test_get_table_info(self, backend, mock_engine, mock_connection):
        """Test getting table info."""
        mock_engine.connect.return_value = mock_connection
        
        # Need to patch the inspect function in the compatibility mixin module
        with patch('mdm.storage.backends.compatibility_mixin.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_columns = [
                {'name': 'id', 'type': 'INTEGER'},
                {'name': 'name', 'type': 'VARCHAR'}
            ]
            mock_inspector.get_columns.return_value = mock_columns
            mock_inspect.return_value = mock_inspector
            
            # Mock row count
            result = Mock()
            result.scalar.return_value = 42
            mock_connection.execute.return_value = result
            
            info = backend.get_table_info('test_table', mock_engine)
            
            assert info['name'] == 'test_table'
            assert info['columns'] == mock_columns
            assert info['row_count'] == 42
            assert info['column_count'] == 2