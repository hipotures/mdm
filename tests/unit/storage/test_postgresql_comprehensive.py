"""Comprehensive unit tests for PostgreSQL storage backend."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from mdm.storage.postgresql import PostgreSQLBackend
from mdm.core.exceptions import StorageError


class TestPostgreSQLBackendInit:
    """Test PostgreSQL backend initialization."""
    
    @patch('mdm.storage.postgresql.psycopg2')
    def test_init_success(self, mock_psycopg2):
        """Test successful initialization."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            },
            'batch_size': 10000
        }
        backend = PostgreSQLBackend(config)
        assert backend.config == config
        assert backend.backend_type == "postgresql"
    
    def test_backend_type(self):
        """Test backend type property."""
        config = {'postgresql': {'user': 'test'}}
        backend = PostgreSQLBackend(config)
        assert backend.backend_type == "postgresql"


class TestPostgreSQLBackendConnectionString:
    """Test PostgreSQL connection string building."""
    
    def test_build_connection_string_basic(self):
        """Test building basic connection string."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
        backend = PostgreSQLBackend(config)
        
        conn_str = backend._build_connection_string('test_db')
        assert conn_str == 'postgresql://test_user:test_pass@localhost:5432/test_db?sslmode=prefer'
    
    def test_build_connection_string_no_password(self):
        """Test building connection string without password."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user'
            }
        }
        backend = PostgreSQLBackend(config)
        
        conn_str = backend._build_connection_string('test_db')
        assert conn_str == 'postgresql://test_user@localhost:5432/test_db?sslmode=prefer'
    
    def test_build_connection_string_admin(self):
        """Test building admin connection string."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
        backend = PostgreSQLBackend(config)
        
        conn_str = backend._build_connection_string(admin=True)
        assert conn_str == 'postgresql://test_user:test_pass@localhost:5432/postgres?sslmode=prefer'
    
    def test_build_connection_string_no_user(self):
        """Test building connection string without user."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432
            }
        }
        backend = PostgreSQLBackend(config)
        
        with pytest.raises(StorageError) as exc_info:
            backend._build_connection_string('test_db')
        assert "PostgreSQL user not configured" in str(exc_info.value)


class TestPostgreSQLBackendDatabaseOperations:
    """Test PostgreSQL database operations."""
    
    @patch('mdm.storage.postgresql.create_engine')
    def test_database_exists_true(self, mock_create_engine):
        """Test checking if database exists - returns true."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
        backend = PostgreSQLBackend(config)
        
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Set up proper context manager
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_conn.execute.return_value = mock_result
        
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_conn
        mock_context.__exit__.return_value = None
        mock_engine.connect.return_value = mock_context
        
        assert backend.database_exists('test_db') is True
        mock_engine.dispose.assert_called_once()
    
    @patch('mdm.storage.postgresql.create_engine')
    def test_database_exists_false(self, mock_create_engine):
        """Test checking if database exists - returns false."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
        backend = PostgreSQLBackend(config)
        
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Set up proper context manager
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = None
        mock_conn.execute.return_value = mock_result
        
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_conn
        mock_context.__exit__.return_value = None
        mock_engine.connect.return_value = mock_context
        
        assert backend.database_exists('test_db') is False
        mock_engine.dispose.assert_called_once()
    
    @patch('mdm.storage.postgresql.psycopg2')
    def test_create_database_success(self, mock_psycopg2):
        """Test successful database creation."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
        backend = PostgreSQLBackend(config)
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn
        
        # Mock the errors module
        mock_psycopg2.errors = Mock()
        mock_psycopg2.errors.DuplicateDatabase = type('DuplicateDatabase', (Exception,), {})
        
        backend.create_database('test_db')
        
        mock_psycopg2.connect.assert_called_once()
        # Check that CREATE DATABASE was executed
        create_db_call = None
        for call in mock_cursor.execute.call_args_list:
            if 'CREATE DATABASE' in str(call):
                create_db_call = call
                break
        assert create_db_call is not None
    
    @patch('mdm.storage.postgresql.psycopg2')
    def test_create_database_duplicate(self, mock_psycopg2):
        """Test creating database that already exists."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
        backend = PostgreSQLBackend(config)
        
        # Mock the errors module and create the exception
        mock_psycopg2.errors = Mock()
        DuplicateDatabase = type('DuplicateDatabase', (Exception,), {})
        mock_psycopg2.errors.DuplicateDatabase = DuplicateDatabase
        
        # Make connect raise DuplicateDatabase
        mock_psycopg2.connect.side_effect = DuplicateDatabase("database already exists")
        
        # Should not raise - it's handled
        backend.create_database('test_db')
    
    @patch('mdm.storage.postgresql.psycopg2')
    def test_create_database_connection_error(self, mock_psycopg2):
        """Test database creation with connection error."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
        backend = PostgreSQLBackend(config)
        
        mock_psycopg2.connect.side_effect = Exception("Connection failed")
        
        # Mock the errors module
        mock_psycopg2.errors = Mock()
        mock_psycopg2.errors.DuplicateDatabase = type('DuplicateDatabase', (Exception,), {})
        
        with pytest.raises(StorageError) as exc_info:
            backend.create_database('test_db')
        assert "Failed to create PostgreSQL database" in str(exc_info.value)


class TestPostgreSQLBackendTableOperations:
    """Test PostgreSQL table operations."""
    
    @pytest.fixture
    def config(self):
        """Default test configuration."""
        return {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
    
    @pytest.fixture
    def backend_with_engine(self, config):
        """Backend with mocked engine."""
        backend = PostgreSQLBackend(config)
        backend._engine = Mock()
        return backend
    
    def test_create_table_from_dataframe_empty(self, backend_with_engine):
        """Test creating table from empty dataframe."""
        df = pd.DataFrame()
        
        with patch('pandas.DataFrame.to_sql') as mock_to_sql:
            backend_with_engine.create_table_from_dataframe(df, 'test_table', backend_with_engine._engine)
            
            mock_to_sql.assert_called_once_with(
                'test_table',
                backend_with_engine._engine,
                if_exists='fail',
                index=False,
                method='multi',
                chunksize=10000
            )
    
    @patch('pandas.DataFrame.to_sql')
    def test_create_table_from_dataframe_success(self, mock_to_sql, backend_with_engine):
        """Test creating table from dataframe."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })
        
        backend_with_engine.create_table_from_dataframe(df, 'test_table', backend_with_engine._engine, if_exists='replace')
        
        mock_to_sql.assert_called_once_with(
            'test_table',
            backend_with_engine._engine,
            if_exists='replace',
            index=False,
            method='multi',
            chunksize=10000
        )
    
    @patch('pandas.DataFrame.to_sql')
    def test_create_table_from_dataframe_large(self, mock_to_sql, backend_with_engine):
        """Test creating table with large dataframe."""
        df = pd.DataFrame({
            'id': range(1000),
            'value': np.random.rand(1000)
        })
        
        backend_with_engine.create_table_from_dataframe(
            df, 'test_table', backend_with_engine._engine
        )
        
        mock_to_sql.assert_called_once()
    
    @patch('pandas.DataFrame.to_sql')
    def test_create_table_from_dataframe_error(self, mock_to_sql, backend_with_engine):
        """Test creating table error handling."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_to_sql.side_effect = Exception("Database error")
        
        with pytest.raises(StorageError) as exc_info:
            backend_with_engine.create_table_from_dataframe(df, 'test_table', backend_with_engine._engine)
        assert "Failed to create table 'test_table'" in str(exc_info.value)
    
    def test_table_exists_true(self, backend_with_engine):
        """Test checking if table exists - returns true."""
        with patch('mdm.storage.base.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_inspect.return_value = mock_inspector
            mock_inspector.get_table_names.return_value = ['test_table', 'other_table']
            
            assert backend_with_engine.table_exists(backend_with_engine._engine, 'test_table') is True
    
    def test_table_exists_false(self, backend_with_engine):
        """Test checking if table exists - returns false."""
        with patch('mdm.storage.base.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_inspect.return_value = mock_inspector
            mock_inspector.get_table_names.return_value = ['other_table']
            
            assert backend_with_engine.table_exists(backend_with_engine._engine, 'test_table') is False


class TestPostgreSQLBackendDataOperations:
    """Test PostgreSQL data operations."""
    
    @pytest.fixture
    def config(self):
        """Default test configuration."""
        return {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
    
    @pytest.fixture
    def backend_with_engine(self, config):
        """Backend with mocked engine."""
        backend = PostgreSQLBackend(config)
        backend._engine = Mock()
        return backend
    
    @patch('pandas.read_sql_query')
    def test_read_table_to_dataframe_success(self, mock_read_sql, backend_with_engine):
        """Test reading table successfully."""
        expected_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        mock_read_sql.return_value = expected_df
        
        result = backend_with_engine.read_table_to_dataframe('test_table', backend_with_engine._engine)
        
        mock_read_sql.assert_called_once()
        pd.testing.assert_frame_equal(result, expected_df)
    
    @patch('pandas.read_sql_query')
    def test_read_table_to_dataframe_with_limit(self, mock_read_sql, backend_with_engine):
        """Test reading table with limit."""
        expected_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob']
        })
        mock_read_sql.return_value = expected_df
        
        result = backend_with_engine.read_table_to_dataframe('test_table', backend_with_engine._engine, limit=2)
        
        mock_read_sql.assert_called_once()
        call_args = mock_read_sql.call_args[0][0]
        assert 'LIMIT 2' in call_args
    
    @patch('pandas.read_sql_query')
    def test_read_table_to_dataframe_error(self, mock_read_sql, backend_with_engine):
        """Test reading table error handling."""
        mock_read_sql.side_effect = Exception("Read error")
        
        with pytest.raises(StorageError) as exc_info:
            backend_with_engine.read_table_to_dataframe('test_table', backend_with_engine._engine)
        assert "Failed to read table 'test_table'" in str(exc_info.value)
    
    @patch('pandas.read_sql_query')
    def test_query_success(self, mock_read_sql_query, backend_with_engine):
        """Test executing query successfully."""
        expected_df = pd.DataFrame({
            'count': [42]
        })
        mock_read_sql_query.return_value = expected_df
        
        result = backend_with_engine.query("SELECT COUNT(*) as count FROM test_table")
        
        mock_read_sql_query.assert_called_once()
        pd.testing.assert_frame_equal(result, expected_df)
    
    @patch('pandas.read_sql_query')
    def test_query_with_where_clause(self, mock_read_sql_query, backend_with_engine):
        """Test executing query with WHERE clause."""
        expected_df = pd.DataFrame({
            'id': [1],
            'name': ['Alice']
        })
        mock_read_sql_query.return_value = expected_df
        
        # Note: query method doesn't support params, so we embed the value directly
        query = "SELECT * FROM test_table WHERE id = 1"
        result = backend_with_engine.query(query)
        
        mock_read_sql_query.assert_called_once()
        pd.testing.assert_frame_equal(result, expected_df)
    
    def test_execute_query_success(self, backend_with_engine):
        """Test executing statement successfully."""
        # Set up proper context manager
        mock_conn = Mock()
        mock_result = Mock()
        mock_conn.execute.return_value = mock_result
        
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_conn
        mock_context.__exit__.return_value = None
        backend_with_engine._engine.begin.return_value = mock_context
        
        with patch('mdm.storage.backends.compatibility_mixin.text') as mock_text:
            mock_text.return_value = "UPDATE test_table SET value = 1"
            
            result = backend_with_engine.execute_query("UPDATE test_table SET value = 1", backend_with_engine._engine)
            
            mock_conn.execute.assert_called_once()
            assert result == mock_result
    
    def test_execute_query_error(self, backend_with_engine):
        """Test executing statement error handling."""
        # Set up context manager that raises error
        mock_conn = Mock()
        mock_conn.execute.side_effect = Exception("Execute error")
        
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_conn
        mock_context.__exit__.return_value = None
        backend_with_engine._engine.begin.return_value = mock_context
        
        with patch('mdm.storage.backends.compatibility_mixin.text') as mock_text:
            mock_text.return_value = "INVALID SQL"
            
            with pytest.raises(StorageError) as exc_info:
                backend_with_engine.execute_query("INVALID SQL", backend_with_engine._engine)
            assert "Failed to execute query" in str(exc_info.value)


class TestPostgreSQLBackendTableInfo:
    """Test PostgreSQL table info operations."""
    
    @pytest.fixture
    def config(self):
        """Default test configuration."""
        return {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
    
    @pytest.fixture
    def backend_with_engine(self, config):
        """Backend with mocked engine."""
        backend = PostgreSQLBackend(config)
        backend._engine = Mock()
        return backend
    
    def test_get_table_info_success(self, backend_with_engine):
        """Test getting table info successfully."""
        with patch('mdm.storage.backends.compatibility_mixin.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_inspect.return_value = mock_inspector
            
            mock_inspector.get_columns.return_value = [
                {
                    'name': 'id',
                    'type': Mock(python_type=int),
                    'nullable': False,
                    'default': None,
                    'autoincrement': True
                },
                {
                    'name': 'name',
                    'type': Mock(python_type=str),
                    'nullable': True,
                    'default': None,
                    'autoincrement': False
                }
            ]
            
            # Mock the connection for row count
            mock_conn = Mock()
            mock_result = Mock()
            mock_result.scalar.return_value = 42
            mock_conn.execute.return_value = mock_result
            
            mock_context = MagicMock()
            mock_context.__enter__.return_value = mock_conn
            mock_context.__exit__.return_value = None
            backend_with_engine._engine.connect.return_value = mock_context
            
            info = backend_with_engine.get_table_info('test_table', backend_with_engine._engine)
            
            assert info['name'] == 'test_table'
            assert info['row_count'] == 42
            assert info['column_count'] == 2
            assert len(info['columns']) == 2
    
    def test_get_table_info_error(self, backend_with_engine):
        """Test getting table info error handling."""
        with patch('mdm.storage.base.inspect') as mock_inspect:
            mock_inspect.side_effect = Exception("Inspector error")
            
            with pytest.raises(Exception):
                backend_with_engine.get_table_info('test_table', backend_with_engine._engine)
    
    def test_get_table_names_success(self, backend_with_engine):
        """Test listing tables successfully."""
        with patch('mdm.storage.base.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_inspect.return_value = mock_inspector
            mock_inspector.get_table_names.return_value = ['table1', 'table2', 'table3']
            
            tables = backend_with_engine.get_table_names(backend_with_engine._engine)
            
            assert tables == ['table1', 'table2', 'table3']
    
    def test_get_table_names_empty(self, backend_with_engine):
        """Test listing tables when database is empty."""
        with patch('mdm.storage.base.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_inspect.return_value = mock_inspector
            mock_inspector.get_table_names.return_value = []
            
            tables = backend_with_engine.get_table_names(backend_with_engine._engine)
            
            assert tables == []


class TestPostgreSQLBackendEngineManagement:
    """Test PostgreSQL engine management."""
    
    def test_create_engine_with_database_name(self):
        """Test creating engine with database name."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
        backend = PostgreSQLBackend(config)
        
        with patch('mdm.storage.postgresql.create_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            
            engine = backend.create_engine('test_db')
            
            mock_create_engine.assert_called_once()
            call_args = mock_create_engine.call_args[0][0]
            assert 'postgresql://test_user:test_pass@localhost:5432/test_db' in call_args
            assert engine == mock_engine
    
    def test_create_engine_with_connection_string(self):
        """Test creating engine with full connection string."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'test_user',
                'password': 'test_pass'
            }
        }
        backend = PostgreSQLBackend(config)
        
        with patch('mdm.storage.postgresql.create_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            
            conn_str = 'postgresql://custom_user:custom_pass@custom_host:5433/custom_db'
            engine = backend.create_engine(conn_str)
            
            mock_create_engine.assert_called_once()
            call_args = mock_create_engine.call_args[0][0]
            assert call_args == conn_str
    
    def test_get_database_path(self):
        """Test getting database path."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'user': 'test_user'
            }
        }
        backend = PostgreSQLBackend(config)
        
        # Default prefix
        path = backend.get_database_path('MyDataset', Path('/data'))
        assert path == 'mdm_mydataset'
        
        # Custom prefix
        config['postgresql']['database_prefix'] = 'custom_'
        backend = PostgreSQLBackend(config)
        path = backend.get_database_path('MyDataset', Path('/data'))
        assert path == 'custom_mydataset'
    
    def test_close_connections(self):
        """Test closing connections."""
        config = {
            'postgresql': {
                'host': 'localhost',
                'user': 'test_user'
            }
        }
        backend = PostgreSQLBackend(config)
        
        # Set up mock engine
        mock_engine = Mock()
        backend._engine = mock_engine
        backend._session_factory = Mock()
        
        backend.close_connections()
        
        mock_engine.dispose.assert_called_once()
        assert backend._engine is None
        assert backend._session_factory is None