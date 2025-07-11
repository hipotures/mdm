"""Simple unit tests for PostgreSQL storage backend to improve coverage."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from pathlib import Path

from mdm.storage.postgresql import PostgreSQLBackend
from mdm.core.exceptions import StorageError


class TestPostgreSQLBackend:
    """Test PostgreSQL backend main functionality."""
    
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
    
    def test_backend_type(self, config):
        """Test backend type property."""
        backend = PostgreSQLBackend(config)
        assert backend.backend_type == "postgresql"
    
    def test_build_connection_string(self, config):
        """Test connection string building."""
        backend = PostgreSQLBackend(config)
        
        # Regular database connection
        conn_str = backend._build_connection_string('test_db')
        assert conn_str == 'postgresql://test_user:test_pass@localhost:5432/test_db?sslmode=prefer'
        
        # Admin connection
        admin_str = backend._build_connection_string(admin=True)
        assert admin_str == 'postgresql://test_user:test_pass@localhost:5432/postgres?sslmode=prefer'
    
    def test_build_connection_string_no_user(self):
        """Test connection string with missing user."""
        config = {'postgresql': {'host': 'localhost'}}
        backend = PostgreSQLBackend(config)
        
        with pytest.raises(StorageError, match="PostgreSQL user not configured"):
            backend._build_connection_string('test_db')
    
    @patch('mdm.storage.postgresql.create_engine')
    def test_create_engine(self, mock_create_engine, config):
        """Test creating SQLAlchemy engine."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        backend = PostgreSQLBackend(config)
        engine = backend.create_engine('test_db')
        
        # Should create engine with proper URL
        expected_url = 'postgresql://test_user:test_pass@localhost:5432/test_db?sslmode=prefer'
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args
        assert call_args[0][0] == expected_url
        assert engine == mock_engine
    
    def test_get_engine(self, config):
        """Test getting engine."""
        backend = PostgreSQLBackend(config)
        
        with patch.object(backend, 'create_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            
            # First call should create engine
            engine1 = backend.get_engine('test_db')
            assert engine1 == mock_engine
            mock_create_engine.assert_called_once_with('test_db')
            
            # Second call should return cached engine
            engine2 = backend.get_engine('test_db')
            assert engine2 == mock_engine
            # Still only called once
            mock_create_engine.assert_called_once()
    
    def test_close_connections(self, config):
        """Test closing database connections."""
        backend = PostgreSQLBackend(config)
        mock_engine = Mock()
        backend._engine = mock_engine
        backend._session_factory = Mock()
        
        backend.close_connections()
        
        mock_engine.dispose.assert_called_once()
        assert backend._engine is None
        assert backend._session_factory is None
    
    @patch('mdm.storage.postgresql.psycopg2')
    def test_create_database_success(self, mock_psycopg2, config):
        """Test successful database creation."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn
        
        # Mock the errors module
        mock_psycopg2.errors = Mock()
        mock_psycopg2.errors.DuplicateDatabase = type('DuplicateDatabase', (Exception,), {})
        
        backend = PostgreSQLBackend(config)
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
    def test_create_database_error(self, mock_psycopg2, config):
        """Test database creation error."""
        mock_psycopg2.connect.side_effect = Exception("Connection failed")
        
        # Mock the errors module
        mock_psycopg2.errors = Mock()
        mock_psycopg2.errors.DuplicateDatabase = type('DuplicateDatabase', (Exception,), {})
        
        backend = PostgreSQLBackend(config)
        with pytest.raises(StorageError, match="Failed to create PostgreSQL database"):
            backend.create_database('test_db')
    
    def test_create_table_from_dataframe(self, config):
        """Test creating table from dataframe."""
        backend = PostgreSQLBackend(config)
        mock_engine = Mock()
        backend._engine = mock_engine
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with patch('pandas.DataFrame.to_sql') as mock_to_sql:
            backend.create_table_from_dataframe(df, 'test_table', mock_engine)
            
            mock_to_sql.assert_called_once_with(
                'test_table',
                mock_engine,
                if_exists='fail',
                index=False
            )
    
    def test_table_exists(self, config):
        """Test checking if table exists."""
        backend = PostgreSQLBackend(config)
        mock_engine = Mock()
        
        with patch('mdm.storage.base.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_inspect.return_value = mock_inspector
            mock_inspector.get_table_names.return_value = ['test_table', 'other_table']
            
            assert backend.table_exists(mock_engine, 'test_table') is True
            assert backend.table_exists(mock_engine, 'nonexistent') is False
    
    def test_read_table_to_dataframe(self, config):
        """Test reading table to dataframe."""
        backend = PostgreSQLBackend(config)
        mock_engine = Mock()
        backend._engine = mock_engine
        
        expected_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with patch('pandas.read_sql_query', return_value=expected_df) as mock_read_sql:
            result = backend.read_table_to_dataframe('test_table', mock_engine)
            
            mock_read_sql.assert_called_once()
            pd.testing.assert_frame_equal(result, expected_df)
    
    def test_query(self, config):
        """Test executing query."""
        backend = PostgreSQLBackend(config)
        mock_engine = Mock()
        backend._engine = mock_engine
        
        expected_df = pd.DataFrame({'count': [42]})
        
        with patch('pandas.read_sql_query', return_value=expected_df) as mock_read_sql_query:
            result = backend.query("SELECT COUNT(*) as count FROM test_table")
            
            mock_read_sql_query.assert_called_once()
            pd.testing.assert_frame_equal(result, expected_df)
    
    def test_execute_query(self, config):
        """Test executing SQL statement."""
        backend = PostgreSQLBackend(config)
        mock_engine = Mock()
        
        # Create a proper mock for the context manager
        mock_conn = Mock()
        mock_result = Mock()
        mock_conn.execute.return_value = mock_result
        
        # Set up the context manager properly
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_conn
        mock_context.__exit__.return_value = None
        mock_engine.begin.return_value = mock_context
        
        with patch('mdm.storage.backends.compatibility_mixin.text') as mock_text:
            mock_text.return_value = "UPDATE test_table SET col1 = 1"
            
            result = backend.execute_query("UPDATE test_table SET col1 = 1", mock_engine)
            
            mock_conn.execute.assert_called_once()
            assert result == mock_result
    
    def test_get_table_info(self, config):
        """Test getting table information."""
        backend = PostgreSQLBackend(config)
        mock_engine = Mock()
        
        with patch('mdm.storage.base.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_inspect.return_value = mock_inspector
            mock_inspector.get_columns.return_value = [
                {'name': 'id', 'type': Mock(python_type=int), 'nullable': False},
                {'name': 'name', 'type': Mock(python_type=str), 'nullable': True}
            ]
            
            # Mock the connection for row count query - proper context manager
            mock_conn = Mock()
            mock_result = Mock()
            mock_result.scalar.return_value = 100
            mock_conn.execute.return_value = mock_result
            
            mock_context = MagicMock()
            mock_context.__enter__.return_value = mock_conn
            mock_context.__exit__.return_value = None
            mock_engine.connect.return_value = mock_context
            
            info = backend.get_table_info('test_table', mock_engine)
            
            assert info['name'] == 'test_table'
            assert len(info['columns']) == 2
            assert info['row_count'] == 100
            assert info['column_count'] == 2
    
    def test_get_table_names(self, config):
        """Test listing tables."""
        backend = PostgreSQLBackend(config)
        mock_engine = Mock()
        
        with patch('mdm.storage.base.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_inspect.return_value = mock_inspector
            mock_inspector.get_table_names.return_value = ['table1', 'table2']
            
            tables = backend.get_table_names(mock_engine)
            assert tables == ['table1', 'table2']
    
    @patch('mdm.storage.postgresql.create_engine')
    def test_database_exists_true(self, mock_create_engine, config):
        """Test checking if database exists - returns true."""
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
    def test_database_exists_false(self, mock_create_engine, config):
        """Test checking if database exists - returns false."""
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
    
    def test_get_database_path(self, config):
        """Test getting database path."""
        backend = PostgreSQLBackend(config)
        
        # Default prefix
        path = backend.get_database_path('MyDataset', Path('/data'))
        assert path == 'mdm_mydataset'
        
        # Custom prefix
        config['postgresql']['database_prefix'] = 'custom_'
        backend = PostgreSQLBackend(config)
        path = backend.get_database_path('MyDataset', Path('/data'))
        assert path == 'custom_mydataset'