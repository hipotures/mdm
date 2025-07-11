"""Comprehensive unit tests for DuckDB storage backend.

NOTE: These tests are for the legacy backend API. The new stateless DuckDB backend
requires different tests. Tests are marked as skip - see test_stateless_backends.py.
"""

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

import pandas as pd
import pytest
from sqlalchemy import text

from mdm.core.exceptions import StorageError
from mdm.storage.duckdb import DuckDBBackend


@pytest.mark.skip(reason="Tests for legacy DuckDB backend API - see test_stateless_backends.py")
class TestDuckDBBackendComplete:
    """Comprehensive test coverage for DuckDB backend."""

    @pytest.fixture
    def config(self):
        """Create backend configuration."""
        return {
            "duckdb": {
                "memory_limit": "1GB",
                "threads": 4,
                "temp_directory": "/tmp/duckdb",
                "access_mode": "READ_WRITE"
            },
            "sqlalchemy": {
                "echo": False,
                "pool_size": 5,
                "max_overflow": 10
            }
        }

    @pytest.fixture
    def mock_engine(self):
        """Create mock SQLAlchemy engine."""
        engine = Mock()
        engine.dispose = Mock()
        engine.url = "duckdb:///test.duckdb"
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
        """Create DuckDB backend instance."""
        return DuckDBBackend(config)

    def test_init_success(self, config):
        """Test successful initialization."""
        backend = DuckDBBackend(config)
        assert backend.config == config
        assert backend.backend_type == "duckdb"

    def test_create_engine_basic(self, backend, tmp_path):
        """Test creating engine with basic configuration."""
        db_path = tmp_path / "test.duckdb"
        
        with patch('mdm.storage.duckdb.create_engine') as mock_create:
            mock_engine = Mock()
            mock_create.return_value = mock_engine
            
            engine = backend.create_engine(str(db_path))
            
            assert engine == mock_engine
            mock_create.assert_called_once()
            
            # Check arguments
            call_args = mock_create.call_args
            assert f"duckdb:///{db_path}" in call_args[0][0]
            
            # Check kwargs
            assert call_args[1]['echo'] is False
            assert call_args[1]['pool_size'] == 5
            assert call_args[1]['max_overflow'] == 10
            
            # Check connect_args
            connect_args = call_args[1]['connect_args']
            assert connect_args['memory_limit'] == "1GB"
            assert connect_args['threads'] == 4
            assert connect_args['temp_directory'] == "/tmp/duckdb"
            assert connect_args['read_only'] is False

    def test_create_engine_read_only(self, config):
        """Test creating engine in read-only mode."""
        config["duckdb"]["access_mode"] = "READ_ONLY"
        backend = DuckDBBackend(config)
        
        with patch('mdm.storage.duckdb.create_engine') as mock_create:
            backend.create_engine("/path/to/db.duckdb")
            
            connect_args = mock_create.call_args[1]['connect_args']
            assert connect_args['read_only'] is True

    def test_create_engine_minimal_config(self):
        """Test creating engine with minimal configuration."""
        config = {}
        backend = DuckDBBackend(config)
        
        with patch('mdm.storage.duckdb.create_engine') as mock_create:
            backend.create_engine("/path/to/db.duckdb")
            
            # Check that connect_args is empty with minimal config
            connect_args = mock_create.call_args[1]['connect_args']
            assert connect_args == {}

    def test_create_engine_expanduser(self, backend):
        """Test that paths with ~ are expanded."""
        with patch('mdm.storage.duckdb.create_engine') as mock_create:
            with patch('pathlib.Path.expanduser') as mock_expand:
                mock_expand.return_value = Path("/home/user/test.duckdb")
                
                backend.create_engine("~/test.duckdb")
                
                mock_expand.assert_called_once()
                call_args = mock_create.call_args[0][0]
                assert "/home/user/test.duckdb" in call_args

    def test_initialize_database(self, backend, mock_engine, mock_connection):
        """Test database initialization."""
        mock_engine.connect.return_value = mock_connection
        
        with patch('mdm.models.base.Base.metadata') as mock_metadata:
            backend.initialize_database(mock_engine)
            
            # Check tables were created
            mock_metadata.create_all.assert_called_once_with(mock_engine)
            
            # Check extensions were installed
            assert mock_connection.execute.call_count == 2
            calls = mock_connection.execute.call_args_list
            assert "INSTALL parquet" in str(calls[0][0][0])
            assert "LOAD parquet" in str(calls[1][0][0])
            mock_connection.commit.assert_called_once()

    def test_get_database_path(self, backend, tmp_path):
        """Test getting database path."""
        base_path = tmp_path / "datasets"
        
        path = backend.get_database_path("MyDataset", base_path)
        
        expected = str(base_path / "mydataset" / "dataset.duckdb")
        assert path == expected

    def test_get_database_path_lowercase(self, backend, tmp_path):
        """Test that dataset names are lowercased."""
        base_path = tmp_path / "datasets"
        
        path = backend.get_database_path("TEST_DATASET", base_path)
        
        expected = str(base_path / "test_dataset" / "dataset.duckdb")
        assert path == expected

    def test_database_exists_true(self, backend, tmp_path):
        """Test checking if database exists."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()  # Create empty file
        
        assert backend.database_exists(str(db_path)) is True

    def test_database_exists_false(self, backend, tmp_path):
        """Test checking if database doesn't exist."""
        db_path = tmp_path / "nonexistent.duckdb"
        
        assert backend.database_exists(str(db_path)) is False

    def test_database_exists_directory(self, backend, tmp_path):
        """Test that directories return False."""
        dir_path = tmp_path / "directory"
        dir_path.mkdir()
        
        assert backend.database_exists(str(dir_path)) is False

    def test_database_exists_expanduser(self, backend):
        """Test that ~ paths are expanded."""
        with patch('pathlib.Path.expanduser') as mock_expand:
            with patch('pathlib.Path.exists') as mock_exists:
                mock_expand.return_value = Path("/home/user/test.duckdb")
                mock_exists.return_value = True
                
                with patch('pathlib.Path.is_file', return_value=True):
                    result = backend.database_exists("~/test.duckdb")
                    
                    assert result is True
                    mock_expand.assert_called_once()

    def test_create_database_success(self, backend, tmp_path):
        """Test successful database creation."""
        db_path = tmp_path / "new_db.duckdb"
        
        with patch('duckdb.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn
            
            backend.create_database(str(db_path))
            
            # Check directory was created
            assert db_path.parent.exists()
            
            # Check duckdb.connect was called
            mock_connect.assert_called_once_with(str(db_path))
            mock_conn.close.assert_called_once()

    def test_create_database_parent_dirs(self, backend, tmp_path):
        """Test that parent directories are created."""
        db_path = tmp_path / "deep" / "nested" / "path" / "db.duckdb"
        
        with patch('duckdb.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn
            
            backend.create_database(str(db_path))
            
            # Check all parent directories were created
            assert db_path.parent.exists()
            assert db_path.parent.parent.exists()

    def test_create_database_error(self, backend, tmp_path):
        """Test database creation error."""
        db_path = tmp_path / "error.duckdb"
        
        with patch('duckdb.connect', side_effect=Exception("Connection failed")):
            with pytest.raises(StorageError, match="Failed to create DuckDB database"):
                backend.create_database(str(db_path))

    def test_drop_database_success(self, backend, tmp_path):
        """Test successful database drop."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()
        
        # Create WAL file
        wal_path = db_path.with_suffix(".duckdb.wal")
        wal_path.touch()
        
        with patch.object(backend, 'close_connections'):
            backend.drop_database(str(db_path))
            
            # Check files were deleted
            assert not db_path.exists()
            assert not wal_path.exists()
            
            # Check connections were closed
            backend.close_connections.assert_called_once()

    def test_drop_database_no_wal(self, backend, tmp_path):
        """Test dropping database without WAL file."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()
        
        with patch.object(backend, 'close_connections'):
            backend.drop_database(str(db_path))
            
            assert not db_path.exists()

    def test_drop_database_not_exists(self, backend, tmp_path):
        """Test dropping non-existent database."""
        db_path = tmp_path / "nonexistent.duckdb"
        
        # Should not raise
        backend.drop_database(str(db_path))

    def test_drop_database_error(self, backend, tmp_path):
        """Test database drop error."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()
        
        with patch.object(backend, 'close_connections'):
            with patch('pathlib.Path.unlink', side_effect=Exception("Permission denied")):
                with pytest.raises(StorageError, match="Failed to drop DuckDB database"):
                    backend.drop_database(str(db_path))

    def test_get_engine_caching(self, config):
        """Test engine caching."""
        # Disable connection pooling for this test
        config['use_connection_pool'] = False
        backend = DuckDBBackend(config)
        
        with patch.object(backend, 'create_engine') as mock_create:
            mock_engine = Mock()
            mock_create.return_value = mock_engine
            
            # First call
            engine1 = backend.get_engine("/path/to/db.duckdb")
            assert engine1 == mock_engine
            mock_create.assert_called_once_with("/path/to/db.duckdb")
            
            # Second call - should use cached engine
            engine2 = backend.get_engine("/path/to/db.duckdb")
            assert engine2 == engine1
            assert mock_create.call_count == 1

    def test_query_method(self, backend, mock_engine):
        """Test query method."""
        backend._engine = mock_engine
        expected_df = pd.DataFrame({'col': [1, 2, 3]})
        
        with patch('pandas.read_sql_query', return_value=expected_df) as mock_read:
            result = backend.query("SELECT * FROM table")
            
            assert result.equals(expected_df)
            mock_read.assert_called_once_with("SELECT * FROM table", mock_engine, params=None)

    def test_table_exists(self, backend, mock_engine):
        """Test checking table existence."""
        with patch('mdm.storage.base.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_inspector.get_table_names.return_value = ['table1', 'table2']
            mock_inspect.return_value = mock_inspector
            
            assert backend.table_exists(mock_engine, 'table1') is True
            assert backend.table_exists(mock_engine, 'table3') is False

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
            
            assert result == expected_result
            mock_connection.execute.assert_called_once()

    def test_close_connections(self, backend, mock_engine):
        """Test closing connections."""
        backend._engine = mock_engine
        backend._session_factory = Mock()
        
        backend.close_connections()
        
        mock_engine.dispose.assert_called_once()
        assert backend._engine is None
        assert backend._session_factory is None

    def test_session_context_manager(self, backend, mock_engine):
        """Test session context manager."""
        mock_session_factory = Mock()
        mock_session = Mock()
        mock_session_factory.return_value = mock_session
        
        backend._engine = mock_engine
        backend._session_factory = mock_session_factory
        
        with backend.session("/path/to/db") as session:
            assert session == mock_session
        
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    def test_session_rollback_on_error(self, backend, mock_engine):
        """Test session rollback on error."""
        mock_session_factory = Mock()
        mock_session = Mock()
        mock_session_factory.return_value = mock_session
        
        backend._engine = mock_engine
        backend._session_factory = mock_session_factory
        
        with pytest.raises(Exception):
            with backend.session("/path/to/db") as session:
                raise Exception("Test error")
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()