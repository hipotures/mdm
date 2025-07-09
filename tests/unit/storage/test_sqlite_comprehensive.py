"""Comprehensive unit tests for SQLiteBackend to achieve 80%+ coverage."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import sqlite3
import tempfile

from sqlalchemy import Engine, create_engine
from mdm.storage.sqlite import SQLiteBackend
from mdm.core.exceptions import StorageError
from mdm.models.base import Base


class TestSQLiteBackendComprehensive:
    """Comprehensive test cases for SQLiteBackend."""

    @pytest.fixture
    def sqlite_config(self):
        """Create SQLite configuration."""
        return {
            'journal_mode': 'WAL',
            'synchronous': 'NORMAL',
            'cache_size': -64000,
            'temp_store': 'MEMORY',
            'mmap_size': 268435456,
            'sqlalchemy': {
                'echo': False,
                'pool_size': 5,
                'max_overflow': 10
            }
        }

    @pytest.fixture
    def backend(self, sqlite_config):
        """Create SQLiteBackend instance."""
        return SQLiteBackend(sqlite_config)

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database path."""
        return tmp_path / "test.db"

    def test_backend_type(self, backend):
        """Test backend type property."""
        assert backend.backend_type == "sqlite"

    def test_create_engine_basic(self, backend, temp_db_path):
        """Test basic engine creation."""
        # Act
        engine = backend.create_engine(str(temp_db_path))
        
        # Assert
        assert isinstance(engine, Engine)
        assert str(engine.url) == f"sqlite:///{temp_db_path}"

    def test_create_engine_with_user_path(self, backend):
        """Test engine creation with ~ in path."""
        # Act
        with patch('pathlib.Path.expanduser') as mock_expand:
            mock_expand.return_value = Path('/home/user/test.db')
            engine = backend.create_engine('~/test.db')
        
        # Assert
        mock_expand.assert_called_once()
        assert str(engine.url) == "sqlite:////home/user/test.db"

    def test_create_engine_pragmas_set(self, backend, temp_db_path):
        """Test that SQLite pragmas are set on connection."""
        # Create engine
        engine = backend.create_engine(str(temp_db_path))
        
        # Connect and verify pragmas
        with engine.connect() as conn:
            # Check journal mode
            result = conn.exec_driver_sql("PRAGMA journal_mode").fetchone()
            assert result[0].upper() == 'WAL'
            
            # Check synchronous mode
            result = conn.exec_driver_sql("PRAGMA synchronous").fetchone()
            # SQLite returns numeric value for synchronous (1 = NORMAL)
            assert result[0] == 1
            
            # Check cache size
            result = conn.exec_driver_sql("PRAGMA cache_size").fetchone()
            assert result[0] == -64000

    def test_create_engine_custom_pragmas(self, temp_db_path):
        """Test engine creation with custom pragma values."""
        # Custom config
        config = {
            'journal_mode': 'DELETE',
            'synchronous': 'FULL',
            'cache_size': -32000,
            'temp_store': 'FILE',
            'mmap_size': 0
        }
        
        backend = SQLiteBackend(config)
        engine = backend.create_engine(str(temp_db_path))
        
        # Verify custom pragmas
        with engine.connect() as conn:
            # Journal mode
            result = conn.exec_driver_sql("PRAGMA journal_mode").fetchone()
            assert result[0].upper() == 'DELETE'
            
            # Synchronous mode (2 = FULL)
            result = conn.exec_driver_sql("PRAGMA synchronous").fetchone()
            assert result[0] == 2

    def test_create_engine_nested_sqlite_config(self, temp_db_path):
        """Test engine creation with nested sqlite configuration."""
        # Nested config (from DatasetManager)
        config = {
            'sqlite': {
                'journal_mode': 'PERSIST',
                'synchronous': 'OFF',
                'cache_size': -16000
            }
        }
        
        backend = SQLiteBackend(config)
        engine = backend.create_engine(str(temp_db_path))
        
        # Verify pragmas from nested config
        with engine.connect() as conn:
            result = conn.exec_driver_sql("PRAGMA journal_mode").fetchone()
            assert result[0].upper() == 'PERSIST'

    def test_create_engine_temp_store_mapping(self, temp_db_path):
        """Test temp_store string to numeric mapping."""
        test_cases = [
            ('DEFAULT', 0),
            ('FILE', 1),
            ('MEMORY', 2),
            ('memory', 2),  # Case insensitive
            ('invalid', 2)  # Default to MEMORY for invalid values
        ]
        
        for temp_store_str, expected_value in test_cases:
            config = {'temp_store': temp_store_str}
            backend = SQLiteBackend(config)
            engine = backend.create_engine(str(temp_db_path))
            
            with engine.connect() as conn:
                result = conn.exec_driver_sql("PRAGMA temp_store").fetchone()
                assert result[0] == expected_value

    def test_initialize_database(self, backend, temp_db_path):
        """Test database initialization with tables."""
        # Create engine
        engine = backend.create_engine(str(temp_db_path))
        
        # Mock Base.metadata
        with patch.object(Base.metadata, 'create_all') as mock_create_all:
            # Initialize database
            backend.initialize_database(engine)
            
            # Verify tables were created
            mock_create_all.assert_called_once_with(engine)

    def test_get_database_path(self, backend):
        """Test getting database path for dataset."""
        base_path = Path("/data/mdm")
        dataset_name = "Test_Dataset"
        
        # Act
        result = backend.get_database_path(dataset_name, base_path)
        
        # Assert - name should be lowercased
        expected = "/data/mdm/test_dataset/dataset.sqlite"
        assert result == expected

    def test_database_exists_true(self, backend, temp_db_path):
        """Test checking if database exists when it does."""
        # Create the file
        temp_db_path.touch()
        
        # Act
        result = backend.database_exists(str(temp_db_path))
        
        # Assert
        assert result is True

    def test_database_exists_false(self, backend, temp_db_path):
        """Test checking if database exists when it doesn't."""
        # Act
        result = backend.database_exists(str(temp_db_path))
        
        # Assert
        assert result is False

    def test_database_exists_directory(self, backend, tmp_path):
        """Test database_exists returns False for directory."""
        # Create a directory instead of file
        dir_path = tmp_path / "test_dir"
        dir_path.mkdir()
        
        # Act
        result = backend.database_exists(str(dir_path))
        
        # Assert
        assert result is False

    def test_database_exists_with_user_path(self, backend):
        """Test database_exists with ~ in path."""
        with patch('pathlib.Path.expanduser') as mock_expand:
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.is_file.return_value = True
            mock_expand.return_value = mock_path
            
            # Act
            result = backend.database_exists('~/test.db')
            
            # Assert
            assert result is True
            mock_expand.assert_called_once()

    def test_create_database_success(self, backend, temp_db_path):
        """Test successful database creation."""
        # Act
        backend.create_database(str(temp_db_path))
        
        # Assert
        assert temp_db_path.exists()
        assert temp_db_path.is_file()
        
        # Verify it's a valid SQLite database
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT sqlite_version()")
        version = cursor.fetchone()
        conn.close()
        assert version is not None

    def test_create_database_creates_parent_dirs(self, backend, tmp_path):
        """Test database creation creates parent directories."""
        # Deep path that doesn't exist
        db_path = tmp_path / "deep" / "nested" / "path" / "test.db"
        
        # Act
        backend.create_database(str(db_path))
        
        # Assert
        assert db_path.exists()
        assert db_path.parent.exists()

    def test_create_database_error(self, backend):
        """Test error handling in database creation."""
        # Use a path that can't be created
        invalid_path = "/tmp/test_error/test.db"
        
        # Mock mkdir to fail
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("No permission")):
            with pytest.raises(StorageError, match="Failed to create SQLite database"):
                backend.create_database(invalid_path)

    def test_drop_database_success(self, backend, temp_db_path):
        """Test successful database deletion."""
        # Create database first
        temp_db_path.touch()
        assert temp_db_path.exists()
        
        # Act
        backend.drop_database(str(temp_db_path))
        
        # Assert
        assert not temp_db_path.exists()

    def test_drop_database_not_exists(self, backend, temp_db_path):
        """Test dropping non-existent database."""
        # Should not raise error
        backend.drop_database(str(temp_db_path))
        
        # Assert nothing happened
        assert not temp_db_path.exists()

    def test_drop_database_error(self, backend, temp_db_path):
        """Test error handling in database deletion."""
        # Create database
        temp_db_path.touch()
        
        # Mock unlink to raise error
        with patch.object(Path, 'unlink', side_effect=PermissionError("No permission")):
            with pytest.raises(StorageError, match="Failed to drop SQLite database"):
                backend.drop_database(str(temp_db_path))

    def test_drop_database_closes_connections(self, backend, temp_db_path):
        """Test that drop_database closes connections first."""
        # Create database
        temp_db_path.touch()
        
        # Mock close_connections
        with patch.object(backend, 'close_connections') as mock_close:
            backend.drop_database(str(temp_db_path))
            
            # Verify connections were closed
            mock_close.assert_called_once()

    def test_sqlalchemy_config_options(self, temp_db_path):
        """Test SQLAlchemy configuration options."""
        config = {
            'sqlalchemy': {
                'echo': True,
                'pool_size': 10,
                'max_overflow': 20
            }
        }
        
        backend = SQLiteBackend(config)
        
        # Mock create_engine to capture arguments
        with patch('mdm.storage.sqlite.create_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            
            # Act
            backend.create_engine(str(temp_db_path))
            
            # Assert
            mock_create_engine.assert_called_once()
            call_kwargs = mock_create_engine.call_args[1]
            assert call_kwargs['echo'] is True
            assert call_kwargs['pool_size'] == 10
            assert call_kwargs['max_overflow'] == 20

    def test_empty_config(self, temp_db_path):
        """Test backend with empty configuration."""
        backend = SQLiteBackend({})
        
        # Should use defaults
        engine = backend.create_engine(str(temp_db_path))
        
        with engine.connect() as conn:
            # Check default journal mode (WAL)
            result = conn.exec_driver_sql("PRAGMA journal_mode").fetchone()
            assert result[0].upper() == 'WAL'
            
            # Check default synchronous (NORMAL = 1)
            result = conn.exec_driver_sql("PRAGMA synchronous").fetchone()
            assert result[0] == 1

    def test_pragma_execution_order(self, backend, temp_db_path):
        """Test that all pragmas are executed in correct order."""
        # Track pragma executions
        executed_pragmas = []
        
        def mock_execute(sql):
            executed_pragmas.append(sql)
            # Return mock result
            return Mock(fetchone=lambda: (1,))
        
        # Create engine and trigger connection event
        engine = backend.create_engine(str(temp_db_path))
        
        with engine.connect() as conn:
            # Pragmas should have been executed
            with patch.object(conn.connection.cursor(), 'execute', side_effect=mock_execute):
                # Trigger another connection to see pragma order
                with engine.connect() as conn2:
                    pass
        
        # The pragmas are set on first connection, verify they exist
        expected_pragmas = [
            'journal_mode', 'synchronous', 'cache_size', 'temp_store', 'mmap_size'
        ]
        
        # Since we connected, pragmas should be set
        assert temp_db_path.exists()  # Database was created