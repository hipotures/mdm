"""Unit tests for stateless storage backends.

These tests are designed for the new stateless backend architecture where
backends fetch configuration from the global config system rather than
accepting it in their constructors.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from mdm.storage.backends.stateless_sqlite import StatelessSQLiteBackend
from mdm.storage.backends.stateless_duckdb import StatelessDuckDBBackend
from mdm.storage.factory import BackendFactory
from mdm.core.exceptions import StorageError, BackendError
from mdm.config import MDMConfig


class TestStatelessSQLiteBackend:
    """Test cases for stateless SQLite backend."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create a mock configuration."""
        config = Mock(spec=MDMConfig)
        config.paths = Mock()
        config.paths.datasets_path = str(temp_dir)
        config.database = Mock()
        config.database.echo_sql = False
        config.database.sqlite = Mock()
        config.database.sqlite.synchronous = "NORMAL"
        config.database.sqlite.journal_mode = "WAL"
        config.database.sqlite.cache_size = -64000
        config.database.sqlite.temp_store = "MEMORY"
        config.database.sqlite.mmap_size = 268435456
        config.performance = Mock()
        config.performance.batch_size = 10000
        config.model_dump.return_value = {"database": {"echo_sql": False}}
        return config
    
    @pytest.fixture
    def backend(self, mock_config):
        """Create a SQLite backend with mocked config."""
        with patch('mdm.storage.backends.stateless_sqlite.get_config', return_value=mock_config):
            return StatelessSQLiteBackend()
    
    def test_backend_type(self, backend):
        """Test backend type property."""
        assert backend.backend_type == "sqlite"
    
    def test_create_dataset(self, backend, temp_dir):
        """Test creating a new dataset."""
        with patch('mdm.storage.backends.stateless_sqlite.get_config') as mock_get_config:
            mock_get_config.return_value = backend.config
            
            dataset_name = "test_dataset"
            config = {"test_key": "test_value"}
            
            backend.create_dataset(dataset_name, config)
            
            # Check dataset directory exists
            dataset_path = temp_dir / dataset_name
            assert dataset_path.exists()
            
            # Check database file exists
            db_path = dataset_path / f"{dataset_name}.sqlite"
            assert db_path.exists()
    
    def test_dataset_exists(self, backend, temp_dir):
        """Test checking if dataset exists."""
        dataset_name = "test_dataset"
        
        # Should not exist initially
        assert not backend.dataset_exists(dataset_name)
        
        # Create dataset
        backend.create_dataset(dataset_name, {})
        
        # Should exist now
        assert backend.dataset_exists(dataset_name)
    
    def test_drop_dataset(self, backend, temp_dir):
        """Test dropping a dataset."""
        dataset_name = "test_dataset"
        
        # Create dataset
        backend.create_dataset(dataset_name, {})
        assert backend.dataset_exists(dataset_name)
        
        # Drop dataset
        backend.drop_dataset(dataset_name)
        assert not backend.dataset_exists(dataset_name)
    
    def test_save_and_load_data(self, backend, temp_dir):
        """Test saving and loading data."""
        dataset_name = "test_dataset"
        backend.create_dataset(dataset_name, {})
        
        # Create test data
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10, 20, 30]
        })
        
        # Save data
        backend.save_data(dataset_name, df, table_name="test_table")
        
        # Load data
        loaded_df = backend.load_data(dataset_name, table_name="test_table")
        
        # Compare
        pd.testing.assert_frame_equal(df, loaded_df)
    
    def test_get_engine_context(self, backend, temp_dir):
        """Test getting engine in context manager."""
        dataset_name = "test_dataset"
        backend.create_dataset(dataset_name, {})
        
        with backend.get_engine_context(dataset_name) as engine:
            assert engine is not None
            # Engine should be connected
            with engine.connect() as conn:
                result = conn.execute("SELECT 1")
                assert result.scalar() == 1
    
    def test_metadata_operations(self, backend, temp_dir):
        """Test metadata get and update operations."""
        dataset_name = "test_dataset"
        initial_metadata = {"key1": "value1", "key2": "value2"}
        
        backend.create_dataset(dataset_name, initial_metadata)
        
        # Get metadata
        metadata = backend.get_metadata(dataset_name)
        assert metadata["key1"] == "value1"
        assert metadata["key2"] == "value2"
        
        # Update metadata
        new_metadata = {"key2": "new_value", "key3": "value3"}
        backend.update_metadata(dataset_name, new_metadata)
        
        # Get updated metadata
        metadata = backend.get_metadata(dataset_name)
        assert metadata["key2"] == "new_value"
        assert metadata["key3"] == "value3"
        assert metadata["key1"] == "value1"  # Original key still there


class TestStatelessDuckDBBackend:
    """Test cases for stateless DuckDB backend."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create a mock configuration."""
        config = Mock(spec=MDMConfig)
        config.paths = Mock()
        config.paths.datasets_path = str(temp_dir)
        config.database = Mock()
        config.database.echo_sql = False
        config.database.duckdb = Mock()
        config.database.duckdb.memory_limit = "2GB"
        config.database.duckdb.threads = 4
        config.database.duckdb.temp_directory = "/tmp/mdm_duckdb"
        config.performance = Mock()
        config.performance.batch_size = 10000
        config.model_dump.return_value = {"database": {"echo_sql": False}}
        return config
    
    @pytest.fixture
    def backend(self, mock_config):
        """Create a DuckDB backend with mocked config."""
        with patch('mdm.storage.backends.stateless_duckdb.get_config', return_value=mock_config):
            return StatelessDuckDBBackend()
    
    def test_backend_type(self, backend):
        """Test backend type property."""
        assert backend.backend_type == "duckdb"
    
    def test_create_dataset(self, backend, temp_dir):
        """Test creating a new dataset."""
        with patch('mdm.storage.backends.stateless_duckdb.get_config') as mock_get_config:
            mock_get_config.return_value = backend.config
            
            dataset_name = "test_dataset"
            config = {"test_key": "test_value"}
            
            backend.create_dataset(dataset_name, config)
            
            # Check dataset directory exists
            dataset_path = temp_dir / dataset_name
            assert dataset_path.exists()
            
            # Check database file exists
            db_path = dataset_path / f"{dataset_name}.duckdb"
            assert db_path.exists()
    
    def test_parquet_export(self, backend, temp_dir):
        """Test DuckDB-specific Parquet export feature."""
        dataset_name = "test_dataset"
        backend.create_dataset(dataset_name, {})
        
        # Create test data
        df = pd.DataFrame({
            'id': range(100),
            'value': range(100, 200)
        })
        
        # Save data
        backend.save_data(dataset_name, df, table_name="test_table")
        
        # Export to Parquet
        parquet_path = temp_dir / "export.parquet"
        backend.export_to_parquet(dataset_name, "test_table", str(parquet_path))
        
        # Verify Parquet file exists
        assert parquet_path.exists()
        
        # Read back and verify
        df_parquet = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(df, df_parquet)
    
    def test_parquet_import(self, backend, temp_dir):
        """Test DuckDB-specific Parquet import feature."""
        dataset_name = "test_dataset"
        backend.create_dataset(dataset_name, {})
        
        # Create test Parquet file
        df = pd.DataFrame({
            'id': range(50),
            'text': [f'text_{i}' for i in range(50)]
        })
        parquet_path = temp_dir / "import.parquet"
        df.to_parquet(parquet_path)
        
        # Import from Parquet
        backend.import_from_parquet(dataset_name, str(parquet_path), "imported_table")
        
        # Load and verify
        loaded_df = backend.load_data(dataset_name, "imported_table")
        pd.testing.assert_frame_equal(df, loaded_df)


class TestBackendFactory:
    """Test cases for backend factory with stateless backends."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=MDMConfig)
        config.paths = Mock()
        config.paths.datasets_path = "/tmp/mdm_test"
        config.database = Mock()
        config.database.echo_sql = False
        config.model_dump.return_value = {"database": {"echo_sql": False}}
        return config
    
    def test_create_sqlite_backend(self, mock_config):
        """Test creating SQLite backend through factory."""
        with patch('mdm.storage.factory.get_config', return_value=mock_config):
            backend = BackendFactory.create('sqlite', {})
            assert isinstance(backend, StatelessSQLiteBackend)
            assert backend.backend_type == "sqlite"
    
    def test_create_duckdb_backend(self, mock_config):
        """Test creating DuckDB backend through factory."""
        with patch('mdm.storage.factory.get_config', return_value=mock_config):
            backend = BackendFactory.create('duckdb', {})
            assert isinstance(backend, StatelessDuckDBBackend)
            assert backend.backend_type == "duckdb"
    
    def test_create_invalid_backend(self):
        """Test error with invalid backend type."""
        with pytest.raises(BackendError, match="Unsupported backend type: invalid"):
            BackendFactory.create('invalid', {})
    
    def test_get_supported_backends(self):
        """Test getting list of supported backends."""
        backends = BackendFactory.get_supported_backends()
        assert 'sqlite' in backends
        assert 'duckdb' in backends
        assert 'postgresql' in backends
    
    def test_backend_registration(self):
        """Test that backends are properly registered in factory."""
        # Access the private _backends dict for testing
        assert 'sqlite' in BackendFactory._backends
        assert 'duckdb' in BackendFactory._backends
        assert 'postgresql' in BackendFactory._backends
        
        # Verify the registered classes
        assert BackendFactory._backends['sqlite'] == StatelessSQLiteBackend
        assert BackendFactory._backends['duckdb'] == StatelessDuckDBBackend