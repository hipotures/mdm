"""Comprehensive unit tests for utils/paths to achieve 80%+ coverage."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

# Import the module first to ensure coverage tracking
import mdm.utils.paths
from mdm.utils.paths import PathManager, get_path_manager


class TestPathManager:
    """Test cases for PathManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.get_full_path = Mock(side_effect=lambda path_key, base_path: {
            "datasets_path": base_path / "datasets",
            "configs_path": base_path / "config/datasets",
            "logs_path": base_path / "logs",
            "custom_features_path": base_path / "config/custom_features",
        }.get(path_key, base_path / path_key))
        return config

    def test_init_default_base_path(self, mock_config):
        """Test initialization with default base path."""
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager()
            assert manager.base_path == Path.home() / ".mdm"

    def test_init_custom_base_path(self, mock_config):
        """Test initialization with custom base path."""
        custom_path = Path("/custom/mdm")
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=custom_path)
            assert manager.base_path == custom_path

    def test_datasets_path(self, mock_config):
        """Test datasets_path property."""
        base_path = Path("/test/mdm")
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            assert manager.datasets_path == base_path / "datasets"
            
            # Verify get_full_path was called correctly
            mock_config.get_full_path.assert_called_with("datasets_path", base_path)

    def test_configs_path(self, mock_config):
        """Test configs_path property."""
        base_path = Path("/test/mdm")
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            assert manager.configs_path == base_path / "config/datasets"
            
            # Verify get_full_path was called correctly
            mock_config.get_full_path.assert_called_with("configs_path", base_path)

    def test_logs_path(self, mock_config):
        """Test logs_path property."""
        base_path = Path("/test/mdm")
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            assert manager.logs_path == base_path / "logs"
            
            # Verify get_full_path was called correctly
            mock_config.get_full_path.assert_called_with("logs_path", base_path)

    def test_custom_features_path(self, mock_config):
        """Test custom_features_path property."""
        base_path = Path("/test/mdm")
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            assert manager.custom_features_path == base_path / "config/custom_features"
            
            # Verify get_full_path was called correctly
            mock_config.get_full_path.assert_called_with("custom_features_path", base_path)

    def test_get_dataset_path(self, mock_config):
        """Test get_dataset_path method."""
        base_path = Path("/test/mdm")
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            
            # Test lowercase conversion
            assert manager.get_dataset_path("TestDataset") == base_path / "datasets" / "testdataset"
            assert manager.get_dataset_path("dataset_123") == base_path / "datasets" / "dataset_123"

    def test_get_dataset_db_path_sqlite(self, mock_config):
        """Test get_dataset_db_path for SQLite."""
        base_path = Path("/test/mdm")
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            
            path = manager.get_dataset_db_path("test_dataset", "sqlite")
            assert path == base_path / "datasets" / "test_dataset" / "dataset.sqlite"

    def test_get_dataset_db_path_duckdb(self, mock_config):
        """Test get_dataset_db_path for DuckDB."""
        base_path = Path("/test/mdm")
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            
            path = manager.get_dataset_db_path("test_dataset", "duckdb")
            assert path == base_path / "datasets" / "test_dataset" / "dataset.duckdb"

    def test_get_dataset_db_path_postgresql(self, mock_config):
        """Test get_dataset_db_path for PostgreSQL."""
        base_path = Path("/test/mdm")
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            
            # PostgreSQL doesn't use file path, returns info file
            path = manager.get_dataset_db_path("test_dataset", "postgresql")
            assert path == base_path / "datasets" / "test_dataset" / "dataset.info"

    def test_get_dataset_db_path_unknown_backend(self, mock_config):
        """Test get_dataset_db_path for unknown backend."""
        base_path = Path("/test/mdm")
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            
            # Unknown backends default to .info file
            path = manager.get_dataset_db_path("test_dataset", "unknown")
            assert path == base_path / "datasets" / "test_dataset" / "dataset.info"

    def test_get_dataset_config_path(self, mock_config):
        """Test get_dataset_config_path method."""
        base_path = Path("/test/mdm")
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            
            # Test lowercase conversion
            assert manager.get_dataset_config_path("TestDataset") == base_path / "config/datasets" / "testdataset.yaml"
            assert manager.get_dataset_config_path("dataset_123") == base_path / "config/datasets" / "dataset_123.yaml"

    def test_ensure_directories(self, mock_config, tmp_path):
        """Test ensure_directories method."""
        base_path = tmp_path / "test_mdm"
        
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            
            # Ensure directories don't exist initially
            assert not (base_path / "datasets").exists()
            assert not (base_path / "config/datasets").exists()
            assert not (base_path / "logs").exists()
            assert not (base_path / "config/custom_features").exists()
            
            # Call ensure_directories
            manager.ensure_directories()
            
            # Verify all directories were created
            assert (base_path / "datasets").exists()
            assert (base_path / "datasets").is_dir()
            assert (base_path / "config/datasets").exists()
            assert (base_path / "config/datasets").is_dir()
            assert (base_path / "logs").exists()
            assert (base_path / "logs").is_dir()
            assert (base_path / "config/custom_features").exists()
            assert (base_path / "config/custom_features").is_dir()

    def test_ensure_directories_already_exist(self, mock_config, tmp_path):
        """Test ensure_directories when directories already exist."""
        base_path = tmp_path / "test_mdm"
        
        # Pre-create directories
        (base_path / "datasets").mkdir(parents=True)
        (base_path / "config/datasets").mkdir(parents=True)
        (base_path / "logs").mkdir(parents=True)
        (base_path / "config/custom_features").mkdir(parents=True)
        
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = PathManager(base_path=base_path)
            
            # Should not raise error
            manager.ensure_directories()
            
            # Verify directories still exist
            assert (base_path / "datasets").exists()
            assert (base_path / "config/datasets").exists()
            assert (base_path / "logs").exists()
            assert (base_path / "config/custom_features").exists()


class TestGetPathManager:
    """Test cases for get_path_manager function."""

    def test_get_path_manager_singleton(self):
        """Test that get_path_manager returns singleton instance."""
        # Reset global state
        import mdm.utils.paths
        mdm.utils.paths._path_manager = None
        
        with patch('mdm.utils.paths.get_config', return_value=Mock()):
            # First call creates instance
            manager1 = get_path_manager()
            assert manager1 is not None
            
            # Second call returns same instance
            manager2 = get_path_manager()
            assert manager2 is manager1

    def test_get_path_manager_initialization(self):
        """Test path manager initialization."""
        # Reset global state
        import mdm.utils.paths
        mdm.utils.paths._path_manager = None
        
        mock_config = Mock()
        mock_config.get_full_path = Mock(side_effect=lambda path_key, base_path: base_path / path_key)
        
        with patch('mdm.utils.paths.get_config', return_value=mock_config):
            manager = get_path_manager()
            
            # Verify it uses default base path
            assert manager.base_path == Path.home() / ".mdm"

    def test_get_path_manager_preserves_state(self):
        """Test that get_path_manager preserves existing instance."""
        # Create a custom instance
        import mdm.utils.paths
        custom_path = Path("/custom/path")
        
        with patch('mdm.utils.paths.get_config', return_value=Mock()):
            custom_manager = PathManager(base_path=custom_path)
            mdm.utils.paths._path_manager = custom_manager
            
            # get_path_manager should return our custom instance
            manager = get_path_manager()
            assert manager is custom_manager
            assert manager.base_path == custom_path