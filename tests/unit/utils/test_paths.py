"""Unit tests for path utilities."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from mdm.utils.paths import PathManager, get_path_manager


class TestPathManager:
    """Test PathManager functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.get_full_path = Mock(side_effect=lambda key, base_path: base_path / key.replace('_path', ''))
        return config
    
    @patch('mdm.utils.paths.get_config')
    def test_init_default_path(self, mock_get_config, mock_config):
        """Test initialization with default path."""
        mock_get_config.return_value = mock_config
        
        manager = PathManager()
        
        assert manager.base_path == Path.home() / ".mdm"
        assert manager._config == mock_config
    
    @patch('mdm.utils.paths.get_config')
    def test_init_custom_path(self, mock_get_config, mock_config):
        """Test initialization with custom path."""
        mock_get_config.return_value = mock_config
        custom_path = Path("/custom/mdm")
        
        manager = PathManager(base_path=custom_path)
        
        assert manager.base_path == custom_path
    
    @patch('mdm.utils.paths.get_config')
    def test_datasets_path(self, mock_get_config, mock_config):
        """Test datasets path property."""
        mock_get_config.return_value = mock_config
        base_path = Path("/test/mdm")
        
        manager = PathManager(base_path=base_path)
        
        assert manager.datasets_path == base_path / "datasets"
        mock_config.get_full_path.assert_called_with("datasets_path", base_path)
    
    @patch('mdm.utils.paths.get_config')
    def test_configs_path(self, mock_get_config, mock_config):
        """Test configs path property."""
        mock_get_config.return_value = mock_config
        base_path = Path("/test/mdm")
        
        manager = PathManager(base_path=base_path)
        
        assert manager.configs_path == base_path / "configs"
        mock_config.get_full_path.assert_called_with("configs_path", base_path)
    
    @patch('mdm.utils.paths.get_config')
    def test_logs_path(self, mock_get_config, mock_config):
        """Test logs path property."""
        mock_get_config.return_value = mock_config
        base_path = Path("/test/mdm")
        
        manager = PathManager(base_path=base_path)
        
        assert manager.logs_path == base_path / "logs"
        mock_config.get_full_path.assert_called_with("logs_path", base_path)
    
    @patch('mdm.utils.paths.get_config')
    def test_custom_features_path(self, mock_get_config, mock_config):
        """Test custom features path property."""
        mock_get_config.return_value = mock_config
        base_path = Path("/test/mdm")
        
        manager = PathManager(base_path=base_path)
        
        assert manager.custom_features_path == base_path / "custom_features"
        mock_config.get_full_path.assert_called_with("custom_features_path", base_path)
    
    @patch('mdm.utils.paths.get_config')
    def test_get_dataset_path(self, mock_get_config, mock_config):
        """Test getting dataset path."""
        mock_get_config.return_value = mock_config
        base_path = Path("/test/mdm")
        
        manager = PathManager(base_path=base_path)
        
        # Test with various dataset names
        assert manager.get_dataset_path("TestDataset") == base_path / "datasets" / "testdataset"
        assert manager.get_dataset_path("UPPERCASE") == base_path / "datasets" / "uppercase"
        assert manager.get_dataset_path("mixed_Case_123") == base_path / "datasets" / "mixed_case_123"
    
    @patch('mdm.utils.paths.get_config')
    def test_get_dataset_db_path_sqlite(self, mock_get_config, mock_config):
        """Test getting SQLite database path."""
        mock_get_config.return_value = mock_config
        base_path = Path("/test/mdm")
        
        manager = PathManager(base_path=base_path)
        
        db_path = manager.get_dataset_db_path("test_dataset", "sqlite")
        assert db_path == base_path / "datasets" / "test_dataset" / "dataset.sqlite"
    
    @patch('mdm.utils.paths.get_config')
    def test_get_dataset_db_path_duckdb(self, mock_get_config, mock_config):
        """Test getting DuckDB database path."""
        mock_get_config.return_value = mock_config
        base_path = Path("/test/mdm")
        
        manager = PathManager(base_path=base_path)
        
        db_path = manager.get_dataset_db_path("test_dataset", "duckdb")
        assert db_path == base_path / "datasets" / "test_dataset" / "dataset.duckdb"
    
    @patch('mdm.utils.paths.get_config')
    def test_get_dataset_db_path_postgresql(self, mock_get_config, mock_config):
        """Test getting PostgreSQL info path."""
        mock_get_config.return_value = mock_config
        base_path = Path("/test/mdm")
        
        manager = PathManager(base_path=base_path)
        
        # PostgreSQL doesn't use file path, returns info file
        db_path = manager.get_dataset_db_path("test_dataset", "postgresql")
        assert db_path == base_path / "datasets" / "test_dataset" / "dataset.info"
    
    @patch('mdm.utils.paths.get_config')
    def test_get_dataset_db_path_unknown_backend(self, mock_get_config, mock_config):
        """Test getting database path for unknown backend."""
        mock_get_config.return_value = mock_config
        base_path = Path("/test/mdm")
        
        manager = PathManager(base_path=base_path)
        
        # Unknown backend defaults to info file
        db_path = manager.get_dataset_db_path("test_dataset", "unknown")
        assert db_path == base_path / "datasets" / "test_dataset" / "dataset.info"
    
    @patch('mdm.utils.paths.get_config')
    def test_get_dataset_config_path(self, mock_get_config, mock_config):
        """Test getting dataset config path."""
        mock_get_config.return_value = mock_config
        base_path = Path("/test/mdm")
        
        manager = PathManager(base_path=base_path)
        
        config_path = manager.get_dataset_config_path("TestDataset")
        assert config_path == base_path / "configs" / "testdataset.yaml"
    
    @patch('mdm.utils.paths.get_config')
    def test_ensure_directories(self, mock_get_config, mock_config):
        """Test ensuring directories exist."""
        mock_get_config.return_value = mock_config
        base_path = Path("/test/mdm")
        
        manager = PathManager(base_path=base_path)
        
        # Mock Path.mkdir
        with patch.object(Path, 'mkdir') as mock_mkdir:
            manager.ensure_directories()
            
            # Should create 4 directories
            assert mock_mkdir.call_count == 4
            
            # Check that mkdir was called with correct arguments
            for call in mock_mkdir.call_args_list:
                assert call[1]['parents'] is True
                assert call[1]['exist_ok'] is True


class TestGetPathManager:
    """Test get_path_manager function."""
    
    @patch('mdm.utils.paths.PathManager')
    def test_get_path_manager_singleton(self, mock_path_manager_class):
        """Test that get_path_manager returns singleton instance."""
        # Reset global
        import mdm.utils.paths
        mdm.utils.paths._path_manager = None
        
        mock_instance = Mock()
        mock_path_manager_class.return_value = mock_instance
        
        # First call creates instance
        manager1 = get_path_manager()
        assert manager1 == mock_instance
        mock_path_manager_class.assert_called_once()
        
        # Second call returns same instance
        manager2 = get_path_manager()
        assert manager2 == manager1
        assert mock_path_manager_class.call_count == 1  # Not called again
    
    @patch('mdm.utils.paths.PathManager')
    def test_get_path_manager_reuse_existing(self, mock_path_manager_class):
        """Test reusing existing path manager."""
        import mdm.utils.paths
        
        # Set existing manager
        existing_manager = Mock()
        mdm.utils.paths._path_manager = existing_manager
        
        # Should return existing without creating new
        manager = get_path_manager()
        assert manager == existing_manager
        mock_path_manager_class.assert_not_called()


class TestPathManagerIntegration:
    """Integration tests for PathManager."""
    
    @patch('mdm.utils.paths.get_config')
    def test_path_hierarchy(self, mock_get_config):
        """Test complete path hierarchy."""
        mock_config = Mock()
        mock_config.get_full_path = Mock(side_effect=lambda key, base_path: base_path / key.replace('_path', ''))
        mock_get_config.return_value = mock_config
        
        base_path = Path("/mdm/test")
        manager = PathManager(base_path=base_path)
        
        # Check all paths are under base path
        assert manager.datasets_path.is_relative_to(base_path)
        assert manager.configs_path.is_relative_to(base_path)
        assert manager.logs_path.is_relative_to(base_path)
        assert manager.custom_features_path.is_relative_to(base_path)
        
        # Check dataset-specific paths
        dataset_name = "my_dataset"
        dataset_path = manager.get_dataset_path(dataset_name)
        assert dataset_path.is_relative_to(manager.datasets_path)
        
        # Check database paths
        sqlite_path = manager.get_dataset_db_path(dataset_name, "sqlite")
        assert sqlite_path.is_relative_to(dataset_path)
        assert sqlite_path.suffix == ".sqlite"
        
        duckdb_path = manager.get_dataset_db_path(dataset_name, "duckdb")
        assert duckdb_path.is_relative_to(dataset_path)
        assert duckdb_path.suffix == ".duckdb"
        
        # Check config path
        config_path = manager.get_dataset_config_path(dataset_name)
        assert config_path.is_relative_to(manager.configs_path)
        assert config_path.suffix == ".yaml"
    
    @patch('mdm.utils.paths.get_config')
    def test_case_insensitive_dataset_names(self, mock_get_config):
        """Test that dataset names are case-insensitive."""
        mock_config = Mock()
        mock_config.get_full_path = Mock(side_effect=lambda key, base_path: base_path / key.replace('_path', ''))
        mock_get_config.return_value = mock_config
        
        manager = PathManager()
        
        # All these should result in the same path
        names = ["TestDataset", "TESTDATASET", "testdataset", "TeSt_DaTaSeT"]
        paths = [manager.get_dataset_path(name) for name in names]
        
        # All paths should end with "testdataset" or "test_dataset"
        for path in paths:
            assert path.name in ["testdataset", "test_dataset"]