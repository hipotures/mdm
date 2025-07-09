"""Unit tests for DatasetManager as repository."""

import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
import json
import yaml
import shutil

from mdm.dataset.manager import DatasetManager
from mdm.models.dataset import DatasetInfo
from mdm.models.enums import ProblemType
from mdm.core.exceptions import DatasetError, StorageError


class TestDatasetManager:
    """Test cases for DatasetManager repository functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.paths.datasets_path = "datasets"
        config.paths.configs_path = "config/datasets"
        config.database.default_backend = "sqlite"
        return config

    @pytest.fixture
    def temp_paths(self, tmp_path):
        """Create temporary paths for testing."""
        base_path = tmp_path / "mdm_test"
        datasets_path = base_path / "datasets"
        configs_path = base_path / "config" / "datasets"
        
        datasets_path.mkdir(parents=True)
        configs_path.mkdir(parents=True)
        
        return {
            'base': base_path,
            'datasets': datasets_path,
            'configs': configs_path
        }

    @pytest.fixture
    def manager(self, mock_config, temp_paths):
        """Create DatasetManager instance."""
        with patch('mdm.dataset.manager.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = temp_paths['base']
            mock_get_config.return_value = mock_manager
            
            manager = DatasetManager()
            return manager

    @pytest.fixture
    def sample_dataset_info(self):
        """Create sample DatasetInfo."""
        return DatasetInfo(
            name="test_dataset",
            problem_type="binary_classification",
            target_column="target",
            id_columns=["id"],
            tables={
                "train": "train_table",
                "test": "test_table"
            },
            shape=(1000, 10),
            description="Test dataset",
            tags=["test", "sample"],
            database={"backend": "sqlite"}
        )

    def test_init_creates_directories(self, mock_config):
        """Test that initialization creates necessary directories."""
        with patch('mdm.dataset.manager.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = Path("/test")
            mock_get_config.return_value = mock_manager
            
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                manager = DatasetManager()
        
        # Should create both datasets and configs directories
        assert mock_mkdir.call_count >= 2

    def test_init_with_custom_path(self, mock_config):
        """Test initialization with custom datasets path."""
        custom_path = Path("/custom/datasets")
        
        with patch('mdm.dataset.manager.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = Path("/test")
            mock_get_config.return_value = mock_manager
            
            manager = DatasetManager(datasets_path=custom_path)
        
        assert manager.datasets_path == custom_path

    def test_register_dataset_success(self, manager, sample_dataset_info):
        """Test successful dataset registration."""
        # Act
        manager.register_dataset(sample_dataset_info)
        
        # Assert
        dataset_path = manager.datasets_path / "test_dataset"
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        
        assert dataset_path.exists()
        assert yaml_path.exists()
        assert (dataset_path / "dataset_info.json").exists()
        assert (dataset_path / "metadata").exists()

    def test_register_dataset_already_exists(self, manager, sample_dataset_info):
        """Test error when registering existing dataset."""
        # First registration
        manager.register_dataset(sample_dataset_info)
        
        # Second registration should fail
        with pytest.raises(DatasetError, match="already exists"):
            manager.register_dataset(sample_dataset_info)

    def test_register_dataset_case_insensitive(self, manager, sample_dataset_info):
        """Test that dataset names are case-insensitive."""
        # Register with original name
        manager.register_dataset(sample_dataset_info)
        
        # Try to register with different case
        sample_dataset_info.name = "TEST_DATASET"
        with pytest.raises(DatasetError, match="already exists"):
            manager.register_dataset(sample_dataset_info)

    def test_register_dataset_cleanup_on_failure(self, manager, sample_dataset_info):
        """Test cleanup when registration fails."""
        # Make JSON dump fail
        with patch('builtins.open', side_effect=IOError("Write failed")):
            with pytest.raises(DatasetError, match="Failed to register dataset"):
                manager.register_dataset(sample_dataset_info)
        
        # Dataset directory should not exist
        dataset_path = manager.datasets_path / "test_dataset"
        assert not dataset_path.exists()

    def test_get_dataset_from_yaml(self, manager, sample_dataset_info):
        """Test getting dataset info from YAML registry."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Get dataset
        result = manager.get_dataset("test_dataset")
        
        # Assert
        assert result is not None
        assert result.name == "test_dataset"
        assert result.problem_type == "binary_classification"
        assert result.target_column == "target"

    def test_get_dataset_case_insensitive(self, manager, sample_dataset_info):
        """Test case-insensitive dataset retrieval."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Get with different cases
        assert manager.get_dataset("test_dataset") is not None
        assert manager.get_dataset("TEST_DATASET") is not None
        assert manager.get_dataset("Test_Dataset") is not None

    def test_get_dataset_not_found(self, manager):
        """Test getting non-existent dataset."""
        result = manager.get_dataset("nonexistent")
        assert result is None

    def test_get_dataset_backend_check(self, manager, sample_dataset_info):
        """Test backend compatibility check when getting dataset."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Change current backend
        manager.config.database.default_backend = "duckdb"
        
        # Should still get dataset info but with warning
        with patch('mdm.dataset.manager.logger') as mock_logger:
            result = manager.get_dataset("test_dataset")
        
        assert result is not None
        mock_logger.warning.assert_called_once()

    def test_dataset_exists(self, manager, sample_dataset_info):
        """Test checking if dataset exists."""
        # Before registration
        assert not manager.dataset_exists("test_dataset")
        
        # After registration
        manager.register_dataset(sample_dataset_info)
        assert manager.dataset_exists("test_dataset")
        
        # Case insensitive
        assert manager.dataset_exists("TEST_DATASET")

    def test_list_datasets(self, manager):
        """Test listing all datasets."""
        # Register multiple datasets
        for i in range(3):
            dataset_info = DatasetInfo(
                name=f"dataset_{i}",
                problem_type="binary_classification",
                tables={"train": f"train_{i}"},
                shape=(100*i, 10),
                database={"backend": "sqlite"}
        )
            manager.register_dataset(dataset_info)
        
        # List datasets
        datasets = manager.list_datasets()
        
        # Assert
        assert len(datasets) == 3
        names = [d.name for d in datasets]
        assert "dataset_0" in names
        assert "dataset_1" in names
        assert "dataset_2" in names

    def test_list_datasets_backend_filter(self, manager):
        """Test listing datasets filters by current backend."""
        # Register dataset with sqlite backend
        dataset_info = DatasetInfo(
            name="sqlite_dataset",
            problem_type="binary_classification",
            tables={"train": "train"},
            shape=(100, 10),
            database={"backend": "sqlite"}
        )
        manager.register_dataset(dataset_info)
        
        # Change backend and create dataset with different backend
        yaml_data = {
            'name': 'duckdb_dataset',
            'database': {'backend': 'duckdb'},
            'tables': {'train': 'train_table'}
        }
        yaml_path = manager.dataset_registry_dir / "duckdb_dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)
        
        # List should only show sqlite dataset
        datasets = manager.list_datasets()
        assert len(datasets) == 1
        assert datasets[0].name == "sqlite_dataset"

    def test_update_dataset(self, manager, sample_dataset_info):
        """Test updating dataset metadata."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Update dataset
        updates = {
            'description': 'Updated description',
            'tags': ['updated', 'test']
        }
        manager.update_dataset("test_dataset", updates)
        
        # Verify updates
        updated = manager.get_dataset("test_dataset")
        assert updated.description == 'Updated description'
        assert updated.tags == ['updated', 'test']

    def test_update_dataset_not_found(self, manager):
        """Test updating non-existent dataset."""
        with pytest.raises(DatasetError, match="not found"):
            manager.update_dataset("nonexistent", {'description': 'new'})

    def test_save_dataset(self, manager, sample_dataset_info):
        """Test saving dataset info (used by registrar)."""
        # Act
        manager.save_dataset(sample_dataset_info)
        
        # Assert - only saves to YAML registry
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        assert yaml_path.exists()
        
        # Verify content
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert data['name'] == 'test_dataset'
        assert data['problem_type'] == 'classification'

    def test_get_backend(self, manager, sample_dataset_info):
        """Test getting storage backend for dataset."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Get backend
        with patch('mdm.dataset.manager.BackendFactory') as mock_factory:
            mock_backend = Mock()
            mock_factory.create.return_value = mock_backend
            
            result = manager.get_backend("test_dataset")
        
        assert result == mock_backend
        mock_factory.create.assert_called_once()

    def test_get_backend_dataset_not_found(self, manager):
        """Test error when getting backend for non-existent dataset."""
        with pytest.raises(DatasetError, match="not found"):
            manager.get_backend("nonexistent")

    def test_yaml_serialization(self, manager, sample_dataset_info):
        """Test proper YAML serialization of dataset info."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Read YAML file
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        # Check serialization
        assert isinstance(data['tables'], dict)
        assert 'train' in data['tables']  # Enum key serialized as string
        assert data['shape'] == [1000, 10]  # Tuple serialized as list
        assert data['problem_type'] == 'classification'  # Enum serialized as string