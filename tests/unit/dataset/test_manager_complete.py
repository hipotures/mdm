"""Comprehensive unit tests for DatasetManager."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from mdm.core.exceptions import DatasetError
from mdm.dataset.manager import DatasetManager
from mdm.models.dataset import DatasetInfo
from mdm.models.enums import ProblemType


class TestDatasetManagerComplete:
    """Comprehensive test coverage for DatasetManager."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.paths.datasets_path = "datasets/"
        config.paths.configs_path = "config/datasets/"
        return config

    @pytest.fixture
    def mock_config_manager(self, mock_config, tmp_path):
        """Create mock config manager."""
        manager = Mock()
        manager.config = mock_config
        manager.base_path = tmp_path
        return manager

    @pytest.fixture
    def manager(self, mock_config_manager):
        """Create DatasetManager instance."""
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            return DatasetManager()

    @pytest.fixture
    def sample_dataset_info(self):
        """Create sample DatasetInfo."""
        return DatasetInfo(
            name="test_dataset",
            description="Test dataset for unit tests",
            source_path="/data/test",
            database={"backend": "sqlite", "path": "test.db"},
            tables={"train": "test_dataset_train"},
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            target_column="target",
            tags=["test", "sample"],
            created_at=datetime.now(timezone.utc),
            last_updated_at=datetime.now(timezone.utc)
        )

    def test_init_default_paths(self, mock_config_manager):
        """Test initialization with default paths."""
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            manager = DatasetManager()
            
            assert manager.datasets_path == mock_config_manager.base_path / "datasets/"
            assert manager.dataset_registry_dir == mock_config_manager.base_path / "config/datasets/"
            
            # Check directories were created
            assert manager.datasets_path.exists()
            assert manager.dataset_registry_dir.exists()

    def test_init_custom_path(self, mock_config_manager, tmp_path):
        """Test initialization with custom datasets path."""
        custom_path = tmp_path / "custom_datasets"
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            manager = DatasetManager(datasets_path=custom_path)
            
            assert manager.datasets_path == custom_path
            assert custom_path.exists()

    def test_register_dataset_success(self, manager, sample_dataset_info):
        """Test successful dataset registration."""
        manager.register_dataset(sample_dataset_info)
        
        # Check dataset directory was created
        dataset_path = manager.datasets_path / "test_dataset"
        assert dataset_path.exists()
        
        # Check JSON file was created
        json_path = dataset_path / "dataset_info.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
            assert data["name"] == "test_dataset"
        
        # Check YAML file was created
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        assert yaml_path.exists()
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            assert data["name"] == "test_dataset"
        
        # Check metadata directory was created
        metadata_path = dataset_path / "metadata"
        assert metadata_path.exists()

    def test_register_dataset_already_exists(self, manager, sample_dataset_info):
        """Test registering dataset that already exists."""
        # Create YAML file to simulate existing dataset
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        yaml_path.write_text("name: test_dataset\n")
        
        with pytest.raises(DatasetError, match="already exists"):
            manager.register_dataset(sample_dataset_info)

    def test_register_dataset_cleanup_on_failure(self, manager, sample_dataset_info):
        """Test that dataset directory is cleaned up on registration failure."""
        dataset_path = manager.datasets_path / "test_dataset"
        
        with patch('json.dump', side_effect=Exception("Write failed")):
            with pytest.raises(DatasetError, match="Failed to register dataset"):
                manager.register_dataset(sample_dataset_info)
        
        # Dataset directory should be cleaned up
        assert not dataset_path.exists()

    def test_get_dataset_from_yaml(self, manager, sample_dataset_info):
        """Test getting dataset from YAML registry."""
        # Create YAML file
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(sample_dataset_info.model_dump(), f)
        
        result = manager.get_dataset("test_dataset")
        
        assert result is not None
        assert result.name == "test_dataset"
        assert result.description == "Test dataset for unit tests"

    def test_get_dataset_from_json_fallback(self, manager, sample_dataset_info):
        """Test getting dataset from JSON when YAML doesn't exist."""
        # Create JSON file
        dataset_path = manager.datasets_path / "test_dataset"
        dataset_path.mkdir(parents=True)
        json_path = dataset_path / "dataset_info.json"
        
        with open(json_path, 'w') as f:
            json.dump(sample_dataset_info.model_dump(), f, default=str)
        
        result = manager.get_dataset("test_dataset")
        
        assert result is not None
        assert result.name == "test_dataset"

    def test_get_dataset_case_insensitive(self, manager, sample_dataset_info):
        """Test that dataset retrieval is case-insensitive."""
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(sample_dataset_info.model_dump(), f)
        
        # Try different cases
        assert manager.get_dataset("TEST_DATASET") is not None
        assert manager.get_dataset("Test_Dataset") is not None
        assert manager.get_dataset("test_dataset") is not None

    def test_get_dataset_not_found(self, manager):
        """Test getting non-existent dataset."""
        result = manager.get_dataset("nonexistent")
        assert result is None

    def test_get_dataset_yaml_error(self, manager):
        """Test handling YAML load error."""
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        yaml_path.write_text("invalid: yaml: content:")
        
        with patch('mdm.dataset.manager.logger') as mock_logger:
            result = manager.get_dataset("test_dataset")
            assert result is None
            mock_logger.error.assert_called()

    def test_update_dataset_success(self, manager, sample_dataset_info):
        """Test successful dataset update."""
        # Register dataset first
        manager.register_dataset(sample_dataset_info)
        
        # Update dataset
        updates = {
            "description": "Updated description",
            "tags": ["test", "updated"],
            "target_column": "new_target"
        }
        
        result = manager.update_dataset("test_dataset", updates)
        
        assert result.description == "Updated description"
        assert result.tags == ["test", "updated"]
        assert result.target_column == "new_target"
        assert result.last_updated_at > sample_dataset_info.last_updated_at
        
        # Check both files were updated
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
            assert yaml_data["description"] == "Updated description"

    def test_update_dataset_not_found(self, manager):
        """Test updating non-existent dataset."""
        with pytest.raises(DatasetError, match="not found"):
            manager.update_dataset("nonexistent", {"description": "test"})

    def test_update_dataset_error(self, manager, sample_dataset_info):
        """Test handling update error."""
        manager.register_dataset(sample_dataset_info)
        
        with patch('yaml.dump', side_effect=Exception("Write failed")):
            with pytest.raises(DatasetError, match="Failed to update dataset"):
                manager.update_dataset("test_dataset", {"description": "test"})

    def test_list_datasets_empty(self, manager):
        """Test listing datasets when none exist."""
        result = manager.list_datasets()
        assert result == []

    def test_list_datasets_from_yaml(self, manager, sample_dataset_info):
        """Test listing datasets from YAML registry."""
        # Create multiple datasets
        for i in range(3):
            dataset_info = sample_dataset_info.model_copy()
            dataset_info.name = f"dataset_{i}"
            yaml_path = manager.dataset_registry_dir / f"dataset_{i}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(dataset_info.model_dump(), f)
        
        result = manager.list_datasets()
        
        assert len(result) == 3
        assert all(d.name.startswith("dataset_") for d in result)
        # Check sorted by name
        assert [d.name for d in result] == ["dataset_0", "dataset_1", "dataset_2"]

    def test_list_datasets_mixed_sources(self, manager, sample_dataset_info):
        """Test listing datasets from both YAML and JSON sources."""
        # Create YAML dataset
        yaml_dataset = sample_dataset_info.model_copy()
        yaml_dataset.name = "yaml_dataset"
        yaml_path = manager.dataset_registry_dir / "yaml_dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_dataset.model_dump(), f)
        
        # Create JSON-only dataset
        json_dataset = sample_dataset_info.model_copy()
        json_dataset.name = "json_dataset"
        dataset_path = manager.datasets_path / "json_dataset"
        dataset_path.mkdir(parents=True)
        json_path = dataset_path / "dataset_info.json"
        with open(json_path, 'w') as f:
            json.dump(json_dataset.model_dump(), f, default=str)
        
        result = manager.list_datasets()
        
        assert len(result) == 2
        assert {d.name for d in result} == {"yaml_dataset", "json_dataset"}

    def test_list_datasets_error_handling(self, manager):
        """Test error handling when loading datasets."""
        # Create invalid YAML file
        yaml_path = manager.dataset_registry_dir / "invalid.yaml"
        yaml_path.write_text("invalid: yaml: content:")
        
        with patch('mdm.dataset.manager.logger') as mock_logger:
            result = manager.list_datasets()
            assert result == []
            mock_logger.error.assert_called()

    def test_dataset_exists_yaml(self, manager):
        """Test checking if dataset exists via YAML."""
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        yaml_path.write_text("name: test_dataset\n")
        
        assert manager.dataset_exists("test_dataset")
        assert manager.dataset_exists("TEST_DATASET")  # Case insensitive
        assert not manager.dataset_exists("nonexistent")

    def test_dataset_exists_directory(self, manager):
        """Test checking if dataset exists via directory."""
        dataset_path = manager.datasets_path / "test_dataset"
        dataset_path.mkdir(parents=True)
        
        assert manager.dataset_exists("test_dataset")

    def test_delete_dataset_success(self, manager, sample_dataset_info):
        """Test successful dataset deletion."""
        # Register dataset first
        manager.register_dataset(sample_dataset_info)
        
        # Delete dataset
        manager.delete_dataset("test_dataset", force=True)
        
        # Check everything was removed
        assert not (manager.datasets_path / "test_dataset").exists()
        assert not (manager.dataset_registry_dir / "test_dataset.yaml").exists()

    def test_delete_dataset_not_found(self, manager):
        """Test deleting non-existent dataset."""
        with pytest.raises(DatasetError, match="not found"):
            manager.delete_dataset("nonexistent")

    def test_delete_dataset_yaml_only(self, manager):
        """Test deleting dataset when only YAML exists."""
        # Create dataset directory and YAML
        dataset_path = manager.datasets_path / "test_dataset"
        dataset_path.mkdir(parents=True)
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        yaml_path.write_text("name: test_dataset\n")
        
        manager.delete_dataset("test_dataset", force=True)
        
        assert not dataset_path.exists()
        assert not yaml_path.exists()

    def test_delete_dataset_error(self, manager, sample_dataset_info):
        """Test handling deletion error."""
        manager.register_dataset(sample_dataset_info)
        
        with patch('shutil.rmtree', side_effect=Exception("Delete failed")):
            with pytest.raises(DatasetError, match="Failed to delete dataset"):
                manager.delete_dataset("test_dataset", force=True)

    def test_remove_dataset_alias(self, manager, sample_dataset_info):
        """Test that remove_dataset is an alias for delete_dataset."""
        manager.register_dataset(sample_dataset_info)
        
        # Use remove_dataset
        manager.remove_dataset("test_dataset")
        
        # Dataset should be gone
        assert not manager.dataset_exists("test_dataset")

    def test_search_datasets(self, manager, sample_dataset_info):
        """Test searching datasets by pattern."""
        # Create datasets with different names
        for name in ["test_dataset", "prod_dataset", "test_data", "sample"]:
            dataset = sample_dataset_info.model_copy()
            dataset.name = name
            yaml_path = manager.dataset_registry_dir / f"{name}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(dataset.model_dump(), f)
        
        # Search by pattern - the search currently returns all datasets that contain "test" in their content
        result = manager.search_datasets("test")
        # All datasets contain "test" in their description or tags
        assert len(result) == 4
        assert all("test" in str(d.model_dump()).lower() for d in result)

    def test_search_datasets_by_tag(self, manager, sample_dataset_info):
        """Test searching datasets by tag."""
        # Create datasets with different tags
        tags_map = {
            "dataset1": ["prod", "ml"],
            "dataset2": ["test", "ml"],
            "dataset3": ["prod", "etl"]
        }
        
        for name, tags in tags_map.items():
            dataset = sample_dataset_info.model_copy()
            dataset.name = name
            dataset.tags = tags
            yaml_path = manager.dataset_registry_dir / f"{name}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(dataset.model_dump(), f)
        
        # Search by tag
        result = manager.search_datasets_by_tag("ml")
        assert len(result) == 2
        assert {d.name for d in result} == {"dataset1", "dataset2"}

    def test_get_statistics(self, manager, sample_dataset_info):
        """Test getting dataset statistics."""
        manager.register_dataset(sample_dataset_info)
        
        # Create statistics file
        stats_dir = manager.datasets_path / "test_dataset" / "metadata"
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_path = stats_dir / "statistics.json"
        stats = {
            "row_count": 1000,
            "column_count": 10,
            "memory_usage_mb": 50.0,
            "computed_at": "2024-01-01T00:00:00Z"
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
        
        result = manager.get_statistics("test_dataset")
        
        assert result is not None
        assert result.row_count == 1000
        assert result.memory_usage_mb == 50.0

    def test_validate_dataset_name(self, manager):
        """Test dataset name validation."""
        # Valid names
        assert manager.validate_dataset_name("test_dataset") == "test_dataset"
        assert manager.validate_dataset_name("Test-Dataset") == "test_dataset"
        assert manager.validate_dataset_name("TEST_DATASET") == "test_dataset"
        
        # Invalid names
        with pytest.raises(DatasetError, match="Dataset name cannot be empty"):
            manager.validate_dataset_name("")
        
        with pytest.raises(DatasetError, match="can only contain alphanumeric"):
            manager.validate_dataset_name("test dataset")  # Space
        
        with pytest.raises(DatasetError, match="can only contain alphanumeric"):
            manager.validate_dataset_name("test/dataset")  # Slash