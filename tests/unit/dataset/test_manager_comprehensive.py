"""Comprehensive unit tests for DatasetManager to achieve 80%+ coverage."""

import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
import json
import yaml
from datetime import datetime, timezone

from mdm.dataset.manager import DatasetManager
from mdm.models.dataset import DatasetInfo, DatasetStatistics
from mdm.models.enums import ProblemType
from mdm.core.exceptions import DatasetError, StorageError


class TestDatasetManagerComprehensive:
    """Comprehensive test cases for DatasetManager."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.paths.datasets_path = "datasets"
        config.paths.configs_path = "config/datasets"
        config.database.default_backend = "sqlite"
        
        # Add backend configs
        sqlite_config = Mock()
        sqlite_config.model_dump.return_value = {"path": "/tmp/test.db"}
        config.database.sqlite = sqlite_config
        
        duckdb_config = Mock()
        duckdb_config.model_dump.return_value = {"path": "/tmp/test.duckdb"}
        config.database.duckdb = duckdb_config
        
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
        with patch('mdm.config.get_config_manager') as mock_get_config:
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
            description="Test dataset",
            tags=["test", "sample"],
            database={"backend": "sqlite", "path": "/path/to/db.db"},
            last_updated_at=datetime.now(timezone.utc)
        )

    @pytest.fixture
    def sample_statistics(self):
        """Create sample DatasetStatistics."""
        return DatasetStatistics(
            row_count=1000,
            column_count=10,
            table_count=2,
            memory_usage_mb=100.5,
            disk_size_mb=50.2,
            features={
                "train": ["feature1", "feature2"],
                "test": ["feature1"]
            },
            computed_at=datetime.now(timezone.utc)
        )

    def test_delete_dataset_success(self, manager, sample_dataset_info):
        """Test successful dataset deletion."""
        # Register dataset first
        manager.register_dataset(sample_dataset_info)
        
        # Delete dataset
        manager.delete_dataset("test_dataset", force=True)
        
        # Verify deletion
        assert not manager.dataset_exists("test_dataset")
        assert not (manager.datasets_path / "test_dataset").exists()
        assert not (manager.dataset_registry_dir / "test_dataset.yaml").exists()

    def test_delete_dataset_not_found(self, manager):
        """Test deleting non-existent dataset."""
        with pytest.raises(DatasetError, match="not found"):
            manager.delete_dataset("nonexistent", force=True)

    def test_delete_dataset_without_force(self, manager, sample_dataset_info):
        """Test dataset deletion without force flag."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Delete without force (should log warning but still delete)
        with patch('mdm.dataset.manager.logger.warning') as mock_warning:
            manager.delete_dataset("test_dataset", force=False)
            mock_warning.assert_called_once()
        
        # Should still be deleted
        assert not manager.dataset_exists("test_dataset")

    def test_delete_dataset_cleanup_error(self, manager, sample_dataset_info):
        """Test error handling during dataset deletion."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Make shutil.rmtree fail
        with patch('shutil.rmtree', side_effect=OSError("Permission denied")):
            with pytest.raises(DatasetError, match="Failed to delete dataset"):
                manager.delete_dataset("test_dataset", force=True)

    def test_remove_dataset_alias(self, manager, sample_dataset_info):
        """Test remove_dataset as alias for delete_dataset."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Remove dataset (should call delete with force=True)
        manager.remove_dataset("test_dataset")
        
        # Verify removal
        assert not manager.dataset_exists("test_dataset")

    def test_save_statistics_success(self, manager, sample_dataset_info, sample_statistics):
        """Test saving dataset statistics."""
        # Register dataset first
        manager.register_dataset(sample_dataset_info)
        
        # Save statistics
        manager.save_statistics("test_dataset", sample_statistics)
        
        # Verify file was created
        stats_path = manager.datasets_path / "test_dataset" / "metadata" / "statistics.json"
        assert stats_path.exists()
        
        # Verify content
        with open(stats_path) as f:
            data = json.load(f)
        assert data['row_count'] == 1000
        assert data['column_count'] == 10

    def test_save_statistics_dataset_not_found(self, manager, sample_statistics):
        """Test saving statistics for non-existent dataset."""
        with pytest.raises(DatasetError, match="not found"):
            manager.save_statistics("nonexistent", sample_statistics)

    def test_save_statistics_write_error(self, manager, sample_dataset_info, sample_statistics):
        """Test error handling when saving statistics fails."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Make write fail
        with patch('builtins.open', side_effect=IOError("Write failed")):
            with pytest.raises(DatasetError, match="Failed to save statistics"):
                manager.save_statistics("test_dataset", sample_statistics)

    def test_get_statistics_success(self, manager, sample_dataset_info, sample_statistics):
        """Test retrieving dataset statistics."""
        # Register dataset and save statistics
        manager.register_dataset(sample_dataset_info)
        manager.save_statistics("test_dataset", sample_statistics)
        
        # Get statistics
        result = manager.get_statistics("test_dataset")
        
        # Verify
        assert result is not None
        assert result.row_count == 1000
        assert result.column_count == 10

    def test_get_statistics_not_found(self, manager, sample_dataset_info):
        """Test getting statistics when file doesn't exist."""
        # Register dataset but don't save statistics
        manager.register_dataset(sample_dataset_info)
        
        result = manager.get_statistics("test_dataset")
        assert result is None

    def test_get_statistics_parse_error(self, manager, sample_dataset_info):
        """Test error handling when statistics file is corrupted."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Create corrupted statistics file
        stats_path = manager.datasets_path / "test_dataset" / "metadata" / "statistics.json"
        stats_path.write_text("invalid json")
        
        # Should return None and log error
        with patch('mdm.dataset.manager.logger.error') as mock_error:
            result = manager.get_statistics("test_dataset")
            assert result is None
            mock_error.assert_called_once()

    def test_get_backend_with_custom_config(self, manager, sample_dataset_info):
        """Test getting backend with dataset-specific configuration."""
        # Modify dataset info to have custom backend config
        sample_dataset_info.database = {
            "backend": "sqlite",
            "path": "/custom/path.db",
            "custom_param": "value"
        }
        
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Get backend
        with patch('mdm.dataset.manager.get_config') as mock_get_config:
            mock_get_config.return_value = manager.config
            
            with patch('mdm.dataset.manager.BackendFactory') as mock_factory:
                mock_backend = Mock()
                mock_factory.create.return_value = mock_backend
                
                result = manager.get_backend("test_dataset")
        
        # Verify custom config was merged
        mock_factory.create.assert_called_once()
        call_args = mock_factory.create.call_args[0]
        assert call_args[0] == "sqlite"  # backend type
        assert call_args[1]["path"] == "/custom/path.db"  # custom path
        assert call_args[1]["custom_param"] == "value"  # custom param

    def test_get_backend_factory_error(self, manager, sample_dataset_info):
        """Test error handling when backend creation fails."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        with patch('mdm.dataset.manager.get_config') as mock_get_config:
            mock_get_config.return_value = manager.config
            
            with patch('mdm.dataset.manager.BackendFactory') as mock_factory:
                mock_factory.create.side_effect = Exception("Backend error")
                
                with pytest.raises(StorageError, match="Failed to create backend"):
                    manager.get_backend("test_dataset")

    def test_validate_dataset_name_valid(self, manager):
        """Test validation of valid dataset names."""
        valid_names = [
            "dataset1",
            "my-dataset",
            "test_data_123",
            "a" * 100  # Max length
        ]
        
        for name in valid_names:
            result = manager.validate_dataset_name(name)
            assert result == name.lower()

    def test_validate_dataset_name_empty(self, manager):
        """Test validation of empty dataset name."""
        with pytest.raises(DatasetError, match="cannot be empty"):
            manager.validate_dataset_name("")

    def test_validate_dataset_name_invalid_chars(self, manager):
        """Test validation of dataset names with invalid characters."""
        invalid_names = [
            "dataset with spaces",
            "dataset@special",
            "dataset.com",
            "dataset/path"
        ]
        
        for name in invalid_names:
            with pytest.raises(DatasetError, match="can only contain alphanumeric"):
                manager.validate_dataset_name(name)

    def test_validate_dataset_name_too_long(self, manager):
        """Test validation of dataset name exceeding max length."""
        long_name = "a" * 101
        with pytest.raises(DatasetError, match="cannot exceed 100 characters"):
            manager.validate_dataset_name(long_name)

    def test_export_metadata_json(self, manager, sample_dataset_info, sample_statistics):
        """Test exporting metadata to JSON file."""
        # Register dataset and save statistics
        manager.register_dataset(sample_dataset_info)
        manager.save_statistics("test_dataset", sample_statistics)
        
        # Export metadata
        output_path = manager.datasets_path / "export.json"
        manager.export_metadata("test_dataset", output_path)
        
        # Verify export
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        
        assert 'dataset_info' in data
        assert 'statistics' in data
        assert data['dataset_info']['name'] == 'test_dataset'
        assert data['statistics']['row_count'] == 1000

    def test_export_metadata_yaml(self, manager, sample_dataset_info):
        """Test exporting metadata to YAML file."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Export metadata
        output_path = manager.datasets_path / "export.yaml"
        manager.export_metadata("test_dataset", output_path)
        
        # Verify export
        assert output_path.exists()
        with open(output_path) as f:
            data = yaml.safe_load(f)
        
        assert 'dataset_info' in data
        assert data['dataset_info']['name'] == 'test_dataset'

    def test_export_metadata_dataset_not_found(self, manager):
        """Test exporting metadata for non-existent dataset."""
        output_path = Path("/tmp/export.json")
        with pytest.raises(DatasetError, match="not found"):
            manager.export_metadata("nonexistent", output_path)

    def test_export_metadata_write_error(self, manager, sample_dataset_info):
        """Test error handling when export fails."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Make write fail only for the export, not for reading
        output_path = Path("/tmp/export.json")
        
        # Mock to allow reads but fail on write
        original_open = open
        def mock_open_func(file, mode='r', *args, **kwargs):
            if 'w' in mode and str(file) == str(output_path):
                raise IOError("Write failed")
            return original_open(file, mode, *args, **kwargs)
        
        with patch('builtins.open', side_effect=mock_open_func):
            with pytest.raises(DatasetError, match="Failed to export metadata"):
                manager.export_metadata("test_dataset", output_path)

    def test_search_datasets_filename_match(self, manager):
        """Test searching datasets by filename."""
        # Register multiple datasets
        for name in ["test_dataset", "prod_dataset", "test_backup"]:
            dataset_info = DatasetInfo(
                name=name,
                problem_type="binary_classification",
                tables={"train": "train"},
                database={"backend": "sqlite"}
            )
            manager.register_dataset(dataset_info)
        
        # Search for "test"
        results = manager.search_datasets("test")
        
        # Should find 2 matches
        assert len(results) == 2
        names = [d.name for d in results]
        assert "test_dataset" in names
        assert "test_backup" in names

    def test_search_datasets_content_match(self, manager, sample_dataset_info):
        """Test searching datasets by content."""
        # Register dataset with specific content
        sample_dataset_info.description = "This is a classification dataset for testing"
        sample_dataset_info.tags = ["binary", "classification"]
        manager.register_dataset(sample_dataset_info)
        
        # Search for content
        results = manager.search_datasets("classification")
        
        # Should find the dataset
        assert len(results) == 1
        assert results[0].name == "test_dataset"

    def test_search_datasets_case_sensitive(self, manager):
        """Test case-sensitive search."""
        # Register dataset
        dataset_info = DatasetInfo(
            name="TestDataset",
            problem_type="binary_classification",
            tables={"train": "train"},
            database={"backend": "sqlite"}
        )
        manager.register_dataset(dataset_info)
        
        # Case-insensitive search (default)
        results = manager.search_datasets("testdataset", case_sensitive=False)
        assert len(results) == 1
        
        # Case-sensitive search
        # Dataset name is normalized to lowercase, so "testdataset" should find it
        results = manager.search_datasets("testdataset", case_sensitive=True)
        assert len(results) == 1
        
        # But "TestDataset" should not find it with case-sensitive search
        results = manager.search_datasets("TestDataset", case_sensitive=True)
        assert len(results) == 0  # Name is stored as lowercase

    def test_search_datasets_error_handling(self, manager, sample_dataset_info):
        """Test error handling during search."""
        # Register dataset
        manager.register_dataset(sample_dataset_info)
        
        # Make file reading fail for one dataset
        with patch('builtins.open', side_effect=[
            mock_open(read_data="corrupted")(),  # First file fails
            mock_open(read_data=yaml.dump(sample_dataset_info.model_dump()))()  # Second succeeds
        ]):
            with patch('mdm.dataset.manager.logger.error') as mock_error:
                results = manager.search_datasets("test")
                # Should log error but continue
                mock_error.assert_called()

    def test_get_dataset_json_fallback(self, manager, sample_dataset_info):
        """Test falling back to JSON when YAML doesn't exist."""
        # Create dataset directory with JSON only
        dataset_name = sample_dataset_info.name.lower()
        dataset_path = manager.datasets_path / dataset_name
        dataset_path.mkdir(parents=True)
        
        # Save JSON file
        info_path = dataset_path / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(sample_dataset_info.model_dump(), f, default=str)
        
        # Get dataset (should fall back to JSON)
        result = manager.get_dataset(dataset_name)
        
        assert result is not None
        assert result.name == dataset_name

    def test_get_dataset_json_error(self, manager):
        """Test error handling when JSON loading fails."""
        # Create dataset directory with corrupted JSON
        dataset_path = manager.datasets_path / "test_dataset"
        dataset_path.mkdir(parents=True)
        
        info_path = dataset_path / "dataset_info.json"
        info_path.write_text("invalid json")
        
        # Should return None and log error
        with patch('mdm.dataset.manager.logger.error') as mock_error:
            result = manager.get_dataset("test_dataset")
            assert result is None
            mock_error.assert_called()

    def test_update_dataset_json_backward_compatibility(self, manager):
        """Test updating dataset with JSON file present."""
        # Create dataset with JSON file
        dataset_path = manager.datasets_path / "test_dataset"
        dataset_path.mkdir(parents=True)
        
        dataset_info = DatasetInfo(
            name="test_dataset",
            problem_type="binary_classification",
            tables={"train": "train"},
            database={"backend": "sqlite"}
        )
        
        # Save JSON
        info_path = dataset_path / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info.model_dump(), f, default=str)
        
        # Also create YAML
        yaml_path = manager.dataset_registry_dir / "test_dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_info.model_dump(), f)
        
        # Update dataset
        updates = {'description': 'Updated'}
        result = manager.update_dataset("test_dataset", updates)
        
        # Both files should be updated
        assert result.description == 'Updated'
        
        # Check JSON was updated
        with open(info_path) as f:
            json_data = json.load(f)
        assert json_data['description'] == 'Updated'

    def test_list_datasets_from_directories(self, manager):
        """Test listing datasets from directories (backward compatibility)."""
        # Create dataset directory without YAML
        dataset_path = manager.datasets_path / "legacy_dataset"
        dataset_path.mkdir(parents=True)
        
        dataset_info = DatasetInfo(
            name="legacy_dataset",
            problem_type="binary_classification",
            tables={"train": "train"},
            database={"backend": "sqlite"}
        )
        
        # Save only JSON
        info_path = dataset_path / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info.model_dump(), f, default=str)
        
        # List should find the dataset
        datasets = manager.list_datasets()
        names = [d.name for d in datasets]
        assert "legacy_dataset" in names

    def test_list_datasets_error_handling(self, manager):
        """Test error handling when listing datasets."""
        # Create corrupted YAML file
        yaml_path = manager.dataset_registry_dir / "corrupted.yaml"
        yaml_path.write_text("invalid: yaml: content:")
        
        # Should log error but continue
        with patch('mdm.dataset.manager.logger.error') as mock_error:
            datasets = manager.list_datasets()
            mock_error.assert_called()
        
        # Should return empty list or skip corrupted file
        assert isinstance(datasets, list)