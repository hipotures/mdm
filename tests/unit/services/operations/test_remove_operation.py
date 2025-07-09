"""Unit tests for RemoveOperation."""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import yaml
import shutil

from mdm.dataset.operations import RemoveOperation
from mdm.core.exceptions import DatasetError


class TestRemoveOperation:
    """Test cases for RemoveOperation."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.paths.configs_path = "config/datasets"
        config.paths.datasets_path = "datasets"
        return config

    @pytest.fixture
    def remove_operation(self, mock_config):
        """Create RemoveOperation instance."""
        with patch('mdm.config.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = Path("/test")
            mock_get_config.return_value = mock_manager
            
            operation = RemoveOperation()
            operation.dataset_registry_dir = Path("/test/config/datasets")
            operation.datasets_dir = Path("/test/datasets")
            return operation

    @pytest.fixture
    def sample_dataset_data(self):
        """Sample dataset YAML data."""
        return {
            'name': 'test_dataset',
            'database': {'backend': 'sqlite'},
            'tables': {'train': 'train_table'}
        }

    def test_execute_dataset_not_found(self, remove_operation):
        """Test error when dataset doesn't exist."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=False):
            # Act & Assert
            with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
                remove_operation.execute("nonexistent")

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.rglob')
    def test_execute_dry_run(self, mock_rglob, mock_exists, remove_operation, sample_dataset_data):
        """Test dry run mode."""
        # Arrange
        yaml_file = remove_operation.dataset_registry_dir / "test_dataset.yaml"
        dataset_dir = remove_operation.datasets_dir / "test_dataset"
        
        yaml_content = yaml.dump(sample_dataset_data)
        
        # Setup exists mock
        def exists_side_effect(self):
            return str(self) == str(yaml_file) or str(self) == str(dataset_dir)
        mock_exists.side_effect = exists_side_effect
        
        # Setup rglob mock
        mock_file1 = Mock()
        mock_file1.is_file.return_value = True
        mock_file1.stat.return_value.st_size = 1000
        
        mock_file2 = Mock()
        mock_file2.is_file.return_value = True
        mock_file2.stat.return_value.st_size = 2000
        
        def rglob_side_effect(self, pattern):
            if str(self) == str(dataset_dir):
                return [mock_file1, mock_file2]
            return []
        mock_rglob.side_effect = rglob_side_effect
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            # Act
            result = remove_operation.execute("test_dataset", dry_run=True)

        # Assert
        assert result['name'] == 'test_dataset'
        assert result['config_file'] == str(yaml_file)
        assert result['dataset_directory'] == str(dataset_dir)
        assert result['size'] == 3000
        assert result['dry_run'] is True

    def test_execute_postgresql_dataset(self, remove_operation):
        """Test removal info for PostgreSQL dataset."""
        # Arrange
        yaml_file = remove_operation.dataset_registry_dir / "pg_dataset.yaml"
        dataset_dir = remove_operation.datasets_dir / "pg_dataset"
        
        pg_data = {
            'name': 'pg_dataset',
            'database': {
                'backend': 'postgresql',
                'database_prefix': 'mdm_test_'
            }
        }
        yaml_content = yaml.dump(pg_data)
        
        with patch.object(yaml_file, 'exists', return_value=True):
            with patch.object(dataset_dir, 'exists', return_value=False):
                with patch('builtins.open', mock_open(read_data=yaml_content)):
                    # Act
                    result = remove_operation.execute("pg_dataset", dry_run=True)

        # Assert
        assert result['postgresql_db'] == 'mdm_test_pg_dataset'

    def test_execute_remove_success(self, remove_operation, sample_dataset_data):
        """Test successful dataset removal."""
        # Arrange
        yaml_file = remove_operation.dataset_registry_dir / "test_dataset.yaml"
        dataset_dir = remove_operation.datasets_dir / "test_dataset"
        
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch.object(yaml_file, 'exists', return_value=True):
            with patch.object(dataset_dir, 'exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=yaml_content)):
                    with patch.object(yaml_file, 'unlink') as mock_unlink:
                        with patch('shutil.rmtree') as mock_rmtree:
                            with patch('mdm.dataset.operations.logger') as mock_logger:
                                # Act
                                result = remove_operation.execute("test_dataset", force=True)

        # Assert
        assert result['removed'] is True
        mock_unlink.assert_called_once()  # YAML file removed
        mock_rmtree.assert_called_once_with(dataset_dir)  # Directory removed
        mock_logger.info.assert_called()

    def test_execute_remove_yaml_only(self, remove_operation, sample_dataset_data):
        """Test removal when only YAML exists (no data directory)."""
        # Arrange
        yaml_file = remove_operation.dataset_registry_dir / "test_dataset.yaml"
        dataset_dir = remove_operation.datasets_dir / "test_dataset"
        
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch.object(yaml_file, 'exists', return_value=True):
            with patch.object(dataset_dir, 'exists', return_value=False):
                with patch('builtins.open', mock_open(read_data=yaml_content)):
                    with patch.object(yaml_file, 'unlink') as mock_unlink:
                        # Act
                        result = remove_operation.execute("test_dataset", force=True)

        # Assert
        assert result['removed'] is True
        assert result['dataset_directory'] is None
        mock_unlink.assert_called_once()

    def test_execute_remove_yaml_error(self, remove_operation, sample_dataset_data):
        """Test error recovery when YAML removal fails."""
        # Arrange
        yaml_file = remove_operation.dataset_registry_dir / "test_dataset.yaml"
        dataset_dir = remove_operation.datasets_dir / "test_dataset"
        
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch.object(yaml_file, 'exists', return_value=True):
            with patch.object(dataset_dir, 'exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=yaml_content)):
                    with patch.object(yaml_file, 'unlink', side_effect=OSError("Permission denied")):
                        # Act & Assert
                        with pytest.raises(DatasetError, match="Failed to remove dataset config"):
                            remove_operation.execute("test_dataset", force=True)

    def test_execute_remove_directory_error(self, remove_operation, sample_dataset_data):
        """Test that directory removal errors cause the operation to fail."""
        # Arrange
        yaml_file = remove_operation.dataset_registry_dir / "test_dataset.yaml"
        dataset_dir = remove_operation.datasets_dir / "test_dataset"
        
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch.object(yaml_file, 'exists', return_value=True):
            with patch.object(dataset_dir, 'exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=yaml_content)):
                    with patch.object(yaml_file, 'unlink'):
                        with patch('shutil.rmtree', side_effect=OSError("Directory in use")):
                            # Act & Assert - directory errors should cause failure
                            with pytest.raises(DatasetError, match="Failed to remove dataset"):
                                remove_operation.execute("test_dataset", force=True)

    def test_execute_size_calculation_with_subdirs(self, remove_operation, sample_dataset_data):
        """Test accurate size calculation with nested directories."""
        # Arrange
        yaml_file = remove_operation.dataset_registry_dir / "test_dataset.yaml"
        dataset_dir = remove_operation.datasets_dir / "test_dataset"
        
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch.object(yaml_file, 'exists', return_value=True):
            with patch.object(dataset_dir, 'exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=yaml_content)):
                    # Mock complex directory structure
                    files = []
                    for i in range(5):
                        mock_file = Mock()
                        mock_file.is_file.return_value = True
                        mock_file.stat.return_value.st_size = 1000 * (i + 1)
                        files.append(mock_file)
                    
                    # Add a directory (should be ignored)
                    mock_dir = Mock()
                    mock_dir.is_file.return_value = False
                    files.append(mock_dir)
                    
                    with patch.object(dataset_dir, 'rglob', return_value=files):
                        # Act
                        result = remove_operation.execute("test_dataset", dry_run=True)

        # Assert
        # Size should be 1000 + 2000 + 3000 + 4000 + 5000 = 15000
        assert result['size'] == 15000