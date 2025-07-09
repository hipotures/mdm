"""Unit tests for UpdateOperation."""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from datetime import datetime
import yaml

from mdm.dataset.operations import UpdateOperation
from mdm.core.exceptions import DatasetError


class TestUpdateOperation:
    """Test cases for UpdateOperation."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.paths.configs_path = "config/datasets"
        config.paths.datasets_path = "datasets"
        return config

    @pytest.fixture
    def update_operation(self, mock_config):
        """Create UpdateOperation instance."""
        with patch('mdm.dataset.operations.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = Path("/test")
            mock_get_config.return_value = mock_manager
            
            operation = UpdateOperation()
            operation.dataset_registry_dir = Path("/test/config/datasets")
            operation.datasets_dir = Path("/test/datasets")
            return operation

    @pytest.fixture
    def sample_dataset_data(self):
        """Sample dataset YAML data."""
        return {
            'name': 'test_dataset',
            'description': 'Original description',
            'target_column': 'target',
            'problem_type': 'classification',
            'id_columns': ['id'],
            'database': {'backend': 'sqlite'},
            'created_at': '2024-01-01T00:00:00'
        }

    def test_execute_update_description(self, update_operation, sample_dataset_data):
        """Test updating dataset description."""
        # Arrange
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml_content)) as mock_file:
                with patch('mdm.dataset.operations.datetime') as mock_datetime:
                    mock_datetime.utcnow.return_value.isoformat.return_value = '2024-01-02T00:00:00'
                    
                    # Act
                    result = update_operation.execute(
                        "test_dataset",
                        description="Updated description"
                    )

        # Assert
        assert result['description'] == 'Updated description'
        assert result['last_updated_at'] == '2024-01-02T00:00:00'
        
        # Verify file was written
        mock_file().write.called

    def test_execute_update_target_column(self, update_operation, sample_dataset_data):
        """Test updating target column."""
        # Arrange
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml_content)):
                # Act
                result = update_operation.execute(
                    "test_dataset",
                    target="new_target"
                )

        # Assert
        assert result['target_column'] == 'new_target'

    def test_execute_update_problem_type(self, update_operation, sample_dataset_data):
        """Test updating problem type."""
        # Arrange
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml_content)):
                # Act
                result = update_operation.execute(
                    "test_dataset",
                    problem_type="regression"
                )

        # Assert
        assert result['problem_type'] == 'regression'

    def test_execute_update_id_columns(self, update_operation, sample_dataset_data):
        """Test updating ID columns."""
        # Arrange
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml_content)):
                # Act
                result = update_operation.execute(
                    "test_dataset",
                    id_columns=["id1", "id2"]
                )

        # Assert
        assert result['id_columns'] == ["id1", "id2"]

    def test_execute_update_multiple_fields(self, update_operation, sample_dataset_data):
        """Test updating multiple fields at once."""
        # Arrange
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml_content)):
                # Act
                result = update_operation.execute(
                    "test_dataset",
                    description="New description",
                    target="new_target",
                    problem_type="regression",
                    id_columns=["new_id"]
                )

        # Assert
        assert result['description'] == 'New description'
        assert result['target_column'] == 'new_target'
        assert result['problem_type'] == 'regression'
        assert result['id_columns'] == ["new_id"]

    def test_execute_dataset_not_found(self, update_operation):
        """Test error when dataset doesn't exist."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=False):
            # Act & Assert
            with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
                update_operation.execute("nonexistent", description="New")

    def test_execute_no_updates(self, update_operation, sample_dataset_data):
        """Test execute with no updates (only timestamp should change)."""
        # Arrange
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml_content)):
                with patch('mdm.dataset.operations.datetime') as mock_datetime:
                    mock_datetime.utcnow.return_value.isoformat.return_value = '2024-01-02T00:00:00'
                    
                    # Act
                    result = update_operation.execute("test_dataset")

        # Assert
        assert result['last_updated_at'] == '2024-01-02T00:00:00'
        # Other fields should remain unchanged
        assert result['description'] == sample_dataset_data['description']

    def test_execute_yaml_write_error(self, update_operation, sample_dataset_data):
        """Test error handling when writing YAML fails."""
        # Arrange
        yaml_content = yaml.dump(sample_dataset_data)
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml_content)) as mock_file:
                # Make write fail
                mock_file.return_value.write.side_effect = IOError("Write failed")
                
                # Act & Assert
                with pytest.raises(DatasetError, match="Failed to update dataset"):
                    update_operation.execute("test_dataset", description="New")

    @patch('mdm.dataset.operations.BackendFactory')
    @patch('pathlib.Path.exists')
    def test_update_database_metadata(self, mock_exists, mock_factory, update_operation, sample_dataset_data):
        """Test database metadata update."""
        # Arrange
        yaml_content = yaml.dump(sample_dataset_data)
        mock_backend = Mock()
        mock_factory.create.return_value = mock_backend
        
        # Mock path exists calls
        mock_exists.side_effect = [True, True]  # yaml exists, dataset dir exists
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            # Act
            result = update_operation.execute(
                "test_dataset",
                description="Updated"
            )

        # Assert
        mock_factory.create.assert_called_once_with('sqlite', {'backend': 'sqlite'})

    @patch('pathlib.Path.exists')
    def test_update_database_metadata_error_handling(self, mock_exists, update_operation, sample_dataset_data):
        """Test that database metadata errors are logged but don't propagate."""
        # Arrange
        yaml_content = yaml.dump(sample_dataset_data)
        
        # Mock path exists calls
        mock_exists.side_effect = [True, True]  # yaml exists, dataset dir exists
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('mdm.dataset.operations.BackendFactory.create', 
                     side_effect=Exception("DB error")):
                with patch('mdm.dataset.operations.logger') as mock_logger:
                    # Act - should succeed because _update_database_metadata catches exceptions
                    result = update_operation.execute(
                        "test_dataset",
                        description="Updated"
                    )

        # Assert - should succeed and log warning
        assert result['description'] == 'Updated'
        mock_logger.warning.assert_called_once()