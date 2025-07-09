"""Unit tests for batch remove operations."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typer.testing import CliRunner

from mdm.cli.batch import batch_app


class TestBatchRemove:
    """Test cases for batch remove operations."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_manager(self):
        """Mock DatasetManager."""
        with patch('mdm.cli.batch.DatasetManager') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_remove_op(self):
        """Mock RemoveOperation."""
        with patch('mdm.cli.batch.RemoveOperation') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance

    def test_batch_remove_success(self, runner, mock_manager, mock_remove_op):
        """Test successful batch remove."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        # First calls are dry run to get info, then actual removal
        mock_remove_op.execute.side_effect = [
            # Dry run for dataset1
            {
                'name': 'dataset1',
                'dry_run': True,
                'config_file': '/path/to/config.yaml',
                'dataset_directory': '/path/to/dataset',
                'size': 1048576  # 1MB
            },
            # Dry run for dataset2
            {
                'name': 'dataset2',
                'dry_run': True,
                'config_file': '/path/to/config.yaml',
                'dataset_directory': '/path/to/dataset',
                'size': 1048576  # 1MB
            },
            # Actual removal for dataset1
            {
                'removed': True,
                'config_file': '/path/to/config.yaml',
                'dataset_directory': '/path/to/dataset',
                'size': 1048576
            },
            # Actual removal for dataset2
            {
                'removed': True,
                'config_file': '/path/to/config.yaml',
                'dataset_directory': '/path/to/dataset',
                'size': 1048576
            }
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "dataset2", "--force"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Removed 2 datasets" in result.output
        assert mock_remove_op.execute.call_count == 4  # 2 dry runs + 2 removals

    def test_batch_remove_dry_run(self, runner, mock_manager, mock_remove_op):
        """Test batch remove with dry run."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_remove_op.execute.return_value = {
            'name': 'dataset1',
            'dry_run': True,
            'config_file': '/path/to/config.yaml',
            'dataset_directory': '/path/to/dataset',
            'size': 2097152  # 2MB
        }

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "--dry-run", "--force"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Datasets to remove: 1" in result.output
        assert "Total size: 2.0 MB" in result.output
        mock_remove_op.execute.assert_called_once_with("dataset1", force=True, dry_run=True)

    def test_batch_remove_without_force(self, runner, mock_manager, mock_remove_op):
        """Test batch remove without force flag (should use dry-run)."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_remove_op.execute.return_value = {
            'name': 'dataset1',
            'dry_run': True,
            'size': 1000000
        }

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1"],
            input="n\n"
        )

        # Assert
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        mock_remove_op.execute.assert_called_once_with("dataset1", force=True, dry_run=True)

    def test_batch_remove_dataset_not_found(self, runner, mock_manager, mock_remove_op):
        """Test batch remove with non-existent dataset."""
        # Arrange
        mock_manager.dataset_exists.side_effect = [True, False, True]
        mock_remove_op.execute.side_effect = [
            {'name': 'dataset1', 'dry_run': True, 'size': 1000000},  # dry_run for dataset1
            {'name': 'dataset3', 'dry_run': True, 'size': 1000000},  # dry_run for dataset3
            {'name': 'dataset1', 'removed': True, 'size': 1000000},  # actual remove for dataset1
            {'name': 'dataset3', 'removed': True, 'size': 1000000}   # actual remove for dataset3
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "dataset2", "dataset3", "--force"]
        )

        # Assert
        assert result.exit_code == 0
        # batch_remove silently skips non-existent datasets
        assert "Removed 2 datasets" in result.output
        # execute is called 2 times for dry_run + 2 times for actual remove
        assert mock_remove_op.execute.call_count == 4

    def test_batch_remove_error_handling(self, runner, mock_manager, mock_remove_op):
        """Test batch remove error handling."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_remove_op.execute.side_effect = [
            # dry_run calls
            {'name': 'dataset1', 'dry_run': True, 'size': 1000000},
            {'name': 'dataset2', 'dry_run': True, 'size': 1500000},
            {'name': 'dataset3', 'dry_run': True, 'size': 2000000},
            # actual remove calls
            {'name': 'dataset1', 'removed': True, 'size': 1000000},
            Exception("Permission denied"),
            {'name': 'dataset3', 'removed': True, 'size': 2000000}
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "dataset2", "dataset3", "--force"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Failed to remove 'dataset2': Permission denied" in result.output
        assert "Removed 2 datasets" in result.output
        assert "Failed to remove 1 datasets" in result.output

    def test_batch_remove_size_formatting(self, runner, mock_manager, mock_remove_op):
        """Test proper size formatting in output."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_remove_op.execute.side_effect = [
            # dry_run calls
            {'name': 'dataset1', 'dry_run': True, 'size': 512},           # 512 B
            {'name': 'dataset2', 'dry_run': True, 'size': 1024},          # 1 KB
            {'name': 'dataset3', 'dry_run': True, 'size': 1048576},       # 1 MB
            {'name': 'dataset4', 'dry_run': True, 'size': 1073741824},    # 1 GB
            # actual remove calls
            {'name': 'dataset1', 'removed': True, 'size': 512},           # 512 B
            {'name': 'dataset2', 'removed': True, 'size': 1024},          # 1 KB
            {'name': 'dataset3', 'removed': True, 'size': 1048576},       # 1 MB
            {'name': 'dataset4', 'removed': True, 'size': 1073741824}     # 1 GB
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "dataset2", "dataset3", "dataset4", "--force"]
        )

        # Assert
        assert result.exit_code == 0
        assert "dataset1 (512.0 B)" in result.output
        assert "dataset2 (1.0 KB)" in result.output
        assert "dataset3 (1.0 MB)" in result.output
        assert "dataset4 (1.0 GB)" in result.output

    def test_batch_remove_empty_list(self, runner):
        """Test batch remove with no datasets."""
        # Act
        result = runner.invoke(batch_app, ["remove"])

        # Assert
        assert result.exit_code != 0  # Should fail with missing argument

    def test_batch_remove_progress_tracking(self, runner, mock_manager, mock_remove_op):
        """Test progress tracking during batch remove."""
        # Arrange
        datasets = [f"dataset{i}" for i in range(5)]
        mock_manager.dataset_exists.return_value = True
        # Create side_effect for both dry_run and actual remove calls
        dry_run_responses = [{'name': f'dataset{i}', 'dry_run': True, 'size': 1000000} for i in range(5)]
        remove_responses = [{'name': f'dataset{i}', 'removed': True, 'size': 1000000} for i in range(5)]
        mock_remove_op.execute.side_effect = dry_run_responses + remove_responses

        # Act
        result = runner.invoke(
            batch_app,
            ["remove"] + datasets + ["--force"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Removed 5 datasets" in result.output
        assert mock_remove_op.execute.call_count == 10  # 5 dry_run + 5 actual

    def test_batch_remove_postgresql_datasets(self, runner, mock_manager, mock_remove_op):
        """Test batch remove with PostgreSQL datasets."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_remove_op.execute.side_effect = [
            # dry_run call
            {
                'name': 'dataset1',
                'dry_run': True,
                'size': 1000000,
                'postgresql_db': 'mdm_dataset1'
            },
            # actual remove call
            {
                'name': 'dataset1',
                'removed': True,
                'size': 1000000,
                'postgresql_db': 'mdm_dataset1'
            }
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "--force"]
        )

        # Assert
        assert result.exit_code == 0
        # The output should still work correctly with PostgreSQL info
        assert "Removed 1 datasets" in result.output

    def test_batch_remove_total_size_calculation(self, runner, mock_manager, mock_remove_op):
        """Test total size calculation in dry run."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_remove_op.execute.side_effect = [
            {'name': 'dataset1', 'dry_run': True, 'size': 1000000},    # 1 MB
            {'name': 'dataset2', 'dry_run': True, 'size': 2000000},    # 2 MB
            {'name': 'dataset3', 'dry_run': True, 'size': 3000000}     # 3 MB
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "dataset2", "dataset3", "--dry-run"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Total size: 5.7 MB" in result.output  # ~6 MB total