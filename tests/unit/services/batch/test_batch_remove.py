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
        assert "Would remove: 1 datasets" in result.output
        assert "Total size: 2.0 MB" in result.output
        mock_remove_op.execute.assert_called_once_with("dataset1", force=True, dry_run=True)

    def test_batch_remove_without_force(self, runner, mock_manager, mock_remove_op):
        """Test batch remove without force flag (should use dry-run)."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_remove_op.execute.return_value = {
            'dry_run': True,
            'size': 1000000
        }

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Use --force to actually remove datasets" in result.output
        mock_remove_op.execute.assert_called_once_with("dataset1", force=False, dry_run=True)

    def test_batch_remove_dataset_not_found(self, runner, mock_manager, mock_remove_op):
        """Test batch remove with non-existent dataset."""
        # Arrange
        mock_manager.dataset_exists.side_effect = [True, False, True]
        mock_remove_op.execute.return_value = {'removed': True, 'size': 1000000}

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "dataset2", "dataset3", "--force"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Dataset 'dataset2' not found, skipping" in result.output
        assert "Successfully removed: 2 datasets" in result.output
        assert "Failed: 1 datasets" in result.output
        assert mock_remove_op.execute.call_count == 2

    def test_batch_remove_error_handling(self, runner, mock_manager, mock_remove_op):
        """Test batch remove error handling."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_remove_op.execute.side_effect = [
            {'removed': True, 'size': 1000000},
            Exception("Permission denied"),
            {'removed': True, 'size': 2000000}
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "dataset2", "dataset3", "--force"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Failed to remove 'dataset2': Permission denied" in result.output
        assert "Successfully removed: 2 datasets" in result.output
        assert "Failed: 1 datasets" in result.output

    def test_batch_remove_size_formatting(self, runner, mock_manager, mock_remove_op):
        """Test proper size formatting in output."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_remove_op.execute.side_effect = [
            {'removed': True, 'size': 512},           # 512 B
            {'removed': True, 'size': 1024},          # 1 KB
            {'removed': True, 'size': 1048576},       # 1 MB
            {'removed': True, 'size': 1073741824}     # 1 GB
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "dataset2", "dataset3", "dataset4", "--force"]
        )

        # Assert
        assert result.exit_code == 0
        assert "dataset1 (512 B)" in result.output
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
        mock_remove_op.execute.return_value = {'removed': True, 'size': 1000000}

        # Act
        result = runner.invoke(
            batch_app,
            ["remove"] + datasets + ["--force"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Successfully removed: 5 datasets" in result.output
        assert mock_remove_op.execute.call_count == 5

    def test_batch_remove_postgresql_datasets(self, runner, mock_manager, mock_remove_op):
        """Test batch remove with PostgreSQL datasets."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_remove_op.execute.return_value = {
            'removed': True,
            'size': 1000000,
            'postgresql_db': 'mdm_dataset1'
        }

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "--force"]
        )

        # Assert
        assert result.exit_code == 0
        # The output should still work correctly with PostgreSQL info
        assert "Successfully removed: 1 datasets" in result.output

    def test_batch_remove_total_size_calculation(self, runner, mock_manager, mock_remove_op):
        """Test total size calculation in dry run."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_remove_op.execute.side_effect = [
            {'dry_run': True, 'size': 1000000},    # 1 MB
            {'dry_run': True, 'size': 2000000},    # 2 MB
            {'dry_run': True, 'size': 3000000}     # 3 MB
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["remove", "dataset1", "dataset2", "dataset3", "--dry-run"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Total size: 5.7 MB" in result.output  # ~6 MB total