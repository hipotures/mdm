"""Unit tests for batch stats operations."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typer.testing import CliRunner

from mdm.cli.batch import batch_app


class TestBatchStats:
    """Test cases for batch stats operations."""

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
    def mock_stats_op(self):
        """Mock StatsOperation."""
        with patch('mdm.cli.batch.StatsOperation') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def sample_stats(self):
        """Sample statistics data."""
        return {
            'dataset_name': 'test_dataset',
            'row_count': 1000,
            'column_count': 10,
            'memory_size_mb': 5.2,
            'missing_values': {'col1': 5, 'col2': 10},
            'column_types': {
                'col1': 'int64',
                'col2': 'float64',
                'col3': 'object'
            }
        }

    def test_batch_stats_success(self, runner, mock_manager, mock_stats_op, sample_stats):
        """Test successful batch stats computation."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_stats_op.execute.return_value = sample_stats

        # Act
        result = runner.invoke(
            batch_app,
            ["stats", "dataset1", "dataset2"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Successfully computed: 2 datasets" in result.output
        assert mock_stats_op.execute.call_count == 2

    def test_batch_stats_full_computation(self, runner, mock_manager, mock_stats_op, sample_stats):
        """Test batch stats with full computation flag."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_stats_op.execute.return_value = sample_stats

        # Act
        result = runner.invoke(
            batch_app,
            ["stats", "dataset1", "--full"]
        )

        # Assert
        assert result.exit_code == 0
        mock_stats_op.execute.assert_called_once_with("dataset1", full=True, export=None)

    def test_batch_stats_export_to_file(self, runner, mock_manager, mock_stats_op, sample_stats):
        """Test batch stats with export option."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_stats_op.execute.return_value = sample_stats

        # Act
        result = runner.invoke(
            batch_app,
            ["stats", "dataset1", "--export-dir", "/tmp/stats"]
        )

        # Assert
        assert result.exit_code == 0
        mock_stats_op.execute.assert_called_once()
        call_args = mock_stats_op.execute.call_args[1]
        assert call_args['export'] == Path("/tmp/stats/dataset1_stats.json")

    def test_batch_stats_dataset_not_found(self, runner, mock_manager, mock_stats_op):
        """Test batch stats with non-existent dataset."""
        # Arrange
        mock_manager.dataset_exists.side_effect = [True, False, True]
        mock_stats_op.execute.return_value = {'row_count': 100}

        # Act
        result = runner.invoke(
            batch_app,
            ["stats", "dataset1", "dataset2", "dataset3"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Dataset 'dataset2' not found, skipping" in result.output
        assert "Successfully computed: 2 datasets" in result.output
        assert "Failed: 1 datasets" in result.output
        assert mock_stats_op.execute.call_count == 2

    def test_batch_stats_error_handling(self, runner, mock_manager, mock_stats_op):
        """Test batch stats error handling."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_stats_op.execute.side_effect = [
            {'row_count': 100},
            Exception("Stats computation failed"),
            {'row_count': 200}
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["stats", "dataset1", "dataset2", "dataset3"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Failed to compute stats for 'dataset2': Stats computation failed" in result.output
        assert "Successfully computed: 2 datasets" in result.output
        assert "Failed: 1 datasets" in result.output

    def test_batch_stats_summary_display(self, runner, mock_manager, mock_stats_op):
        """Test batch stats summary display."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_stats_op.execute.side_effect = [
            {'row_count': 1000, 'column_count': 10},
            {'row_count': 2000, 'column_count': 20},
            {'row_count': 500, 'column_count': 5}
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["stats", "dataset1", "dataset2", "dataset3"]
        )

        # Assert
        assert result.exit_code == 0
        assert "dataset1 (1000 rows, 10 columns)" in result.output
        assert "dataset2 (2000 rows, 20 columns)" in result.output
        assert "dataset3 (500 rows, 5 columns)" in result.output

    def test_batch_stats_empty_list(self, runner):
        """Test batch stats with no datasets."""
        # Act
        result = runner.invoke(batch_app, ["stats"])

        # Assert
        assert result.exit_code != 0  # Should fail with missing argument

    def test_batch_stats_progress_tracking(self, runner, mock_manager, mock_stats_op):
        """Test progress tracking during batch stats."""
        # Arrange
        datasets = [f"dataset{i}" for i in range(10)]
        mock_manager.dataset_exists.return_value = True
        mock_stats_op.execute.return_value = {'row_count': 100}

        # Act
        result = runner.invoke(
            batch_app,
            ["stats"] + datasets
        )

        # Assert
        assert result.exit_code == 0
        assert "Successfully computed: 10 datasets" in result.output
        assert mock_stats_op.execute.call_count == 10

    @patch('pathlib.Path.mkdir')
    def test_batch_stats_export_directory_creation(self, mock_mkdir, runner, mock_manager, mock_stats_op):
        """Test that export directories are created properly."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_stats_op.execute.return_value = {'row_count': 100}

        # Act
        result = runner.invoke(
            batch_app,
            ["stats", "dataset1", "--export-dir", "new_stats"]
        )

        # Assert
        assert result.exit_code == 0
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_batch_stats_missing_values_display(self, runner, mock_manager, mock_stats_op):
        """Test display of missing values in stats."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_stats_op.execute.return_value = {
            'row_count': 1000,
            'column_count': 5,
            'missing_values': {'col1': 10, 'col2': 20}
        }

        # Act
        result = runner.invoke(
            batch_app,
            ["stats", "dataset1"]
        )

        # Assert
        assert result.exit_code == 0
        # The summary should mention the dataset was processed successfully
        assert "dataset1" in result.output