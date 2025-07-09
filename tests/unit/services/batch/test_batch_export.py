"""Unit tests for batch export operations."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from typer.testing import CliRunner

from mdm.cli.batch import batch_app


class TestBatchExport:
    """Test cases for batch export operations."""

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
    def mock_export_op(self):
        """Mock ExportOperation."""
        with patch('mdm.cli.batch.ExportOperation') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance

    def test_batch_export_success(self, runner, mock_manager, mock_export_op):
        """Test successful batch export."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_export_op.execute.side_effect = [
            [Path("exports/dataset1/train.csv"), Path("exports/dataset1/test.csv")],
            [Path("exports/dataset2/train.csv")]
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["export", "dataset1", "dataset2", "--output-dir", "exports"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Successfully exported: 2 datasets" in result.output
        assert "dataset1 (2 files)" in result.output
        assert "dataset2 (1 files)" in result.output
        assert mock_export_op.execute.call_count == 2

    def test_batch_export_dataset_not_found(self, runner, mock_manager, mock_export_op):
        """Test batch export with non-existent dataset."""
        # Arrange
        mock_manager.dataset_exists.side_effect = [True, False, True]
        mock_export_op.execute.side_effect = [
            [Path("exports/dataset1/train.csv")],
            [Path("exports/dataset3/train.csv")]
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["export", "dataset1", "dataset2", "dataset3"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Dataset 'dataset2' not found, skipping" in result.output
        assert "Successfully exported: 2 datasets" in result.output
        assert "Failed: 1 datasets" in result.output
        assert mock_export_op.execute.call_count == 2

    def test_batch_export_with_options(self, runner, mock_manager, mock_export_op):
        """Test batch export with format and compression options."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_export_op.execute.return_value = [Path("exports/dataset1/train.parquet.gz")]

        # Act
        result = runner.invoke(
            batch_app,
            ["export", "dataset1", "--format", "parquet", "--compression", "gzip"]
        )

        # Assert
        assert result.exit_code == 0
        mock_export_op.execute.assert_called_once_with(
            name="dataset1",
            format="parquet",
            output_dir=Path("exports/dataset1"),
            compression="gzip",
            metadata_only=False
        )

    def test_batch_export_metadata_only(self, runner, mock_manager, mock_export_op):
        """Test batch export with metadata-only flag."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_export_op.execute.return_value = [Path("exports/dataset1/metadata.yaml")]

        # Act
        result = runner.invoke(
            batch_app,
            ["export", "dataset1", "--metadata-only"]
        )

        # Assert
        assert result.exit_code == 0
        mock_export_op.execute.assert_called_once()
        call_args = mock_export_op.execute.call_args[1]
        assert call_args['metadata_only'] is True

    def test_batch_export_error_handling(self, runner, mock_manager, mock_export_op):
        """Test batch export error handling."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_export_op.execute.side_effect = [
            [Path("exports/dataset1/train.csv")],
            Exception("Export failed"),
            [Path("exports/dataset3/train.csv")]
        ]

        # Act
        result = runner.invoke(
            batch_app,
            ["export", "dataset1", "dataset2", "dataset3"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Failed to export 'dataset2': Export failed" in result.output
        assert "Successfully exported: 2 datasets" in result.output
        assert "Failed: 1 datasets" in result.output

    def test_batch_export_custom_output_dir(self, runner, mock_manager, mock_export_op):
        """Test batch export with custom output directory."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_export_op.execute.return_value = [Path("custom/dataset1/train.csv")]

        # Act
        result = runner.invoke(
            batch_app,
            ["export", "dataset1", "--output-dir", "/tmp/custom_exports"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Output directory: /tmp/custom_exports" in result.output
        mock_export_op.execute.assert_called_once()
        call_args = mock_export_op.execute.call_args[1]
        assert call_args['output_dir'] == Path("/tmp/custom_exports/dataset1")

    def test_batch_export_empty_list(self, runner):
        """Test batch export with no datasets."""
        # Act
        result = runner.invoke(batch_app, ["export"])

        # Assert
        assert result.exit_code != 0  # Should fail with missing argument

    def test_batch_export_progress_tracking(self, runner, mock_manager, mock_export_op):
        """Test progress tracking during batch export."""
        # Arrange
        datasets = [f"dataset{i}" for i in range(5)]
        mock_manager.dataset_exists.return_value = True
        mock_export_op.execute.return_value = [Path("exports/data.csv")]

        # Act
        result = runner.invoke(
            batch_app,
            ["export"] + datasets
        )

        # Assert
        assert result.exit_code == 0
        assert "Successfully exported: 5 datasets" in result.output
        assert mock_export_op.execute.call_count == 5

    def test_batch_export_all_formats(self, runner, mock_manager, mock_export_op):
        """Test batch export with different formats."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        
        for format_type in ["csv", "parquet", "json"]:
            mock_export_op.execute.return_value = [
                Path(f"exports/dataset1/train.{format_type}")
            ]

            # Act
            result = runner.invoke(
                batch_app,
                ["export", "dataset1", "--format", format_type]
            )

            # Assert
            assert result.exit_code == 0
            assert "Successfully exported: 1 datasets" in result.output

    @patch('pathlib.Path.mkdir')
    def test_batch_export_directory_creation(self, mock_mkdir, runner, mock_manager, mock_export_op):
        """Test that directories are created properly."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        mock_export_op.execute.return_value = []

        # Act
        result = runner.invoke(
            batch_app,
            ["export", "dataset1", "--output-dir", "new_exports"]
        )

        # Assert
        assert result.exit_code == 0
        # Check that mkdir was called for both base and dataset-specific dirs
        assert mock_mkdir.call_count >= 2