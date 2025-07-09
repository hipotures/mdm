"""Unit tests for ExportOperation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from mdm.dataset.operations import ExportOperation


class TestExportOperation:
    """Test cases for ExportOperation."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.paths.configs_path = "config/datasets"
        config.paths.datasets_path = "datasets"
        config.export.default_format = "csv"
        config.export.compression = None
        return config

    @pytest.fixture
    def export_operation(self, mock_config):
        """Create ExportOperation instance."""
        with patch('mdm.config.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = Path("/test")
            mock_get_config.return_value = mock_manager
            
            operation = ExportOperation()
            return operation

    @pytest.fixture
    def mock_exporter(self):
        """Mock DatasetExporter."""
        with patch('mdm.dataset.operations.DatasetExporter') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance

    def test_execute_default_format(self, export_operation, mock_exporter):
        """Test export with default format from config."""
        # Arrange
        mock_exporter.export.return_value = ["/output/dataset.csv"]

        # Act
        result = export_operation.execute("test_dataset")

        # Assert
        mock_exporter.export.assert_called_once_with(
            dataset_name="test_dataset",
            format="csv",  # Default from config
            output_dir=None,
            table=None,
            compression=None,
            metadata_only=False,
            no_header=False
        )
        assert result == ["/output/dataset.csv"]

    def test_execute_custom_format(self, export_operation, mock_exporter):
        """Test export with custom format."""
        # Arrange
        mock_exporter.export.return_value = ["/output/dataset.parquet"]

        # Act
        result = export_operation.execute(
            "test_dataset",
            format="parquet"
        )

        # Assert
        mock_exporter.export.assert_called_once_with(
            dataset_name="test_dataset",
            format="parquet",
            output_dir=None,
            table=None,
            compression=None,
            metadata_only=False,
            no_header=False
        )

    def test_execute_with_output_dir(self, export_operation, mock_exporter):
        """Test export with custom output directory."""
        # Arrange
        mock_exporter.export.return_value = ["/custom/path/dataset.csv"]

        # Act
        result = export_operation.execute(
            "test_dataset",
            output_dir=Path("/custom/path")
        )

        # Assert
        mock_exporter.export.assert_called_once()
        call_args = mock_exporter.export.call_args[1]
        assert call_args['output_dir'] == Path("/custom/path")

    def test_execute_specific_table(self, export_operation, mock_exporter):
        """Test export of specific table."""
        # Arrange
        mock_exporter.export.return_value = ["/output/train.csv"]

        # Act
        result = export_operation.execute(
            "test_dataset",
            table="train"
        )

        # Assert
        mock_exporter.export.assert_called_once()
        call_args = mock_exporter.export.call_args[1]
        assert call_args['table'] == "train"

    def test_execute_with_compression(self, export_operation, mock_exporter):
        """Test export with compression."""
        # Arrange
        mock_exporter.export.return_value = ["/output/dataset.csv.gz"]

        # Act
        result = export_operation.execute(
            "test_dataset",
            compression="gzip"
        )

        # Assert
        mock_exporter.export.assert_called_once()
        call_args = mock_exporter.export.call_args[1]
        assert call_args['compression'] == "gzip"

    def test_execute_default_compression_from_config(self, export_operation, mock_exporter):
        """Test export uses default compression from config."""
        # Arrange
        export_operation.config.export.compression = "zip"
        mock_exporter.export.return_value = ["/output/dataset.csv.zip"]

        # Act
        result = export_operation.execute("test_dataset")

        # Assert
        mock_exporter.export.assert_called_once()
        call_args = mock_exporter.export.call_args[1]
        assert call_args['compression'] == "zip"

    def test_execute_metadata_only(self, export_operation, mock_exporter):
        """Test metadata-only export."""
        # Arrange
        mock_exporter.export.return_value = ["/output/metadata.json"]

        # Act
        result = export_operation.execute(
            "test_dataset",
            metadata_only=True
        )

        # Assert
        mock_exporter.export.assert_called_once()
        call_args = mock_exporter.export.call_args[1]
        assert call_args['metadata_only'] is True

    def test_execute_no_header(self, export_operation, mock_exporter):
        """Test CSV export without header."""
        # Arrange
        mock_exporter.export.return_value = ["/output/dataset.csv"]

        # Act
        result = export_operation.execute(
            "test_dataset",
            no_header=True
        )

        # Assert
        mock_exporter.export.assert_called_once()
        call_args = mock_exporter.export.call_args[1]
        assert call_args['no_header'] is True

    def test_execute_all_options(self, export_operation, mock_exporter):
        """Test export with all options."""
        # Arrange
        mock_exporter.export.return_value = [
            "/custom/train.json.gz",
            "/custom/test.json.gz"
        ]

        # Act
        result = export_operation.execute(
            "test_dataset",
            format="json",
            output_dir=Path("/custom"),
            table="all",
            compression="gzip",
            metadata_only=False,
            no_header=False
        )

        # Assert
        mock_exporter.export.assert_called_once_with(
            dataset_name="test_dataset",
            format="json",
            output_dir=Path("/custom"),
            table="all",
            compression="gzip",
            metadata_only=False,
            no_header=False
        )
        assert len(result) == 2

    def test_execute_exporter_error(self, export_operation, mock_exporter):
        """Test handling of exporter errors."""
        # Arrange
        mock_exporter.export.side_effect = Exception("Export failed")

        # Act & Assert
        with pytest.raises(Exception, match="Export failed"):
            export_operation.execute("test_dataset")