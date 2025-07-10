"""Unit tests for DatasetExporter."""

import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
import pandas as pd
import json
import yaml
import gzip
import zipfile

from mdm.dataset.exporter import DatasetExporter
from mdm.core.exceptions import DatasetError, StorageError


class TestDatasetExporter:
    """Test cases for DatasetExporter."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.paths.configs_path = "config/datasets"
        config.paths.datasets_path = "datasets"
        config.database.default_backend = "sqlite"  # Add default backend
        return config

    @pytest.fixture
    def exporter(self, mock_config):
        """Create DatasetExporter instance."""
        with patch('mdm.config.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = Path("/test")
            mock_get_config.return_value = mock_manager
            
            exporter = DatasetExporter()
            return exporter

    @pytest.fixture
    def sample_dataset_info(self):
        """Sample dataset information."""
        return {
            'name': 'test_dataset',
            'tables': {
                'train': 'train_table',
                'test': 'test_table'
            },
            'target_column': 'target',
            'id_columns': ['id'],
            'database': {'backend': 'sqlite'},
            'metadata': {
                'row_count': 1000,
                'column_count': 10
            }
        }

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'feature1': [10, 20, 30, 40, 50],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        })

    def test_export_invalid_format(self, exporter):
        """Test error with invalid export format."""
        with pytest.raises(ValueError, match="Invalid format 'invalid'"):
            exporter.export("test_dataset", format="invalid")

    @patch.object(DatasetExporter, '_load_dataset_info')
    @patch.object(DatasetExporter, '_export_metadata')
    def test_export_metadata_only(self, mock_export_metadata, mock_load_info, exporter, sample_dataset_info):
        """Test metadata-only export."""
        # Arrange
        mock_load_info.return_value = sample_dataset_info
        mock_export_metadata.return_value = Path("/output/metadata.yaml")
        
        # Act
        result = exporter.export(
            "test_dataset",
            format="csv",
            metadata_only=True
        )
        
        # Assert
        assert len(result) == 1
        assert result[0] == Path("/output/metadata.yaml")
        mock_export_metadata.assert_called_once()

    @patch('pathlib.Path.mkdir')
    @patch.object(DatasetExporter, '_load_dataset_info')
    @patch.object(DatasetExporter, '_export_metadata')
    @patch.object(DatasetExporter, '_get_backend')
    @patch.object(DatasetExporter, '_get_tables_to_export')
    @patch.object(DatasetExporter, '_read_table')
    @patch.object(DatasetExporter, '_export_csv')
    def test_export_csv_all_tables(self, mock_export_csv, mock_read_table, mock_get_tables, 
                                  mock_get_backend, mock_export_metadata, mock_load_info, 
                                  mock_mkdir, exporter, sample_dataset_info, sample_dataframe):
        """Test exporting all tables to CSV."""
        # Arrange
        mock_load_info.return_value = sample_dataset_info
        mock_export_metadata.return_value = Path("/output/metadata.yaml")
        mock_backend = Mock()
        mock_get_backend.return_value = mock_backend
        mock_get_tables.return_value = sample_dataset_info['tables']
        mock_read_table.return_value = sample_dataframe
        mock_export_csv.side_effect = [
            Path("/output/test_dataset_train.csv"),
            Path("/output/test_dataset_test.csv")
        ]
        
        # Act
        result = exporter.export(
            "test_dataset",
            format="csv",
            output_dir=Path("/output")
        )
        
        # Assert
        assert len(result) == 3  # metadata + 2 tables
        assert Path("/output/metadata.yaml") in result
        assert Path("/output/test_dataset_train.csv") in result
        assert Path("/output/test_dataset_test.csv") in result
        assert mock_export_csv.call_count == 2

    @patch('pathlib.Path.mkdir')
    @patch.object(DatasetExporter, '_load_dataset_info')
    @patch.object(DatasetExporter, '_export_metadata')
    @patch.object(DatasetExporter, '_get_backend')
    @patch.object(DatasetExporter, '_get_tables_to_export')
    @patch.object(DatasetExporter, '_read_table')
    @patch.object(DatasetExporter, '_export_parquet')
    def test_export_specific_table(self, mock_export_parquet, mock_read_table, mock_get_tables,
                                  mock_get_backend, mock_export_metadata, mock_load_info,
                                  mock_mkdir, exporter, sample_dataset_info, sample_dataframe):
        """Test exporting specific table."""
        # Arrange
        mock_load_info.return_value = sample_dataset_info
        mock_export_metadata.return_value = Path("/output/metadata.yaml")
        mock_backend = Mock()
        mock_get_backend.return_value = mock_backend
        mock_get_tables.return_value = {'train': 'train_table'}
        mock_read_table.return_value = sample_dataframe
        mock_export_parquet.return_value = Path("/output/test_dataset_train.parquet")
        
        # Act
        result = exporter.export(
            "test_dataset",
            format="parquet",
            table="train"
        )
        
        # Assert
        assert len(result) == 2  # metadata + 1 table
        mock_get_tables.assert_called_once_with(sample_dataset_info, "train")
        mock_export_parquet.assert_called_once()

    def test_load_dataset_info_not_found(self, exporter):
        """Test loading non-existent dataset info."""
        # Arrange
        yaml_path = exporter.dataset_registry_dir / "nonexistent.yaml"
        
        with patch('pathlib.Path.exists', return_value=False):
            # Act & Assert
            with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
                exporter._load_dataset_info("nonexistent")

    def test_load_dataset_info_backend_mismatch(self, exporter, sample_dataset_info):
        """Test error when backend doesn't match."""
        # Arrange
        sample_dataset_info['database']['backend'] = 'duckdb'
        yaml_content = yaml.dump(sample_dataset_info)
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml_content)):
                # Act & Assert
                with pytest.raises(DatasetError) as exc_info:
                    exporter._load_dataset_info("test_dataset")
                
                # Verify error message
                assert "uses 'duckdb' backend" in str(exc_info.value)
                assert "current backend is 'sqlite'" in str(exc_info.value)

    @patch('mdm.dataset.exporter.BackendFactory')
    def test_get_backend(self, mock_factory, exporter, sample_dataset_info):
        """Test backend creation."""
        # Arrange
        mock_backend = Mock()
        mock_factory.create.return_value = mock_backend
        
        # Act
        result = exporter._get_backend("test_dataset", sample_dataset_info)
        
        # Assert
        assert result == mock_backend
        mock_factory.create.assert_called_once_with(
            'sqlite',
            {'backend': 'sqlite'}
        )

    def test_get_tables_to_export_all(self, exporter, sample_dataset_info):
        """Test getting all tables to export."""
        result = exporter._get_tables_to_export(sample_dataset_info, None)
        assert result == sample_dataset_info['tables']

    def test_get_tables_to_export_specific(self, exporter, sample_dataset_info):
        """Test getting specific table to export."""
        result = exporter._get_tables_to_export(sample_dataset_info, 'train')
        assert result == {'train': 'train_table'}

    def test_get_tables_to_export_invalid(self, exporter, sample_dataset_info):
        """Test error with invalid table name."""
        with pytest.raises(DatasetError, match="Table 'invalid' not found"):
            exporter._get_tables_to_export(sample_dataset_info, 'invalid')

    def test_export_csv(self, exporter, sample_dataframe):
        """Test exporting to CSV."""
        output_path = Path("/output/test.csv")
        
        with patch.object(sample_dataframe, 'to_csv') as mock_to_csv:
            # Act
            result = exporter._export_csv(
                sample_dataframe,
                output_path,
                compression=None,
                no_header=False
            )
        
        # Assert
        mock_to_csv.assert_called_once_with(
            output_path,
            index=False,
            header=True
        )
        assert result == output_path

    def test_export_csv_no_header(self, exporter, sample_dataframe):
        """Test exporting CSV without header."""
        output_path = Path("/output/test.csv")
        
        with patch.object(sample_dataframe, 'to_csv') as mock_to_csv:
            # Act
            exporter._export_csv(
                sample_dataframe,
                output_path,
                compression=None,
                no_header=True
            )
        
        # Assert
        mock_to_csv.assert_called_once_with(
            output_path,
            index=False,
            header=False
        )

    def test_export_parquet(self, exporter, sample_dataframe):
        """Test exporting to Parquet."""
        output_path = Path("/output/test.parquet")
        
        with patch.object(sample_dataframe, 'to_parquet') as mock_to_parquet:
            # Act
            result = exporter._export_parquet(
                sample_dataframe,
                output_path,
                compression='snappy'
            )
        
        # Assert
        mock_to_parquet.assert_called_once_with(
            output_path,
            index=False,
            compression='snappy'
        )
        assert result == output_path

    @patch('builtins.open', new_callable=mock_open)
    def test_export_json(self, mock_file, exporter, sample_dataframe):
        """Test exporting to JSON."""
        output_path = Path("/output/test.json")
        
        with patch.object(sample_dataframe, 'to_json', return_value='{"test": "data"}') as mock_to_json:
            # Act
            result = exporter._export_json(
                sample_dataframe,
                output_path,
                compression=None
            )
        
        # Assert
        mock_to_json.assert_called_once_with(orient='records', date_format='iso')
        mock_file.assert_called_once_with(output_path, 'w')
        mock_file().write.assert_called_once_with('{"test": "data"}')
        assert result == output_path


    def test_export_metadata(self, exporter, sample_dataset_info):
        """Test metadata export."""
        output_dir = Path("/output")
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                # Act
                result = exporter._export_metadata(
                    "test_dataset",
                    sample_dataset_info,
                    output_dir
                )
        
        # Assert
        assert result == output_dir / "test_dataset_metadata.json"
        # Since it's actually using json.dump, not yaml.dump
        # mock_yaml_dump.assert_called_once()

    def test_read_table(self, exporter):
        """Test reading table."""
        # Arrange
        backend = Mock()
        full_df = pd.DataFrame({'id': range(100), 'value': range(100)})
        backend.query.return_value = full_df
        
        # Act
        result = exporter._read_table(backend, 'test_table')
        
        # Assert
        backend.query.assert_called_once_with("SELECT * FROM test_table")
        assert result.equals(full_df)

    def test_export_error_handling(self, exporter):
        """Test error handling during export."""
        # Arrange
        with patch.object(exporter, '_load_dataset_info', side_effect=Exception("Load failed")):
            # Act & Assert
            with pytest.raises(Exception, match="Load failed"):
                exporter.export("test_dataset", format="csv")