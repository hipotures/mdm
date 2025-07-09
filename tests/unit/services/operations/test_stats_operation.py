"""Unit tests for StatsOperation."""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json
import yaml

from mdm.dataset.operations import StatsOperation


class TestStatsOperation:
    """Test cases for StatsOperation."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.paths.configs_path = "config/datasets"
        config.paths.datasets_path = "datasets"
        return config

    @pytest.fixture
    def stats_operation(self, mock_config):
        """Create StatsOperation instance."""
        with patch('mdm.dataset.operations.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = Path("/test")
            mock_get_config.return_value = mock_manager
            
            operation = StatsOperation()
            return operation

    @pytest.fixture
    def mock_statistics(self):
        """Mock DatasetStatistics."""
        with patch('mdm.dataset.operations.DatasetStatistics') as mock_class:
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
            'missing_values': {
                'col1': 5,
                'col2': 10
            },
            'column_types': {
                'col1': 'int64',
                'col2': 'float64',
                'col3': 'object'
            },
            'computed_at': '2024-01-01T00:00:00'
        }

    def test_execute_load_existing_stats(self, stats_operation, mock_statistics, sample_stats):
        """Test loading pre-computed statistics."""
        # Arrange
        mock_statistics.load_statistics.return_value = sample_stats

        # Act
        result = stats_operation.execute("test_dataset")

        # Assert
        mock_statistics.load_statistics.assert_called_once_with("test_dataset")
        mock_statistics.compute_statistics.assert_not_called()
        assert result == sample_stats

    def test_execute_compute_new_stats(self, stats_operation, mock_statistics, sample_stats):
        """Test computing new statistics when none exist."""
        # Arrange
        mock_statistics.load_statistics.return_value = None
        mock_statistics.compute_statistics.return_value = sample_stats

        # Act
        result = stats_operation.execute("test_dataset")

        # Assert
        mock_statistics.load_statistics.assert_called_once_with("test_dataset")
        mock_statistics.compute_statistics.assert_called_once_with(
            "test_dataset",
            full=False,
            save=True
        )
        assert result == sample_stats

    def test_execute_force_full_computation(self, stats_operation, mock_statistics, sample_stats):
        """Test forcing full statistics computation."""
        # Arrange
        old_stats = {'dataset_name': 'test_dataset', 'row_count': 500}
        mock_statistics.load_statistics.return_value = old_stats
        mock_statistics.compute_statistics.return_value = sample_stats

        # Act
        result = stats_operation.execute("test_dataset", full=True)

        # Assert
        mock_statistics.compute_statistics.assert_called_once_with(
            "test_dataset",
            full=True,
            save=True
        )
        assert result == sample_stats

    def test_execute_export_to_json(self, stats_operation, mock_statistics, sample_stats):
        """Test exporting statistics to JSON."""
        # Arrange
        mock_statistics.load_statistics.return_value = sample_stats
        export_path = Path("/tmp/stats.json")

        # Act
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                result = stats_operation.execute(
                    "test_dataset",
                    export=export_path
                )

        # Assert
        mock_file.assert_called_once_with(export_path, 'w')
        mock_json_dump.assert_called_once()
        call_args = mock_json_dump.call_args[0]
        assert call_args[0] == sample_stats
        assert call_args[1] == mock_file()

    def test_execute_export_to_yaml(self, stats_operation, mock_statistics, sample_stats):
        """Test exporting statistics to YAML."""
        # Arrange
        mock_statistics.load_statistics.return_value = sample_stats
        export_path = Path("/tmp/stats.yaml")

        # Act
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('yaml.dump') as mock_yaml_dump:
                result = stats_operation.execute(
                    "test_dataset",
                    export=export_path
                )

        # Assert
        mock_file.assert_called_once_with(export_path, 'w')
        mock_yaml_dump.assert_called_once()
        call_args = mock_yaml_dump.call_args[0]
        assert call_args[0] == sample_stats
        assert call_args[1] == mock_file()

    def test_execute_export_default_format(self, stats_operation, mock_statistics, sample_stats):
        """Test export defaults to JSON for unknown extensions."""
        # Arrange
        mock_statistics.load_statistics.return_value = sample_stats
        export_path = Path("/tmp/stats.txt")

        # Act
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                result = stats_operation.execute(
                    "test_dataset",
                    export=export_path
                )

        # Assert
        mock_json_dump.assert_called_once()

    def test_execute_export_error_handling(self, stats_operation, mock_statistics, sample_stats):
        """Test error handling during export."""
        # Arrange
        mock_statistics.load_statistics.return_value = sample_stats
        export_path = Path("/tmp/stats.json")

        # Act
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with patch('mdm.dataset.operations.logger') as mock_logger:
                result = stats_operation.execute(
                    "test_dataset",
                    export=export_path
                )

        # Assert
        mock_logger.error.assert_called_once()
        assert "Failed to export statistics" in mock_logger.error.call_args[0][0]
        assert result == sample_stats  # Should still return stats

    def test_execute_compute_and_export(self, stats_operation, mock_statistics, sample_stats):
        """Test computing new stats and exporting in one call."""
        # Arrange
        mock_statistics.load_statistics.return_value = None
        mock_statistics.compute_statistics.return_value = sample_stats
        export_path = Path("/tmp/stats.json")

        # Act
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                result = stats_operation.execute(
                    "test_dataset",
                    full=True,
                    export=export_path
                )

        # Assert
        mock_statistics.compute_statistics.assert_called_once()
        mock_json_dump.assert_called_once()
        assert result == sample_stats

    def test_execute_statistics_error(self, stats_operation, mock_statistics):
        """Test handling of statistics computation errors."""
        # Arrange
        mock_statistics.load_statistics.return_value = None
        mock_statistics.compute_statistics.side_effect = Exception("Computation failed")

        # Act & Assert
        with pytest.raises(Exception, match="Computation failed"):
            stats_operation.execute("test_dataset")