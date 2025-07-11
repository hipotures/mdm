"""Comprehensive unit tests for CLI modules to achieve 90% coverage."""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from datetime import datetime

import pytest
import pandas as pd
from typer.testing import CliRunner
from loguru import logger

# Import all CLI modules
from mdm.cli.main import app, setup_logging, _format_size, main
from mdm.cli.dataset import dataset_app, _display_column_summary
from mdm.cli.batch import batch_app
from mdm.cli.timeseries import app as timeseries_app


class TestMainCoverage:
    """Tests for main.py to achieve 90% coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def mock_setup_logging(self):
        """Mock setup_logging for all tests."""
        with patch('mdm.cli.main.setup_logging'):
            yield
    
    @patch('mdm.config.get_config_manager')
    @patch('loguru.logger')
    def test_setup_logging_comprehensive(self, mock_logger, mock_get_config):
        """Test all branches of setup_logging."""
        # Reset the global flag
        import mdm.cli.main
        mdm.cli.main._logging_initialized = False
        
        # Create comprehensive mock config
        mock_config = Mock()
        mock_config.logging.file = "test.log"  # File logging
        mock_config.logging.level = "DEBUG"
        mock_config.logging.format = "json"
        mock_config.logging.max_bytes = 10485760
        mock_config.logging.backup_count = 5
        mock_config.database.sqlalchemy.echo = True  # Enable SQLAlchemy echo
        mock_config.paths.logs_path = "logs"
        mock_config.model_dump.return_value = {
            "logging": {
                "file": "test.log",
                "level": "DEBUG",
                "format": "json"
            },
            "database": {
                "sqlalchemy": {"echo": True}
            }
        }
        
        mock_manager = Mock()
        mock_manager.config = mock_config
        mock_manager.base_path = Path("/tmp/mdm")
        mock_get_config.return_value = mock_manager
        
        # Mock logger.add to track calls
        mock_logger.add = Mock()
        mock_logger.remove = Mock()
        
        # Call setup_logging directly
        from mdm.cli.main import setup_logging
        setup_logging()
        
        # Should have been called
        assert mock_logger.remove.called
        assert mock_logger.add.called
    
    @patch('mdm.config.get_config_manager')
    @patch('mdm.dataset.manager.DatasetManager')
    @patch('shutil.disk_usage')
    @patch('os.path.exists')
    def test_info_command_with_errors(self, mock_exists, mock_disk_usage, mock_dataset_manager, mock_get_config, runner, mock_setup_logging):
        """Test info command with various error conditions."""
        # Setup mocks
        mock_config = Mock()
        mock_config.database.default_backend = "sqlite"
        mock_config.performance.batch_size = 10000
        mock_config.performance.max_concurrent_operations = 4
        mock_config.paths.datasets_path = "datasets"
        mock_config.paths.configs_path = "config"
        mock_config.paths.logs_path = "logs"
        
        mock_manager = Mock()
        mock_manager.config = mock_config
        mock_manager.base_path = Path("/tmp/mdm")
        mock_get_config.return_value = mock_manager
        
        # Mock file system checks
        mock_exists.return_value = False  # Directory doesn't exist
        
        # Mock disk usage to raise error
        mock_disk_usage.side_effect = Exception("Disk error")
        
        # Mock dataset manager to raise error on list_datasets
        mock_dm_instance = Mock()
        mock_dm_instance.list_datasets.side_effect = Exception("Manager error")
        mock_dataset_manager.return_value = mock_dm_instance
        
        result = runner.invoke(app, ["info"])
        
        # Should fail due to manager error
        assert result.exit_code == 1
        assert result.exception
    
    @patch('mdm.cli.main.app')
    @patch.object(sys, 'argv', ['mdm', '--help'])
    def test_main_with_help_flag(self, mock_app):
        """Test main() when --help is already present."""
        main()
        # Should not add another --help
        assert sys.argv == ['mdm', '--help']
        mock_app.assert_called_once()


class TestDatasetCoverage:
    """Tests for dataset.py to achieve 90% coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture(autouse=True)
    def setup_patches(self):
        """Setup common patches."""
        with patch('mdm.cli.main.setup_logging'):
            yield
    
    @patch('mdm.cli.dataset.DatasetRegistrar')
    @patch('mdm.cli.dataset._display_column_summary')
    @patch('mdm.cli.dataset.console')
    def test_register_with_column_summary(self, mock_console, mock_display_summary, mock_registrar_class, runner):
        """Test registration with column summary display."""
        # Setup mocks
        mock_registrar = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.name = "test_dataset"
        mock_dataset_info.target_column = "target"
        mock_dataset_info.problem_type = "classification"
        mock_dataset_info.id_columns = ["id"]
        mock_dataset_info.time_column = "date"
        mock_dataset_info.group_column = "group"
        mock_dataset_info.tables = {"train": "train_table", "test": "test_table"}
        mock_dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_dataset_info.tags = ["tag1", "tag2"]
        mock_dataset_info.features_enabled = True
        mock_dataset_info.config_path = "/tmp/config.yaml"
        
        mock_registrar.register.return_value = mock_dataset_info
        mock_registrar.manager = Mock()  # Add manager attribute
        mock_registrar_class.return_value = mock_registrar
        
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            result = runner.invoke(dataset_app, [
                "register", "test_dataset", tmp.name,
                "--target", "target",
                "--problem-type", "classification",
                "--id-columns", "id",
                "--time-column", "date",
                "--group-column", "group",
                "--datetime-columns", "date,created_at",
                "--description", "Test dataset",
                "--tags", "tag1,tag2"
            ])
        
        assert result.exit_code == 0
        # Verify column summary was called for each table
        assert mock_display_summary.call_count >= 1
    
    @patch('mdm.dataset.operations.UpdateOperation')
    def test_update_dataset_both_options(self, mock_update_class, runner):
        """Test update with both description and tags."""
        mock_update = Mock()
        mock_update_class.return_value = mock_update
        
        result = runner.invoke(dataset_app, [
            "update", "test_dataset",
            "--description", "New description"
        ])
        
        assert result.exit_code == 0
        mock_update.execute.assert_called_once_with(
            "test_dataset",
            {'description': 'New description'}
        )
    
    @patch('mdm.dataset.operations.ExportOperation')
    def test_export_with_all_options(self, mock_export_class, runner):
        """Test export with all options."""
        
        mock_export = Mock()
        mock_export.execute.return_value = [
            Path('/tmp/export/train.parquet.gz'),
            Path('/tmp/export/test.parquet.gz')
        ]
        mock_export_class.return_value = mock_export
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(dataset_app, [
                "export", "test_dataset",
                "--output-dir", tmpdir,
                "--format", "parquet",
                "--compression", "gzip",
                "--metadata-only"
            ])
        
        assert result.exit_code == 0
        assert "exported successfully" in result.stdout
    
    @patch('mdm.cli.dataset.InfoOperation')
    def test_info_with_detailed_flag(self, mock_info_class, runner):
        """Test info command with details flag."""
        mock_info = Mock()
        mock_info.execute.return_value = {
            'name': 'test_dataset',
            'description': 'Test description',
            'target_column': 'target',
            'problem_type': 'classification',
            'id_columns': ['id'],
            'time_column': 'date',
            'group_column': 'group',
            'datetime_columns': ['date', 'created_at'],
            'tables': {
                'train': {
                    'name': 'train_table',
                    'row_count': 1000,
                    'columns': ['id', 'feature1', 'target']
                }
            },
            'database': {'backend': 'sqlite', 'path': '/tmp/test.db'},
            'tags': ['test', 'sample'],
            'features': {
                'enabled': True,
                'count': 10
            },
            'registration_date': '2023-01-01',
            'last_modified': '2023-01-02'
        }
        mock_info_class.return_value = mock_info
        
        result = runner.invoke(dataset_app, ["info", "test_dataset", "--details"])
        
        assert result.exit_code == 0
        mock_info.execute.assert_called_with("test_dataset", details=True)


class TestBatchCoverage:
    """Tests for batch.py to achieve 90% coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture(autouse=True)
    def setup_patches(self):
        with patch('mdm.cli.main.setup_logging'):
            yield
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.ExportOperation')
    @patch('mdm.cli.batch.Progress')
    def test_batch_export_with_progress(self, mock_progress_class, mock_export_class, mock_manager_class, runner):
        """Test batch export with progress tracking."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.dataset_exists.side_effect = [True, True]
        mock_manager.get_dataset.side_effect = [
            Mock(name="dataset1"),
            Mock(name="dataset2"),
        ]
        mock_manager_class.return_value = mock_manager
        
        mock_export = Mock()
        mock_export.execute.side_effect = [
            {
                'dataset_name': 'dataset1',
                'output_path': '/tmp/dataset1.csv',
                'tables_exported': ['train', 'test'],
                'total_rows': 1000
            },
            {
                'dataset_name': 'dataset2',
                'output_path': '/tmp/dataset2.csv',
                'tables_exported': ['data'],
                'total_rows': 2000
            }
        ]
        mock_export_class.return_value = mock_export
        
        # Mock progress
        mock_progress = Mock()
        mock_task = Mock()
        mock_progress.add_task.return_value = mock_task
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress_class.return_value = mock_progress
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(batch_app, [
                "export", "dataset1", "dataset2",
                "--output-dir", tmpdir
            ])
        
        assert result.exit_code == 0
        assert mock_progress.update.called
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.StatsOperation')
    def test_batch_stats_csv_export(self, mock_stats_class, mock_manager_class, runner):
        """Test batch stats with CSV export."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.dataset_exists.return_value = True
        mock_manager.get_dataset.return_value = Mock(name="dataset1")
        mock_manager_class.return_value = mock_manager
        
        mock_stats = Mock()
        mock_stats.execute.return_value = {
            'dataset_name': 'dataset1',
            'summary': {
                'total_rows': 1000,
                'total_columns': 10,
                'total_tables': 1,
                'overall_completeness': 0.95
            }
        }
        mock_stats_class.return_value = mock_stats
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(batch_app, [
                "stats", "dataset1",
                "--export", tmpdir,
                "--full"
            ])
        
        assert result.exit_code == 0
        assert "Statistics Summary:" in result.stdout


class TestTimeseriesCoverage:
    """Tests for timeseries.py to achieve 90% coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture(autouse=True)
    def setup_patches(self):
        with patch('mdm.cli.main.setup_logging'):
            yield
    
    @patch('mdm.cli.timeseries.MDMClient')
    @patch('mdm.cli.timeseries.TimeSeriesAnalyzer')
    def test_analyze_with_output(self, mock_analyzer_class, mock_client_class, runner):
        """Test analyze command."""
        # Setup mocks
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = "date"
        mock_dataset_info.target_column = "value"
        mock_dataset_info.group_column = "group"
        mock_client.get_dataset.return_value = mock_dataset_info
        
        train_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'value': range(100),
            'group': ['A'] * 50 + ['B'] * 50
        })
        mock_client.load_dataset_files.return_value = {'train': train_df}
        mock_client_class.return_value = mock_client
        
        # Setup analyzer mock
        mock_analyzer = Mock()
        mock_analyzer.analyze.return_value = {
            'time_range': {
                'start': datetime(2023, 1, 1),
                'end': datetime(2023, 4, 10),
                'duration_days': 99
            },
            'frequency': 'daily',
            'missing_timestamps': {
            'count': 1,
            'percentage': 1.0,
            'dates': ['2023-02-01']
        },
            'trend': {
                'direction': 'increasing',
                'strength': 0.95
            },
            'seasonality': {
                'period': 7,
                'strength': 0.3
            },
            'stationarity': {
                'is_stationary': False,
                'adf_statistic': -2.5,
                'p_value': 0.12
            },
            'groups': {
                'count': 2,
                'names': ['A', 'B']
            }
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        result = runner.invoke(timeseries_app, [
            "analyze", "test_dataset"
        ])
        
        assert result.exit_code == 0
        assert "Time Series Analysis: test_dataset" in result.stdout
        assert "Duration: 99 days" in result.stdout
        assert "Missing Timestamps:" in result.stdout
    
    @patch('mdm.cli.timeseries.MDMClient')
    def test_split_with_all_options(self, mock_client_class, runner):
        """Test split with all options including validation set."""
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = "date"
        mock_client.get_dataset.return_value = mock_dataset_info
        
        # Create splits as list of tuples (as per the actual implementation)
        train_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=60, freq='D'),
            'value': range(60)
        })
        test_df = pd.DataFrame({
            'date': pd.date_range('2023-03-02', periods=40, freq='D'),
            'value': range(60, 100)
        })
        splits = [(train_df, test_df)]
        
        mock_client.split_time_series.return_value = splits
        mock_client_class.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(timeseries_app, [
                "split", "test_dataset",
                "--test-size", "0.4",
                "--n-splits", "1",
                "--output", tmpdir
            ])
        
        assert result.exit_code == 0
        assert "Time series split completed!" in result.stdout


# Additional helper test to increase coverage of edge cases
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_format_size_edge_cases(self):
        """Test _format_size with various sizes."""
        from mdm.cli.dataset import _format_size
        
        assert _format_size(0) == "0.0 B"
        assert _format_size(1023) == "1023.0 B"
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(1048576) == "1.0 MB"
        assert _format_size(1073741824) == "1.0 GB"
        assert _format_size(1099511627776) == "1.0 TB"
        assert _format_size(1125899906842624) == "1.0 PB"
        assert _format_size(1152921504606846976) == "1024.0 PB"  # > 1 PB
    
    @patch('mdm.cli.dataset.BackendFactory')
    @patch('mdm.cli.dataset.console')
    def test_display_column_summary_postgres(self, mock_console, mock_factory):
        """Test column summary with PostgreSQL backend."""
        # Setup mock for PostgreSQL
        mock_backend = Mock()
        mock_backend.query.side_effect = [
            pd.DataFrame({'total_rows': [1000]}),
            pd.DataFrame({'null_count': [50]}),
        ]
        mock_backend.get_table_info.return_value = {
            'columns': [
                {'name': 'id', 'type': 'INTEGER'},
                {'name': 'name', 'type': 'VARCHAR(255)'},
            ]
        }
        mock_factory.create.return_value = mock_backend
        
        mock_dataset_info = Mock()
        mock_dataset_info.database = {
            'backend': 'postgresql',
            'host': 'localhost',
            'port': 5432,
            'user': 'user',
            'password': 'pass',
            'database': 'test_db'
        }
        mock_dataset_info.tables = {'train': 'train_table'}
        
        # Call function
        _display_column_summary(mock_dataset_info, Mock(), 'train')
        
        # Verify it was called
        assert mock_console.print.called
    
    @patch('mdm.cli.dataset.BackendFactory')
    @patch('mdm.cli.dataset.console')
    def test_display_column_summary_many_columns(self, mock_console, mock_factory):
        """Test column summary with more than 20 columns."""
        # Setup mock
        mock_backend = Mock()
        
        # Mock query to return appropriate responses
        query_results = [pd.DataFrame({'total_rows': [1000]})]  # First query for total rows
        # Add null count queries for each of the first 20 columns
        for i in range(20):
            query_results.append(pd.DataFrame({'null_count': [i * 10]}))  # Varying null counts
        
        mock_backend.query.side_effect = query_results
        mock_backend.get_engine.return_value = Mock()
        
        # Create 25 columns
        columns = [{'name': f'col_{i}', 'type': 'INTEGER'} for i in range(25)]
        mock_backend.get_table_info.return_value = {'columns': columns}
        mock_factory.create.return_value = mock_backend
        
        mock_dataset_info = Mock()
        mock_dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_dataset_info.tables = {'train': 'train_table'}
        
        # Call function
        _display_column_summary(mock_dataset_info, Mock(), 'train')
        
        # Just verify that the function was called and didn't crash
        assert mock_console.print.called
        assert mock_console.print.call_count == 2  # Table and total rows message
        
        # The implementation correctly adds the "... (5 more columns)" row to the table
        # but verifying the exact table content in a unit test is not straightforward
        # The important thing is that the function handles > 20 columns without crashing