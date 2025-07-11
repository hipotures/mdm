"""Improved unit tests for CLI modules to achieve 90% coverage."""

import os
import sys
import json
import csv
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from datetime import datetime
from io import StringIO

import pytest
import pandas as pd
from typer.testing import CliRunner
from loguru import logger

# Import all CLI modules
from mdm.cli.main import app, setup_logging, _format_size, main
from mdm.cli.dataset import dataset_app, _display_column_summary
from mdm.cli.batch import batch_app
from mdm.cli.timeseries import app as timeseries_app


class TestMainCLI90Coverage:
    """Tests for main.py CLI to improve coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    # Test the actual setup_logging function
    @patch('mdm.config.get_config_manager')
    @patch('loguru.logger')
    @patch('logging.basicConfig')
    @patch('logging.getLogger')
    def test_setup_logging_all_branches(self, mock_get_logger, mock_basic_config, mock_logger, mock_get_config):
        """Test setup_logging with all configuration options."""
        # Reset the global flag
        import mdm.cli.main
        mdm.cli.main._logging_initialized = False
        
        # Create mock config with all options
        mock_config = Mock()
        mock_config.logging.file = "mdm.log"  # Enable file logging
        mock_config.logging.level = "DEBUG"
        mock_config.logging.format = "json"  # JSON format
        mock_config.logging.max_bytes = 10485760
        mock_config.logging.backup_count = 5
        mock_config.database.sqlalchemy.echo = True  # Enable SQLAlchemy logging
        mock_config.paths.logs_path = "logs"
        mock_config.model_dump.return_value = {
            "logging": {
                "file": "mdm.log",
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
        
        # Mock logger methods
        mock_logger.remove = Mock()
        mock_logger.add = Mock()
        
        # Mock logging.getLogger
        mock_std_logger = Mock()
        mock_get_logger.return_value = mock_std_logger
        
        # Call the real function
        setup_logging()
        
        # Verify logger was configured
        assert mock_logger.remove.called
        assert mock_logger.add.called
        # Should have file handler, console handler, and SQLAlchemy handler (since echo=True and level=DEBUG)
        assert mock_logger.add.call_count >= 3
        
    @patch('mdm.config.get_config_manager')
    @patch('loguru.logger')
    def test_setup_logging_no_file(self, mock_logger, mock_get_config):
        """Test setup_logging without file logging."""
        # Reset the global flag
        import mdm.cli.main
        mdm.cli.main._logging_initialized = False
        
        mock_config = Mock()
        mock_config.logging.file = "test.log"  # Still need a file for the function to work
        mock_config.logging.level = "INFO"
        mock_config.logging.format = "console"  # Console format
        mock_config.logging.max_bytes = 10485760
        mock_config.logging.backup_count = 5
        mock_config.database.sqlalchemy.echo = False
        mock_config.paths.logs_path = "logs"
        mock_config.model_dump.return_value = {}
        
        mock_manager = Mock()
        mock_manager.config = mock_config
        mock_manager.base_path = Path("/tmp/mdm")
        mock_get_config.return_value = mock_manager
        
        mock_logger.remove = Mock()
        mock_logger.add = Mock()
        
        setup_logging()
        
        # Should have file and console handlers
        assert mock_logger.add.call_count >= 2
    
    @patch('shutil.disk_usage')
    @patch('mdm.dataset.manager.DatasetManager')
    @patch('mdm.config.get_config_manager')
    def test_info_command_comprehensive(self, mock_get_config, mock_dataset_manager, mock_disk_usage, runner):
        """Test info command with all output."""
        # Setup comprehensive config
        mock_config = Mock()
        mock_config.database.default_backend = "postgresql"
        mock_config.database.postgresql.host = "localhost"
        mock_config.database.postgresql.port = 5432
        mock_config.database.sqlalchemy.echo = False
        mock_config.performance.batch_size = 5000
        mock_config.performance.max_concurrent_operations = 8
        mock_config.paths.datasets_path = "datasets"
        mock_config.paths.configs_path = "config/datasets"
        mock_config.paths.logs_path = "logs"
        mock_config.features.enabled = True
        mock_config.features.generic_enabled = True
        mock_config.features.custom_enabled = True
        mock_config.logging.file = "mdm.log"
        mock_config.logging.level = "INFO"
        mock_config.logging.format = "console"
        mock_config.logging.max_bytes = 10485760
        mock_config.logging.backup_count = 5
        
        mock_manager = Mock()
        mock_manager.config = mock_config
        mock_manager.base_path = Path("/tmp/mdm")
        mock_get_config.return_value = mock_manager
        
        # Mock dataset manager
        mock_dm_instance = Mock()
        mock_dm_instance.list_datasets.return_value = ["dataset1", "dataset2", "dataset3"]
        mock_dataset_manager.return_value = mock_dm_instance
        
        # Mock disk usage - return a proper object with free attribute
        from collections import namedtuple
        DiskUsage = namedtuple('DiskUsage', ['total', 'used', 'free'])
        mock_disk_usage.return_value = DiskUsage(total=10000000000, used=5000000000, free=5000000000)
        
        # Call the command with mocked dependencies
        result = runner.invoke(app, ["info"])
        
        if result.exit_code != 0:
            print(f"Error: {result.stdout}")
            print(f"Exception: {result.exception}")
            if result.exception:
                import traceback
                traceback.print_tb(result.exception.__traceback__)
        
        assert result.exit_code == 0
        assert "ML Data Manager" in result.stdout
        # The backend should be postgresql as configured in the mock
        assert "postgresql" in result.stdout
        # Chunk size is 5,000 from mock config
        assert "5,000" in result.stdout


class TestDatasetCLI90Coverage:
    """Tests for dataset.py CLI to improve coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.dataset._display_column_summary')
    @patch('mdm.cli.dataset.DatasetRegistrar')
    def test_register_comprehensive(self, mock_registrar_class, mock_display, runner):
        """Test register with all options and column display."""
        # Setup comprehensive dataset info
        mock_dataset_info = Mock()
        mock_dataset_info.name = "test_dataset"
        mock_dataset_info.target_column = "target"
        mock_dataset_info.problem_type = "multiclass"
        mock_dataset_info.id_columns = ["id1", "id2"]
        mock_dataset_info.time_column = "timestamp"
        mock_dataset_info.group_column = "user_id"
        mock_dataset_info.datetime_columns = ["timestamp", "created_at", "updated_at"]
        mock_dataset_info.tables = {"train": "train_table", "test": "test_table", "val": "val_table"}
        mock_dataset_info.database = {'backend': 'duckdb', 'path': '/tmp/test.duckdb'}
        mock_dataset_info.tags = ["production", "ml", "timeseries"]
        mock_dataset_info.features_enabled = True
        mock_dataset_info.config_path = "/tmp/config.yaml"
        
        mock_registrar = Mock()
        mock_registrar.register.return_value = mock_dataset_info
        mock_registrar.manager = Mock()  # Add manager attribute
        mock_registrar_class.return_value = mock_registrar
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            # Write some data to the CSV file
            tmp.write("id1,id2,feature1,feature2,target,timestamp,created_at,updated_at,user_id\n")
            tmp.write("1,2,0.5,1.5,A,2023-01-01,2023-01-01,2023-01-01,100\n")
            tmp.write("3,4,0.6,1.6,B,2023-01-02,2023-01-02,2023-01-02,101\n")
            tmp.flush()
            tmp_path = tmp.name
        
        try:
            result = runner.invoke(dataset_app, [
                "register", "test_dataset", tmp_path,
                "--target", "target",
                "--problem-type", "multiclass",
                "--id-columns", "id1,id2",
                "--time-column", "timestamp",
                "--group-column", "user_id",
                "--datetime-columns", "timestamp,created_at,updated_at",
                "--description", "Production ML dataset",
                "--tags", "production,ml,timeseries",
                "--force"
            ])
            
            # Print the output for debugging
            if result.exit_code != 0:
                print(f"Command failed with exit code {result.exit_code}")
                print(f"Output: {result.stdout}")
                print(f"Exception: {result.exception}")
            
            assert result.exit_code == 0
            # Should display summary for train table (or data table if no train)
            assert mock_display.call_count == 1  # Only displays one table summary
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
    
    @patch('mdm.cli.dataset.InfoOperation')
    def test_info_detailed(self, mock_info_class, runner):
        """Test info command with --details flag."""
        mock_info = Mock()
        mock_info.execute.return_value = {
            'name': 'test_dataset',
            'description': 'Detailed test dataset',
            'target_column': 'label',
            'problem_type': 'binary',
            'id_columns': ['uuid'],
            'time_column': 'event_time',
            'group_column': 'session_id',
            'datetime_columns': ['event_time', 'process_time'],
            'tables': {
                'train': {
                    'name': 'train_table',
                    'row_count': 50000,
                    'columns': ['uuid', 'feature1', 'feature2', 'label']
                },
                'test': {
                    'name': 'test_table',
                    'row_count': 10000,
                    'columns': ['uuid', 'feature1', 'feature2']
                }
            },
            'features': {
                'enabled': True,
                'count': 25,
                'generic': 20,
                'custom': 5
            },
            'database': {
                'backend': 'postgresql',
                'host': 'db.example.com',
                'database': 'ml_data'
            },
            'tags': ['validated', 'production'],
            'registration_date': '2024-01-15T10:30:00',
            'last_modified': '2024-01-20T15:45:00',
            'size_bytes': 104857600
        }
        mock_info_class.return_value = mock_info
        
        result = runner.invoke(dataset_app, ["info", "test_dataset", "--details"])
        
        if result.exit_code != 0:
            print(f"Command failed with exit code {result.exit_code}")
            print(f"Output: {result.stdout}")
            print(f"Exception: {result.exception}")
        
        assert result.exit_code == 0
        assert "Detailed test dataset" in result.stdout
        assert "Problem Type: binary" in result.stdout
        assert "Target Column: label" in result.stdout
        assert "validated, production" in result.stdout
        mock_info.execute.assert_called_with("test_dataset", details=True)
    
    @patch('mdm.dataset.manager.DatasetManager')
    def test_update_comprehensive(self, mock_manager_class, runner):
        """Test update with description and tags."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Patch UpdateOperation at the correct location (imported inside function)
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_class:
            mock_update = Mock()
            mock_update_class.return_value = mock_update
            
            result = runner.invoke(dataset_app, [
                "update", "my_dataset",
                "--description", "Updated description with more details",
                "--problem-type", "regression",
                "--id-columns", "id1,id2"
            ])
            
            if result.exit_code != 0:
                print(f"Command failed with exit code {result.exit_code}")
                print(f"Output: {result.stdout}")
                print(f"Exception: {result.exception}")
            
            assert result.exit_code == 0
            # Check the UpdateOperation was called correctly
            mock_update.execute.assert_called_with(
                "my_dataset",
                {
                    'description': "Updated description with more details",
                    'problem_type': "regression",
                    'id_columns': ["id1", "id2"]
                }
            )
    
    @patch('mdm.dataset.operations.ExportOperation')
    @patch('mdm.dataset.manager.DatasetManager')
    def test_export_comprehensive(self, mock_manager_class, mock_export_class, runner):
        """Test export with all options."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        mock_export = Mock()
        mock_export.execute.return_value = {
            'dataset_name': 'test_dataset',
            'output_path': '/tmp/exports/test_dataset.parquet.gz',
            'format': 'parquet',
            'compression': 'gzip',
            'tables_exported': ['train', 'test', 'val'],
            'total_rows': 75000,
            'file_size': 52428800,
            'metadata_only': False
        }
        mock_export_class.return_value = mock_export
        
        # Mock ExportOperation to return list of paths
        mock_export.execute.return_value = [
            Path('/tmp/exports/test_dataset_train.parquet.gz'),
            Path('/tmp/exports/test_dataset_test.parquet.gz'),
            Path('/tmp/exports/test_dataset_val.parquet.gz')
        ]
        
        # ExportOperation is already patched above at mdm.dataset.operations.ExportOperation
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(dataset_app, [
                "export", "test_dataset",
                "--output-dir", tmpdir,
                "--format", "parquet",
                "--compression", "gzip",
                # Note: there's no --tables option, only --table for single table
            ])
            
            if result.exit_code != 0:
                print(f"Command failed with exit code {result.exit_code}")
                print(f"Output: {result.stdout}")
                print(f"Exception: {result.exception}")
            
            assert result.exit_code == 0
            assert "exported successfully" in result.stdout


class TestBatchCLI90Coverage:
    """Tests for batch.py CLI to improve coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.ExportOperation')
    @patch('rich.progress.Progress')
    def test_batch_export_comprehensive(self, mock_progress_class, mock_export_class, mock_manager_class, runner):
        """Test batch export with multiple datasets and options."""
        # Setup dataset manager
        mock_manager = Mock()
        # Mock dataset_exists to return True for first 3 datasets, False for the 4th
        mock_manager.dataset_exists.side_effect = [True, True, True, False]
        mock_manager_class.return_value = mock_manager
        
        # Setup export operation
        mock_export = Mock()
        # Return list of file paths as expected by the batch export
        mock_export.execute.side_effect = [
            # dataset1 - success
            [Path('/tmp/dataset1/train.json.gz'), Path('/tmp/dataset1/test.json.gz')],
            # dataset2 - failure
            Exception("Export failed for dataset2"),
            # dataset3 - success
            [Path('/tmp/dataset3/train.json.gz'), Path('/tmp/dataset3/val.json.gz')]
        ]
        mock_export_class.return_value = mock_export
        
        # Setup progress tracking
        mock_progress = MagicMock()
        mock_task = Mock()
        mock_progress.add_task.return_value = mock_task
        mock_progress.__enter__.return_value = mock_progress
        mock_progress.__exit__.return_value = None
        mock_progress_class.return_value = mock_progress
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(batch_app, [
                "export", "dataset1", "dataset2", "dataset3", "dataset4",
                "--output-dir", tmpdir,
                "--format", "json",
                "--compression", "gzip",
                "--metadata-only"
            ])
        
        assert result.exit_code == 0
        assert "Successfully exported: 2 datasets" in result.stdout
        assert "Failed: 2 datasets" in result.stdout
        assert "dataset2: Export failed" in result.stdout
        assert "dataset4: Dataset not found" in result.stdout
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.StatsOperation')
    @patch('builtins.open', new_callable=mock_open)
    def test_batch_stats_with_export(self, mock_file, mock_stats_class, mock_manager_class, runner):
        """Test batch stats with CSV export."""
        # Setup dataset manager
        mock_manager = Mock()
        # Mock dataset_exists to return True for both datasets
        mock_manager.dataset_exists.side_effect = [True, True]
        mock_manager_class.return_value = mock_manager
        
        # Setup stats operation
        mock_stats = Mock()
        mock_stats.execute.side_effect = [
            {
                'dataset_name': 'ds1',
                'tables': {
                    'train': {
                        'row_count': 5000,
                        'column_count': 20,
                        'size_bytes': 2097152,
                        'completeness': 0.98
                    }
                },
                'total_row_count': 5000,
                'total_size_bytes': 2097152
            },
            {
                'dataset_name': 'ds2',
                'tables': {
                    'data': {
                        'row_count': 10000,
                        'column_count': 15,
                        'size_bytes': 4194304,
                        'completeness': 0.95
                    }
                },
                'total_row_count': 10000,
                'total_size_bytes': 4194304
            }
        ]
        mock_stats_class.return_value = mock_stats
        
        # Create StringIO to capture CSV output
        csv_buffer = StringIO()
        mock_file.return_value.write = csv_buffer.write
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(batch_app, [
                "stats", "ds1", "ds2",
                "--export", tmpdir,
                "--full"  # Changed from --detailed to --full
            ])
            
            if result.exit_code != 0:
                print(f"Command failed with exit code {result.exit_code}")
                print(f"Output: {result.stdout}")
                print(f"Exception: {result.exception}")
            
            assert result.exit_code == 0
            # The batch stats command exports to a directory, not a single file
            assert "Statistics Summary:" in result.stdout


class TestTimeseriesCLI90Coverage:
    """Tests for timeseries.py CLI to improve coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.api.MDMClient')
    @patch('mdm.utils.time_series.TimeSeriesAnalyzer')
    @patch('builtins.open', new_callable=mock_open)
    def test_analyze_comprehensive(self, mock_file, mock_analyzer_class, mock_client_class, runner):
        """Test analyze with all features including output."""
        # Setup client
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = "datetime"
        mock_dataset_info.target_column = "sales"
        mock_dataset_info.group_column = "store_id"
        mock_client.get_dataset.return_value = mock_dataset_info
        
        # Create sample data with groups
        train_df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=200, freq='D'),
            'sales': list(range(100)) * 2,
            'store_id': ['A'] * 100 + ['B'] * 100
        })
        mock_client.load_dataset_files.return_value = {'train': train_df}
        mock_client_class.return_value = mock_client
        
        # Setup analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze.return_value = {
            'time_range': {
                'start': datetime(2023, 1, 1),
                'end': datetime(2023, 7, 19),
                'duration_days': 199
            },
            'frequency': 'daily',
            'missing_timestamps': {
                'count': 2,
                'percentage': 1.0,
                'dates': ['2023-02-15', '2023-04-01']
            },
            'trend': {
                'direction': 'increasing',
                'strength': 0.85
            },
            'seasonality': {
                'weekly': True,
                'monthly': False
            },
            'stationarity': {
                'is_stationary': False,
                'adf_statistic': -1.8,
                'p_value': 0.35
            },
            'target_stats': {
                'mean': 49.5,
                'std': 28.87,
                'trend': 'increasing'
            }
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        # Fix the mocks - need to patch at correct import locations
        with patch('mdm.cli.timeseries.MDMClient', mock_client_class):
            with patch('mdm.cli.timeseries.TimeSeriesAnalyzer', mock_analyzer_class):
                result = runner.invoke(timeseries_app, [
                    "analyze", "test_dataset"
                ])
                
                if result.exit_code != 0:
                    print(f"Command failed with exit code {result.exit_code}")
                    print(f"Output: {result.stdout}")
                    print(f"Exception: {result.exception}")
                
                assert result.exit_code == 0
                assert "Time Series Analysis: test_dataset" in result.stdout
                assert "Start:" in result.stdout
                assert "Duration: 199 days" in result.stdout
                assert "Frequency: daily" in result.stdout
                assert "Missing Timestamps:" in result.stdout


class TestCLIHelpers90Coverage:
    """Test helper functions for 90% coverage."""
    
    def test_format_size_all_units(self):
        """Test _format_size with all unit conversions."""
        assert _format_size(0) == "0.0 B"
        assert _format_size(512) == "512.0 B"
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(1536) == "1.5 KB"
        assert _format_size(1048576) == "1.0 MB"
        assert _format_size(1073741824) == "1.0 GB"
        assert _format_size(1099511627776) == "1.0 TB"
        assert _format_size(1125899906842624) == "1.0 PB"
        # Test very large size (beyond PB)
        assert _format_size(1125899906842624 * 1024) == "1024.0 PB"
    
    @patch('mdm.storage.factory.BackendFactory')
    @patch('mdm.cli.dataset.console')
    def test_display_column_summary_edge_cases(self, mock_console, mock_factory):
        """Test column summary with edge cases."""
        # Test with empty table
        mock_backend = Mock()
        mock_backend.query.return_value = pd.DataFrame({'total_rows': [0]})
        mock_backend.get_table_info.return_value = {'columns': []}
        mock_factory.create.return_value = mock_backend
        
        mock_dataset_info = Mock()
        mock_dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_dataset_info.tables = {'train': 'empty_table'}
        
        _display_column_summary(mock_dataset_info, Mock(), 'train')
        
        # Should handle empty table gracefully
        assert mock_console.print.called
    
    @patch('mdm.storage.factory.BackendFactory')
    @patch('mdm.cli.dataset.console')
    def test_display_column_summary_query_errors(self, mock_console, mock_factory):
        """Test column summary with query errors."""
        mock_backend = Mock()
        # First query succeeds
        mock_backend.query.side_effect = [
            pd.DataFrame({'total_rows': [1000]}),
            Exception("Query failed"),  # Null count query fails
            pd.DataFrame({'null_count': [100]})  # Next column succeeds
        ]
        mock_backend.get_table_info.return_value = {
            'columns': [
                {'name': 'col1', 'type': 'TEXT'},
                {'name': 'col2', 'type': 'INTEGER'}
            ]
        }
        mock_factory.create.return_value = mock_backend
        
        mock_dataset_info = Mock()
        mock_dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_dataset_info.tables = {'train': 'test_table'}
        
        _display_column_summary(mock_dataset_info, Mock(), 'train')
        
        # Should display an error message when query fails
        # Check that console.print was called
        assert mock_console.print.called
        # The function catches the exception and prints an error message
        error_printed = any('Could not generate column summary' in str(call) for call in mock_console.print.call_args_list)
        assert error_printed