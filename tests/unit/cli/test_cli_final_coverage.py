"""Final comprehensive tests for CLI modules to achieve 90% coverage."""

import os
import sys
import json
import csv
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, mock_open, ANY
from datetime import datetime
from io import StringIO

import pytest
import pandas as pd
from typer.testing import CliRunner
from loguru import logger
from rich.table import Table

# Import CLI apps
from mdm.cli.main import app
from mdm.cli.dataset import dataset_app
from mdm.cli.batch import batch_app
from mdm.cli.timeseries import app as timeseries_app


class TestMainCLIFinal:
    """Final tests for main.py to achieve 90% coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    # Test setup_logging directly without mocking it
    @patch('mdm.config.get_config_manager')
    @patch('loguru.logger')
    @patch('logging.basicConfig')
    @patch('logging.getLogger')
    def test_setup_logging_real_function(self, mock_get_logger, mock_basic_config, mock_logger, mock_get_config):
        """Test the real setup_logging function."""
        # Reset the global flag
        import mdm.cli.main
        mdm.cli.main._logging_initialized = False
        
        from mdm.cli.main import setup_logging
        
        # Create comprehensive mock config
        mock_config = Mock()
        mock_config.logging.file = "mdm.log"
        mock_config.logging.level = "DEBUG"
        mock_config.logging.format = "json"
        mock_config.logging.max_bytes = 10485760
        mock_config.logging.backup_count = 5
        mock_config.database.sqlalchemy.echo = True
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
        mock_manager.base_path = Path("/tmp/mdm_test")
        mock_get_config.return_value = mock_manager
        
        # Mock logger methods
        mock_logger.remove = Mock()
        mock_logger.add = Mock()
        
        # Mock logging.getLogger
        mock_sql_logger = Mock()
        mock_get_logger.return_value = mock_sql_logger
        
        # Call the real function
        setup_logging()
        
        # Verify logger was configured properly
        assert mock_logger.remove.called
        assert mock_logger.add.called
        # Should have file and console handlers, plus SQLAlchemy since echo=True and level=DEBUG
        assert mock_logger.add.call_count >= 3
        
    @patch('mdm.config.get_config_manager')
    @patch('loguru.logger')
    def test_setup_logging_absolute_path(self, mock_logger, mock_get_config):
        """Test setup_logging with absolute log file path."""
        # Reset the global flag
        import mdm.cli.main
        mdm.cli.main._logging_initialized = False
        
        from mdm.cli.main import setup_logging
        
        mock_config = Mock()
        mock_config.logging.file = "/var/log/mdm.log"  # Absolute path
        mock_config.logging.level = "INFO"
        mock_config.logging.format = "console"
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
        
        # Check that logger.add was called
        assert mock_logger.add.called
        # Check that the first call used the absolute path
        if mock_logger.add.call_args_list:
            file_add_call = mock_logger.add.call_args_list[0]
            log_path = file_add_call[0][0]
            assert str(log_path) == "/var/log/mdm.log" 
    
    @patch.dict(os.environ, {'MDM_LOGGING_LEVEL': 'ERROR'})
    @patch('mdm.config.get_config_manager')
    @patch('loguru.logger')
    def test_setup_logging_env_override(self, mock_logger, mock_get_config):
        """Test setup_logging with environment override."""
        # Reset the global flag
        import mdm.cli.main
        mdm.cli.main._logging_initialized = False
        
        from mdm.cli.main import setup_logging
        
        mock_config = Mock()
        mock_config.logging.file = "mdm.log"
        mock_config.logging.level = "INFO"  # Will be overridden by env
        mock_config.logging.format = "console"
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
        
        # Check that ERROR level from env was used
        assert mock_logger.add.called
        if mock_logger.add.call_args_list:
            file_add_call = mock_logger.add.call_args_list[0]
            assert file_add_call[1]['level'] == 'ERROR'
    
    def test_format_size_helper(self):
        """Test _format_size helper function."""
        from mdm.cli.main import _format_size
        
        assert _format_size(500) == "500.0 B"
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(1048576) == "1.0 MB"
        assert _format_size(1073741824) == "1.0 GB"
        assert _format_size(1099511627776) == "1.0 TB"
        assert _format_size(1125899906842624) == "1.0 PB"
    
    @patch('mdm.cli.main.app')
    def test_main_function_no_args(self, mock_app):
        """Test main() function with no arguments."""
        from mdm.cli.main import main
        
        with patch.object(sys, 'argv', ['mdm']):
            main()
            # Should append --help
            assert sys.argv == ['mdm', '--help']
            mock_app.assert_called_once()
    
    @patch('mdm.cli.main.app')
    def test_main_function_with_args(self, mock_app):
        """Test main() function with arguments."""
        from mdm.cli.main import main
        
        with patch.object(sys, 'argv', ['mdm', 'dataset', 'list']):
            main()
            # Should not modify argv
            assert sys.argv == ['mdm', 'dataset', 'list']
            mock_app.assert_called_once()


class TestDatasetCLIFinal:
    """Final tests for dataset.py to achieve 90% coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.main.setup_logging')
    @patch('mdm.cli.dataset.ListOperation')
    def test_dataset_list_json_format(self, mock_list_class, mock_setup, runner):
        """Test dataset list with JSON format."""
        mock_list = Mock()
        mock_list.execute.return_value = [
            {
                'name': 'dataset1',
                'problem_type': 'classification',
                'target_column': 'label'
            }
        ]
        mock_list_class.return_value = mock_list
        
        result = runner.invoke(dataset_app, ["list", "--format", "json"])
        
        assert result.exit_code == 0
        # Should output JSON
        data = json.loads(result.stdout)
        assert data[0]['name'] == 'dataset1'
        
        # Verify ListOperation was called with json format
        mock_list.execute.assert_called_with(
            format="json",
            filter_str=None,
            sort_by="name",
            limit=None
        )
    
    @patch('mdm.cli.main.setup_logging')
    def test_dataset_stats_tables_option(self, mock_setup, runner):
        """Test dataset stats with full option."""
        # Patch StatsOperation inside the function where it's imported
        with patch('mdm.dataset.operations.StatsOperation') as mock_stats_class:
            mock_stats = Mock()
            mock_stats.execute.return_value = {
                'dataset_name': 'test_dataset',
                'computed_at': '2025-07-08T12:00:00',
                'mode': 'full',
                'summary': {
                    'total_tables': 1,
                    'total_rows': 1000,
                    'total_columns': 5,
                    'overall_completeness': 0.95
                },
                'tables': {
                    'train': {'row_count': 1000}
                }
            }
            mock_stats_class.return_value = mock_stats
            
            result = runner.invoke(dataset_app, [
                "stats", "test_dataset",
                "--full"
            ])
            
            assert result.exit_code == 0
            assert "Statistics for dataset: test_dataset" in result.stdout
            mock_stats.execute.assert_called_with("test_dataset", full=True)
    
    @patch('mdm.cli.main.setup_logging')
    @patch('mdm.cli.dataset.RemoveOperation')
    def test_dataset_remove_dry_run(self, mock_remove_class, mock_setup, runner):
        """Test dataset remove with dry-run."""
        mock_remove = Mock()
        mock_remove.execute.return_value = {
            'name': 'test_dataset',
            'size': 1048576,
            'config_file': '/tmp/test_dataset.yaml',
            'dataset_directory': '/tmp/datasets/test_dataset'
        }
        mock_remove_class.return_value = mock_remove
        
        result = runner.invoke(dataset_app, [
            "remove", "test_dataset",
            "--force",
            "--dry-run"
        ])
        
        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout
        # Should be called once for dry-run
        mock_remove.execute.assert_called_once_with(
            "test_dataset",
            force=True,
            dry_run=True
        )
    
    @patch('mdm.cli.dataset.BackendFactory')
    @patch('mdm.cli.dataset.console')
    def test_display_column_summary_many_columns(self, mock_console, mock_factory):
        """Test _display_column_summary with more than 20 columns."""
        from mdm.cli.dataset import _display_column_summary
        
        # Create mock backend
        mock_backend = Mock()
        mock_backend.query.side_effect = [
            pd.DataFrame({'total_rows': [5000]}),
            # Return null counts for first 20 columns
            *[pd.DataFrame({'null_count': [i * 10]}) for i in range(20)]
        ]
        
        # Create 25 columns
        columns = [{'name': f'col_{i}', 'type': 'INTEGER'} for i in range(25)]
        mock_backend.get_table_info.return_value = {'columns': columns}
        mock_backend.get_engine.return_value = Mock()
        mock_factory.create.return_value = mock_backend
        
        # Create mock dataset info
        mock_info = Mock()
        mock_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_info.tables = {'train': 'train_table'}
        
        # Mock table for output
        mock_table = Mock(spec=Table)
        with patch('mdm.cli.dataset.Table', return_value=mock_table):
            _display_column_summary(mock_info, Mock(), 'train')
        
        # Should print message about additional columns
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        # Check if the expected message was printed to the table or console
        assert any("5 more columns" in str(call) for call in print_calls) or \
               any("5 more columns" in str(call) for call in mock_table.add_row.call_args_list if hasattr(mock_table, 'add_row'))
    
    @patch('mdm.cli.dataset.BackendFactory')
    @patch('mdm.cli.dataset.console')
    def test_display_column_summary_postgresql(self, mock_console, mock_factory):
        """Test _display_column_summary with PostgreSQL backend."""
        from mdm.cli.dataset import _display_column_summary
        
        mock_backend = Mock()
        mock_backend.query.return_value = pd.DataFrame({'total_rows': [1000]})
        mock_backend.get_table_info.return_value = {
            'columns': [{'name': 'id', 'type': 'INTEGER'}]
        }
        mock_backend.get_engine.return_value = Mock()
        mock_factory.create.return_value = mock_backend
        
        # Mock dataset info with PostgreSQL
        mock_info = Mock()
        mock_info.database = {
            'backend': 'postgresql',
            'host': 'localhost',
            'port': 5432,
            'user': 'user',
            'password': 'pass',
            'database': 'testdb'
        }
        mock_info.tables = {'train': 'train_table'}
        
        _display_column_summary(mock_info, Mock(), 'train')
        
        # Should create backend with postgres config
        mock_factory.create.assert_called_with('postgresql', mock_info.database)


class TestBatchCLIFinal:
    """Final tests for batch.py to achieve 90% coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.main.setup_logging')
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.RemoveOperation')
    def test_batch_remove_dry_run(self, mock_remove_class, mock_manager_class, mock_setup, runner):
        """Test batch remove with dry-run mode."""
        mock_manager = Mock()
        mock_manager.dataset_exists.side_effect = [True, True]  # Both datasets exist
        mock_manager_class.return_value = mock_manager
        
        mock_remove = Mock()
        mock_remove.execute.side_effect = [
            {'name': 'ds1', 'size': 1048576},
            {'name': 'ds2', 'size': 2097152}
        ]
        mock_remove_class.return_value = mock_remove
        
        result = runner.invoke(batch_app, [
            "remove", "ds1", "ds2",
            "--force",
            "--dry-run"
        ])
        
        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout
        assert "ds1" in result.stdout
        assert "ds2" in result.stdout
    
    @patch('mdm.cli.main.setup_logging')
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.StatsOperation')
    def test_batch_stats_export_option(self, mock_stats_class, mock_manager_class, mock_setup, runner):
        """Test batch stats with export option."""
        mock_manager = Mock()
        mock_manager.dataset_exists.return_value = True
        mock_manager_class.return_value = mock_manager
        
        mock_stats = Mock()
        mock_stats.execute.return_value = {
            'dataset_name': 'ds1',
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
                "stats", "ds1",
                "--export", tmpdir
            ])
            
            assert result.exit_code == 0
            assert "Statistics Summary:" in result.stdout
            assert "ds1" in result.stdout


class TestTimeseriesCLIFinal:
    """Final tests for timeseries.py to achieve 90% coverage."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.main.setup_logging')
    @patch('mdm.cli.timeseries.MDMClient')
    def test_timeseries_split_error_handling(self, mock_client_class, mock_setup, runner):
        """Test timeseries split with error conditions."""
        mock_client = Mock()
        mock_client.get_dataset.return_value = None  # Dataset not found
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(timeseries_app, ["split", "nonexistent"])
        
        assert result.exit_code == 1
        assert "Dataset 'nonexistent' not found" in result.stdout
    
    @patch('mdm.cli.main.setup_logging')
    @patch('mdm.cli.timeseries.MDMClient')
    def test_timeseries_validate_no_time_column(self, mock_client_class, mock_setup, runner):
        """Test timeseries validate without time column."""
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = None
        mock_client.get_dataset.return_value = mock_dataset_info
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(timeseries_app, ["validate", "test_dataset"])
        
        assert result.exit_code == 1
        assert "has no time column configured" in result.stdout


# Direct function tests to maximize coverage
class TestDirectFunctions:
    """Test functions directly for maximum coverage."""
    
    def test_all_format_size_branches(self):
        """Test all branches of _format_size."""
        from mdm.cli.main import _format_size
        from mdm.cli.dataset import _format_size as dataset_format_size
        
        # Test all size units
        sizes = [
            (0, "0.0 B"),
            (500, "500.0 B"),
            (1024, "1.0 KB"),
            (1024 * 1024, "1.0 MB"),
            (1024 * 1024 * 1024, "1.0 GB"),
            (1024 * 1024 * 1024 * 1024, "1.0 TB"),
            (1024 * 1024 * 1024 * 1024 * 1024, "1.0 PB"),
            (1024 * 1024 * 1024 * 1024 * 1024 * 1024, "1024.0 PB"),  # Beyond PB
        ]
        
        for size, expected in sizes:
            assert _format_size(size) == expected
            assert dataset_format_size(size) == expected
    
    @patch('mdm.storage.factory.BackendFactory')
    @patch('mdm.cli.dataset.console')
    def test_display_column_summary_error_handling(self, mock_console, mock_factory):
        """Test _display_column_summary error handling."""
        from mdm.cli.dataset import _display_column_summary
        
        # Mock backend that raises exception immediately
        mock_factory.create.side_effect = Exception("Connection failed")
        
        mock_info = Mock()
        mock_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_info.tables = {'train': 'train_table'}
        
        # Should not raise, just print error
        _display_column_summary(mock_info, Mock(), 'train')
        
        # Check error was printed
        error_printed = any(
            "Could not generate column summary" in str(call)
            for call in mock_console.print.call_args_list
        )
        assert error_printed