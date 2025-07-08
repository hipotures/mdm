"""Unit tests for main CLI module."""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import shutil

import pytest
from typer.testing import CliRunner
from loguru import logger

from mdm.cli.main import app, setup_logging, _format_size, main


class TestSetupLogging:
    """Test logging setup functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        mock_config = Mock()
        mock_config.logging.file = "mdm.log"
        mock_config.logging.level = "INFO"
        mock_config.logging.format = "console"
        mock_config.logging.max_bytes = 10485760
        mock_config.logging.backup_count = 5
        mock_config.database.sqlalchemy.echo = False
        mock_config.paths.logs_path = "logs"
        # Add model_dump method to return empty dict
        mock_config.model_dump.return_value = {}
        return mock_config
    
    @pytest.fixture
    def mock_config_manager(self, mock_config):
        """Create mock config manager."""
        mock_manager = Mock()
        mock_manager.config = mock_config
        mock_manager.base_path = Path("/tmp/test_mdm")
        return mock_manager
    
    @patch('mdm.cli.main.get_config_manager')
    @patch('mdm.cli.main.logger')
    def test_setup_logging_basic(self, mock_logger, mock_get_config, mock_config_manager):
        """Test basic logging setup."""
        mock_get_config.return_value = mock_config_manager
        
        setup_logging()
        
        # Verify logger setup was called
        mock_logger.remove.assert_called_once()
        assert mock_logger.add.call_count >= 2  # File and console handlers
        
    @patch('mdm.cli.main.get_config_manager')
    @patch('mdm.cli.main.logger')
    def test_setup_logging_with_absolute_path(self, mock_logger, mock_get_config, mock_config_manager):
        """Test logging setup with absolute log file path."""
        mock_config_manager.config.logging.file = "/var/log/mdm.log"
        mock_get_config.return_value = mock_config_manager
        
        setup_logging()
        
        # Verify absolute path is used
        file_handler_call = mock_logger.add.call_args_list[0]
        assert file_handler_call[0][0] == Path("/var/log/mdm.log")
    
    @patch('mdm.cli.main.get_config_manager')
    @patch('mdm.cli.main.logger')
    @patch.dict(os.environ, {'MDM_LOGGING_LEVEL': 'DEBUG'})
    def test_setup_logging_env_override(self, mock_logger, mock_get_config, mock_config_manager):
        """Test logging setup with environment variable override."""
        mock_get_config.return_value = mock_config_manager
        
        setup_logging()
        
        # Verify DEBUG level is used from env var
        file_handler_call = mock_logger.add.call_args_list[0]
        assert file_handler_call[1]['level'] == 'DEBUG'
    
    @patch('mdm.cli.main.get_config_manager')
    @patch('mdm.cli.main.logger')
    def test_setup_logging_json_format(self, mock_logger, mock_get_config, mock_config_manager):
        """Test logging setup with JSON format."""
        mock_config_manager.config.logging.format = "json"
        mock_get_config.return_value = mock_config_manager
        
        setup_logging()
        
        # Verify JSON format is used
        file_handler_call = mock_logger.add.call_args_list[0]
        assert file_handler_call[1]['serialize'] is True
    
    @patch('mdm.cli.main.get_config_manager')
    @patch('mdm.cli.main.logger')
    def test_setup_logging_sqlalchemy_echo(self, mock_logger, mock_get_config, mock_config_manager):
        """Test logging setup with SQLAlchemy echo enabled."""
        mock_config_manager.config.database.sqlalchemy.echo = True
        mock_config_manager.config.logging.level = "DEBUG"
        mock_get_config.return_value = mock_config_manager
        
        setup_logging()
        
        # Should have additional handler for SQLAlchemy
        assert mock_logger.add.call_count >= 3
    
    @patch('mdm.cli.main.get_config_manager')
    @patch('mdm.cli.main.logger')
    @patch('mdm.cli.main.logging')
    def test_setup_logging_intercept_handler(self, mock_logging, mock_logger, mock_get_config, mock_config_manager):
        """Test InterceptHandler setup for standard logging."""
        mock_get_config.return_value = mock_config_manager
        
        setup_logging()
        
        # Verify logging.basicConfig was called
        mock_logging.basicConfig.assert_called_once()
        
        # Verify specific loggers were configured
        for logger_name in ["mdm", "sqlalchemy.engine", "sqlalchemy.pool"]:
            mock_logging.getLogger.assert_any_call(logger_name)


class TestCLICommands:
    """Test CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture(autouse=True)
    def mock_setup_logging(self):
        """Mock setup_logging for all CLI tests."""
        with patch('mdm.cli.main.setup_logging'):
            yield
    
    def test_version_command(self, runner):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "MDM" in result.stdout
        assert "0.1.0" in result.stdout
    
    @patch('mdm.cli.main.get_config_manager')
    @patch('mdm.cli.main.DatasetManager')
    @patch('mdm.cli.main.shutil.disk_usage')
    def test_info_command(self, mock_disk_usage, mock_dataset_manager, mock_get_config, runner):
        """Test info command."""
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
        mock_manager.base_path = Path("/home/user/.mdm")
        mock_get_config.return_value = mock_manager
        
        mock_dataset_manager_instance = Mock()
        mock_dataset_manager_instance.list_datasets.return_value = ["dataset1", "dataset2"]
        mock_dataset_manager.return_value = mock_dataset_manager_instance
        
        mock_disk_usage.return_value = Mock(free=1000000000)
        
        # Create temp directory to avoid file system errors
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_manager.base_path = Path(tmpdir)
            (Path(tmpdir) / "datasets").mkdir()
            
            result = runner.invoke(app, ["info"])
        
        assert result.exit_code == 0
        assert "ML Data Manager" in result.stdout
        assert "Configuration:" in result.stdout
        assert "Storage paths:" in result.stdout
        assert "Database settings:" in result.stdout
        assert "System status:" in result.stdout
        assert "Registered datasets: 2" in result.stdout
        assert "sqlite" in result.stdout
    
    def test_help_command(self, runner):
        """Test help command."""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "ML Data Manager" in result.stdout
        assert "dataset" in result.stdout
        assert "batch" in result.stdout
        assert "timeseries" in result.stdout
    
    def test_invalid_command(self, runner):
        """Test invalid command."""
        result = runner.invoke(app, ["invalid-command"])
        
        assert result.exit_code != 0
        # Typer outputs errors to stderr, not stdout
        assert "No such command" in result.stderr or "Error" in result.stderr


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_format_size_bytes(self):
        """Test formatting bytes."""
        assert _format_size(500) == "500.0 B"
        assert _format_size(1023) == "1023.0 B"
    
    def test_format_size_kilobytes(self):
        """Test formatting kilobytes."""
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(1536) == "1.5 KB"
        assert _format_size(1048575) == "1024.0 KB"
    
    def test_format_size_megabytes(self):
        """Test formatting megabytes."""
        assert _format_size(1048576) == "1.0 MB"
        assert _format_size(1572864) == "1.5 MB"
    
    def test_format_size_gigabytes(self):
        """Test formatting gigabytes."""
        assert _format_size(1073741824) == "1.0 GB"
        assert _format_size(1610612736) == "1.5 GB"
    
    def test_format_size_terabytes(self):
        """Test formatting terabytes."""
        assert _format_size(1099511627776) == "1.0 TB"
    
    def test_format_size_petabytes(self):
        """Test formatting petabytes."""
        assert _format_size(1125899906842624) == "1.0 PB"


class TestMainEntryPoint:
    """Test main entry point."""
    
    @patch('mdm.cli.main.app')
    @patch.object(sys, 'argv', ['mdm'])
    def test_main_no_args_shows_help(self, mock_app):
        """Test main() with no arguments shows help."""
        main()
        
        # Verify --help was added
        assert sys.argv == ['mdm', '--help']
        mock_app.assert_called_once()
    
    @patch('mdm.cli.main.app')
    @patch.object(sys, 'argv', ['mdm', 'version'])
    def test_main_with_args(self, mock_app):
        """Test main() with arguments."""
        main()
        
        # Verify argv was not modified
        assert sys.argv == ['mdm', 'version']
        mock_app.assert_called_once()
    
    @patch('mdm.cli.main.app')
    def test_main_module_execution(self, mock_app):
        """Test module execution."""
        # Import the module first
        import mdm.cli.main
        
        # Mock sys.argv and execute main()
        with patch.object(sys, 'argv', ['mdm', 'info']):
            # Call main() directly to simulate module execution
            mdm.cli.main.main()
            
            # Verify app was called
            mock_app.assert_called_once()


class TestMainCallback:
    """Test main callback function."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('mdm.cli.main.setup_logging')
    def test_main_callback_called_before_commands(self, mock_setup_logging, runner):
        """Test that main_callback (setup_logging) is called before any command."""
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        mock_setup_logging.assert_called_once()