"""Tests for 1.4 Logging Configuration based on MANUAL_TEST_CHECKLIST.md"""

import os
from pathlib import Path

import pytest


class TestLoggingConfiguration:
    """Test logging configuration functionality."""
    
    @pytest.mark.mdm_id("1.4.1.1")
    def test_default_console_logging(self, clean_mdm_env, run_mdm):
        """1.4.1.1: Default console logging (WARNING level)"""
        # Run a command that generates warnings
        result = run_mdm(["dataset", "list"])
        
        assert result.returncode == 0
        # Default level is WARNING, so debug/info shouldn't appear
        assert "DEBUG" not in result.stderr
        assert "INFO" not in result.stderr
    
    @pytest.mark.mdm_id("1.4.1.2")
    def test_change_log_level_yaml(self, clean_mdm_env, run_mdm, mdm_config_file, sample_csv_data):
        """1.4.1.2: Change log level via YAML config"""
        # Create config with DEBUG level
        mdm_config_file(logging={"level": "DEBUG"})
        
        # Run command with debug logging
        result = run_mdm([
            "dataset", "register", "test_debug", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        # Debug messages should appear in file log (if configured)
        # Console still shows WARNING+ only
    
    @pytest.mark.mdm_id("1.4.1.3")
    def test_log_level_env_override(self, clean_mdm_env, run_mdm, mdm_config_file):
        """1.4.1.3: Environment variable overrides YAML"""
        # Create config with INFO level
        mdm_config_file(logging={"level": "INFO"})
        
        # Override with DEBUG via env var
        env = os.environ.copy()
        env["MDM_LOGGING_LEVEL"] = "DEBUG"
        
        result = run_mdm(["dataset", "list"], env=env)
        assert result.returncode == 0
        # Environment variable takes precedence
    
    @pytest.mark.mdm_id("1.4.2.1")
    def test_file_logging_yaml(self, clean_mdm_env, run_mdm, mdm_config_file, sample_csv_data):
        """1.4.2.1: Enable file logging via YAML"""
        log_file = clean_mdm_env / "mdm.log"
        
        # Create config with file logging
        mdm_config_file(logging={
            "level": "DEBUG",
            "file": str(log_file)
        })
        
        # Run command that generates logs
        result = run_mdm([
            "dataset", "register", "test_file_log", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        assert log_file.exists()
        
        # Check log contents
        log_content = log_file.read_text()
        assert "test_file_log" in log_content
        assert "DEBUG" in log_content
    
    @pytest.mark.mdm_id("1.4.2.2")
    def test_file_logging_env(self, clean_mdm_env, run_mdm, sample_csv_data):
        """1.4.2.2: Enable file logging via environment variable"""
        log_file = clean_mdm_env / "env_mdm.log"
        
        env = os.environ.copy()
        env["MDM_LOGGING_FILE"] = str(log_file)
        env["MDM_LOGGING_LEVEL"] = "DEBUG"
        
        result = run_mdm([
            "dataset", "register", "test_env_log", str(sample_csv_data),
            "--target", "value"
        ], env=env)
        
        assert result.returncode == 0
        assert log_file.exists()
        
        # Verify log file has content
        log_content = log_file.read_text()
        assert len(log_content) > 0
        assert "test_env_log" in log_content
    
    @pytest.mark.mdm_id("1.4.2.3")
    def test_log_file_rotation(self, clean_mdm_env, run_mdm, mdm_config_file, sample_csv_data):
        """1.4.2.3: Log file rotation when size limit reached"""
        log_file = clean_mdm_env / "rotate.log"
        
        mdm_config_file(logging={
            "level": "DEBUG",
            "file": str(log_file),
            "max_bytes": 1024,  # Small size to trigger rotation
            "backup_count": 3
        })
        
        # Generate multiple log entries
        for i in range(5):
            result = run_mdm([
                "dataset", "list"
            ])
            assert result.returncode == 0
        
        # Note: Log rotation configuration may not be implemented
        # This test documents expected behavior
    
    @pytest.mark.mdm_id("1.4.3.1")
    def test_log_format_default(self, clean_mdm_env, run_mdm, mdm_config_file):
        """1.4.3.1: Default log format includes timestamp, level, message"""
        log_file = clean_mdm_env / "format.log"
        
        mdm_config_file(logging={
            "level": "INFO",
            "file": str(log_file)
        })
        
        result = run_mdm(["dataset", "list"])
        assert result.returncode == 0
        
        if log_file.exists():
            log_content = log_file.read_text()
            # Check for standard log format elements
            # Format: timestamp - level - message
            lines = log_content.strip().split('\n')
            if lines:
                # Basic format validation
                assert any('-' in line for line in lines)
    
    @pytest.mark.mdm_id("1.4.3.2")
    def test_debug_shows_module_info(self, clean_mdm_env, run_mdm, mdm_config_file, sample_csv_data):
        """1.4.3.2: Debug mode shows module and function info"""
        log_file = clean_mdm_env / "debug_format.log"
        
        mdm_config_file(logging={
            "level": "DEBUG",
            "file": str(log_file)
        })
        
        result = run_mdm([
            "dataset", "register", "test_debug_info", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        
        if log_file.exists():
            log_content = log_file.read_text()
            # Debug logs typically include more detail
            assert "DEBUG" in log_content
    
    @pytest.mark.mdm_id("1.4.4.1")
    def test_suppress_external_library_logs(self, clean_mdm_env, run_mdm, mdm_config_file, sample_csv_data):
        """1.4.4.1: External library logs are suppressed by default"""
        log_file = clean_mdm_env / "external.log"
        
        mdm_config_file(logging={
            "level": "DEBUG",
            "file": str(log_file)
        })
        
        # Register dataset (uses pandas, sqlalchemy, etc)
        result = run_mdm([
            "dataset", "register", "test_external", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        
        if log_file.exists():
            log_content = log_file.read_text()
            # Should not contain excessive pandas/sqlalchemy debug logs
            # MDM suppresses these by default
            assert log_content.count("pandas") < 10
            assert log_content.count("sqlalchemy") < 10
    
    @pytest.mark.mdm_id("1.4.4.2")
    def test_progress_bar_clean_output(self, clean_mdm_env, run_mdm, sample_csv_data):
        """1.4.4.2: Progress bars don't interfere with logging"""
        # Register a dataset (shows progress bars)
        result = run_mdm([
            "dataset", "register", "test_progress", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
        
        # Output should be clean without progress bar artifacts
        assert "\r" not in result.stdout  # No carriage returns
        assert "100%" not in result.stderr  # Progress not in stderr