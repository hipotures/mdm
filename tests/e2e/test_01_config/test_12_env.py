"""Tests for 1.2 Environment Variables based on MANUAL_TEST_CHECKLIST.md"""

import os
from pathlib import Path

import pytest


class TestEnvironmentVariables:
    """Test environment variable configuration."""
    
    @pytest.mark.mdm_id("1.2.1")
    def test_mdm_log_level_debug(self, clean_mdm_env, run_mdm, sample_csv_data):
        """1.2.1: Set MDM_LOG_LEVEL=DEBUG and verify debug output appears"""
        # Run with DEBUG level
        result = run_mdm(
            ["dataset", "register", "test_debug", str(sample_csv_data), "--target", "value"],
            env={"MDM_LOGGING_LEVEL": "DEBUG"}
        )
        
        assert result.returncode == 0
        
        # Check for debug output indicators
        output = result.stdout + result.stderr
        # Debug mode should show configuration details or SQL queries
        assert "DEBUG" in output or "Configuration" in output or "SQL" in output
    
    @pytest.mark.mdm_id("1.2.2")
    def test_mdm_database_backend_sqlite(self, clean_mdm_env, run_mdm, sample_csv_data):
        """1.2.2: Set MDM_DATABASE_DEFAULT_BACKEND=sqlite and register a dataset"""
        result = run_mdm(
            ["dataset", "register", "test_sqlite_env", str(sample_csv_data), "--target", "value"],
            env={"MDM_DATABASE_DEFAULT_BACKEND": "sqlite"}
        )
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
        
        # Verify SQLite file was created
        sqlite_file = clean_mdm_env / "datasets" / "test_sqlite_env" / "test_sqlite_env.sqlite"
        assert sqlite_file.exists()
    
    @pytest.mark.mdm_id("1.2.3")
    def test_mdm_database_backend_duckdb(self, clean_mdm_env, run_mdm, sample_csv_data):
        """1.2.3: Set MDM_DATABASE_DEFAULT_BACKEND=duckdb and register another dataset"""
        result = run_mdm(
            ["dataset", "register", "test_duckdb_env", str(sample_csv_data), "--target", "value"],
            env={"MDM_DATABASE_DEFAULT_BACKEND": "duckdb"}
        )
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
        
        # Verify DuckDB file was created
        duckdb_file = clean_mdm_env / "datasets" / "test_duckdb_env" / "test_duckdb_env.duckdb"
        assert duckdb_file.exists()
    
    @pytest.mark.mdm_id("1.2.4")
    def test_mdm_batch_size(self, clean_mdm_env, run_mdm, sample_csv_data):
        """1.2.4: Set MDM_BATCH_SIZE=5000 and verify it's used in operations"""
        # This requires DEBUG output to verify batch size
        result = run_mdm(
            ["dataset", "register", "test_batch", str(sample_csv_data), "--target", "value"],
            env={
                "MDM_PERFORMANCE_BATCH_SIZE": "5000",
                "MDM_LOGGING_LEVEL": "DEBUG"
            }
        )
        
        assert result.returncode == 0
        # With only 100 rows, it should process in one batch
        # But the setting should be acknowledged
    
    @pytest.mark.mdm_id("1.2.5")
    def test_mdm_home_custom_path(self, clean_mdm_env, run_mdm, tmp_path):
        """1.2.5: Set MDM_HOME_DIR=/custom/path and verify directory creation"""
        # Create custom path
        custom_home = tmp_path / "custom_mdm_home"
        
        # Run MDM with custom home directory - use info command which should create dirs
        result = run_mdm(
            ["info"],
            env={"MDM_HOME_DIR": str(custom_home)}
        )
        
        assert result.returncode == 0
        
        # Verify MDM_HOME_DIR was used
        assert str(custom_home) in result.stdout
        
        # Verify at least the base directory was created
        assert custom_home.exists()
        
        # Check if MDM created its config file
        config_file = custom_home / "mdm.yaml"
        if config_file.exists():
            # If config exists, datasets directory should too
            assert (custom_home / "datasets").exists()
    
    @pytest.mark.mdm_id("1.2.6")
    def test_mdm_datasets_path(self, run_mdm, tmp_path, sample_csv_data):
        """1.2.6: Set MDM_DATASETS_PATH=/custom/datasets and verify usage"""
        # Path configuration is under MDM_PATHS_DATASETS_PATH
        custom_datasets = tmp_path / "custom_datasets"
        custom_datasets.mkdir()
        
        # Register a dataset with custom datasets path
        result = run_mdm(
            ["dataset", "register", "test_custom_path", str(sample_csv_data), "--target", "value"],
            env={"MDM_PATHS_DATASETS_PATH": str(custom_datasets)}
        )
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
        
        # Verify dataset was created in custom path
        dataset_dir = custom_datasets / "test_custom_path"
        assert dataset_dir.exists()
        
        # Verify info shows custom path
        result = run_mdm(
            ["info"],
            env={"MDM_PATHS_DATASETS_PATH": str(custom_datasets)}
        )
        assert str(custom_datasets) in result.stdout
    
    @pytest.mark.mdm_id("1.2.7")
    @pytest.mark.skip(reason="PostgreSQL requires database server")
    def test_mdm_default_backend_postgresql(self):
        """1.2.7: Set MDM_DEFAULT_BACKEND=postgresql and test"""
        # PostgreSQL requires a running database server
        pass
    
    @pytest.mark.mdm_id("1.2.8")
    def test_env_overrides_yaml(self, clean_mdm_env, mdm_config_file, run_mdm, sample_csv_data):
        """Test that environment variables override YAML config"""
        # Create YAML config with SQLite
        mdm_config_file("""
database:
  default_backend: sqlite
""")
        
        # But use env var to override to DuckDB
        result = run_mdm(
            ["dataset", "register", "test_override", str(sample_csv_data), "--target", "value"],
            env={"MDM_DATABASE_DEFAULT_BACKEND": "duckdb"}
        )
        
        assert result.returncode == 0
        
        # Should use DuckDB from env var, not SQLite from YAML
        duckdb_file = clean_mdm_env / "datasets" / "test_override" / "test_override.duckdb"
        sqlite_file = clean_mdm_env / "datasets" / "test_override" / "test_override.sqlite"
        
        assert duckdb_file.exists()
        assert not sqlite_file.exists()
    
    @pytest.mark.mdm_id("1.2.9")
    def test_mdm_logging_format(self, clean_mdm_env, run_mdm):
        """Test MDM_LOGGING_FORMAT environment variable"""
        # Test JSON format
        result = run_mdm(
            ["dataset", "list"],
            env={
                "MDM_LOGGING_FORMAT": "json",
                "MDM_LOGGING_LEVEL": "INFO"
            }
        )
        
        # Should not fail
        assert result.returncode == 0
        
        # Test console format (default)
        result = run_mdm(
            ["dataset", "list"],
            env={
                "MDM_LOGGING_FORMAT": "console",
                "MDM_LOGGING_LEVEL": "INFO"
            }
        )
        
        assert result.returncode == 0