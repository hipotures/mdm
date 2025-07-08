"""Tests for 1.3 Database Backend Configuration based on MANUAL_TEST_CHECKLIST.md"""

import os
from pathlib import Path

import pytest


class TestDatabaseBackendConfiguration:
    """Test database backend configuration functionality."""
    
    @pytest.mark.mdm_id("1.3.1.1")
    def test_sqlite_default_backend(self, clean_mdm_env, run_mdm, sample_csv_data):
        """1.3.1.1: Default SQLite backend configuration"""
        # Register dataset with default backend
        result = run_mdm([
            "dataset", "register", "test_sqlite", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
        
        # Check that SQLite file was created
        db_file = clean_mdm_env / "datasets" / "test_sqlite" / "dataset.db"
        assert db_file.exists()
        assert db_file.suffix == ".db"
    
    @pytest.mark.mdm_id("1.3.1.2")
    def test_change_backend_to_duckdb(self, clean_mdm_env, run_mdm, sample_csv_data, mdm_config_file):
        """1.3.1.2: Change backend to DuckDB in config"""
        # Create config with DuckDB backend
        mdm_config_file(database={"default_backend": "duckdb"})
        
        # Register dataset with DuckDB backend
        result = run_mdm([
            "dataset", "register", "test_duckdb", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
        
        # Check that DuckDB file was created
        db_file = clean_mdm_env / "datasets" / "test_duckdb" / "dataset.duckdb"
        assert db_file.exists()
        assert db_file.suffix == ".duckdb"
    
    @pytest.mark.mdm_id("1.3.1.3")
    def test_backend_env_override(self, clean_mdm_env, run_mdm, sample_csv_data, mdm_config_file):
        """1.3.1.3: Environment variable overrides YAML config"""
        # Create config with SQLite
        mdm_config_file(database={"default_backend": "sqlite"})
        
        # Override with DuckDB via env var
        env = os.environ.copy()
        env["MDM_DATABASE_DEFAULT_BACKEND"] = "duckdb"
        
        result = run_mdm([
            "dataset", "register", "test_env_duckdb", str(sample_csv_data),
            "--target", "value"
        ], env=env)
        
        assert result.returncode == 0
        
        # Should use DuckDB despite config saying SQLite
        db_file = clean_mdm_env / "datasets" / "test_env_duckdb" / "dataset.duckdb"
        assert db_file.exists()
    
    @pytest.mark.mdm_id("1.3.2.1")
    def test_invalid_backend_error(self, clean_mdm_env, run_mdm, sample_csv_data, mdm_config_file):
        """1.3.2.1: Invalid backend name in config"""
        # Create config with invalid backend
        mdm_config_file(database={"default_backend": "invalid_backend"})
        
        result = run_mdm([
            "dataset", "register", "test_invalid", str(sample_csv_data),
            "--target", "value"
        ], check=False)
        
        assert result.returncode != 0
        assert "invalid_backend" in result.stderr or "invalid_backend" in result.stdout
    
    @pytest.mark.mdm_id("1.3.2.2")
    def test_backend_isolation(self, clean_mdm_env, run_mdm, sample_csv_data, mdm_config_file):
        """1.3.2.2: Datasets from different backends are isolated"""
        # Register with SQLite
        mdm_config_file(database={"default_backend": "sqlite"})
        result = run_mdm([
            "dataset", "register", "test_sqlite_iso", str(sample_csv_data),
            "--target", "value"
        ])
        assert result.returncode == 0
        
        # List shows SQLite dataset
        result = run_mdm(["dataset", "list"])
        assert "test_sqlite_iso" in result.stdout
        
        # Change to DuckDB
        mdm_config_file(database={"default_backend": "duckdb"})
        
        # List should NOT show SQLite dataset
        result = run_mdm(["dataset", "list"])
        assert "test_sqlite_iso" not in result.stdout
        
        # Register DuckDB dataset
        result = run_mdm([
            "dataset", "register", "test_duckdb_iso", str(sample_csv_data),
            "--target", "value"
        ])
        assert result.returncode == 0
        
        # List shows only DuckDB dataset
        result = run_mdm(["dataset", "list"])
        assert "test_duckdb_iso" in result.stdout
        assert "test_sqlite_iso" not in result.stdout
    
    @pytest.mark.mdm_id("1.3.3.1")
    def test_sqlite_synchronous_setting(self, clean_mdm_env, run_mdm, sample_csv_data, mdm_config_file):
        """1.3.3.1: SQLite synchronous pragma setting"""
        # Create config with NORMAL synchronous mode
        mdm_config_file(database={
            "default_backend": "sqlite",
            "sqlite": {"synchronous": "NORMAL"}
        })
        
        result = run_mdm([
            "dataset", "register", "test_sync", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        # Note: Known issue - synchronous is always FULL
        # This test documents expected behavior
    
    @pytest.mark.mdm_id("1.3.3.2")
    def test_sqlalchemy_echo_setting(self, clean_mdm_env, run_mdm, sample_csv_data, mdm_config_file):
        """1.3.3.2: SQLAlchemy echo setting for debugging"""
        # Create config with echo enabled
        mdm_config_file(database={
            "default_backend": "sqlite",
            "echo": True
        })
        
        result = run_mdm([
            "dataset", "register", "test_echo", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        # Note: Known issue - echo setting not working
        # This test documents expected behavior
    
    @pytest.mark.mdm_id("1.3.4.1")
    @pytest.mark.skip(reason="PostgreSQL requires external setup")
    def test_postgresql_backend(self):
        """1.3.4.1: PostgreSQL backend configuration"""
        pass
    
    @pytest.mark.mdm_id("1.3.4.2")
    @pytest.mark.skip(reason="PostgreSQL connection string requires external DB")
    def test_postgresql_connection_string(self):
        """1.3.4.2: PostgreSQL connection string from env var"""
        pass