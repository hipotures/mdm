"""Tests for 1.1 YAML Configuration based on MANUAL_TEST_CHECKLIST.md"""

import os
from pathlib import Path

import pytest
import yaml


class TestYAMLConfiguration:
    """Test YAML configuration file handling."""
    
    @pytest.mark.mdm_id("1.1.1")
    def test_create_yaml_with_custom_settings(self, clean_mdm_env, mdm_config_file, run_mdm):
        """1.1.1: Create ~/.mdm/mdm.yaml with custom settings"""
        # Create config with custom settings
        config_content = """
database:
  default_backend: sqlite
  connection_timeout: 120
  sqlite:
    journal_mode: "WAL"
    synchronous: "NORMAL"
    cache_size: -64000

logging:
  level: DEBUG
  format: "console"

performance:
  batch_size: 5000
"""
        config_file = mdm_config_file(config_content)
        
        # Verify file was created
        assert config_file.exists()
        assert config_file.read_text() == config_content
        
        # Run MDM to verify it loads without errors
        result = run_mdm(["--help"])
        assert result.returncode == 0
    
    @pytest.mark.mdm_id("1.1.2")
    def test_verify_yaml_settings_applied(self, clean_mdm_env, mdm_config_file, run_mdm, sample_csv_data):
        """1.1.2: Run MDM and verify settings from YAML are applied"""
        # Create config with specific backend
        config_content = """
database:
  default_backend: sqlite
logging:
  level: DEBUG
"""
        mdm_config_file(config_content)
        
        # Set debug environment to see config loading
        result = run_mdm(
            ["dataset", "register", "test_yaml", str(sample_csv_data), "--target", "value"],
            env={"MDM_LOGGING_LEVEL": "DEBUG"}
        )
        
        # Should succeed
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
        
        # Verify SQLite backend was used
        datasets_dir = clean_mdm_env / "datasets" / "test_yaml"
        assert (datasets_dir / "test_yaml.sqlite").exists()
        assert not (datasets_dir / "test_yaml.duckdb").exists()
    
    @pytest.mark.mdm_id("1.1.3")
    def test_modify_yaml_changes_take_effect(self, clean_mdm_env, mdm_config_file, run_mdm, sample_csv_data):
        """1.1.3: Modify mdm.yaml and check if changes take effect"""
        # Start with DuckDB
        config_content_1 = """
database:
  default_backend: duckdb
"""
        config_file = mdm_config_file(config_content_1)
        
        # Register first dataset
        result = run_mdm(["dataset", "register", "test_duckdb", str(sample_csv_data), "--target", "value"])
        assert result.returncode == 0
        
        # Verify DuckDB was used
        assert (clean_mdm_env / "datasets" / "test_duckdb" / "test_duckdb.duckdb").exists()
        
        # Now change to SQLite
        config_content_2 = """
database:
  default_backend: sqlite
"""
        config_file.write_text(config_content_2)
        
        # Register second dataset
        result = run_mdm(["dataset", "register", "test_sqlite", str(sample_csv_data), "--target", "value"])
        assert result.returncode == 0
        
        # Verify SQLite was used for second dataset
        assert (clean_mdm_env / "datasets" / "test_sqlite" / "test_sqlite.sqlite").exists()
    
    @pytest.mark.mdm_id("1.1.4")
    def test_delete_yaml_uses_defaults(self, clean_mdm_env, mdm_config_file, run_mdm, sample_csv_data):
        """1.1.4: Delete mdm.yaml and verify MDM still works with defaults"""
        # First create a config
        config_file = mdm_config_file("""
database:
  default_backend: sqlite
""")
        
        # Register dataset with config
        result = run_mdm(["dataset", "register", "test_with_config", str(sample_csv_data), "--target", "value"])
        assert result.returncode == 0
        
        # Delete config file
        config_file.unlink()
        assert not config_file.exists()
        
        # MDM should still work with defaults
        result = run_mdm(["dataset", "register", "test_no_config", str(sample_csv_data), "--target", "value"])
        assert result.returncode == 0
        
        # Default backend in code is SQLite
        assert (clean_mdm_env / "datasets" / "test_no_config" / "test_no_config.sqlite").exists()
    
    @pytest.mark.mdm_id("1.1.5")
    def test_invalid_yaml_syntax(self, clean_mdm_env, mdm_config_file, run_mdm):
        """1.1.5: Test with invalid YAML syntax"""
        # Create invalid YAML
        invalid_yaml = """
database:
  default_backend: sqlite
  invalid line without colon
logging:
  level: DEBUG
"""
        mdm_config_file(invalid_yaml)
        
        # MDM should fail gracefully
        result = run_mdm(["dataset", "list"], check=False)
        
        # Should fail with error message about YAML
        assert result.returncode != 0
        assert "yaml" in result.stderr.lower() or "config" in result.stderr.lower()
    
    @pytest.mark.mdm_id("1.1.6")
    def test_unknown_configuration_keys(self, clean_mdm_env, mdm_config_file, run_mdm, sample_csv_data):
        """1.1.6: Test with unknown configuration keys"""
        # Create config with unknown keys
        config_content = """
database:
  default_backend: sqlite
  
# Unknown top-level key
unknown_section:
  some_key: some_value
  
logging:
  level: DEBUG
  # Unknown key in known section
  unknown_key: unknown_value
  
# Another unknown section
future_feature:
  enabled: true
"""
        mdm_config_file(config_content)
        
        # MDM should ignore unknown keys and work normally
        result = run_mdm(["dataset", "register", "test_unknown_keys", str(sample_csv_data), "--target", "value"])
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
        
        # Verify dataset was created despite unknown keys
        assert (clean_mdm_env / "datasets" / "test_unknown_keys").exists()