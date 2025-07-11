"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from mdm.config import ConfigManager, get_config, get_config_manager
from mdm.core.exceptions import ConfigError
from mdm.models.config import MDMConfig
import mdm.config.config


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_default_config(self):
        """Test loading default configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mdm.yaml"
            manager = ConfigManager(config_path)
            
            # Load default config
            config = manager.load()
            
            # Check defaults
            assert config.database.default_backend == "sqlite"
            assert config.performance.batch_size == 10000
            assert config.logging.level == "INFO"
            assert config.cli.default_output_format == "rich"
    
    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mdm.yaml"
            
            # Write test config
            test_config = {
                "database": {
                    "default_backend": "postgresql",
                    "connection_timeout": 60,
                },
                "performance": {
                    "batch_size": 50000,
                },
            }
            
            with open(config_path, "w") as f:
                yaml.dump(test_config, f)
            
            # Load config
            manager = ConfigManager(config_path)
            config = manager.load()
            
            # Check loaded values
            assert config.database.default_backend == "postgresql"
            assert config.database.connection_timeout == 60
            assert config.performance.batch_size == 50000
            
            # Check defaults still work
            assert config.logging.level == "INFO"
    
    def test_environment_variables(self):
        """Test environment variable override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mdm.yaml"
            
            # Set environment variables
            env_vars = {
                "MDM_DATABASE_DEFAULT_BACKEND": "duckdb",
                "MDM_DATABASE_CONNECTION_TIMEOUT": "120",
                "MDM_PERFORMANCE_BATCH_SIZE": "25000",
                "MDM_LOGGING_LEVEL": "DEBUG",
                "MDM_CLI_SHOW_PROGRESS": "false",
                "MDM_DATABASE_POSTGRESQL_HOST": "db.example.com",
                "MDM_DATABASE_POSTGRESQL_PORT": "5433",
            }
            
            # Apply env vars
            for key, value in env_vars.items():
                os.environ[key] = value
            
            try:
                manager = ConfigManager(config_path)
                config = manager.load()
                
                # Check environment overrides
                assert config.database.default_backend == "duckdb"
                assert config.database.connection_timeout == 120
                assert config.performance.batch_size == 25000
                assert config.logging.level == "DEBUG"
                assert config.cli.show_progress is False
                assert config.database.postgresql.host == "db.example.com"
                assert config.database.postgresql.port == 5433
                
            finally:
                # Clean up env vars
                for key in env_vars:
                    os.environ.pop(key, None)
    
    def test_save_config(self):
        """Test saving configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mdm.yaml"
            manager = ConfigManager(config_path)
            
            # Create custom config
            config = MDMConfig(
                database={"default_backend": "postgresql"},
                performance={"batch_size": 20000},
            )
            
            # Save config
            manager.save(config)
            
            # Verify file exists
            assert config_path.exists()
            
            # Load and verify
            with open(config_path, "r") as f:
                saved_data = yaml.safe_load(f)
            
            assert saved_data["database"]["default_backend"] == "postgresql"
            assert saved_data["performance"]["batch_size"] == 20000
    
    def test_initialize_defaults(self):
        """Test initializing default configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / ".mdm"
            config_path = base_path / "mdm.yaml"
            
            manager = ConfigManager(config_path)
            manager.base_path = base_path
            
            # Initialize defaults
            manager.initialize_defaults()
            
            # Check config file created
            assert config_path.exists()
            
            # Check directories created
            assert (base_path / "datasets").exists()
            assert (base_path / "config" / "datasets").exists()
            assert (base_path / "logs").exists()
            assert (base_path / "config" / "custom_features").exists()
    
    def test_invalid_yaml(self):
        """Test error handling for invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mdm.yaml"
            
            # Write invalid YAML
            with open(config_path, "w") as f:
                f.write("invalid: yaml: content: [")
            
            manager = ConfigManager(config_path)
            
            with pytest.raises(ConfigError, match="Failed to load configuration"):
                manager.load()
    
    def test_validation_error(self):
        """Test validation error handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mdm.yaml"
            
            # Write config with invalid values
            test_config = {
                "database": {
                    "default_backend": "invalid_backend",
                },
            }
            
            with open(config_path, "w") as f:
                yaml.dump(test_config, f)
            
            manager = ConfigManager(config_path)
            
            with pytest.raises(ConfigError, match="Invalid configuration"):
                manager.load()
    
    def test_get_full_path(self):
        """Test getting full paths."""
        config = MDMConfig()
        base_path = Path("/home/user/.mdm")
        
        # Test path resolution
        datasets_path = config.get_full_path("datasets_path", base_path)
        assert datasets_path == base_path / "datasets/"
        
        logs_path = config.get_full_path("logs_path", base_path)
        assert logs_path == base_path / "logs/"
    
    def test_list_environment_variable(self):
        """Test parsing list from environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mdm.yaml"
            
            # Set list env var
            os.environ["MDM_FEATURE_ENGINEERING_GENERIC_BINNING_N_BINS"] = "3,6,9,12"
            
            try:
                manager = ConfigManager(config_path)
                config = manager.load()
                
                # Check list parsed correctly
                assert config.feature_engineering.generic.binning.n_bins == [3, 6, 9, 12]
                
            finally:
                os.environ.pop("MDM_FEATURE_ENGINEERING_GENERIC_BINNING_N_BINS", None)


class TestGlobalConfig:
    """Test global configuration functions."""
    
    def setup_method(self):
        """Reset global config manager before each test."""
        mdm.config.config._config_manager = None
    
    def test_get_config_manager(self):
        """Test getting global config manager."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        # Should return same instance
        assert manager1 is manager2
    
    def test_get_config(self):
        """Test getting configuration."""
        # Create temporary config to avoid side effects
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mdm.yaml"
            mdm.config.config._config_manager = ConfigManager(config_path)
            
            config = get_config()
            
            # Should return valid config
            assert isinstance(config, MDMConfig)
            assert config.database.default_backend in ["sqlite", "duckdb", "postgresql"]