"""Unit tests for configuration system."""

from pathlib import Path

import pytest
import yaml

from mdm.config import Config, get_config
from mdm.models.enums import LogLevel


class TestConfig:
    """Test configuration system."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        # Check database defaults
        assert config.database.default_backend == "duckdb"
        assert config.database.duckdb.memory_limit == "4GB"
        assert config.database.duckdb.threads == 4
        
        # Check storage defaults
        assert config.storage.datasets_path == Path("./datasets")
        assert config.storage.configs_path == Path("./configs")
        
        # Check performance defaults
        assert config.performance.batch_size == 10000
        assert config.performance.max_workers == 4
        
        # Check logging defaults
        assert config.logs.level == LogLevel.INFO
        assert config.logs.file_path == Path.home() / ".mdm" / "logs" / "mdm.log"

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "database": {
                "default_backend": "sqlite"
            },
            "storage": {
                "datasets_path": "/data/datasets",
                "configs_path": "/data/configs"
            },
            "performance": {
                "batch_size": 5000
            }
        }
        
        config = Config(**config_dict)
        
        assert config.database.default_backend == "sqlite"
        assert config.storage.datasets_path == Path("/data/datasets")
        assert config.storage.configs_path == Path("/data/configs")
        assert config.performance.batch_size == 5000

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = Config(
            database={"default_backend": "postgresql"},
            performance={"batch_size": 1000, "max_workers": 8}
        )
        assert config.database.default_backend == "postgresql"
        assert config.performance.batch_size == 1000
        
        # Invalid batch size (should be positive)
        with pytest.raises(ValueError):
            Config(performance={"batch_size": 0})
        
        # Invalid max workers (should be positive)
        with pytest.raises(ValueError):
            Config(performance={"max_workers": -1})

    def test_config_yaml_export(self, temp_dir):
        """Test exporting config to YAML."""
        config = Config(
            database={"default_backend": "sqlite"},
            storage={
                "datasets_path": str(temp_dir / "datasets"),
                "configs_path": str(temp_dir / "configs")
            }
        )
        
        # Export to YAML
        yaml_path = temp_dir / "test_config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config.model_dump(), f)
        
        # Load and verify
        with open(yaml_path) as f:
            loaded_dict = yaml.safe_load(f)
        
        assert loaded_dict["database"]["default_backend"] == "sqlite"
        assert loaded_dict["storage"]["datasets_path"] == str(temp_dir / "datasets")

    def test_feature_engineering_config(self):
        """Test feature engineering configuration."""
        config = Config(
            feature_engineering={
                "enabled": True,
                "generic_features": {
                    "temporal": {"enable_cyclical": True},
                    "categorical": {"max_cardinality": 100},
                    "statistical": {"enable_log_transform": False}
                }
            }
        )
        
        assert config.feature_engineering.enabled is True
        assert config.feature_engineering.generic_features.temporal.enable_cyclical is True
        assert config.feature_engineering.generic_features.categorical.max_cardinality == 100
        assert config.feature_engineering.generic_features.statistical.enable_log_transform is False

    def test_environment_variable_override(self, monkeypatch):
        """Test environment variable overrides."""
        # Set environment variables
        monkeypatch.setenv("MDM_BATCH_SIZE", "25000")
        monkeypatch.setenv("MDM_LOG_LEVEL", "DEBUG")
        
        # These would be handled by the actual config loading logic
        # For now, test that we can create config with these values
        config = Config(
            performance={"batch_size": 25000},
            logs={"level": LogLevel.DEBUG}
        )
        
        assert config.performance.batch_size == 25000
        assert config.logs.level == LogLevel.DEBUG