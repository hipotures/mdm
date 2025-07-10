"""Test configuration migration functionality."""
import pytest
from pathlib import Path
import tempfile
import os

from mdm.core import feature_flags
from mdm.adapters.config_adapters import get_config_manager, get_config
from mdm.migration.config_migration import (
    ConfigurationMigrator,
    ConfigurationValidator,
)
from mdm.core.config_new import NewMDMConfig


def test_config_adapter_interface():
    """Test that both config systems work through adapters."""
    # Test legacy
    feature_flags.set("use_new_config", False)
    legacy_config = get_config()
    
    assert legacy_config.home_dir.exists()
    assert legacy_config.default_backend in ["sqlite", "duckdb", "postgresql"]
    assert legacy_config.batch_size > 0
    
    # Test new
    feature_flags.set("use_new_config", True)
    new_config = get_config()
    
    assert new_config.home_dir.exists()
    assert new_config.default_backend in ["sqlite", "duckdb", "postgresql"]
    assert new_config.batch_size > 0


def test_config_migration():
    """Test migrating from legacy to new format."""
    migrator = ConfigurationMigrator()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a legacy config
        yaml_content = """
database:
  default_backend: duckdb
  duckdb:
    memory_limit: 16GB
performance:
  batch_size: 25000
logging:
  level: DEBUG
"""
        legacy_path = Path(tmpdir) / "legacy.yaml"
        legacy_path.write_text(yaml_content)
        
        # Migrate it
        new_config = migrator.migrate_from_legacy(legacy_path)
        
        # Verify key settings
        assert new_config.database.default_backend == "duckdb"
        assert new_config.database.duckdb_memory_limit == "16GB"
        assert new_config.performance.batch_size == 25000
        assert new_config.logging.level == "DEBUG"


def test_config_validation():
    """Test configuration validation."""
    validator = ConfigurationValidator()
    
    # Test valid config
    config = NewMDMConfig()
    assert validator.validate_config(config)
    
    # Test invalid config
    config.database.default_backend = "invalid_backend"
    assert not validator.validate_config(config)
    
    report = validator.get_report()
    assert report["error_count"] > 0
    assert "Invalid backend" in str(report["errors"])


def test_environment_override():
    """Test that environment variables override config."""
    # Set test env vars for both systems
    os.environ["MDM_DATABASE_DEFAULT_BACKEND"] = "postgresql"  # Legacy
    os.environ["MDM_DATABASE__DEFAULT_BACKEND"] = "postgresql"  # New (nested)
    
    try:
        # Reset configs
        from mdm.config.config import reset_config_manager
        from mdm.core.config_new import reset_new_config
        
        reset_config_manager()
        reset_new_config()
        
        # Test both systems pick up env var
        feature_flags.set("use_new_config", False)
        legacy_config = get_config()
        assert legacy_config.default_backend == "postgresql"
        
        feature_flags.set("use_new_config", True)
        new_config = get_config()
        assert new_config.default_backend == "postgresql"
        
    finally:
        # Clean up
        os.environ.pop("MDM_DATABASE_DEFAULT_BACKEND", None)
        os.environ.pop("MDM_DATABASE__DEFAULT_BACKEND", None)
        reset_config_manager()
        reset_new_config()


def test_path_compatibility():
    """Test that path resolution is compatible."""
    feature_flags.set("use_new_config", False)
    legacy_config = get_config()
    
    feature_flags.set("use_new_config", True)
    new_config = get_config()
    
    # Check key paths exist and are Path objects
    assert isinstance(legacy_config.config_dir, Path)
    assert isinstance(new_config.config_dir, Path)
    
    # Both should have datasets and logs directories
    assert legacy_config.datasets_dir.name == new_config.datasets_dir.name
    assert legacy_config.logs_dir.name == new_config.logs_dir.name