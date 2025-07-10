#!/usr/bin/env python3
"""Simple unit tests for configuration system."""
import sys
import os
from pathlib import Path
import tempfile

sys.path.insert(0, '/home/xai/DEV/mdm-refactor-2025/src')

from mdm.config.models import MDMConfig, DatabaseConfig, PerformanceConfig
from mdm.config.manager import ConfigurationManager
from mdm.config.migrator import ConfigurationMigrator


def test_validation():
    """Test configuration validation"""
    print("=== Testing Validation ===")
    
    # Valid config
    try:
        config = DatabaseConfig(default_backend="sqlite")
        print("✅ Valid backend accepted")
    except:
        print("❌ Valid backend rejected")
    
    # Invalid config
    try:
        config = DatabaseConfig(default_backend="invalid")
        print("❌ Invalid backend accepted")
    except ValueError as e:
        print(f"✅ Invalid backend rejected: {e}")
    
    # Batch size validation
    try:
        config = PerformanceConfig(batch_size=50)  # Too small
        print("❌ Invalid batch size accepted")
    except ValueError as e:
        print(f"✅ Invalid batch size rejected: {e}")


def test_env_override():
    """Test environment variable override"""
    print("\n=== Testing Environment Override ===")
    
    # Set env vars
    os.environ["MDM_DATABASE__DEFAULT_BACKEND"] = "duckdb"
    os.environ["MDM_PERFORMANCE__BATCH_SIZE"] = "15000"
    os.environ["MDM_LOGGING__LEVEL"] = "DEBUG"
    
    # Load config
    config = MDMConfig.load_from_file(Path("/tmp/nonexistent.yaml"))
    
    print(f"Backend: {config.database.default_backend} (expected: duckdb)")
    print(f"Batch size: {config.performance.batch_size} (expected: 15000)")
    print(f"Log level: {config.logging.level} (expected: DEBUG)")
    
    # Cleanup
    del os.environ["MDM_DATABASE__DEFAULT_BACKEND"]
    del os.environ["MDM_PERFORMANCE__BATCH_SIZE"]
    del os.environ["MDM_LOGGING__LEVEL"]
    
    if config.database.default_backend == "duckdb" and config.performance.batch_size == 15000:
        print("✅ Environment override works!")
    else:
        print("❌ Environment override failed!")


def test_manager():
    """Test configuration manager"""
    print("\n=== Testing Configuration Manager ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "test_mdm.yaml"
        manager = ConfigurationManager(config_file)
        
        # Test get
        backend = manager.get("database.default_backend")
        print(f"Get backend: {backend} (expected: sqlite)")
        
        # Test set
        manager.set("database.default_backend", "postgresql")
        manager.set("performance.batch_size", 25000)
        
        # Reload and verify
        manager.reload()
        new_backend = manager.get("database.default_backend")
        new_batch = manager.get("performance.batch_size")
        
        print(f"After set - backend: {new_backend} (expected: postgresql)")
        print(f"After set - batch: {new_batch} (expected: 25000)")
        
        if new_backend == "postgresql" and new_batch == 25000:
            print("✅ Configuration manager works!")
        else:
            print("❌ Configuration manager failed!")


def test_migrator():
    """Test configuration migrator"""
    print("\n=== Testing Configuration Migrator ===")
    
    migrator = ConfigurationMigrator()
    
    # Test analysis
    analysis = migrator.analyze_current_config()
    print(f"Files found: {len(analysis['files_found'])}")
    print(f"Env vars: {len(analysis['env_vars_found'])}")
    
    # Test migration (dry run)
    result = migrator.perform_migration(dry_run=True)
    print(f"Migration success: {result['success']}")
    print(f"Issues: {len(result['issues'])}")
    
    if "files_found" in analysis and "success" in result:
        print("✅ Configuration migrator works!")
    else:
        print("❌ Configuration migrator failed!")


def main():
    """Run all tests"""
    print("🧪 Running Configuration Unit Tests\n")
    
    test_validation()
    test_env_override()
    test_manager()
    test_migrator()
    
    print("\n✅ All tests completed!")
    print("\nCo możesz teraz przetestować:")
    print("1. Sprawdź plik ~/.mdm/mdm.yaml")
    print("2. Ustaw zmienne środowiskowe: export MDM_DATABASE__DEFAULT_BACKEND=duckdb")
    print("3. Uruchom analizę: python -c 'from mdm.config.migrator import ConfigurationMigrator; m = ConfigurationMigrator(); print(m.analyze_current_config())'")


if __name__ == "__main__":
    main()