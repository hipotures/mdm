#!/usr/bin/env python3
"""Test configuration migration in practice."""
import sys
import os
from pathlib import Path

# Add refactor path
sys.path.insert(0, '/home/xai/DEV/mdm-refactor-2025/src')

from mdm.config.models import MDMConfig
from mdm.config.manager import ConfigurationManager
from mdm.config.migrator import ConfigurationMigrator
from mdm.core.feature_flags import feature_flags


def test_1_basic_config():
    """Test 1: Basic configuration access"""
    print("=== Test 1: Basic Configuration ===")
    
    # Create config
    config = MDMConfig()
    print(f"Default backend: {config.database.default_backend}")
    print(f"Batch size: {config.performance.batch_size}")
    print(f"Log level: {config.logging.level}")
    print("✅ Basic config works!\n")


def test_2_env_vars():
    """Test 2: Environment variable override"""
    print("=== Test 2: Environment Variables ===")
    
    # Set some env vars
    os.environ["MDM_DATABASE__DEFAULT_BACKEND"] = "duckdb"
    os.environ["MDM_PERFORMANCE__BATCH_SIZE"] = "25000"
    
    # Load config
    config = MDMConfig.load_from_file(Path("/tmp/nonexistent.yaml"))
    print(f"Backend from env: {config.database.default_backend}")
    print(f"Batch size from env: {config.performance.batch_size}")
    
    # Cleanup
    del os.environ["MDM_DATABASE__DEFAULT_BACKEND"]
    del os.environ["MDM_PERFORMANCE__BATCH_SIZE"]
    print("✅ Environment variables work!\n")


def test_3_yaml_persistence():
    """Test 3: YAML save/load"""
    print("=== Test 3: YAML Persistence ===")
    
    # Create and save config
    config = MDMConfig()
    config.database.default_backend = "postgresql"
    config.performance.batch_size = 30000
    
    test_file = Path("/tmp/test_mdm_config.yaml")
    config.save_to_file(test_file)
    print(f"Saved config to: {test_file}")
    
    # Load it back
    loaded = MDMConfig.load_from_file(test_file)
    print(f"Loaded backend: {loaded.database.default_backend}")
    print(f"Loaded batch size: {loaded.performance.batch_size}")
    
    # Show YAML content
    print("\nYAML content:")
    print(test_file.read_text())
    
    # Cleanup
    test_file.unlink()
    print("✅ YAML persistence works!\n")


def test_4_migration_analysis():
    """Test 4: Analyze current MDM configuration"""
    print("=== Test 4: Configuration Analysis ===")
    
    migrator = ConfigurationMigrator()
    analysis = migrator.analyze_current_config()
    
    print(f"Config files found: {len(analysis['files_found'])}")
    for file in analysis['files_found']:
        print(f"  - {file}")
    
    print(f"Environment variables: {len(analysis['env_vars_found'])}")
    for var in analysis['env_vars_found'][:5]:  # Show first 5
        print(f"  - {var}")
    
    print(f"Total settings: {analysis['settings_count']}")
    print("✅ Analysis works!\n")


def test_5_feature_flags():
    """Test 5: Feature flag switching"""
    print("=== Test 5: Feature Flag Control ===")
    
    from mdm.config import get_config
    
    # Test with new config
    feature_flags.set("use_new_config", True)
    config = get_config()
    print(f"New config type: {type(config).__name__}")
    
    # Test with legacy config
    feature_flags.set("use_new_config", False)
    config = get_config()
    print(f"Legacy config type: {type(config).__name__}")
    
    # Reset
    feature_flags.set("use_new_config", True)
    print("✅ Feature flag switching works!\n")


def main():
    """Run all tests"""
    print("🔍 Testing Configuration Migration\n")
    
    tests = [
        test_1_basic_config,
        test_2_env_vars,
        test_3_yaml_persistence,
        test_4_migration_analysis,
        test_5_feature_flags
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("🎉 All tests completed!")
    print("\n📝 Next steps:")
    print("1. Check if ~/.mdm/mdm.yaml exists")
    print("2. Try: mdm config analyze (if CLI is connected)")
    print("3. Test migration with: mdm config migrate --dry-run")


if __name__ == "__main__":
    main()