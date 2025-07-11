#!/usr/bin/env python3
"""Test real configuration migration."""
import sys
from pathlib import Path

sys.path.insert(0, '/home/xai/DEV/mdm-refactor-2025/src')

from mdm.config.migrator import ConfigurationMigrator
from mdm.config.models import MDMConfig
import yaml


def test_real_migration():
    """Test migration of real MDM config"""
    print("🔍 Testing Real Configuration Migration\n")
    
    # Load existing config
    existing_config_path = Path.home() / ".mdm" / "mdm.yaml"
    print(f"📄 Loading existing config from: {existing_config_path}")
    
    with open(existing_config_path) as f:
        existing_data = yaml.safe_load(f)
    
    print("\n🔧 Existing configuration summary:")
    print(f"  - Database backend: {existing_data.get('database', {}).get('default_backend')}")
    print(f"  - Batch size: {existing_data.get('performance', {}).get('batch_size')}")
    print(f"  - Log level: {existing_data.get('logging', {}).get('level')}")
    print(f"  - SQLite journal mode: {existing_data.get('database', {}).get('sqlite', {}).get('journal_mode')}")
    
    # Test migration
    print("\n🚀 Testing migration...")
    migrator = ConfigurationMigrator()
    
    # Dry run
    result = migrator.perform_migration(dry_run=True)
    
    print(f"\n📊 Migration result:")
    print(f"  - Success: {result['success']}")
    print(f"  - Issues: {len(result['issues'])}")
    print(f"  - Validation errors: {len(result['validation_errors'])}")
    
    if result['issues']:
        print("\n⚠️  Issues found:")
        for issue in result['issues']:
            print(f"  - {issue}")
    
    if result['validation_errors']:
        print("\n❌ Validation errors:")
        for error in result['validation_errors']:
            print(f"  - {error}")
    
    # Test new config format
    print("\n🆕 Testing new config format...")
    new_config = MDMConfig()
    
    # Apply some settings from old config
    new_config.database.default_backend = existing_data.get('database', {}).get('default_backend', 'sqlite')
    new_config.performance.batch_size = existing_data.get('performance', {}).get('batch_size', 10000)
    new_config.logging.level = existing_data.get('logging', {}).get('level', 'INFO')
    
    # SQLite specific settings
    sqlite_config = existing_data.get('database', {}).get('sqlite', {})
    if sqlite_config.get('journal_mode'):
        new_config.database.sqlite_journal_mode = sqlite_config['journal_mode']
    if sqlite_config.get('synchronous'):
        new_config.database.sqlite_synchronous = sqlite_config['synchronous']
    
    # Save to test file
    test_path = Path("/tmp/mdm_new_config.yaml")
    new_config.save_to_file(test_path)
    
    print(f"\n💾 Saved new format config to: {test_path}")
    print("\nNew config content (first 20 lines):")
    lines = test_path.read_text().splitlines()[:20]
    for line in lines:
        print(f"  {line}")
    
    # Cleanup
    test_path.unlink()
    
    print("\n✅ Migration test completed!")


if __name__ == "__main__":
    test_real_migration()