#!/usr/bin/env python3
"""
Example demonstrating configuration migration between old and new systems.

This example shows:
- Loading configurations with both systems
- Using adapters for compatibility
- Migrating from legacy to new format
- Validating configuration parity
- Testing with feature flags
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mdm.core import feature_flags, metrics_collector
from mdm.adapters.config_adapters import get_config_manager, get_config
from mdm.migration.config_migration import (
    ConfigurationMigrator,
    ConfigurationValidator,
    migrate_config_file
)
from mdm.testing.config_comparison import ConfigComparisonTester

print("Configuration Migration Example")
print("=" * 60)

# 1. Show current configuration system
print("\n1. Current Configuration System")
print("-" * 40)

# Use legacy by default
feature_flags.set("use_new_config", False)
print(f"Using legacy configuration: {not feature_flags.get('use_new_config')}")

config = get_config()
print(f"  Home directory: {config.home_dir}")
print(f"  Default backend: {config.default_backend}")
print(f"  Batch size: {config.batch_size}")
print(f"  Auto-detect enabled: {config.enable_auto_detect}")

# 2. Switch to new configuration
print("\n2. New Configuration System")
print("-" * 40)

feature_flags.set("use_new_config", True)
print(f"Using new configuration: {feature_flags.get('use_new_config')}")

config = get_config()
print(f"  Home directory: {config.home_dir}")
print(f"  Default backend: {config.default_backend}")
print(f"  Batch size: {config.batch_size}")
print(f"  Auto-detect enabled: {config.enable_auto_detect}")

# 3. Demonstrate migration
print("\n3. Configuration Migration")
print("-" * 40)

migrator = ConfigurationMigrator()

# Check if legacy config exists
legacy_path = Path.home() / ".mdm" / "mdm.yaml"
if legacy_path.exists():
    print(f"Found legacy config at: {legacy_path}")
    
    # Perform migration
    new_config = migrator.migrate_from_legacy(legacy_path)
    print("Migration completed successfully")
    
    # Validate migration
    is_valid = migrator.validate_migration(legacy_path)
    print(f"Migration validation: {'PASSED' if is_valid else 'FAILED'}")
    
    if not is_valid and migrator.differences:
        print("\nDifferences found:")
        for key, old_val, new_val in migrator.differences[:5]:
            print(f"  {key}: {old_val} -> {new_val}")
else:
    print("No legacy config found, creating example")
    
    # Create example legacy config
    from mdm.config.config import ConfigManager
    from mdm.models.config import MDMConfig as LegacyMDMConfig
    
    manager = ConfigManager()
    legacy_config = LegacyMDMConfig()
    
    # Customize some values
    legacy_config.database.default_backend = "duckdb"
    legacy_config.performance.batch_size = 25000
    legacy_config.logging.level = "DEBUG"
    
    # Save it
    example_path = Path.home() / ".mdm" / "example_legacy.yaml"
    manager.save(legacy_config, example_path)
    print(f"Created example legacy config at: {example_path}")
    
    # Now migrate it
    new_config = migrator.migrate_from_legacy(example_path)
    new_path = Path.home() / ".mdm" / "example_new.yaml"
    new_config.to_yaml(new_path)
    print(f"Migrated to new format at: {new_path}")

# 4. Validate new configuration
print("\n4. Configuration Validation")
print("-" * 40)

validator = ConfigurationValidator()
is_valid = validator.validate_config(new_config)

report = validator.get_report()
print(f"Validation result: {'PASSED' if is_valid else 'FAILED'}")
print(f"  Errors: {report['error_count']}")
print(f"  Warnings: {report['warning_count']}")

if report['errors']:
    print("\nErrors:")
    for error in report['errors']:
        print(f"  - {error}")
        
if report['warnings']:
    print("\nWarnings:")
    for warning in report['warnings']:
        print(f"  - {warning}")

# 5. Run comparison tests
print("\n5. Configuration Comparison Tests")
print("-" * 40)

tester = ConfigComparisonTester()
results = tester.run_all_tests()

print(f"Total tests: {results['total_tests']}")
print(f"Passed: {results['passed']}")
print(f"Failed: {results['failed']}")
print(f"Success rate: {results['success_rate']:.1f}%")

print("\nTest results:")
for result in results['results']:
    status = "✅" if result.get('passed', False) else "❌"
    print(f"  {status} {result['test_name']}")

# 6. Performance comparison
print("\n6. Performance Comparison")
print("-" * 40)

import time

# Time legacy loading
feature_flags.set("use_new_config", False)
start = time.perf_counter()
for _ in range(100):
    _ = get_config_manager().load()
legacy_time = time.perf_counter() - start

# Time new loading
feature_flags.set("use_new_config", True)
start = time.perf_counter()
for _ in range(100):
    _ = get_config_manager().load()
new_time = time.perf_counter() - start

print(f"Legacy loading (100x): {legacy_time:.3f}s")
print(f"New loading (100x): {new_time:.3f}s")
print(f"Performance delta: {((new_time - legacy_time) / legacy_time * 100):.1f}%")

# 7. Environment variable handling
print("\n7. Environment Variable Handling")
print("-" * 40)

# Set some env vars
os.environ["MDM_DATABASE_DEFAULT_BACKEND"] = "postgresql"
os.environ["MDM_PERFORMANCE_BATCH_SIZE"] = "50000"

# Reset configs to pick up env vars
from mdm.config.config import reset_config_manager
from mdm.core.config_new import reset_new_config

reset_config_manager()
reset_new_config()

# Check both systems
feature_flags.set("use_new_config", False)
legacy_config = get_config()
print(f"Legacy - Backend from env: {legacy_config.default_backend}")
print(f"Legacy - Batch size from env: {legacy_config.batch_size}")

feature_flags.set("use_new_config", True)
new_config = get_config()
print(f"New - Backend from env: {new_config.default_backend}")
print(f"New - Batch size from env: {new_config.batch_size}")

# Clean up env vars
os.environ.pop("MDM_DATABASE_DEFAULT_BACKEND", None)
os.environ.pop("MDM_PERFORMANCE_BATCH_SIZE", None)

# 8. Generate reports
print("\n8. Migration Reports")
print("-" * 40)

# Create migration report
report_path = migrator.create_migration_report(
    Path.home() / ".mdm" / "migration_reports" / "config_migration.json"
)
print(f"Migration report: {report_path}")

# Create comparison report
comparison_path = tester.generate_report()
print(f"Comparison report: {comparison_path}")

# Export metrics
metrics_path = metrics_collector.export("config_migration_metrics.json")
print(f"Metrics: {metrics_path}")

print("\n" + "=" * 60)
print("Configuration migration example complete!")
print("\nKey takeaways:")
print("- Both configuration systems work side-by-side")
print("- Feature flags control which system is active") 
print("- Adapters provide a common interface")
print("- Migration preserves all settings")
print("- Validation ensures correctness")
print("- Performance is comparable between systems")