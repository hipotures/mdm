# MDM Migration Guide

## Overview

This guide helps developers migrate from the legacy MDM implementation to the new refactored architecture. The migration is designed to be gradual and non-disruptive using feature flags.

## Table of Contents

1. [Pre-Migration Checklist](#pre-migration-checklist)
2. [Migration Strategy](#migration-strategy)
3. [Step-by-Step Migration](#step-by-step-migration)
4. [Code Changes](#code-changes)
5. [Testing During Migration](#testing-during-migration)
6. [Rollback Procedures](#rollback-procedures)
7. [Common Issues](#common-issues)

## Pre-Migration Checklist

Before starting the migration:

- [ ] Backup all datasets in `~/.mdm/`
- [ ] Document current MDM version and configuration
- [ ] Review breaking changes in the new architecture
- [ ] Test migration in a development environment
- [ ] Plan rollback strategy
- [ ] Schedule migration during low-usage period

## Migration Strategy

### Feature Flag Approach

The migration uses feature flags to enable gradual rollout:

```python
# Feature flags control which implementation is used
use_new_storage = False    # New storage backends
use_new_features = False   # New feature engineering
use_new_dataset = False    # New dataset management
use_new_config = False     # New configuration system
use_new_cli = False        # New CLI implementation
```

### Phased Migration

1. **Phase 1: Configuration Migration**
   - Migrate configuration to new format
   - Test with legacy implementation

2. **Phase 2: Storage Backend Migration**
   - Enable new storage backends
   - Validate data integrity

3. **Phase 3: Feature Engineering Migration**
   - Switch to new feature generation
   - Verify feature compatibility

4. **Phase 4: Dataset Management Migration**
   - Migrate dataset operations
   - Update automation scripts

5. **Phase 5: CLI Migration**
   - Switch to new CLI commands
   - Update documentation

## Step-by-Step Migration

### Step 1: Install New Version

```bash
# Backup current installation
pip freeze | grep mdm > mdm_version_backup.txt

# Install new version
pip install --upgrade mdm-refactor

# Verify installation
mdm version
```

### Step 2: Migrate Configuration

```python
from mdm.migration import ConfigMigrator

# Create migrator
migrator = ConfigMigrator()

# Backup current config
migrator.backup_config("~/.mdm/config_backup")

# Migrate to new format
new_config = migrator.migrate_config()

# Validate migrated config
if migrator.validate_config(new_config):
    print("Configuration migrated successfully")
else:
    print("Configuration migration failed")
    migrator.restore_config("~/.mdm/config_backup")
```

### Step 3: Enable Storage Migration

```python
from mdm.core import feature_flags
from mdm.migration import StorageMigrator

# Enable new storage backend
feature_flags.set("use_new_storage", True)

# Migrate each dataset
migrator = StorageMigrator()
for dataset_name in migrator.list_datasets():
    result = migrator.migrate_dataset(dataset_name)
    if not result.success:
        print(f"Failed to migrate {dataset_name}: {result.error}")
        # Rollback this dataset
        migrator.rollback_dataset(dataset_name)
```

### Step 4: Validate Data Integrity

```python
from mdm.migration import DataValidator

# Validate all migrated datasets
validator = DataValidator()
validation_report = validator.validate_all_datasets()

# Check for issues
if validation_report.has_errors():
    print("Validation errors found:")
    for error in validation_report.errors:
        print(f"  - {error}")
else:
    print("All datasets validated successfully")
```

### Step 5: Enable Remaining Features

```python
# Gradually enable other features
feature_flags.set("use_new_features", True)
feature_flags.set("use_new_dataset", True)
feature_flags.set("use_new_config", True)
feature_flags.set("use_new_cli", True)

# Or enable all at once (after testing)
feature_flags.enable_all_new_features()
```

## Code Changes

### Import Changes

**Legacy imports:**
```python
from mdm.storage import SQLiteBackend
from mdm.dataset import DatasetRegistrar
from mdm.features import FeatureGenerator
```

**New imports:**
```python
from mdm.adapters import get_storage_backend
from mdm.adapters import get_dataset_registrar
from mdm.adapters import get_feature_generator
```

### API Changes

**Dataset Registration:**

Legacy:
```python
registrar = DatasetRegistrar()
registrar.register(name, path, target)
```

New:
```python
registrar = get_dataset_registrar()
dataset_info = registrar.register_dataset(
    name=name,
    path=path,
    target_column=target,
    problem_type="classification"
)
```

**Storage Backend:**

Legacy:
```python
backend = SQLiteBackend()
engine = backend.create_engine(db_path)
```

New:
```python
backend = get_storage_backend("sqlite")
engine = backend.get_engine(db_path)
```

**Feature Generation:**

Legacy:
```python
generator = FeatureGenerator()
features = generator.generate(df)
```

New:
```python
generator = get_feature_generator()
features = generator.generate_features(
    df=df,
    column_types={"col1": "numeric", "col2": "categorical"}
)
```

### Configuration Changes

**Environment Variables:**

Legacy:
```bash
export MDM_BACKEND=sqlite
export MDM_BATCH_SIZE=1000
```

New:
```bash
export MDM_DATABASE_DEFAULT_BACKEND=sqlite
export MDM_PERFORMANCE_BATCH_SIZE=1000
```

**Configuration File:**

Legacy (`~/.mdm/config.yaml`):
```yaml
backend: sqlite
batch_size: 1000
```

New (`~/.mdm/mdm.yaml`):
```yaml
database:
  default_backend: sqlite
performance:
  batch_size: 1000
```

## Testing During Migration

### Integration Tests

```python
from mdm.testing import IntegrationTestFramework

# Run integration tests
framework = IntegrationTestFramework()
results = framework.run_all_tests()

# Check results
if results['passed'] == results['total']:
    print("All integration tests passed")
else:
    print(f"Failed tests: {results['failed_tests']}")
```

### Migration Tests

```python
from mdm.testing import MigrationTestSuite

# Test migration readiness
suite = MigrationTestSuite()
readiness = suite.test_migration_readiness()

print(f"Migration readiness: {readiness['overall_score']}%")
print(f"Status: {readiness['status']}")
```

### Performance Comparison

```python
from mdm.testing import PerformanceBenchmark

# Compare legacy vs new performance
benchmark = PerformanceBenchmark()
comparison = benchmark.compare_implementations(
    test_dataset="sample_data.csv"
)

# Review results
for operation, metrics in comparison.items():
    print(f"{operation}:")
    print(f"  Legacy: {metrics['legacy']['duration']:.2f}s")
    print(f"  New: {metrics['new']['duration']:.2f}s")
    print(f"  Speedup: {metrics['speedup']:.2f}x")
```

## Rollback Procedures

### Quick Rollback

```python
from mdm.migration import RollbackManager

# Create rollback manager
rollback = RollbackManager()

# Rollback all changes
rollback.rollback_all()

# Or rollback specific components
rollback.rollback_storage()
rollback.rollback_config()
```

### Manual Rollback

1. **Disable feature flags:**
```python
from mdm.core import feature_flags
feature_flags.disable_all_new_features()
```

2. **Restore configuration:**
```bash
cp ~/.mdm/config_backup/config.yaml ~/.mdm/config.yaml
```

3. **Restore datasets (if needed):**
```bash
rm -rf ~/.mdm/datasets_new
mv ~/.mdm/datasets_backup ~/.mdm/datasets
```

## Common Issues

### Issue 1: Dataset Not Found After Migration

**Symptom:** Datasets that existed before migration are not visible.

**Solution:**
```python
from mdm.migration import DatasetDiscovery

# Re-discover datasets
discovery = DatasetDiscovery()
discovery.scan_and_register_datasets()
```

### Issue 2: Performance Degradation

**Symptom:** Operations are slower after migration.

**Solution:**
```python
# Enable performance optimizations
from mdm.adapters import get_config_manager

config_manager = get_config_manager()
config_manager.update_config({
    "performance": {
        "enable_optimizations": True,
        "cache_size_mb": 100,
        "connection_pool_size": 10
    }
})
```

### Issue 3: Feature Incompatibility

**Symptom:** Generated features differ between versions.

**Solution:**
```python
# Use compatibility mode
from mdm.core import feature_flags

feature_flags.set("feature_compatibility_mode", True)
```

### Issue 4: Configuration Not Loading

**Symptom:** Custom configuration not being applied.

**Solution:**
```bash
# Check configuration location
mdm config --show-path

# Validate configuration
mdm config --validate

# Fix permissions
chmod 644 ~/.mdm/mdm.yaml
```

## Migration Monitoring

### Track Migration Progress

```python
from mdm.migration import MigrationMonitor

# Create monitor
monitor = MigrationMonitor()

# Get migration status
status = monitor.get_migration_status()
print(f"Migrated components: {status['completed']}")
print(f"Pending components: {status['pending']}")
print(f"Failed components: {status['failed']}")

# Get detailed report
report = monitor.generate_report()
report.save("migration_report.html")
```

### Monitor System Health

```python
from mdm.performance import get_monitor

# Get performance monitor
monitor = get_monitor()

# Check system metrics
metrics = monitor.get_system_metrics()
print(f"CPU Usage: {metrics['cpu_percent']}%")
print(f"Memory Usage: {metrics['memory_percent']}%")
print(f"Active Operations: {metrics['active_operations']}")
```

## Post-Migration

### Cleanup

```python
from mdm.migration import MigrationCleanup

# Remove migration artifacts
cleanup = MigrationCleanup()
cleanup.remove_backups()
cleanup.remove_legacy_configs()
cleanup.optimize_storage()
```

### Update Documentation

1. Update internal documentation
2. Update automation scripts
3. Update CI/CD pipelines
4. Train team on new features

### Performance Tuning

```python
from mdm.performance import PerformanceTuner

# Auto-tune performance settings
tuner = PerformanceTuner()
optimal_config = tuner.auto_tune()
tuner.apply_config(optimal_config)
```

## Support

For migration issues:

1. Check the [Troubleshooting Guide](./Troubleshooting_Guide.md)
2. Review [API Reference](./API_Reference.md) for new APIs
3. File issues at https://github.com/mdm/mdm-refactor/issues
4. Contact support with migration logs

## Appendix

### Migration Checklist Template

```markdown
## Migration Checklist for [ENVIRONMENT]

**Date:** ___________
**Version:** From _______ to _______

### Pre-Migration
- [ ] Backup completed
- [ ] Test environment validated
- [ ] Team notified
- [ ] Rollback plan documented

### Migration Steps
- [ ] Configuration migrated
- [ ] Storage backends migrated
- [ ] Features migrated
- [ ] Dataset operations migrated
- [ ] CLI migrated

### Validation
- [ ] Integration tests passed
- [ ] Performance benchmarks acceptable
- [ ] Data integrity verified
- [ ] User acceptance tested

### Post-Migration
- [ ] Documentation updated
- [ ] Team trained
- [ ] Monitoring enabled
- [ ] Backups cleaned up
```