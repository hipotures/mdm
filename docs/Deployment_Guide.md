# MDM Deployment Guide

## Overview

This guide provides detailed instructions for deploying the refactored MDM system to production. It covers pre-deployment preparation, the deployment process, and post-deployment verification.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Process](#deployment-process)
3. [Monitoring During Deployment](#monitoring-during-deployment)
4. [Rollback Procedures](#rollback-procedures)
5. [Post-Deployment Validation](#post-deployment-validation)
6. [Troubleshooting](#troubleshooting)

## Pre-Deployment Checklist

### System Requirements

Ensure the target environment meets these requirements:

- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Disk Space**: At least 10GB free space
- **CPU**: Minimum 2 cores (4+ recommended)
- **OS**: Linux, macOS, or Windows with WSL2

### Access Requirements

- [ ] SSH access to production servers
- [ ] Database credentials (if using PostgreSQL)
- [ ] Write permissions to MDM directories
- [ ] Access to monitoring systems
- [ ] Ability to modify environment variables

### Backup Verification

```bash
# Create full backup before deployment
mdm backup create ~/mdm_production_backup_$(date +%Y%m%d)

# Verify backup integrity
mdm backup verify ~/mdm_production_backup_*

# List backup contents
ls -la ~/mdm_production_backup_*/
```

### Test Migration

Always test the migration in a staging environment first:

```bash
# Run migration in dry-run mode
./scripts/final_migration.py --dry-run

# Review the output carefully
# Check for any warnings or potential issues
```

## Deployment Process

### Step 1: Prepare Environment

```bash
# Set environment for production
export MDM_ENVIRONMENT=production

# Enable production logging
export MDM_LOGGING_LEVEL=INFO
export MDM_LOGGING_FILE=/var/log/mdm/migration.log

# Set performance parameters
export MDM_PERFORMANCE_BATCH_SIZE=50000
export MDM_PERFORMANCE_MAX_WORKERS=8
```

### Step 2: Run Pre-Deployment Validation

```bash
# Run comprehensive validation
python -m mdm.rollout.validator

# Check specific components
mdm doctor --all
mdm dataset validate --all
```

### Step 3: Create Rollback Point

```bash
# Create rollback point before starting
python -c "
from mdm.rollout import RollbackManager
manager = RollbackManager()
point = manager.create_rollback_point('Pre-deployment snapshot')
print(f'Created rollback point: {point.id}')
"
```

### Step 4: Execute Migration

```bash
# Run the final migration script
./scripts/final_migration.py

# For automated deployments (CI/CD)
./scripts/final_migration.py --auto-approve
```

### Step 5: Monitor Progress

The migration script provides real-time progress updates. Monitor for:

- ✓ Green checkmarks indicate successful steps
- ✗ Red X marks indicate failures
- ⚠ Yellow warnings require attention but aren't blocking

### Step 6: Verify Deployment

```bash
# Run post-deployment validation
mdm validation post-deployment

# Check system health
mdm health check

# Verify feature flags
mdm flags status
```

## Monitoring During Deployment

### Real-Time Dashboard

Start the monitoring dashboard in a separate terminal:

```python
from mdm.rollout import RolloutMonitor

monitor = RolloutMonitor()
monitor.start_monitoring()
monitor.display_dashboard()
```

### Key Metrics to Watch

1. **System Metrics**
   - CPU usage < 80%
   - Memory usage < 90%
   - Disk I/O within normal range

2. **Application Metrics**
   - Error rate < 1%
   - Response time < 1s for most operations
   - Active operations count stable

3. **Migration Metrics**
   - Dataset migration success rate > 99%
   - Feature generation completion rate 100%
   - No data integrity errors

### Alert Handling

If alerts occur during deployment:

1. **Warning Alerts**: Can usually continue, but investigate after deployment
2. **Error Alerts**: Pause and investigate immediately
3. **Critical Alerts**: Consider rollback

## Rollback Procedures

### Automatic Rollback

If the migration fails, it will offer automatic rollback:

```
Migration did not complete successfully.
Would you like to rollback? (y/n):
```

### Manual Rollback

To manually rollback at any time:

```bash
# List available rollback points
python -c "
from mdm.rollout import RollbackManager
manager = RollbackManager()
manager.list_rollback_points()
"

# Rollback to specific point
./scripts/final_migration.py --rollback

# Or rollback specific components
python -c "
from mdm.rollout import RollbackManager, RollbackType
manager = RollbackManager()
result = manager.rollback(
    rollback_type=RollbackType.FEATURE_FLAGS,
    dry_run=False
)
"
```

### Rollback Verification

After rollback:

```bash
# Verify system state
mdm doctor --all

# Check feature flags are disabled
mdm flags status

# Verify datasets are accessible
mdm dataset list
```

## Post-Deployment Validation

### Automated Tests

```bash
# Run integration tests
python -m mdm.testing.integration_framework

# Run migration validation
python -m mdm.testing.migration_tests

# Performance benchmarks
python -m mdm.testing.performance_benchmark
```

### Manual Verification

1. **Dataset Operations**
```bash
# List datasets
mdm dataset list

# Test dataset registration
mdm dataset register test_deployment /tmp/test.csv --dry-run

# Verify statistics
mdm dataset stats <existing_dataset>
```

2. **Feature Engineering**
```python
from mdm.adapters import get_feature_generator

generator = get_feature_generator()
# Test feature generation
transformers = generator.get_available_transformers()
print(f"Available transformers: {transformers}")
```

3. **Storage Backends**
```python
from mdm.adapters import get_storage_backend

# Test each backend
for backend_type in ['sqlite', 'duckdb', 'postgresql']:
    try:
        backend = get_storage_backend(backend_type)
        print(f"{backend_type}: OK")
    except Exception as e:
        print(f"{backend_type}: FAILED - {e}")
```

### Performance Validation

Compare performance before and after:

```python
from mdm.performance import get_monitor

monitor = get_monitor()
report = monitor.get_report()

# Check for performance regressions
if 'summary' in report and 'timers' in report['summary']:
    for op, stats in report['summary']['timers'].items():
        if stats['avg'] > 1.0:  # Operations taking > 1s
            print(f"Warning: {op} averaging {stats['avg']:.2f}s")
```

## Troubleshooting

### Common Issues

#### 1. Migration Hangs

**Symptom**: Progress stops without error

**Solution**:
```bash
# Check system resources
top
df -h

# Check for locks
lsof | grep mdm

# Kill stuck processes if needed
pkill -f mdm
```

#### 2. Feature Flag Issues

**Symptom**: Old implementation still being used

**Solution**:
```python
# Force clear all caches
from mdm.adapters import (
    clear_storage_cache,
    clear_feature_cache,
    clear_dataset_cache,
    clear_cli_cache
)

clear_storage_cache()
clear_feature_cache()
clear_dataset_cache()
clear_cli_cache()

# Verify flags
from mdm.core import feature_flags
print(feature_flags.get_all())
```

#### 3. Database Connection Errors

**Symptom**: Storage backend failures

**Solution**:
```bash
# Check database service
systemctl status postgresql  # or appropriate service

# Test connection manually
psql -h localhost -U mdm_user -d mdm_db -c "SELECT 1"

# Check connection limits
mdm config show | grep -A5 "pool"
```

### Debug Mode

For detailed debugging during deployment:

```bash
# Maximum verbosity
export MDM_LOGGING_LEVEL=DEBUG
export MDM_LOGGING_FILE=/tmp/mdm_deployment_debug.log

# Run with profiling
./scripts/final_migration.py --dry-run 2>&1 | tee deployment_debug.log
```

### Getting Help

If issues persist:

1. Collect diagnostic information:
```bash
mdm debug report > mdm_deployment_issue.txt
```

2. Include:
   - Deployment logs
   - Error messages
   - System specifications
   - Steps to reproduce

3. Contact support with collected information

## Deployment Checklist Template

```markdown
## MDM Production Deployment - [DATE]

### Pre-Deployment
- [ ] Backup created and verified
- [ ] Staging test successful
- [ ] Team notified
- [ ] Maintenance window scheduled
- [ ] Rollback plan documented

### During Deployment
- [ ] Pre-flight checks passed
- [ ] Configuration migrated
- [ ] Feature flags enabled
- [ ] Storage backends migrated
- [ ] Datasets migrated
- [ ] Validation passed

### Post-Deployment
- [ ] Integration tests passed
- [ ] Performance acceptable
- [ ] Monitoring active
- [ ] Documentation updated
- [ ] Team notified
- [ ] Backup cleaned up

### Sign-off
- Deployed by: ___________
- Verified by: ___________
- Date/Time: ___________
```

## Best Practices

1. **Always Test First**: Never deploy directly to production without staging tests
2. **Monitor Actively**: Keep monitoring dashboard open during deployment
3. **Document Everything**: Record any issues or deviations from plan
4. **Have Rollback Ready**: Know your rollback procedure before starting
5. **Communicate**: Keep stakeholders informed of progress
6. **Verify Twice**: Double-check critical operations before proceeding

## Next Steps

After successful deployment:

1. Monitor system for 24-48 hours
2. Gather performance metrics
3. Document any issues encountered
4. Plan optimization based on production usage
5. Schedule follow-up review meeting