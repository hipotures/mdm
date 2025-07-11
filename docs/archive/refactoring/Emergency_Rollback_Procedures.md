# Emergency Rollback Procedures

## Overview
This document provides detailed procedures for rolling back MDM refactoring changes at any stage of the migration. Each rollback procedure is designed to restore the system to a known working state with minimal data loss.

## Pre-Rollback Checklist

Before initiating any rollback:
1. **STOP** all MDM operations immediately
2. **BACKUP** current state: `./scripts/backup_mdm_state.sh`
3. **NOTIFY** all users of the rollback
4. **DOCUMENT** the issue that triggered rollback
5. **PRESERVE** logs for analysis: `./scripts/preserve_logs.sh`

## Rollback Procedures by Stage

### Stage 1: Test Stabilization Rollback
**When to use**: Tests are failing after test framework changes

```bash
# 1. Revert test changes
git checkout main -- tests/
git checkout main -- scripts/run_tests.sh

# 2. Restore test configuration
cp ~/.mdm/backup/test_config.yaml ~/.mdm/test_config.yaml

# 3. Verify rollback
./scripts/run_tests.sh --validate-rollback
```

### Stage 2: Abstraction Layer Rollback
**When to use**: New interfaces breaking existing functionality

```bash
# 1. Disable abstraction layer
export MDM_FEATURE_FLAGS_USE_ABSTRACTION=false

# 2. Revert code changes
git checkout main -- src/mdm/interfaces/
git checkout main -- src/mdm/adapters/

# 3. Clear any cached interfaces
rm -rf ~/.mdm/cache/interfaces/

# 4. Restart MDM services
mdm restart --force
```

### Stage 3: Configuration System Rollback
**When to use**: Configuration loading failures or incorrect settings

```bash
# 1. Restore original configuration
cp ~/.mdm/backup/mdm.yaml ~/.mdm/mdm.yaml
cp -r ~/.mdm/backup/config/ ~/.mdm/config/

# 2. Clear configuration cache
rm -rf ~/.mdm/cache/config/

# 3. Revert to legacy configuration loader
export MDM_USE_LEGACY_CONFIG=true

# 4. Validate configuration
mdm config validate --legacy
```

### Stage 4: Storage Backend Rollback
**When to use**: Data corruption, connection failures, or performance issues

```bash
# CRITICAL: This is the most dangerous rollback - follow carefully

# 1. Stop all dataset operations
mdm dataset lock --all

# 2. Export current data (if possible)
./scripts/emergency_export.sh --output /backup/emergency/

# 3. Restore backend configuration
export MDM_STORAGE_USE_LEGACY=true
export MDM_STORAGE_DISABLE_POOLS=true

# 4. Revert database schemas
./scripts/revert_schema.sh --backend all

# 5. Restore from backup
./scripts/restore_datasets.sh --from ~/.mdm/backup/datasets/

# 6. Validate data integrity
mdm dataset validate --all --deep
```

### Stage 5: Feature Engineering Rollback
**When to use**: Feature computation errors or incompatible features

```bash
# 1. Disable new feature system
export MDM_FEATURES_USE_LEGACY=true

# 2. Clear feature cache
rm -rf ~/.mdm/cache/features/

# 3. Revert feature definitions
cp -r ~/.mdm/backup/features/ ~/.mdm/config/features/

# 4. Regenerate features with legacy system
mdm features regenerate --all --legacy

# 5. Validate feature consistency
mdm features validate --compare-with-backup
```

### Stage 6: Dataset Registration Rollback
**When to use**: Registration failures or corrupted metadata

```bash
# 1. Disable new registration pipeline
export MDM_REGISTRATION_USE_LEGACY=true

# 2. Restore registration metadata
cp -r ~/.mdm/backup/metadata/ ~/.mdm/datasets/

# 3. Rebuild dataset registry
mdm registry rebuild --from-backup

# 4. Validate all datasets
mdm dataset validate --all --check-registration
```

## Complete System Rollback

For catastrophic failures requiring full rollback:

```bash
#!/bin/bash
# emergency_full_rollback.sh

echo "EMERGENCY FULL ROLLBACK - Starting..."

# 1. Create emergency backup of current state
./scripts/backup_mdm_state.sh --emergency --output /backup/emergency_$(date +%Y%m%d_%H%M%S)/

# 2. Stop all MDM processes
pkill -f mdm
sleep 5

# 3. Restore from last known good backup
BACKUP_DATE=$(cat ~/.mdm/last_good_backup)
./scripts/restore_full_system.sh --from-date $BACKUP_DATE

# 4. Reset all feature flags
cat > ~/.mdm/feature_flags.yaml << EOF
use_legacy_config: true
use_legacy_storage: true
use_legacy_features: true
use_legacy_registration: true
enable_new_architecture: false
EOF

# 5. Clear all caches
rm -rf ~/.mdm/cache/

# 6. Restore original codebase
git checkout $LAST_STABLE_TAG

# 7. Reinstall dependencies
pip install -r requirements_stable.txt

# 8. Run validation
./scripts/validate_rollback.sh --full

echo "EMERGENCY FULL ROLLBACK - Complete"
```

## Post-Rollback Actions

After successful rollback:

1. **Validate System Health**
   ```bash
   mdm health check --comprehensive
   mdm dataset list --verify-all
   mdm test --smoke
   ```

2. **Generate Rollback Report**
   ```bash
   ./scripts/generate_rollback_report.sh > rollback_report_$(date +%Y%m%d).md
   ```

3. **Communicate Status**
   - Send notification to all users
   - Update status dashboard
   - Schedule post-mortem meeting

4. **Analyze Root Cause**
   - Collect all logs: `./scripts/collect_logs.sh`
   - Run diagnostics: `mdm diagnose --deep`
   - Create issue report

## Rollback Validation Tests

Run after each rollback to ensure system stability:

```bash
#!/bin/bash
# validate_rollback.sh

echo "=== Rollback Validation ==="

# 1. Configuration tests
echo "Testing configuration..."
mdm config test || exit 1

# 2. Storage tests
echo "Testing storage backends..."
mdm storage test --all-backends || exit 1

# 3. Dataset integrity
echo "Testing dataset integrity..."
mdm dataset validate --all --checksum || exit 1

# 4. Feature consistency
echo "Testing features..."
mdm features test --consistency || exit 1

# 5. API compatibility
echo "Testing API..."
mdm api test --compatibility || exit 1

# 6. Performance baseline
echo "Testing performance..."
mdm benchmark --compare-baseline || exit 1

echo "=== Validation Complete ==="
```

## Rollback Decision Matrix

| Symptom | Severity | Rollback Stage | Time Estimate |
|---------|----------|----------------|---------------|
| Test failures (>10%) | Medium | Stage 1 | 5-10 min |
| Configuration errors | Low-Medium | Stage 3 | 10-15 min |
| Data corruption | CRITICAL | Stage 4 | 30-60 min |
| Performance degradation (>50%) | High | Stage 4-5 | 20-30 min |
| Registration failures | High | Stage 6 | 15-20 min |
| Complete system failure | CRITICAL | Full | 60-90 min |

## Emergency Contacts

- **Primary**: DevOps Team - oncall@company.com
- **Secondary**: Data Platform Team - data-platform@company.com
- **Escalation**: CTO - emergency@company.com

## Automated Rollback Triggers

The system will automatically initiate rollback if:
- Dataset corruption detected (checksum mismatch)
- Storage backend unavailable for >5 minutes
- Memory usage >90% for >10 minutes
- Error rate >25% for >5 minutes
- Critical services down for >2 minutes

## Recovery Time Objectives (RTO)

- Configuration issues: <15 minutes
- Feature engineering issues: <30 minutes
- Storage backend issues: <60 minutes
- Complete system failure: <90 minutes

## Important Notes

1. **NEVER** skip the backup step before rollback
2. **ALWAYS** validate data integrity after rollback
3. **DOCUMENT** every step taken during rollback
4. **TEST** rollback procedures monthly
5. **UPDATE** this document after each rollback incident

## Rollback Automation Scripts

All rollback scripts are available in:
```
scripts/rollback/
├── stage1_test_rollback.sh
├── stage2_abstraction_rollback.sh
├── stage3_config_rollback.sh
├── stage4_storage_rollback.sh
├── stage5_features_rollback.sh
├── stage6_registration_rollback.sh
├── emergency_full_rollback.sh
├── validate_rollback.sh
└── generate_rollback_report.sh
```

Run with: `./scripts/rollback/stage{N}_*.sh --validate --backup`