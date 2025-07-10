# Step 0: Prerequisites and Preparation

## Overview

Before starting the migration, ensure all prerequisites are met and the team is prepared. This step is crucial for migration success.

## Duration

1-2 days (not included in the 21-week timeline)

## Objectives

1. Verify system readiness
2. Set up development environment
3. Create backup and recovery plans
4. Set up simple monitoring for migration tracking
5. Prepare workspace for safe refactoring

## Detailed Steps

### 1. System Readiness Check

#### 1.1 Verify Git Repository State
```bash
# Ensure you're on the correct branch
git checkout main
git pull origin main

# Check for uncommitted changes
git status

# Verify repository health
git fsck --full

# Check current branch structure
git branch -a
```

#### 1.2 Document Current State
```bash
# Generate current code statistics
find src/ -name "*.py" | xargs wc -l > docs/refactoring/migration-steps/logs/initial-code-stats.txt

# Document current test status
./scripts/run_tests.sh --all > docs/refactoring/migration-steps/logs/initial-test-status.txt

# Save current coverage report
pytest --cov=src/mdm --cov-report=html:docs/refactoring/migration-steps/logs/initial-coverage
```

#### 1.3 Verify Development Tools
```bash
# Check Python version (should be 3.8+)
python --version

# Verify uv is installed
uv --version

# Check all development dependencies
uv pip list

# Verify test tools
pytest --version
ruff --version
black --version
mypy --version
```

### 2. Environment Setup

#### 2.1 Create Migration Workspace
```bash
# Create migration directories
mkdir -p ~/DEV/mdm-migration/{backups,logs,scripts,comparison-tests}

# Set up environment variables

# MDM Migration Environment
export MDM_MIGRATION_ROOT=~/DEV/mdm-migration
export MDM_ORIGINAL_ROOT=/home/xai/DEV/mdm.wt.dev2
export MDM_REFACTOR_ROOT=/home/xai/DEV2/mdm
export MDM_MIGRATION_LOG=$MDM_MIGRATION_ROOT/logs/migration.log


```

#### 2.2 Install Additional Tools
```bash
# Install migration-specific tools
uv pip install --user pytest-benchmark pytest-timeout pytest-xdist memory_profiler

# Install comparison tools
uv pip install --user deepdiff jsondiff
```

#### 2.3 Set Up Git Worktrees
```bash
# Create worktree for refactoring work
cd $MDM_ORIGINAL_ROOT
git worktree add ../mdm-refactor-2025 -b refactor-2025

# Create worktree for comparison testing
git worktree add ../mdm-comparison -b comparison-testing
```

### 3. Backup and Recovery Plans

#### 3.1 Create Full System Backup
```bash
# Backup current MDM installation
tar -czf $MDM_MIGRATION_ROOT/backups/mdm-pre-migration-$(date +%Y%m%d).tar.gz ~/.mdm/

# Backup current codebase
cd $MDM_ORIGINAL_ROOT
git bundle create $MDM_MIGRATION_ROOT/backups/mdm-repo-$(date +%Y%m%d).bundle --all

# Document current datasets
mdm dataset list > $MDM_MIGRATION_ROOT/backups/datasets-list-$(date +%Y%m%d).txt
```

#### 3.2 Create Recovery Scripts
```bash
# Create recovery script
cat > $MDM_MIGRATION_ROOT/scripts/emergency-recovery.sh << 'EOF'
#!/bin/bash
# Emergency Recovery Script
set -e

echo "Starting emergency recovery..."

# Restore MDM data
if [ -f "$1" ]; then
    tar -xzf "$1" -C ~/
    echo "MDM data restored from $1"
else
    echo "Usage: $0 <backup-file.tar.gz>"
    exit 1
fi

# Reset feature flags
cat > ~/.mdm/mdm.yaml << 'YAML'
database:
  default_backend: sqlite
refactoring:
  use_new_backend: false
  use_new_registrar: false
  use_new_features: false
YAML

echo "Recovery complete. Please restart your shell."
EOF

chmod +x $MDM_MIGRATION_ROOT/scripts/emergency-recovery.sh
```

### 4. Team Setup

#### 4.1 Define Roles
Create a team roster in `migration-team.md`:
```markdown
# Migration Team Roles

## Core Team
- **Migration Lead**: [Name] - Overall coordination and decisions
- **Test Lead**: [Name] - Test strategy and validation
- **Backend Developer**: [Name] - Storage and infrastructure
- **Feature Developer**: [Name] - Feature engineering migration

## Responsibilities
- Code reviews: All changes require 2 reviews
- Testing: Developer tests + Test Lead validation
- Documentation: Update as you go
- Communication: Daily standup during active migration
```

#### 4.2 Communication Channels
```bash
# Set up migration log
touch $MDM_MIGRATION_LOG

# Create daily status template
cat > $MDM_MIGRATION_ROOT/logs/daily-status-template.md << 'EOF'
# Daily Migration Status - [DATE]

## Completed Today
- [ ] Task 1
- [ ] Task 2

## Blockers
- None

## Tomorrow's Plan
- [ ] Task 1
- [ ] Task 2

## Metrics
- Tests Passing: X/Y
- Coverage: X%
- Performance Delta: X%
EOF
```

### 5. Simple Monitoring Setup (Single-User)

Since MDM is designed for single-user deployment, we use a lightweight, built-in monitoring system that requires no external infrastructure.

#### 5.1 Enable Built-in Monitoring

MDM's monitoring starts automatically. To verify it's working:

```bash
# Test that monitoring is available
python -c "from mdm.monitoring import SimpleMonitor; print('Monitoring ready')"

# Check monitoring files are created
ls -la ~/.mdm/metrics.db ~/.mdm/logs/

# View initial statistics (will be empty initially)
mdm stats summary
```

#### 5.2 Create Migration Baseline

Before starting migration, capture baseline metrics:

```bash
# Run any existing datasets through operations to establish baseline
# (Skip if no datasets registered yet)
mdm dataset list

# If you have test datasets, run some operations
mdm dataset info test_dataset 2>/dev/null || echo "No datasets yet"

# Generate initial dashboard to verify everything works
mdm stats dashboard --output ~/mdm-migration/baseline-dashboard.html --no-open

# View current logs
mdm stats logs --tail 20
```

#### 5.3 Migration Monitoring Script

Create a simple script to track migration progress:

```bash
# $MDM_MIGRATION_ROOT/scripts/track_migration.sh
#!/bin/bash

echo "=== MDM Migration Status ==="
echo "Date: $(date)"

# Check current metrics
echo -e "\nCurrent Statistics:"
mdm stats summary

# Check for errors
echo -e "\nRecent Errors:"
mdm stats logs --level ERROR --tail 10

# Save dashboard snapshot
mdm stats dashboard --output "$MDM_MIGRATION_ROOT/logs/dashboard_$(date +%Y%m%d_%H%M%S).html" --no-open

echo -e "\nMonitoring data saved to $MDM_MIGRATION_ROOT/logs/"
```

Make it executable:
```bash
chmod +x $MDM_MIGRATION_ROOT/scripts/track_migration.sh
```

#### 5.4 Set Up Simple Automated Checks
```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Run tests before committing during migration

if [[ $(git rev-parse --abbrev-ref HEAD) == refactor-* ]]; then
    echo "Running migration validation checks..."
    
    # Check for broken imports
    python -m py_compile src/**/*.py || exit 1
    
    # Run quick tests
    pytest tests/unit/test_config.py -v || exit 1
    
    echo "Pre-commit checks passed!"
fi
EOF

chmod +x .git/hooks/pre-commit
```

## Validation Checklist

Before proceeding to Step 1, ensure:

- [ ] Repository access verified
- [ ] Backup created and recovery script tested
- [ ] Git worktrees set up successfully
- [ ] Development tools installed and working
- [ ] Migration directories created
- [ ] Simple monitoring verified (mdm stats works)
- [ ] Pre-commit hooks installed
- [ ] Baseline metrics captured

## Success Criteria

- All prerequisites checked and verified
- Backup and recovery tested successfully
- Simple monitoring system operational
- Development environment fully prepared
- Safe workspace for refactoring established

## Troubleshooting

### Issue: Git worktree creation fails
```bash
# Clean up and retry
git worktree prune
git worktree add ../mdm-refactor-2025 -b refactor-2025
```

### Issue: Test tools not found
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
uv pip install -e .
```

### Issue: Backup too large
```bash
# Create incremental backups
tar -czf backup-code.tar.gz --exclude=.venv --exclude=__pycache__ .
tar -czf backup-data.tar.gz ~/.mdm/datasets/
```

## Next Steps

Once all prerequisites are complete, proceed to [01-test-stabilization.md](01-test-stabilization.md).

## Notes

- Keep this checklist updated throughout the migration
- Document any deviations or additional steps needed
- Review with team before starting actual migration