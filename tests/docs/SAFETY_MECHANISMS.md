# E2E Test Safety Mechanisms

## Overview

The E2E test suite includes multiple safety mechanisms to prevent accidental modification or destruction of user data in `~/.mdm`.

## Safety Features

### 1. Automatic Safe Directory Assignment

If `MDM_HOME_DIR` is not set or points to an unsafe location, tests automatically use a temporary directory:

```python
# In conftest.py - clean_mdm_env fixture
if not current_mdm_home or current_mdm_home == home_mdm or not current_mdm_home.startswith("/tmp"):
    safety_dir = Path(f"/tmp/mdm_test_safety_{os.getpid()}_{uuid.uuid4().hex[:4]}")
    os.environ["MDM_HOME_DIR"] = str(safety_dir)
```

### 2. Safety Warnings

Tests display clear warnings when safety mechanisms activate:

```
⚠️  SAFETY: MDM_HOME_DIR not set, using temporary: /tmp/mdm_test_safety_134250_40e8
   This prevents tests from accidentally modifying production data.
```

### 3. Production Directory Protection

The following scenarios trigger safety mechanisms:
- `MDM_HOME_DIR` not set
- `MDM_HOME_DIR` set to `~/.mdm` 
- `MDM_HOME_DIR` set to any path outside `/tmp`

### 4. Unique Test Environments

Each test gets a completely isolated environment:
```
/tmp/mdm_test_<uuid>/     # Unique for each test
├── datasets/
├── config/
│   └── datasets/
├── logs/
└── cache/
```

### 5. Automatic Cleanup

Test directories are automatically removed after each test completes.

## Running Tests Safely

### Safe (Recommended)
```bash
# Let tests use automatic safety
pytest tests/e2e/

# Or explicitly set safe directory
export MDM_HOME_DIR=/tmp/mdm_tests
pytest tests/e2e/
```

### Unsafe (Will be automatically redirected)
```bash
# These will trigger safety mechanisms:
unset MDM_HOME_DIR
pytest tests/e2e/  # ✓ Safe - redirected to /tmp

export MDM_HOME_DIR=~/.mdm  
pytest tests/e2e/  # ✓ Safe - redirected to /tmp

export MDM_HOME_DIR=/home/user/production
pytest tests/e2e/  # ✓ Safe - redirected to /tmp
```

## Implementation Details

### ConfigManager Fix
The `ConfigManager` now respects `MDM_HOME_DIR`:
```python
self.base_path = Path(os.environ.get("MDM_HOME_DIR", str(Path.home() / ".mdm")))
```

### Test Isolation
- Each test calls `reset_config()` and `reset_config_manager()`
- Environment variables are saved and restored
- Temporary directories are created and destroyed per test

## Verification

Run the isolation test to verify safety mechanisms:
```bash
python -m pytest tests/e2e/test_isolation.py -v
```

This will verify:
1. Tests use isolated environments
2. Parallel tests don't interfere
3. No access to production `~/.mdm`