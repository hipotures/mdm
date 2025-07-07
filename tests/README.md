# MDM Testing Guide

This directory contains the test suite for MDM (ML Data Manager).

## Test Structure

```
tests/
├── conftest.py          # Pytest configuration and fixtures
├── unit/                # Unit tests for individual components
│   ├── test_models.py   # Test data models
│   ├── test_config.py   # Test configuration system
│   ├── test_serialization.py  # Test serialization utilities
│   └── test_time_series.py    # Test time series utilities
├── integration/         # Integration tests
│   ├── test_dataset_lifecycle.py  # Test complete dataset lifecycle
│   └── test_storage_backends.py   # Test storage backend implementations
└── README.md           # This file
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/unit/test_models.py
```

### Run with coverage
```bash
pytest --cov=mdm --cov-report=html
```

### Run with verbose output
```bash
pytest -v
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

## End-to-End Testing

In addition to unit and integration tests, MDM provides comprehensive end-to-end testing scripts in the `scripts/` directory:

### Available E2E Scripts

1. **test_e2e_nocolor.sh** - Complete testing without colors (CI/CD friendly)
   ```bash
   ./scripts/test_e2e_nocolor.sh test_dataset ./data/sample
   ```

2. **test_e2e_quick.sh** - Quick testing with reduced delays
   ```bash
   ./scripts/test_e2e_quick.sh test_dataset ./data/sample
   ```

3. **test_e2e_demo.sh** - Interactive demo with colored output
   ```bash
   ./scripts/test_e2e_demo.sh demo_dataset
   ```

4. **test_e2e_safe.sh** - Non-destructive testing
   ```bash
   ./scripts/test_e2e_safe.sh existing_dataset ./data/sample
   ```

5. **test_e2e_simple.sh** - Minimal testing script
   ```bash
   ./scripts/test_e2e_simple.sh test_dataset ./data/sample
   ```

### What E2E Scripts Test

- System information and configuration
- Dataset registration with auto-detection
- Dataset information and statistics
- Search and discovery operations
- Export to multiple formats (CSV, JSON, Parquet)
- Metadata updates
- Batch operations
- Dataset removal

### Running E2E Tests After Changes

Always run E2E tests after making changes to verify functionality:

```bash
# Quick smoke test
./scripts/test_e2e_quick.sh smoke_test ./data/sample

# Full test
./scripts/test_e2e_nocolor.sh full_test ./data/sample
```

## Test Data

Sample test data is provided in `data/sample/`:
- `train.csv` - Training data with features and target
- `test.csv` - Test data without target
- `sample_submission.csv` - Submission template

## Writing New Tests

### Unit Test Template
```python
import pytest
from mdm.module import YourClass

class TestYourClass:
    def test_basic_functionality(self):
        """Test basic functionality of YourClass."""
        instance = YourClass()
        result = instance.method()
        assert result == expected_value
```

### Integration Test Template
```python
import pytest
from mdm.api import MDMClient

class TestIntegration:
    def test_workflow(self, test_config, sample_data):
        """Test complete workflow."""
        client = MDMClient(config=test_config)
        # Test workflow steps
```

## Test Fixtures

Common fixtures available in `conftest.py`:
- `temp_dir` - Temporary directory for testing
- `test_config` - Test configuration with temp paths
- `dataset_manager` - Configured DatasetManager instance
- `sample_data` - Path to sample CSV data

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        pytest --cov=mdm
    - name: Run E2E tests
      run: |
        ./scripts/test_e2e_nocolor.sh ci_test ./data/sample
```

## Debugging Test Failures

1. **Check test output**: Use `-v` flag for verbose output
2. **Check logs**: Review `~/.mdm/logs/mdm.log`
3. **Run specific test**: Isolate failing test
4. **Use debugger**: Add `import pdb; pdb.set_trace()`
5. **Check fixtures**: Ensure test fixtures are set up correctly

## Performance Testing

For performance testing, use the integration tests with larger datasets:

```python
def test_large_dataset_performance(self, temp_dir):
    """Test performance with large dataset."""
    # Create large dataset
    n_rows = 100000
    df = create_large_dataframe(n_rows)
    
    # Time operations
    import time
    start = time.time()
    # ... operations ...
    elapsed = time.time() - start
    
    assert elapsed < 10.0  # Should complete within 10 seconds
```