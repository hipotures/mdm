# MDM End-to-End Tests

This directory contains end-to-end tests based on the MANUAL_TEST_CHECKLIST.md document. Tests are organized hierarchically to match the checklist structure.

## Test Organization

Tests are organized by category matching the manual checklist:

```
test_01_config/          # 1. Configuration System
├── test_11_yaml.py     # 1.1 YAML Configuration
├── test_12_env.py      # 1.2 Environment Variables
├── test_13_backends.py # 1.3 Database Backend Configuration
├── test_14_logging.py  # 1.4 Logging Configuration
└── test_15_perf.py     # 1.5 Performance Configuration

test_02_dataset/         # 2. Dataset Operations  
├── test_21_register.py # 2.1 Dataset Registration
├── test_22_list.py     # 2.2 Dataset Listing
├── test_23_info.py     # 2.3 Dataset Information
└── ...
```

## Running Tests

### Run all tests
```bash
python tests/run_e2e_tests.py

# Or using pytest directly
pytest tests/e2e/ -v
```

### Run specific test by ID
```bash
# Run test 1.1.1
python tests/run_e2e_tests.py 1.1.1

# Using pytest markers
pytest tests/e2e/ -m "mdm_id('1.1.1')" -v
```

### Run category of tests
```bash
# Run all YAML configuration tests (1.1)
python tests/run_e2e_tests.py 1.1

# Run all configuration tests (1.*)
python tests/run_e2e_tests.py 1
```

### Generate test report
```bash
# Generate report to stdout
python tests/run_e2e_tests.py

# Save report to file
python tests/run_e2e_tests.py --output test_report.md
```

## Test Environment

All tests run in an isolated environment in `/tmp` to avoid affecting the user's MDM installation:

- Test MDM home: `/tmp/mdm_test_<uuid>/`
- Each test gets a clean environment
- No interference with `~/.mdm/`

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `clean_mdm_env`: Provides clean MDM environment in /tmp
- `mdm_config_file`: Helper to create mdm.yaml config
- `sample_csv_data`: Creates sample CSV data
- `kaggle_dataset_structure`: Creates Kaggle-style dataset
- `run_mdm`: Runs MDM commands and captures output

## Adding New Tests

1. Find the appropriate category file or create a new one
2. Add test method with appropriate marker:
   ```python
   @pytest.mark.mdm_id("1.2.3")
   def test_feature_name(self, fixtures...):
       """1.2.3: Description from checklist"""
       # Test implementation
   ```

3. Use fixtures for common operations
4. Follow naming convention: `test_<number>_<name>.py`

## Skipped Tests

Some tests are marked as skipped because:
- Feature not implemented (e.g., `--source`, `--skip-analysis`)
- Requires external dependencies (e.g., PostgreSQL)
- Known issues being tracked

## Test Report Format

The generated report includes:
- Summary statistics (total, passed, failed, skipped)
- Detailed results by category
- Failed test details with error messages
- Checklist item mapping

## Continuous Integration

These tests can be integrated into CI/CD:
```yaml
# .github/workflows/test.yml
- name: Run E2E Tests
  run: |
    python tests/run_e2e_tests.py --output e2e_report.md
    
- name: Upload Test Report
  uses: actions/upload-artifact@v3
  with:
    name: e2e-test-report
    path: e2e_report.md
```