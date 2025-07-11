# Testing and Validation

This guide covers testing strategies that should be implemented for MDM, including unit tests, integration tests, and end-to-end testing scripts.

## End-to-End Testing Scripts

MDM should include comprehensive end-to-end (e2e) testing scripts in the `scripts/` directory that test the complete lifecycle of dataset management. These scripts should be essential for verifying that all MDM features work correctly together after implementation or changes.

### Purpose of E2E Scripts

End-to-end testing scripts should serve several critical purposes:

1. **Full Workflow Testing**: Test the complete dataset lifecycle from registration through removal
2. **Integration Verification**: Ensure all components work together correctly
3. **Regression Testing**: Quickly identify if changes break existing functionality
4. **Documentation by Example**: Demonstrate proper usage of all MDM commands
5. **Performance Baseline**: Establish expected execution times for operations

### Required E2E Scripts

The `scripts/` directory should contain multiple e2e testing scripts, each optimized for different scenarios:

#### 1. test_e2e_nocolor.sh (Recommended)
```bash
./scripts/test_e2e_nocolor.sh <dataset_name> <path_to_data>
```
- **Purpose**: Complete testing without terminal colors (CI/CD friendly)
- **Features**: Full lifecycle testing with clear output
- **Use Case**: Automated testing, CI pipelines, log analysis

#### 2. test_e2e_demo.sh
```bash
./scripts/test_e2e_demo.sh <dataset_name>
```
- **Purpose**: Interactive demonstration with colored output
- **Features**: Visual feedback, progress indicators
- **Use Case**: Live demonstrations, manual testing

#### 3. test_e2e_simple.sh
```bash
./scripts/test_e2e_simple.sh <dataset_name> <path_to_data>
```
- **Purpose**: Minimal script for basic testing
- **Features**: No colors, no clear commands, simple output
- **Use Case**: Quick validation, debugging

#### 4. test_e2e_quick.sh
```bash
./scripts/test_e2e_quick.sh <dataset_name> <path_to_data>
```
- **Purpose**: Faster testing with reduced delays
- **Features**: 1-second delays between commands
- **Use Case**: Rapid development testing

#### 5. test_e2e_safe.sh
```bash
./scripts/test_e2e_safe.sh <dataset_name> <path_to_data>
```
- **Purpose**: Safe testing that doesn't remove datasets
- **Features**: Skips destructive operations
- **Use Case**: Testing on production data

### What E2E Scripts Should Test

A complete e2e script should test the following operations in sequence:

1. **System Information**
   - Display MDM configuration
   - Show current backend settings

2. **Dataset Registration**
   - List existing datasets (before)
   - Register new dataset with auto-detection
   - Verify registration success

3. **Dataset Information**
   - Display basic dataset info
   - Show detailed information with --details flag
   - List dataset statistics

4. **Search and Discovery**
   - Search datasets by partial name
   - Validate search results

5. **Export Operations**
   - Export to JSON format
   - Export to Parquet format
   - Export to CSV format
   - Create dataset package (.mdm file)

6. **Metadata Updates**
   - Update dataset description
   - Modify display name
   - Add tags

7. **Batch Operations**
   - Batch export operations

8. **Dataset Lifecycle**
    - Remove dataset (with confirmation)

### Script Implementation Pattern

All e2e scripts should follow a similar pattern:

```bash
#!/bin/bash
set -e  # Exit on error

# Configuration
DATASET_NAME=$1
DATA_PATH=$2
DELAY=3  # Seconds between commands

# Helper functions
print_section() {
    echo "==== $1 ===="
}

run_command() {
    echo "Running: $1"
    eval "$1"
    sleep $DELAY
}

# Test sequence
print_section "Dataset Registration"
run_command "mdm dataset register '$DATASET_NAME' '$DATA_PATH'"

print_section "Dataset Information"
run_command "mdm dataset info '$DATASET_NAME'"

# ... more tests ...
```

### Creating Custom E2E Scripts

It should be possible to create custom e2e scripts for specific testing needs:

```bash
#!/bin/bash
# Custom e2e script for specific workflow

set -e

# Add custom configuration
BACKEND="sqlite"  # Test specific backend

# Modify mdm.yaml for testing
cat > ~/.mdm/mdm.yaml << EOF
database:
  default_backend: $BACKEND
EOF

# Run tests with specific options
mdm dataset register test_data /path/to/data \
    --description "Test with $BACKEND backend"

# Add custom validation
if mdm dataset info test_data | grep -q "$BACKEND"; then
    echo "✓ Backend correctly set to $BACKEND"
else
    echo "✗ Backend not set correctly"
    exit 1
fi
```

### Best Practices for E2E Testing

1. **Run After Implementation**: E2E tests should always be run after completing documentation or implementation
2. **Test Multiple Backends**: Tests should be run with different database backends (DuckDB, SQLite, PostgreSQL)
3. **Use Real Data**: Testing should use actual datasets to catch edge cases
4. **Check Logs**: Test execution should include reviewing `~/.mdm/logs/mdm.log` for warnings or errors
5. **Automate Testing**: E2E scripts should be included in CI/CD pipelines
6. **Document Failures**: Test failures should be documented with issue and resolution

### Continuous Integration

Example GitHub Actions workflow using e2e scripts:

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        backend: [duckdb, sqlite]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install MDM
      run: |
        # CI uses pip for compatibility
        pip install -e .
    
    - name: Configure backend
      run: |
        mkdir -p ~/.mdm
        echo "database:" > ~/.mdm/mdm.yaml
        echo "  default_backend: ${{ matrix.backend }}" >> ~/.mdm/mdm.yaml
    
    - name: Run E2E tests
      run: |
        ./scripts/test_e2e_nocolor.sh test_ci ./data/sample
```

### Debugging E2E Test Failures

When e2e tests fail, the following steps should be taken:

1. **Check the error message**: The script uses `set -e` to stop on first error
2. **Review logs**: Check `~/.mdm/logs/mdm.log` for detailed error information
3. **Run commands manually**: Execute failing command separately for debugging
4. **Verify configuration**: Ensure `~/.mdm/mdm.yaml` has correct settings
5. **Check file permissions**: Ensure MDM has write access to required directories

### Performance Considerations

- **Delays**: Scripts should include delays to ensure operations complete
- **Batch size**: Large datasets should support custom chunk sizes
- **Backend selection**: Performance should vary based on backend choice
- **Parallel execution**: Batch operations should be parallelizable where possible

## Unit Testing

MDM should use pytest for unit testing:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_storage_backends.py

# Run with coverage
pytest --cov=mdm --cov-report=html
```

## Integration Testing

Integration tests should verify component interactions:

```bash
# Run integration tests
pytest tests/integration/

# Test specific backend
pytest tests/integration/test_sqlite_integration.py
```

## Summary

End-to-end testing scripts should be essential tools for validating MDM functionality. They should:
- Be run after every significant change to verify implementation correctness
- Be used to verify that implementation matches documentation
- Be included in automated testing pipelines for continuous validation
- Be customizable for specific testing needs and scenarios

The implementation should provide a variety of e2e scripts to allow flexibility in testing approaches while ensuring comprehensive validation of the entire dataset lifecycle. After completing MDM documentation, developers should use these scripts to verify that their implementation correctly realizes all specified functionality.