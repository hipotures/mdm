# Testing Guide

## Overview

MDM has a comprehensive automated test suite with over 1100 tests covering 95.4% of the codebase.

## Running Tests

### All Tests
```bash
./scripts/run_tests.sh
```

### Test Categories
```bash
# Unit tests only
./scripts/run_tests.sh --unit-only

# Integration tests only  
./scripts/run_tests.sh --integration-only

# End-to-end tests only
./scripts/run_tests.sh --e2e-only

# With coverage report
./scripts/run_tests.sh --coverage
```

### Specific Tests
```bash
# Run single test file
pytest tests/unit/test_config.py -v

# Run specific test
pytest tests/unit/test_config.py::test_function_name -v

# Run tests matching pattern
pytest -k "test_dataset_register" -v
```

## Test Structure

```
tests/
├── unit/              # Fast, isolated component tests
│   ├── cli/          # CLI command tests
│   ├── services/     # Service layer tests
│   └── storage/      # Storage backend tests
├── integration/       # Tests with real databases
└── e2e/              # Full workflow tests
```

## Writing Tests

### Unit Test Example
```python
def test_dataset_registration():
    """Test dataset registration logic."""
    mock_storage = Mock(spec=StorageBackend)
    registrar = DatasetRegistrar(storage=mock_storage)
    
    result = registrar.register("test", "path/to/data.csv")
    
    assert result.success
    mock_storage.create_tables.assert_called_once()
```

### Integration Test Example
```python
@pytest.mark.integration
def test_sqlite_backend():
    """Test SQLite backend with real database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = SQLiteBackend(tmpdir)
        backend.initialize("test_dataset")
        
        # Test real operations
        backend.insert_data(test_data)
        result = backend.query("SELECT COUNT(*) FROM train")
        assert result[0][0] == len(test_data)
```

## Common Test Patterns

See [Test Fix Patterns](../../tests/docs/Test_Fix_Patterns.md) for solutions to common test failures.

## Continuous Integration

Tests run automatically on:
- Every pull request
- Push to main branch
- Nightly builds

## Performance Testing

For performance-critical code:
```python
@pytest.mark.benchmark
def test_batch_processing_performance(benchmark):
    """Benchmark batch processing."""
    data = generate_large_dataset(rows=100000)
    result = benchmark(process_batch, data)
    assert result.total_time < 1.0  # Should complete in under 1 second
```

## Test Data

Test fixtures are provided in `tests/fixtures/`:
- `sample_data.csv` - Small dataset for quick tests
- `kaggle_structure/` - Kaggle competition format
- `edge_cases/` - Files with special characters, encodings

## Debugging Tests

```bash
# Run with verbose output
pytest -vv tests/unit/test_failing.py

# Drop into debugger on failure
pytest --pdb tests/unit/test_failing.py

# Show local variables on failure
pytest -l tests/unit/test_failing.py

# Run last failed tests
pytest --lf
```

## Test Coverage

Current coverage: **95.4%**

To generate coverage report:
```bash
./scripts/run_tests.sh --coverage
# Open htmlcov/index.html in browser
```

Target coverage for new code: **90%+**