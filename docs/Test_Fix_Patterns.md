# MDM Test Fix Patterns Guide

This guide documents common test failure patterns encountered in MDM and their fixes. Use this as a reference when writing or fixing tests.

## Table of Contents
1. [Performance Test Timeouts](#performance-test-timeouts)
2. [Mock Configuration Issues](#mock-configuration-issues)
3. [Type Detection Mismatches](#type-detection-mismatches)
4. [Handler Count Assertions](#handler-count-assertions)
5. [Missing Test Data](#missing-test-data)
6. [Parameter Signature Mismatches](#parameter-signature-mismatches)
7. [Backend Expectation Errors](#backend-expectation-errors)
8. [Error Handling Issues](#error-handling-issues)
9. [Test Isolation Problems](#test-isolation-problems)

## Performance Test Timeouts

### Problem
Performance tests fail on slower systems due to strict timeout limits.

### Example
```python
# BAD - Too strict for CI/slower systems
assert end - start < 5.0  # Fails on slower machines
```

### Fix
```python
# GOOD - More generous timeout
assert end - start < 10.0  # Increased to 10 seconds for safety
```

### When to Apply
- Any test measuring execution time
- Tests that might run on CI servers
- Operations involving file I/O or database access

## Mock Configuration Issues

### Problem
Mocks don't match actual method signatures or return incorrect types.

### Example
```python
# BAD - Mock returns wrong type
mock_backend.get_table_info.return_value = "wrong"

# BAD - Missing required parameters
def get_table_info(table):  # Missing engine parameter
    pass
```

### Fix
```python
# GOOD - Correct return type
mock_backend.get_table_info.return_value = {
    'columns': [...],
    'row_count': 100
}

# GOOD - Correct signature
def get_table_info(table, engine):
    pass
```

### When to Apply
- Creating mocks for storage backends
- Mocking manager or registrar methods
- Any mock that interfaces with SQLAlchemy

## Type Detection Mismatches

### Problem
Tests expect different column types than what MDM actually detects.

### Example
```python
# BAD - Assumes categorical detection
assert column_types['color'] == ColumnType.CATEGORICAL
```

### Fix
```python
# GOOD - MDM doesn't distinguish categorical from text
assert column_types['color'] == ColumnType.TEXT
```

### Current MDM Type Detection
- **Detects**: NUMERIC, TEXT, ID, TARGET
- **Does NOT detect**: CATEGORICAL, DATETIME, BOOLEAN

## Handler Count Assertions

### Problem
Log handler count varies based on configuration and log level.

### Example
```python
# BAD - Assumes fixed handler count
assert len(logger.handlers) == 3
```

### Fix
```python
# GOOD - Check handler types instead
console_handlers = [h for h in logger.handlers if isinstance(h, RichHandler)]
assert len(console_handlers) == 1

# Or check minimum handlers
assert len(logger.handlers) >= 2  # At least console and file
```

### Handler Rules
- Console handler: Always present
- File handler: Added if log file configured
- SQLAlchemy handler: Added only if echo=True AND level is DEBUG/INFO

## Missing Test Data

### Problem
Test data structures missing required fields like 'sample_data'.

### Example
```python
# BAD - Missing sample_data
column_info = {
    'name': {
        'type': 'TEXT',
        'nullable': True
    }
}
```

### Fix
```python
# GOOD - Include all required fields
column_info = {
    'name': {
        'type': 'TEXT',
        'nullable': True,
        'sample_data': ['Alice', 'Bob', 'Charlie']
    }
}
```

### Required Column Info Fields
- `type`: Data type
- `nullable`: Boolean
- `sample_data`: List of sample values
- `unique_count`: Number of unique values (optional)

## Parameter Signature Mismatches

### Problem
Test expects different parameters than method accepts.

### Example
```python
# BAD - Passing force to manager
client.manager.remove_dataset.assert_called_with('test', force=True)
```

### Fix
```python
# GOOD - Manager doesn't accept force parameter
client.manager.remove_dataset.assert_called_with('test')
```

### Common Mismatches
- `remove_dataset`: Client accepts force, manager doesn't
- `get_table_info`: Requires engine parameter
- `generate_features`: Requires tmp_path in tests

## Backend Expectation Errors

### Problem
Tests assume wrong backend type or don't account for backend initialization.

### Example
```python
# BAD - Assumes backend is set
if backend and hasattr(backend, 'close_connections'):
    backend.close_connections()  # UnboundLocalError if backend not initialized
```

### Fix
```python
# GOOD - Initialize backend first
backend = None
try:
    backend = BackendFactory.create(...)
finally:
    if backend and hasattr(backend, 'close_connections'):
        backend.close_connections()
```

## Error Handling Issues

### Problem
Error handling code has bugs or doesn't properly clean up.

### Example
```python
# BAD - backend might not be defined
try:
    backend = create_backend()
    # ... use backend
finally:
    backend.cleanup()  # UnboundLocalError if creation failed
```

### Fix
```python
# GOOD - Safe error handling
backend = None
try:
    backend = create_backend()
    # ... use backend
finally:
    if backend:
        backend.cleanup()
```

## Test Isolation Problems

### Problem
Tests affect each other through shared state or file system.

### Example
```python
# BAD - Uses real home directory
config_path = Path.home() / '.mdm' / 'config.yaml'
```

### Fix
```python
# GOOD - Use tmp_path fixture
def test_something(tmp_path):
    config_path = tmp_path / '.mdm' / 'config.yaml'
```

### Best Practices
- Always use `tmp_path` for file operations
- Mock `Path.home()` when needed
- Clean up created resources in teardown
- Don't rely on test execution order

## Common Test Patterns

### Testing DataFrame Operations
```python
# Create test DataFrame with proper types
df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['A', 'B', 'C'],
    'value': [10.5, 20.5, 30.5]
})

# Include dtypes when needed
dtypes = {
    'id': 'int64',
    'name': 'object',
    'value': 'float64'
}
```

### Testing Column Type Detection
```python
# Set up registrar state before testing
registrar._column_types = {
    'date_col': ColumnType.TEXT,  # Not DATETIME
    'status': ColumnType.TEXT,     # Not CATEGORICAL
}

# For datetime conversion tests
registrar._detected_datetime_columns = ['date_col']
```

### Testing File Compression
```python
# Don't assert exact size for small files
# Compression can increase size due to headers
assert compressed_size > 0  # Just verify it worked
```

### Testing Progress Bars
```python
# Mock or suppress progress bars in tests
with patch('mdm.utils.Progress'):
    # Test code here
    pass
```

## Debugging Tips

### 1. Check Mock Calls
```python
# See what was actually called
print(mock_object.call_args_list)
print(mock_object.method_calls)
```

### 2. Verify Data Structures
```python
# Print actual vs expected
print(f"Expected: {expected}")
print(f"Actual: {actual}")
print(f"Difference: {set(expected) - set(actual)}")
```

### 3. Test in Isolation
```bash
# Run single test with verbose output
pytest path/to/test.py::test_function -vv -s
```

### 4. Check Test Environment
```python
# Ensure clean state
assert not Path('~/.mdm').expanduser().exists()
```

## Future Considerations

### When MDM Adds Features
If MDM adds categorical or datetime detection:
1. Update type detection tests
2. Add new column type test cases
3. Update this guide

### When Adding New Tests
1. Follow existing patterns
2. Include all required fields
3. Use generous timeouts
4. Mock external dependencies
5. Test both success and error cases