# Tests to Remove/Simplify for Single-User MDM

## Quick Reference List

### 🗑️ Tests to Remove Completely

1. **test_concurrent_operation_limits** (`test_system_resources.py:276-308`)
   - Uses ThreadPoolExecutor for concurrent registrations
   - Single-user won't do concurrent operations

2. **test_network_interruption_during_transfer** (`test_system_resources.py:225-236`)
   - Tests network failures that don't apply to local operations

3. **test_retry_on_temporary_network_failure** (`test_system_resources.py:237-260`)
   - Network retry logic unnecessary for local file access

### ⚡ Tests to Simplify

1. **test_postgresql_connection_timeout** (`test_system_resources.py:154-187`)
   - Keep only if PostgreSQL is essential
   - Otherwise remove (SQLite/DuckDB sufficient)

2. **SQL Injection Tests** (`test_security.py:16-102`)
   - Reduce from 9 attack vectors to 2-3 basic cases
   - Focus on accidental malformed input, not malicious attacks

3. **test_file_handle_limits** (`test_system_resources.py:310-321`)
   - Make optional or reduce scope
   - Less critical for single-user scenarios

### ✅ Tests to Keep

1. **Disk Space Tests**
   - test_disk_space_check_before_operation
   - test_disk_full_during_write
   - test_cleanup_after_disk_full
   - test_export_with_insufficient_space

2. **Core Functionality Tests**
   - All dataset registration (non-concurrent)
   - Data integrity verification
   - File format detection
   - Query operations
   - Export/import

3. **Error Handling Tests**
   - File not found
   - Corrupt data files
   - Invalid configurations
   - Basic validation

4. **Security Tests (Simplified)**
   - Basic input validation
   - Path traversal protection (keep simple version)
   - Dataset name validation

## Implementation Script

```python
# pytest.ini configuration
[pytest]
markers =
    single_user: Essential tests for single-user operation
    multi_user: Tests for concurrent/multi-user scenarios (skip)
    network: Tests requiring network operations (skip)
    optional: Nice-to-have tests for comprehensive coverage

# Run only single-user tests
pytest -m "single_user and not multi_user and not network"
```

## Test Counts

- **Total tests identified for removal**: 3
- **Tests to simplify**: 3
- **Approximate time saved**: 30-40% of test suite runtime
- **Complexity reduction**: Significant

## Migration Testing Focus

During refactoring migration, prioritize:
1. Core functionality preservation
2. Data integrity
3. Performance comparisons (old vs new)
4. Error handling consistency

Skip:
1. Concurrent operation testing
2. Network resilience
3. Multi-user scenarios
4. Enterprise security hardening