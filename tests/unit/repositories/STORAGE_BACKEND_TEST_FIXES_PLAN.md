# Storage Backend Test Fixes Plan

**Date:** 2025-07-09  
**Tests to Fix:** 20 storage backend tests

## API Mapping

### Method Name Changes
| Test Expects | Actual Method | Notes |
|--------------|---------------|--------|
| `create_table()` | `create_table_from_dataframe()` | Requires engine parameter |
| `read_table()` | `read_table_to_dataframe()` | Requires engine parameter |
| `list_tables()` | `get_table_names()` | Requires engine parameter |
| `drop_table()` | Not implemented | Need to add or remove test |
| `execute_query()` | `execute_query()` | Exists but requires engine parameter |
| `get_row_count()` | Not implemented | Need to calculate from query |
| `close()` | `close_connections()` | Different method name |

### Attribute Issues
| Test Expects | Actual State |
|--------------|--------------|
| `backend.db_path` | Not stored as attribute |
| `backend.engine` | Private `_engine` attribute |
| `backend.metadata` | Not implemented |

## Test Categories to Fix

### 1. Base Class Tests (2 tests)
- `test_create_table` - Need to use `create_table_from_dataframe` with engine
- `test_read_table` - Need to use `read_table_to_dataframe` with engine

### 2. SQLite Backend Tests (9 tests)
- `test_init_creates_database` - Expects `db_path` attribute
- `test_create_and_read_table` - Wrong method names
- `test_table_exists` - Missing engine parameter
- `test_get_table_info` - Wrong method name for create
- `test_list_tables` - Wrong method names
- `test_drop_table` - Method doesn't exist
- `test_execute_query` - Wrong method name for create
- `test_get_row_count` - Method doesn't exist
- `test_close` - Wrong method name

### 3. DuckDB Backend Tests (4 tests)
- `test_init_creates_database` - Expects `db_path` attribute
- `test_create_and_read_table` - Wrong method names
- `test_read_table_with_limit` - Wrong method names
- `test_parquet_export` - Method doesn't exist

### 4. Backend Factory Tests (5 tests)
- `test_create_sqlite_backend` - Wrong patching
- `test_create_duckdb_backend` - Wrong patching
- `test_create_postgresql_backend` - Wrong patching
- `test_create_invalid_backend` - Test expects exception
- `test_get_backend_class` - Method doesn't exist

## Fix Strategy

### Option 1: Update All Tests (Recommended)
- Modify tests to use actual API with engine parameter
- Remove tests for non-existent methods
- Add proper mocking for engine creation

### Option 2: Create Wrapper Methods
- Add convenience methods that match test expectations
- Would require modifying production code

### Option 3: Mix Approach
- Update most tests to match implementation
- Add a few critical convenience methods

## Example Fix Pattern

```python
# Before (failing):
def test_create_table(self, backend):
    df = pd.DataFrame({'id': [1, 2, 3]})
    backend.create_table("test_table", df)

# After (fixed):
def test_create_table(self, backend):
    df = pd.DataFrame({'id': [1, 2, 3]})
    database_path = backend.get_database_path("test_dataset", Path("/tmp"))
    engine = backend.get_engine(database_path)
    backend.create_table_from_dataframe(df, "test_table", engine)
```