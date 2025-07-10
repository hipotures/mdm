# API Usage Report: StorageBackend

## Summary

- **Total method calls**: 62
- **Unique methods**: 14
- **Total attribute access**: 1
- **Unique attributes**: 1

## Methods by Usage

| Method | Count | Example Locations |
|--------|-------|------------------|
| `get_engine()` | 11 | src/mdm/api.py, src/mdm/api.py, src/mdm/api.py |
| `create_table_from_dataframe()` | 10 | src/mdm/dataset/registrar.py, src/mdm/dataset/registrar.py, src/mdm/dataset/registrar.py |
| `query()` | 9 | src/mdm/api.py, src/mdm/cli/dataset.py, src/mdm/cli/dataset.py |
| `read_table_to_dataframe()` | 7 | src/mdm/api.py, src/mdm/api.py, src/mdm/api.py |
| `close_connections()` | 7 | src/mdm/cli/dataset.py, src/mdm/dataset/exporter.py, src/mdm/dataset/registrar.py |
| `read_table()` | 7 | src/mdm/features/engine.py, src/mdm/services/dataset_service.py, src/mdm/services/dataset_service.py |
| `write_table()` | 3 | src/mdm/features/engine.py, src/mdm/features/engine.py, src/mdm/features/engine.py |
| `get_table_info()` | 2 | src/mdm/cli/dataset.py, src/mdm/dataset/registrar.py |
| `execute_query()` | 1 | src/mdm/api.py |
| `get_connection()` | 1 | src/mdm/api.py |
| `get_columns()` | 1 | src/mdm/api.py |
| `analyze_column()` | 1 | src/mdm/api.py |
| `database_exists()` | 1 | src/mdm/dataset/registrar.py |
| `create_database()` | 1 | src/mdm/dataset/registrar.py |

## Attributes by Usage

| Attribute | Count | Example Locations |
|-----------|-------|------------------|
| `connection` | 1 | src/mdm/api.py |