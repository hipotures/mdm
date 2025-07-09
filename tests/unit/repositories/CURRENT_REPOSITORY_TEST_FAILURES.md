# Current MDM Repository Unit Test Failures

**Generated:** 2025-07-09 12:42:17

## Summary: 25 failures across 2 test suites

## Failures by Error Type

### Attribute error (11 failures)
- **StorageBackend**: test_storage_backend.py::TestSQLiteBackend::test_get_row_count
- **StorageBackend**: test_storage_backend.py::TestSQLiteBackend::test_close
- **StorageBackend**: test_storage_backend.py::TestDuckDBBackend::test_init_creates_database
- **StorageBackend**: test_storage_backend.py::TestDuckDBBackend::test_create_and_read_table
- **StorageBackend**: test_storage_backend.py::TestDuckDBBackend::test_read_table_with_limit
- ... and 6 more

### Unknown error (10 failures)
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_init_with_custom_path
- **StorageBackend**: test_storage_backend.py::TestStorageBackendBase::test_create_table
- **StorageBackend**: test_storage_backend.py::TestStorageBackendBase::test_read_table
- **StorageBackend**: test_storage_backend.py::TestSQLiteBackend::test_init_creates_database
- **StorageBackend**: test_storage_backend.py::TestSQLiteBackend::test_create_and_read_table
- ... and 5 more

### File/Directory not found (4 failures)
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_get_dataset_backend_check
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_list_datasets_backend_filter
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_save_dataset
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_yaml_serialization

## Detailed Failures by Test Suite

### DatasetManager (5 failures)

#### test_init_with_custom_path
- **Error Type**: Unknown error

#### test_get_dataset_backend_check
- **Error Type**: File/Directory not found
- **Message**: E   FileNotFoundError: [Errno 2] No such file or directory: '/custom/datasets'

#### test_list_datasets_backend_filter
- **Error Type**: File/Directory not found
- **Message**: E   FileNotFoundError: [Errno 2] No such file or directory: '/custom/datasets'

#### test_save_dataset
- **Error Type**: File/Directory not found
- **Message**: E   FileNotFoundError: [Errno 2] No such file or directory: '/custom/datasets'

#### test_yaml_serialization
- **Error Type**: File/Directory not found
- **Message**: E   FileNotFoundError: [Errno 2] No such file or directory: '/custom/datasets'

### StorageBackend (20 failures)

#### test_create_table
- **Error Type**: Unknown error

#### test_read_table
- **Error Type**: Unknown error

#### test_init_creates_database
- **Error Type**: Unknown error

#### test_create_and_read_table
- **Error Type**: Unknown error

#### test_table_exists
- **Error Type**: Unknown error

## Common Issues and Recommended Fixes