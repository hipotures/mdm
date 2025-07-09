# Current MDM Repository Unit Test Failures

**Generated:** 2025-07-09 10:21:59

## Summary: 39 failures across 2 test suites

## Failures by Error Type

### Unknown error (22 failures)
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_init_creates_directories
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_init_with_custom_path
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_register_dataset_success
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_register_dataset_already_exists
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_register_dataset_case_insensitive
- ... and 17 more

### Missing attribute error (15 failures)
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_list_datasets
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_list_datasets_backend_filter
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_update_dataset
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_update_dataset_not_found
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_save_dataset
- ... and 10 more

### Attribute error (2 failures)
- **DatasetManager**: test_dataset_manager.py::TestDatasetManager::test_dataset_exists
- **StorageBackend**: test_storage_backend.py::TestDuckDBBackend::test_create_and_read_table

## Detailed Failures by Test Suite

### DatasetManager (19 failures)

#### test_init_creates_directories
- **Error Type**: Unknown error

#### test_init_with_custom_path
- **Error Type**: Unknown error

#### test_register_dataset_success
- **Error Type**: Unknown error

#### test_register_dataset_already_exists
- **Error Type**: Unknown error

#### test_register_dataset_case_insensitive
- **Error Type**: Unknown error

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

### Missing Attribute Error
- **Issue**: Tests are trying to patch attributes that don't exist
- **Fix**: Check import paths and update patches to correct locations