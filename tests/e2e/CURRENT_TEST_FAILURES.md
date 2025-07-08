# Current MDM E2E Test Failures

**Generated:** 2025-07-08 12:49:11

## Summary: 45 failures across 6 categories

## Failures by Error Type

### Type error (16 failures)
- **Logging Configuration**: test_14_logging.py::TestLoggingConfiguration::test_change_log_level_yaml
- **Logging Configuration**: test_14_logging.py::TestLoggingConfiguration::test_log_level_env_override
- **Logging Configuration**: test_14_logging.py::TestLoggingConfiguration::test_file_logging_yaml
- **Logging Configuration**: test_14_logging.py::TestLoggingConfiguration::test_log_file_rotation
- **Logging Configuration**: test_14_logging.py::TestLoggingConfiguration::test_log_format_default
- ... and 11 more

### Assertion failed (16 failures)
- **Dataset Listing**: test_22_list.py::TestDatasetListingFiltering::test_list_sort_by_date
- **Dataset Listing**: test_22_list.py::TestDatasetListingFiltering::test_list_sort_by_size
- **Dataset Listing**: test_22_list.py::TestDatasetListingFiltering::test_filter_by_tag_single
- **Dataset Listing**: test_22_list.py::TestDatasetListingFiltering::test_filter_by_multiple_tags
- **Dataset Listing**: test_22_list.py::TestDatasetListingFiltering::test_filter_by_problem_type
- ... and 11 more

### File/Directory not found (7 failures)
- **Database Backends**: test_13_backends.py::TestDatabaseBackendConfiguration::test_sqlite_default_backend
- **Database Backends**: test_13_backends.py::TestDatabaseBackendConfiguration::test_change_backend_to_duckdb
- **Database Backends**: test_13_backends.py::TestDatabaseBackendConfiguration::test_backend_env_override
- **Database Backends**: test_13_backends.py::TestDatabaseBackendConfiguration::test_invalid_backend_error
- **Database Backends**: test_13_backends.py::TestDatabaseBackendConfiguration::test_backend_isolation
- ... and 2 more

### Unknown (6 failures)
- **Dataset Registration**: test_21_register.py::TestDatasetRegistration::test_force_flag_overwrites
- **Dataset Listing**: test_22_list.py::TestDatasetListingFiltering::test_list_all_datasets_default
- **Dataset Listing**: test_22_list.py::TestDatasetListingFiltering::test_list_with_limit
- **Dataset Listing**: test_22_list.py::TestDatasetListingFiltering::test_list_sort_by_name
- **Dataset Info/Stats**: test_23_info.py::TestDatasetInformationStatistics::test_dataset_info_basic
- ... and 1 more

## Detailed Failures by Category

### Database Backends (7 failures)

#### test_sqlite_default_backend
- **Error Type**: File/Directory not found
- **Message**: assert db_file.exists()

#### test_change_backend_to_duckdb
- **Error Type**: File/Directory not found
- **Message**: assert db_file.exists()

#### test_backend_env_override
- **Error Type**: File/Directory not found
- **Message**: assert db_file.exists()

#### test_invalid_backend_error
- **Error Type**: File/Directory not found
- **Message**: assert db_file.exists()

#### test_backend_isolation
- **Error Type**: File/Directory not found
- **Message**: assert db_file.exists()

#### test_sqlite_synchronous_setting
- **Error Type**: File/Directory not found
- **Message**: assert db_file.exists()

#### test_sqlalchemy_echo_setting
- **Error Type**: File/Directory not found
- **Message**: assert db_file.exists()

### Logging Configuration (7 failures)

#### test_change_log_level_yaml
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'logging'

#### test_log_level_env_override
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'logging'

#### test_file_logging_yaml
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'logging'

#### test_log_file_rotation
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'logging'

#### test_log_format_default
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'logging'

#### test_debug_shows_module_info
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'logging'

#### test_suppress_external_library_logs
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'logging'

### Performance Configuration (9 failures)

#### test_custom_batch_size_yaml
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'performance'

#### test_batch_size_env_override
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'performance'

#### test_parallel_processing_enabled
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'performance'

#### test_n_jobs_setting
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'performance'

#### test_memory_limit_setting
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'performance'

#### test_cache_size_setting
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'performance'

#### test_disable_progress_bars
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'performance'

#### test_optimize_dtypes_enabled
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'performance'

#### test_compression_settings
- **Error Type**: Type error
- **Message**: E   TypeError: mdm_config_file.<locals>._create_config() got an unexpected keyword argument 'performance'

### Dataset Registration (1 failures)

#### test_force_flag_overwrites
- **Error Type**: Unknown
- **Message**: assert "already exists" in result.stderr

### Dataset Listing (11 failures)

#### test_list_all_datasets_default
- **Error Type**: Unknown

#### test_list_with_limit
- **Error Type**: Unknown

#### test_list_sort_by_name
- **Error Type**: Unknown
- **Message**: assert dataset in result.stdout

#### test_list_sort_by_date
- **Error Type**: Assertion failed
- **Message**: assert dataset in result.stdout

#### test_list_sort_by_size
- **Error Type**: Assertion failed
- **Message**: assert dataset in result.stdout

#### test_filter_by_tag_single
- **Error Type**: Assertion failed
- **Message**: assert dataset in result.stdout

#### test_filter_by_multiple_tags
- **Error Type**: Assertion failed
- **Message**: assert dataset in result.stdout

#### test_filter_by_problem_type
- **Error Type**: Assertion failed
- **Message**: assert dataset in result.stdout

#### test_verbose_output
- **Error Type**: Assertion failed
- **Message**: assert dataset in result.stdout

#### test_no_matches_filter
- **Error Type**: Assertion failed
- **Message**: assert dataset in result.stdout

#### test_list_performance
- **Error Type**: Assertion failed
- **Message**: assert dataset in result.stdout

### Dataset Info/Stats (10 failures)

#### test_dataset_info_basic
- **Error Type**: Unknown

#### test_dataset_info_schema
- **Error Type**: Unknown
- **Message**: assert "Name" in result.stdout

#### test_dataset_info_size
- **Error Type**: Assertion failed
- **Message**: assert "Name" in result.stdout

#### test_dataset_stats_basic
- **Error Type**: Assertion failed
- **Message**: assert "Name" in result.stdout

#### test_dataset_stats_percentiles
- **Error Type**: Assertion failed
- **Message**: assert "Name" in result.stdout

#### test_dataset_stats_categorical
- **Error Type**: Assertion failed
- **Message**: assert "Name" in result.stdout

#### test_dataset_stats_target_distribution
- **Error Type**: Assertion failed
- **Message**: assert "Name" in result.stdout

#### test_verbose_info
- **Error Type**: Assertion failed
- **Message**: assert "Name" in result.stdout

#### test_info_large_dataset
- **Error Type**: Assertion failed
- **Message**: assert "Name" in result.stdout

#### test_stats_computation_caching
- **Error Type**: Assertion failed
- **Message**: assert "Name" in result.stdout

## Common Issues and Recommended Fixes

### File/Directory Not Found
- **Issue**: Tests expect files/directories in wrong locations
- **Fix**: Update file paths or directory structure expectations