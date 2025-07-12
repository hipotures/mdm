# Current MDM E2E Test Failures

**Generated:** 2025-07-11 16:06:36

## Summary: 6 failures across 3 categories

## Failures by Error Type

### Command failed (6 failures)
- **YAML Configuration**: test_11_yaml.py::TestYAMLConfiguration::test_modify_yaml_changes_take_effect
- **Environment Variables**: test_12_env.py::TestEnvironmentVariables::test_mdm_database_backend_duckdb
- **Environment Variables**: test_12_env.py::TestEnvironmentVariables::test_env_overrides_yaml
- **Database Backends**: test_13_backends.py::TestDatabaseBackendConfiguration::test_change_backend_to_duckdb
- **Database Backends**: test_13_backends.py::TestDatabaseBackendConfiguration::test_backend_env_override
- ... and 1 more

## Detailed Failures by Category

### YAML Configuration (1 failures)

#### test_modify_yaml_changes_take_effect
- **Error Type**: Command failed
- **Message**: E   subprocess.CalledProcessError: Command '['/home/xai/DEV/mdm-refactor-20250711/.venv/bin/python', '-m', 'mdm.cli.main', 'dataset', 'register', 'test_duckdb', '/tmp/mdm_test_a480b54f/test_data/sample_data.csv', '--target', 'value']' returned non-zero exit status 1.

### Environment Variables (2 failures)

#### test_mdm_database_backend_duckdb
- **Error Type**: Command failed
- **Message**: E   subprocess.CalledProcessError: Command '['/home/xai/DEV/mdm-refactor-20250711/.venv/bin/python', '-m', 'mdm.cli.main', 'dataset', 'register', 'test_duckdb_env', '/tmp/mdm_test_2d8b678b/test_data/sample_data.csv', '--target', 'value']' returned non-zero exit status 1.

#### test_env_overrides_yaml
- **Error Type**: Command failed
- **Message**: E   subprocess.CalledProcessError: Command '['/home/xai/DEV/mdm-refactor-20250711/.venv/bin/python', '-m', 'mdm.cli.main', 'dataset', 'register', 'test_duckdb_env', '/tmp/mdm_test_2d8b678b/test_data/sample_data.csv', '--target', 'value']' returned non-zero exit status 1.

### Database Backends (3 failures)

#### test_change_backend_to_duckdb
- **Error Type**: Command failed
- **Message**: E   subprocess.CalledProcessError: Command '['/home/xai/DEV/mdm-refactor-20250711/.venv/bin/python', '-m', 'mdm.cli.main', 'dataset', 'register', 'test_duckdb', '/tmp/mdm_test_620ce7b1/test_data/sample_data.csv', '--target', 'value']' returned non-zero exit status 1.

#### test_backend_env_override
- **Error Type**: Command failed
- **Message**: E   subprocess.CalledProcessError: Command '['/home/xai/DEV/mdm-refactor-20250711/.venv/bin/python', '-m', 'mdm.cli.main', 'dataset', 'register', 'test_duckdb', '/tmp/mdm_test_620ce7b1/test_data/sample_data.csv', '--target', 'value']' returned non-zero exit status 1.

#### test_backend_isolation
- **Error Type**: Command failed
- **Message**: E   subprocess.CalledProcessError: Command '['/home/xai/DEV/mdm-refactor-20250711/.venv/bin/python', '-m', 'mdm.cli.main', 'dataset', 'register', 'test_duckdb', '/tmp/mdm_test_620ce7b1/test_data/sample_data.csv', '--target', 'value']' returned non-zero exit status 1.

## Common Issues and Recommended Fixes

### Command Failed
- **Issue**: MDM commands returning non-zero exit codes
- **Fix**: Check for missing parameters or invalid operations