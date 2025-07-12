# Comprehensive Test Report

**Generated:** 2025-07-12 14:04:38 CEST
**Last Updated:** 2025-07-12 14:25:00 CEST

## Summary
- Total tests: 1163 (1026 unit + 38 integration + 99 E2E)
- Passed: 1110 (987 unit + 37 integration + 86 E2E)
- Failed: 23 (all in unit tests)
- Skipped: 30 (16 unit + 1 integration + 13 E2E)
- Coverage: 28%

## Test Results by Category

### Unit Tests
- Total: 1026
- Passed: 987
- Failed: 23
- Skipped: 16
- Execution time: 74.95s

### Integration Tests
- Total: 38
- Passed: 37
- Failed: 0
- Skipped: 1 (test_update_persistence_yaml - YAML file not found)
- Execution time: 23.86s

### E2E Tests
- Total: 99
- Configuration tests (45): 42 passed, 3 skipped
- Dataset tests (52): 42 passed, 10 skipped
- Isolation tests (2): 2 passed
- Execution time: 415.63s total (145.59s config + 262.32s dataset + 7.72s isolation)
- Note: E2E tests are slow due to full dataset operations

## Failed Tests Details

### CLI Tests (11 failures)
1. **test_dataset_update_comprehensive.py** (10 failures)
   - `TestDatasetUpdateBasic::test_update_tags_multiple` - Mock call signature mismatch
   - `TestDatasetUpdateBasic::test_update_problem_type` - Mock call signature mismatch
   - `TestDatasetUpdateBasic::test_update_description` - Mock call signature mismatch
   - `TestDatasetUpdateBasic::test_update_all_fields` - Mock call signature mismatch
   - `TestDatasetUpdateEdgeCases::test_update_with_empty_values` - Mock call signature mismatch
   - `TestDatasetUpdateEdgeCases::test_update_with_whitespace_values` - Mock call signature mismatch
   - `TestDatasetUpdateEdgeCases::test_update_dataset_name_with_special_chars` - Mock call signature mismatch
   - Problem: UpdateManager.execute() being called with keyword arguments instead of dictionary

2. **test_cli_90_coverage.py** (1 failure)
   - `TestDatasetCoverage::test_update_dataset_both_options` - Similar mock signature issue

### Feature Registry Tests (3 failures)
1. **test_feature_registry.py**
   - `TestFeatureRegistry::test_get_all_transformers` - Expected 4 transformers, got 6
   - `TestFeatureRegistry::test_get_all_transformers_with_custom` - Expected 5 transformers, got 7
   - `TestFeatureRegistry::test_empty_registry` - Expected empty list, got 2 GlobalFeatureAdapter objects
   - Problem: GlobalFeatureAdapter objects being included unexpectedly

### Dataset Service Tests (1 failure)
1. **test_dataset_exporter.py**
   - `TestDatasetExporter::test_load_dataset_info_backend_mismatch` - Expected DatasetError not raised
   - Problem: Backend mismatch validation not working as expected

### Feature Generator Tests (6 failures)
1. **test_feature_generator.py**
   - All tests failing with "TypeError: 'Mock' object is not iterable"
   - Problem: Mock setup for feature registry returning wrong type

### CLI Basic Tests (2 failures)
1. **test_cli_basic.py**
   - `TestMainCommands::test_version_command` - Version assertion mismatch
   - `TestMainCommands::test_help_command` - Version assertion mismatch
   - Problem: Tests expect version 0.2.0, actual version is 0.3.1

## Key Issues Identified

### 1. Mock Signature Mismatches (48% of failures)
- UpdateManager.execute() expects keyword arguments but tests pass dictionary
- Feature registry mocks not properly configured

### 2. Version Mismatch (9% of failures)
- Tests hardcoded with version 0.2.0
- Actual version is 0.3.1

### 3. Feature Registry Issues (13% of failures)
- GlobalFeatureAdapter objects being created unexpectedly
- Registry not properly isolated in tests

### 4. Backend Validation (4% of failures)
- Backend mismatch validation not raising expected errors

### 5. Mock Configuration (26% of failures)
- Feature generator tests have improperly configured mocks

## Coverage Report

### Overall Coverage: 28%

### Key Areas with Low Coverage:
- `mdm.api.client`: 0%
- `mdm.cli.commands.monitoring`: 0%
- `mdm.cli.commands.system`: 47%
- `mdm.cli.commands.timeseries`: 0%
- `mdm.monitoring`: 0-11%
- `mdm.services.dataset_service`: 0%
- `mdm.storage.postgresql`: 16%
- `mdm.utils.integration`: 10%
- `mdm.utils.performance`: 12%

### Well-Tested Areas:
- `mdm.config`: 89%
- `mdm.core.exceptions`: 100%
- `mdm.dataset.discovery`: 89%
- `mdm.dataset.manager`: 91%
- `mdm.dataset.registrar`: 83%
- `mdm.features.generic`: 70-93%
- `mdm.storage.factory`: 80%

## E2E Tests - Skipped Features

The following features are not implemented or have known issues:
1. `--time-column` and `--group-column` - cause "multiple values for keyword argument" error
2. `--skip-analysis` - option not implemented
3. `--tag` filtering - option not implemented
4. `--has-target` filter - not implemented
5. Output formats (`--format json/csv`) - not implemented for list/info commands
6. Verbose flag (`-v`) - not implemented for dataset list

## Recommendations

1. **Fix Mock Signatures**: Update all UpdateManager mock calls to use keyword arguments
2. **Update Version Assertions**: Change hardcoded version checks from 0.2.0 to 0.3.1 or make dynamic
3. **Isolate Feature Registry**: Ensure feature registry is properly reset between tests
4. **Fix Feature Generator Mocks**: Configure mocks to return iterable objects
5. **Add Missing Tests**: Focus on areas with 0% coverage, especially:
   - API client
   - Monitoring commands
   - Time series functionality
   - Dataset service
   - PostgreSQL backend

## Next Steps

1. Fix the 23 failing unit tests (mostly mock configuration issues)
2. Complete E2E test execution with proper timeouts
3. Increase coverage for critical components with 0% coverage
4. Add integration tests for PostgreSQL backend
5. Implement missing time series and monitoring tests