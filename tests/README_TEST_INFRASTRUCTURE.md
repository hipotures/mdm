# MDM Test Infrastructure

This document describes the refactored test infrastructure for MDM, which provides unified test runners with GitHub integration.

## Overview

The test infrastructure has been consolidated to provide:
- Shared GitHub integration for all test types
- Consistent CLI parameters across all runners
- Unified error analysis and categorization
- Rate limiting for GitHub API calls
- Configuration from `.env` file

## Test Runners

### 1. Unified Test Analyzer
**Script:** `analyze_test_failures.py`

The main test analyzer that can run all test types and create GitHub issues.

```bash
# Analyze all tests
./analyze_test_failures.py

# Analyze only unit tests
./analyze_test_failures.py --scope unit

# Analyze E2E tests and create GitHub issues (dry run)
./analyze_test_failures.py --scope e2e --github

# Create up to 5 real GitHub issues
./analyze_test_failures.py --github --no-dry-run --limit 5

# Analyze specific category
./analyze_test_failures.py --category "CLI*"

# Save report to file
./analyze_test_failures.py --output report.json
```

### 2. Specialized Runners

#### Unit Tests
**Script:** `run_unit_tests.py`

```bash
# Run all unit tests
./run_unit_tests.py

# Run tests and create GitHub issues
./run_unit_tests.py --github --no-dry-run
```

#### Integration Tests
**Script:** `run_integration_tests.py`

```bash
# Run all integration tests
./run_integration_tests.py

# Run tests with GitHub integration
./run_integration_tests.py --github
```

#### E2E Tests
**Script:** `run_e2e_tests_enhanced.py`

Enhanced E2E runner with GitHub integration and backward compatibility:

```bash
# Run all E2E tests
./run_e2e_tests_enhanced.py

# Run specific test category (legacy mode)
./run_e2e_tests_enhanced.py 1.1

# Run specific test (legacy mode)
./run_e2e_tests_enhanced.py 1.1.1

# Run with GitHub integration
./run_e2e_tests_enhanced.py --github
```

The original `run_e2e_tests.py` still works as before for backward compatibility.

### 3. Utility Scripts

#### Check MDM Installation
**Script:** `check_mdm_installation.py`

Verifies MDM is properly installed and configured:

```bash
./check_mdm_installation.py
```

Checks:
- MDM command availability
- Python import
- Environment variables
- Directory structure
- Basic command execution

#### Debug Test Environment
**Script:** `debug_test_environment.py`

Helps debug test isolation and environment issues:

```bash
# General environment debugging
./debug_test_environment.py

# Debug specific test file
./debug_test_environment.py tests/unit/test_example.py
```

## Shared Modules

### GitHub Integration (`utils/github_integration.py`)

Provides:
- `GitHubConfig`: Configuration from environment
- `GitHubIssueManager`: Issue creation and updates
- `RateLimiter`: API rate limiting (default: 30 issues/hour)
- Issue deduplication using MD5 hashes

### Test Runner Base (`utils/test_runner.py`)

Provides:
- `BaseTestRunner`: Abstract base class for test runners
- `TestResult`: Individual test result data
- `TestSuite`: Collection of test results
- JSON report generation
- Rich console output support

### Error Analyzer (`utils/error_analyzer.py`)

Provides:
- `ErrorAnalyzer`: Categorizes test failures
- `ErrorPattern`: Pattern matching for error types
- Suggested fixes for common error types
- Failure grouping for issue creation

## Configuration

### Environment Variables (in `.env`)

```bash
# GitHub Integration
GITHUB_TOKEN=your_github_token
GITHUB_REPO=hipotures/mdm
GITHUB_RATE_LIMIT=30  # Max issues per hour

# MDM Configuration
MDM_HOME_DIR=/path/to/mdm/home
MDM_DATABASE_DEFAULT_BACKEND=sqlite
```

### CLI Parameters (Standardized)

All test runners support:
```
--github              # Enable GitHub issue creation
--github-token TOKEN  # Override GITHUB_TOKEN from env
--github-repo REPO    # Override GITHUB_REPO from env  
--github-limit N      # Max issues per run (default: 30)
--dry-run            # Preview without creating (default)
--no-dry-run         # Actually create issues
--output FILE        # Save report to file
--quiet              # Minimal output
```

## GitHub Issue Creation

### Issue Format

Issues are created with:
- **Title:** `[Test Failure] {test_name} - {error_type} [{issue_id}]`
- **Labels:** `test-failure`, `automated`, `category-{category}`, `error-{type}`
- **Body:** Error details, reproduction steps, suggested fixes
- **ID:** 8-character MD5 hash for deduplication

### Rate Limiting

- Default: 30 issues per hour (configurable)
- Tracks API calls with timestamps
- Provides wait time when limit reached
- Continues with remaining tests after limit

### Deduplication

- Each test failure gets a unique ID based on test name, error type, and category
- Existing open issues are updated with comments instead of creating duplicates
- Issue ID is included in title and body for tracking

## Migration from Old Scripts

### Archived Scripts

The following scripts have been archived to `tests/archive/`:
- `analyze_cli_failures_github.py` → Use `analyze_test_failures.py --scope unit --category "CLI*"`
- `e2e/analyze_current_failures_github.py` → Use `analyze_test_failures.py --scope e2e`
- `unit/analyze_unit_test_failures_github.py` → Use `analyze_test_failures.py --scope unit`
- `e2e/analyze_tests.py` → Use `analyze_test_failures.py`
- `e2e/simple_test_runner.py` → Use `run_e2e_tests_enhanced.py`
- `e2e/quick_test_check.py` → Use `check_mdm_installation.py`
- `e2e/debug_isolation.py` → Use `debug_test_environment.py`

### Benefits of New Infrastructure

1. **Unified Interface**: Same CLI parameters for all test types
2. **Better Error Analysis**: Consistent error categorization and suggested fixes
3. **GitHub Integration**: Built-in support for issue creation with rate limiting
4. **Configuration Management**: Settings from `.env` file instead of hardcoded values
5. **Extensibility**: Easy to add new test categories or runners
6. **Performance**: Batch processing and progress tracking
7. **Debugging**: Better tools for environment and isolation issues

## Examples

### Daily Test Run with GitHub Reporting
```bash
# Run all tests and create issues for failures
./analyze_test_failures.py --github --no-dry-run --limit 50
```

### CI/CD Integration
```bash
# Run unit tests, fail on errors, save report
./run_unit_tests.py --output unit-report.json --quiet || exit 1

# Run integration tests
./run_integration_tests.py --output integration-report.json --quiet || exit 1

# Create GitHub issues if failures (dry run in CI)
./analyze_test_failures.py --github --output full-report.json
```

### Debugging Test Failures
```bash
# Check environment first
./check_mdm_installation.py

# Debug specific test
./debug_test_environment.py tests/unit/cli/test_main.py

# Analyze failures with verbose output
./analyze_test_failures.py --scope unit --category "CLI*" -v
```

## Future Enhancements

1. **Parallel Test Execution**: Run test categories in parallel
2. **Test Result Caching**: Skip unchanged tests
3. **GitHub Actions Integration**: Direct integration with GHA workflows
4. **Metrics Dashboard**: Test pass rates over time
5. **Auto-fix Suggestions**: AI-powered fix suggestions
6. **Test Flakiness Detection**: Identify and track flaky tests