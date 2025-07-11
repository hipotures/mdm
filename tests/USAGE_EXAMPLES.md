# MDM Test Scripts - Usage Examples

## Quick Start: Run All Tests with GitHub Issues

The `analyze_test_failures.py` script is the main unified test runner that can:
- Run all test types (unit, integration, E2E)
- Create GitHub issues for failures
- Generate reports

### Basic Usage

```bash
# Run ALL tests and create GitHub issues (dry run - safe to test)
./tests/analyze_test_failures.py --github

# Run ALL tests and ACTUALLY create GitHub issues
./tests/analyze_test_failures.py --github --no-dry-run

# Run with custom limit (default is 30 issues per hour)
./tests/analyze_test_failures.py --github --no-dry-run --github-limit 10

# Save report to file
./tests/analyze_test_failures.py --github --output report.json
```

### Convenience Script

Use the wrapper script for the most common use case:

```bash
# Dry run (default - safe)
./tests/run_all_tests_with_github.sh

# Actually create issues
./tests/run_all_tests_with_github.sh --no-dry-run
```

## Specific Test Scopes

```bash
# Run only unit tests
./tests/analyze_test_failures.py --scope unit --github

# Run only integration tests
./tests/analyze_test_failures.py --scope integration --github

# Run only E2E tests
./tests/analyze_test_failures.py --scope e2e --github

# Run tests matching pattern
./tests/analyze_test_failures.py --category "CLI*" --github
```

## Using Specialized Runners

```bash
# Unit tests only
./tests/run_unit_tests.py --github --no-dry-run

# Integration tests only
./tests/run_integration_tests.py --github --no-dry-run

# E2E tests only
./tests/run_e2e_tests_enhanced.py --github --no-dry-run
```

## Configuration

The scripts read from `.env` file:
```bash
GITHUB_TOKEN=your_github_token
GITHUB_REPO=hipotures/mdm
GITHUB_RATE_LIMIT=30
```

You can override these:
```bash
./tests/analyze_test_failures.py \
    --github \
    --github-token YOUR_TOKEN \
    --github-repo owner/repo \
    --github-limit 20 \
    --no-dry-run
```

## Debugging

```bash
# Check MDM installation
./tests/check_mdm_installation.py

# Debug test environment
./tests/debug_test_environment.py

# Run in quiet mode (less output)
./tests/analyze_test_failures.py --github --quiet

# Run with specific test file
./tests/debug_test_environment.py tests/unit/cli/test_main.py
```

## Common Workflows

### Daily CI Run
```bash
# Run all tests, create up to 50 issues, save report
./tests/analyze_test_failures.py \
    --github \
    --no-dry-run \
    --github-limit 50 \
    --output daily-report-$(date +%Y%m%d).json
```

### Pre-commit Check
```bash
# Run unit tests only, fail on errors
./tests/run_unit_tests.py --quiet || exit 1
```

### Investigation Mode
```bash
# Dry run to see what issues would be created
./tests/analyze_test_failures.py --github --category "CLI*"

# Check specific scope
./tests/analyze_test_failures.py --scope unit --github
```

## Notes

1. **Dry Run by Default**: All scripts default to `--dry-run` mode for safety
2. **Rate Limiting**: Default 30 issues/hour to avoid GitHub API limits
3. **Deduplication**: Existing issues are updated with comments, not duplicated
4. **Issue Format**: Issues include test name, error type, reproduction steps, and suggested fixes
5. **Exit Codes**: Scripts return 0 if all tests pass, 1 if any failures