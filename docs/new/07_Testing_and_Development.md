# 7. Testing and Development

This section provides guidance for developers who want to contribute to MDM or run the test suite.

## Running Tests

MDM has a comprehensive test suite that includes unit, integration, and end-to-end (E2E) tests. The tests are managed by a shell script.

**Run all tests:**

```bash
./scripts/run_tests.sh
```

**Run specific test suites:**

```bash
# Run only unit tests
./scripts/run_tests.sh --unit-only

# Run only integration tests
./scripts/run_tests.sh --integration-only

# Run only E2E tests
./scripts/run_tests.sh --e2e-only
```

**Run individual tests:**

You can also run individual test files or specific test functions using `pytest`:

```bash
# Run a single test file
pytest tests/unit/test_config.py -v

# Run a specific test function
pytest tests/unit/test_config.py::test_function_name -v
```

## Code Quality

MDM uses several tools to maintain code quality:

*   **`ruff`** for linting.
*   **`black`** for code formatting.
*   **`mypy`** for static type checking.

**Run all quality checks:**

```bash
ruff check src/ && black src/ tests/ --line-length 100 --check && mypy src/mdm
```

**Individual checks:**

```bash
# Linting
ruff check src/

# Formatting
black src/ tests/ --line-length 100

# Type checking
mypy src/mdm
```

## Pre-commit Hooks

This project uses `pre-commit` to automatically run checks before each commit. To set it up, run:

```bash
pre-commit install
```
