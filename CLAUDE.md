# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MDM (ML Data Manager) is a standalone, enterprise-grade dataset management system for machine learning. It uses a decentralized architecture where each dataset is self-contained with its own database, supporting SQLite, DuckDB, and PostgreSQL backends via SQLAlchemy.

## Development Commands

### Installation and Setup
```bash
# Using uv (recommended - project uses uv for dependency management)
uv venv
source .venv/bin/activate
uv pip install -e .

# Generate missing lock file
uv lock
```

### Testing
```bash
# Run all tests (unit, integration, E2E)
./scripts/run_tests.sh

# Run specific test suites
./scripts/run_tests.sh --unit-only
./scripts/run_tests.sh --integration-only
./scripts/run_tests.sh --e2e-only

# Run with coverage
./scripts/run_tests.sh --coverage

# Run individual test file
pytest tests/unit/test_config.py -v

# Run specific test
pytest tests/unit/test_config.py::test_function_name -v

# Run E2E test scripts
./scripts/test_e2e_quick.sh test_name ./data/sample
./scripts/test_e2e_simple.sh
./scripts/test_e2e_safe.sh
```

### Code Quality
```bash
# Linting with ruff
ruff check src/

# Formatting with black
black src/ tests/ --line-length 100

# Type checking with mypy
mypy src/mdm

# Run all quality checks
ruff check src/ && black src/ tests/ --line-length 100 --check && mypy src/mdm
```

### Common MDM Commands
```bash
# Register dataset
mdm dataset register <name> <path> [--target <col>] [--problem-type <type>] [--force]

# List datasets
mdm dataset list [--limit N] [--sort-by name|registration_date]

# Dataset operations
mdm dataset info <name>
mdm dataset stats <name>
mdm dataset export <name> [--format csv|parquet|json] [--output <path>]
mdm dataset remove <name> [--force]
mdm dataset update <name> [--description "text"] [--tags "tag1,tag2"]
mdm dataset search <pattern> [--tag <tag>]
```

## Architecture and Key Components

### Core Architecture
MDM uses a two-tier database system:
1. **Dataset-Specific Databases**: Each dataset has its own SQLite/DuckDB/PostgreSQL database in `~/.mdm/datasets/{name}/`
2. **Discovery Mechanism**: YAML configuration files in `~/.mdm/config/datasets/` serve as lightweight pointers

### Single Backend Principle
- MDM uses ONE backend type for all datasets at any time
- Backend is set via `database.default_backend` in `~/.mdm/mdm.yaml`
- Changing backends makes datasets from other backends invisible
- No CLI parameter to override backend selection
- To change backend: edit config BEFORE registering datasets

### Key Modules
- `src/mdm/storage/`: Storage backend implementations (SQLite, DuckDB, PostgreSQL)
- `src/mdm/dataset/`: Dataset management logic (registrar, manager, operations)
- `src/mdm/features/`: Feature engineering system (generic and custom transformers)
- `src/mdm/cli/`: Typer-based CLI implementation with Rich formatting
- `src/mdm/config/`: Configuration management using Pydantic Settings
- `src/mdm/api/`: Programmatic API (MDMClient)

### Important Classes
- `StorageBackend` (base.py): Abstract base for all storage backends
- `DatasetRegistrar`: Handles dataset registration and auto-detection (12-step process)
- `DatasetManager`: Core dataset operations
- `FeatureGenerator`: Two-tier feature engineering system
- `MDMClient`: High-level programmatic API

### Dataset Registration Process
1. Validates dataset name
2. Checks if dataset exists
3. Validates path
4. Auto-detects structure (Kaggle format, CSV files, etc.)
5. Discovers data files
6. Detects ID columns and target
7. Creates storage backend
8. Loads data with batch processing (10k rows per chunk)
9. Detects column types using ydata-profiling
10. Generates features
11. Computes statistics
12. Saves configuration

## Configuration System

### Configuration Hierarchy
1. Defaults (in code)
2. YAML file (`~/.mdm/mdm.yaml`)
3. Environment variables (highest priority)

### Environment Variable Mapping
- Pattern: `MDM_<SECTION>_<KEY>` (nested with underscores)
- Examples:
  - `MDM_DATABASE_DEFAULT_BACKEND=duckdb`
  - `MDM_PERFORMANCE_BATCH_SIZE=10000`
  - `MDM_LOGGING_LEVEL=DEBUG`
  - `MDM_LOGGING_FILE=/tmp/mdm.log`

### Logging Configuration
- File logging: Set via `logging.file` in config or `MDM_LOGGING_FILE` env var
- Log level: DEBUG shows all operations including batch processing
- Console output: WARNING and above only (clean CLI experience)

## Recent Improvements

### Progress Bar and Output Management
- Batch loading with Rich progress bars during registration
- Suppressed ydata-profiling/tqdm progress bars globally
- Missing column warnings changed to debug level
- Clean, unified progress tracking from data loading to feature generation
- Enhanced configuration display with Rich panels after registration

### Memory Efficiency
- Data loaded in configurable chunks (default: 10,000 rows)
- Feature generation processes data in batches
- Prevents memory exhaustion on large datasets

## Known Issues and Workarounds

### Critical Bugs
1. **--time-column and --group-column**: Cause "multiple values for keyword argument" error
2. **--id-columns with multiple values**: Similar error when specifying multiple columns
3. **SQLite synchronous setting**: Always FULL instead of configured NORMAL
4. **Custom features**: Not loaded from `~/.mdm/config/custom_features/`

### Missing Features
- Many CLI options in test checklist don't exist (--source, --datetime-columns, etc.)
- Compressed file support (.csv.gz)
- Excel file support (.xlsx)
- SQLAlchemy echo configuration not working
- Automatic datetime detection (datetime columns stored as TEXT)

### Testing Artifacts
The codebase contains many test datasets from manual testing. When running tests, be aware that `~/.mdm/datasets/` may contain numerous test datasets.

## Development Guidelines

### Language Requirement
All code, documentation, and communication must be in English.

### Testing Approach
- Use MANUAL_TEST_CHECKLIST.md for comprehensive testing (617 test items)
- Document issues in ISSUES.md
- Track progress in test_progress.md

### Common Patterns
- Use SQLAlchemy ORM for all database operations
- Feature engineering follows a two-tier system (generic + custom)
- CLI uses Typer with Rich for formatting
- All models use Pydantic for validation
- Batch processing for memory efficiency
- Progress tracking with Rich Progress bars

### Error Handling
- DatasetError for dataset-specific issues
- MDMError for general errors
- Rich console for user-friendly error display

## Useful Files for Context
- `docs/03_Database_Architecture.md`: Authoritative backend selection explanation
- `docs/MANUAL_TEST_CHECKLIST.md`: 617-item comprehensive test checklist
- `ISSUES.md`: Documented bugs and limitations
- `test_progress.md`: Testing status and findings
- `pyproject.toml`: Dependencies and project metadata