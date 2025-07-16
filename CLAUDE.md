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

# Generate lock file if missing
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

# Check test import paths (pre-commit hook)
./scripts/check_test_imports.py
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
mdm dataset register <name> <path> [--target <col>] [--problem-type <type>] [--force] [--no-features]

# List datasets
mdm dataset list [--limit N] [--sort-by name|registration_date|size]

# Dataset operations
mdm dataset info <name>
mdm dataset stats <name>
mdm dataset export <name> [--format csv|parquet|json] [--output-dir <path>] [--compression none|gzip|zip]
mdm dataset remove <name> [--force]
mdm dataset update <name> [--description "text"] [--tags "tag1,tag2"] [--problem-type type]
mdm dataset search <pattern> [--tag <tag>]

# Show system info
mdm info
mdm version
```

## Architecture and Key Components

### Core Architecture
MDM uses a two-tier database system:
1. **Dataset-Specific Databases**: Each dataset has its own SQLite/DuckDB/PostgreSQL database in `~/.mdm/datasets/{name}/`
2. **Discovery Mechanism**: YAML configuration files in `~/.mdm/config/datasets/` serve as lightweight pointers

### Database File Naming
- **Primary database file**: `{dataset_name}.sqlite` (contains all data and features)
- **Secondary file**: `dataset.db` (may be empty or a symlink)
- Configuration points to the primary file via `database.path`

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
- `src/mdm/monitoring/`: Simple monitoring and dashboard capabilities
- `src/mdm/core/`: Core implementations including logging and exceptions

### Important Classes
- `StorageBackend` (base.py): Abstract base for all storage backends
- `DatasetRegistrar`: Handles dataset registration and auto-detection (12-step process)
- `DatasetManager`: Core dataset operations
- `FeatureGenerator`: Two-tier feature engineering system
- `MDMClient`: High-level programmatic API
- `SimpleMonitor`: Basic monitoring and metrics collection

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
10. Generates features (can be skipped with --no-features)
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
  - `MDM_PATHS_DATASETS_PATH=custom/path`

### Logging Configuration
- File logging: Set via `logging.file` in config or `MDM_LOGGING_FILE` env var
- Log level: DEBUG shows all operations including batch processing
- Console output: WARNING and above only (clean CLI experience)
- Interceptor pattern used to unify standard logging with loguru
- Log files located in `~/.mdm/logs/`

## Recent Improvements (2025-07)

### Encoding Support (v1.0.6)
- Automatic encoding detection using chardet for non-UTF8 files
- Support for Latin-1, Windows-1252, and other common encodings
- Intelligent file format detection (CSV, Parquet, Excel)
- Proper error handling for malformed CSV files

### Performance Optimizations
- CLI startup time reduced from 6.5s to 0.1s using lazy loading
- Special fast path for `mdm version` command
- Lazy imports with `__getattr__` in modules
- Batch loading with Rich progress bars during registration
- Memory-efficient chunk processing (10k rows default)

### Code Cleanup (2025-07-11)
- Removed all legacy migration code and adapters
- Consolidated configuration from `config_new.py` to `config.py`
- Eliminated feature flag system complexity
- Cleaned up ~37,745 lines of legacy code

### Progress Bar and Output Management
- Batch loading with Rich progress bars during registration
- Suppressed ydata-profiling/tqdm progress bars globally
- Missing column warnings changed to debug level
- Clean, unified progress tracking from data loading to feature generation
- Enhanced configuration display with Rich panels after registration

## Known Issues and Workarounds

### Critical Bugs
1. **--time-column and --group-column**: Cause "multiple values for keyword argument" error
2. **--id-columns with multiple values**: Similar error when specifying multiple columns
3. **SQLite synchronous setting**: Always FULL instead of configured NORMAL
4. **Custom features**: Not loaded from `~/.mdm/config/custom_features/`

### File Discovery Priority
- CSV files are now prioritized over Parquet files during discovery
- If both exist, CSV will be used unless Parquet is explicitly specified

### Missing Features
- Many CLI options in test checklist don't exist (--source, --datetime-columns, etc.)
- Compressed file support (.csv.gz) partially implemented
- Excel file support (.xlsx) exists but has issues
- SQLAlchemy echo configuration not working
- Automatic datetime detection (datetime columns stored as TEXT)

## Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Component tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
└── utils/         # Test utilities
```

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
- StorageError for backend-specific issues
- Rich console for user-friendly error display

### Test Fixes Pattern
When fixing tests:
- Check for mock method signatures matching actual implementation
- Ensure _detected_datetime_columns is initialized in DatasetRegistrar tests
- _load_data_files returns Dict[str, str] not nested dicts
- Backend initialization must handle errors properly
- Use appropriate timeouts for performance tests (10s default)
- E2E tests may be slow due to ydata-profiling - use --no-features flag for speed

## Feature Generation

### Generic Features
MDM automatically generates ~90+ features including:
- Statistical aggregations (mean, std, ratios)
- Frequency-based features for categorical columns
- Interaction features between numeric columns
- Time-based features if datetime columns detected

### Feature Generation Time
- Small datasets (<100k rows): 2-5 minutes
- Medium datasets (100k-1M rows): 10-30 minutes
- Large datasets (>1M rows): Can take hours
- Use `--no-features` flag to skip feature generation

### Export Functionality
```bash
# Export dataset with features
mdm dataset export <name> --output-dir ./exports --format parquet

# Export creates:
# - {name}_train.parquet
# - {name}_train_features.parquet
# - {name}_test.parquet
# - {name}_test_features.parquet
# - {name}_metadata.json
```

## Useful Files for Context
- `docs/03_Database_Architecture.md`: Authoritative backend selection explanation
- `docs/Architecture_Decisions.md`: ADRs for key design choices
- `docs/Migration_Guide.md`: Detailed migration instructions
- `docs/MANUAL_TEST_CHECKLIST.md`: 617-item comprehensive test checklist
- `MDM_Architecture_Analysis_Report.md`: Comprehensive architecture analysis
- `CHANGELOG.md`: Recent changes and version history
- `pyproject.toml`: Dependencies and project metadata (currently v1.0.6)
- `scripts/check_test_imports.py`: Pre-commit hook for test imports