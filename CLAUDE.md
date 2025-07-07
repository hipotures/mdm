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

# Note: uv.lock file is missing and should be generated with:
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
```

### Common MDM Commands
```bash
# Register dataset
mdm dataset register <name> <path> [--target <col>] [--problem-type <type>]

# List datasets
mdm dataset list [--limit N] [--sort-by name|registration_date]

# Dataset operations
mdm dataset info <name>
mdm dataset stats <name>
mdm dataset export <name> [--format csv|parquet|json]
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

### Key Modules
- `src/mdm/storage/`: Storage backend implementations (SQLite, DuckDB, PostgreSQL)
- `src/mdm/dataset/`: Dataset management logic (registrar, manager, operations)
- `src/mdm/features/`: Feature engineering system (generic and custom transformers)
- `src/mdm/cli/`: Typer-based CLI implementation
- `src/mdm/config/`: Configuration management using Pydantic Settings

### Important Classes
- `StorageBackend` (base.py): Abstract base for all storage backends
- `DatasetRegistrar`: Handles dataset registration and auto-detection
- `DatasetManager`: Core dataset operations
- `FeatureGenerator`: Two-tier feature engineering system

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

## Known Issues and Workarounds

### Critical Bugs
1. **--time-column and --group-column**: Cause "multiple values for keyword argument" error
2. **SQLite synchronous setting**: Always FULL instead of configured NORMAL
3. **Custom features**: Not loaded from `~/.mdm/config/custom_features/`

### Missing Features
- Many CLI options in test checklist don't exist (--source, --datetime-columns, etc.)
- Compressed file support (.csv.gz)
- Excel file support (.xlsx)
- SQLAlchemy echo configuration not working
- Log level configuration partially working

### Testing Artifacts
The codebase contains many test datasets from manual testing. When running tests, be aware that `~/.mdm/datasets/` may contain numerous test datasets.

## Development Guidelines

### Language Requirement
All code, documentation, and communication must be in English as specified in the project overview.

### Testing Approach
- Use MANUAL_TEST_CHECKLIST.md for comprehensive testing
- Document issues in ISSUES.md
- Track progress in test_progress.md

### Common Patterns
- Use SQLAlchemy ORM for all database operations
- Feature engineering follows a two-tier system (generic + custom)
- CLI uses Typer with Rich for formatting
- All models use Pydantic for validation

## Useful Files for Context
- `docs/03_Database_Architecture.md`: Authoritative backend selection explanation
- `docs/MANUAL_TEST_CHECKLIST.md`: 617-item comprehensive test checklist
- `ISSUES.md`: Documented bugs and limitations
- `test_progress.md`: Testing status and findings