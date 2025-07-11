# MDM Changelog

All notable changes to ML Data Manager (MDM) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-07-11

### Added
- **Performance Optimizations**
  - CLI startup time reduced from 6.5s to 0.1s through lazy loading
  - Special fast path for `mdm version` command
  - Batch processing with configurable chunk size (default: 10,000 rows)
  - Rich progress bars during dataset registration
  - Memory-efficient data loading for large datasets

- **Feature Flag System**
  - Gradual migration support from legacy to new implementation
  - Environment variable control for feature flags
  - Safe rollback capability

- **Monitoring and Observability**
  - `SimpleMonitor` class for tracking metrics
  - Dashboard view for system statistics
  - Performance metrics collection

- **Documentation**
  - Quick Start guide for 5-minute onboarding
  - Comprehensive FAQ section
  - Complete environment variables reference
  - Reorganized documentation structure

### Changed
- **Logging System**
  - Default log format changed from 'json' to 'console' for better readability
  - Unified logging with loguru across all modules
  - Suppressed external library progress bars (ydata-profiling, tqdm)

- **Configuration**
  - Environment variables now have highest priority
  - Improved configuration validation
  - Better error messages for invalid settings

### Fixed
- **Update Command**
  - Fixed exit code behavior (returns 0 when no updates specified)
  - Added input validation for --id-columns and --problem-type
  - Improved error handling with user-friendly messages

- **Test Infrastructure**
  - Fixed test isolation issues
  - CLI subcommands now available in test environment
  - E2E tests run in isolated /tmp directories
  - Fixed logging configuration conflicts

- **Backend Compatibility**
  - PostgreSQL backend fully operational
  - Fixed SQLAlchemy compatibility issues
  - Proper connection handling and cleanup

### Known Issues
- `--time-column` and `--group-column` cause "multiple values for keyword argument" error
- Custom features not loaded from `~/.mdm/config/custom_features/`
- SQLite synchronous setting always FULL instead of configured NORMAL
- Automatic datetime detection stores columns as TEXT

## [0.1.0] - 2025-01-15

### Added
- Initial release after major refactoring
- Clean architecture with interfaces and adapters
- Multi-backend support (SQLite, DuckDB, PostgreSQL)
- Two-tier database system
- Comprehensive feature engineering
- Rich CLI with Typer and Rich
- Batch processing support
- Dataset auto-detection
- Kaggle competition format support

### Architecture
- Interface-based design for extensibility
- Adapter pattern for legacy compatibility
- Storage backend abstraction
- Performance optimization layer
- Configuration management with Pydantic

### Documentation
- Complete user documentation (00-14 series)
- API reference
- Developer guide
- Migration guide
- Architecture decision records (ADRs)

## [Pre-0.1.0]

Legacy versions before the 2025 refactoring. See git history for details.