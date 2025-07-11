# Changelog

All notable changes to MDM (ML Data Manager) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-07-11

### Changed
- Major codebase cleanup - removed all legacy migration code
- Consolidated configuration system from `config_new.py` to `config.py`
- Simplified architecture by removing feature flag system

### Removed
- `src/mdm/cli/legacy_adapters.py` - legacy adapter implementations
- `tests/old/` directory with outdated tests
- `docs/archive/migration-summaries/` and `docs/archive/refactoring/` directories
- All `*_migration_example.py` files from examples directory
- `scripts/post_deployment_validation.py` and `scripts/final_migration.py`
- Migration test files (`test_config_migration.py`, `test_feature_migration.py`)

### Fixed
- Configuration system now properly ignores extra environment variables
- Improved error handling in configuration loading

## [0.3.0] - Previous Release

### Added
- Initial release with complete dataset management functionality
- Support for SQLite, DuckDB, and PostgreSQL backends
- Feature engineering system with generic and custom transformers
- Rich CLI interface with progress tracking
- Comprehensive API through MDMClient