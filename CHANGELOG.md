# Changelog

All notable changes to MDM (ML Data Manager) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-07-12

### Documentation
- Complete documentation overhaul - created comprehensive new documentation
- Fixed package name from "mdm-ml" to "mdm" throughout documentation
- Added automatic Kaggle structure recognition to key features
- Moved test documentation to `tests/docs/` directory
- Removed redundant manual test checklist in favor of automated tests

### Fixed
- Architecture documentation now accurately reflects implementation:
  - Corrected database table names (removed `_table` suffix)
  - Fixed metadata table name (`_mdm_metadata` → `_metadata`)
  - Removed reference to non-existent `_mdm_features` table
  - Updated backend classes to show stateless implementations
  - Fixed custom feature examples to use `BaseDomainFeatures`
  - Corrected `StorageBackend` interface documentation
  - Removed MongoDB from examples (not implemented)
  - Fixed incorrect multi-instance deployment diagram
  
### Changed
- Reorganized documentation structure - moved `docs-new/` to `docs/`
- Removed all outdated documentation directories
- Cleaned up documentation to focus on actual features, not planned ones

## [1.0.0] - 2025-07-12

### Added
- Production-ready status - MDM is now stable for production use
- Comprehensive test suite with 95.4% passing rate (1110/1163 tests)
- Full documentation suite including architecture, API reference, and migration guides
- GitHub issue integration for automated test failure tracking

### Changed
- Updated project status from Alpha to Production/Stable
- Improved test coverage and fixed all critical test failures

### Fixed
- Fixed UpdateOperation API in dataset commands to use keyword arguments
- Fixed feature generator tests by properly mocking get_global_transformers
- Fixed feature registry tests to account for global transformer instances
- Fixed all dataset update comprehensive tests (9 tests)
- Resolved mock signature mismatches across test suite

### Technical Improvements
- Test suite now properly handles all mock operations
- Improved test isolation and reliability
- Enhanced error messages and debugging capabilities

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