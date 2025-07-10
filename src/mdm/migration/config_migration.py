"""Configuration migration utilities.

This module provides tools for migrating between old and new configuration
systems, including validation and comparison functionality.
"""
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging
import json
from datetime import datetime

from mdm.config.config import ConfigManager
from mdm.models.config import MDMConfig as LegacyMDMConfig
from mdm.core.config_new import NewMDMConfig, get_new_config
from mdm.core import metrics_collector
from mdm.testing import ComparisonTester

logger = logging.getLogger(__name__)


class ConfigurationMigrator:
    """Handles migration between configuration systems."""
    
    def __init__(self):
        """Initialize migrator."""
        self.legacy_manager = ConfigManager()
        self.differences: List[Tuple[str, Any, Any]] = []
        
    def migrate_from_legacy(self, legacy_path: Optional[Path] = None) -> NewMDMConfig:
        """Migrate from legacy configuration to new format.
        
        Args:
            legacy_path: Optional path to legacy config file
            
        Returns:
            New configuration instance
        """
        logger.info("Starting configuration migration from legacy format")
        
        # Load legacy config
        if legacy_path:
            self.legacy_manager.config_path = legacy_path
        legacy_config = self.legacy_manager.load()
        
        # Map to new structure
        new_config_data = self._map_legacy_to_new(legacy_config)
        
        # Create new config
        new_config = NewMDMConfig(**new_config_data)
        
        logger.info("Configuration migration completed")
        return new_config
    
    def _map_legacy_to_new(self, legacy: LegacyMDMConfig) -> Dict[str, Any]:
        """Map legacy config structure to new format.
        
        Args:
            legacy: Legacy configuration
            
        Returns:
            Dictionary for new configuration
        """
        return {
            # Base settings
            "home_dir": self.legacy_manager.base_path,
            "enable_auto_detect": True,  # Not in legacy
            "enable_validation": legacy.validation.before_features.check_duplicates,
            
            # Paths
            "paths": {
                "datasets_path": legacy.paths.datasets_path,
                "configs_path": legacy.paths.configs_path,
                "logs_path": legacy.paths.logs_path,
                "custom_features_path": legacy.paths.custom_features_path,
                "cache_path": "cache/",  # Not in legacy
            },
            
            # Database
            "database": {
                "default_backend": legacy.database.default_backend,
                "connection_timeout": legacy.database.connection_timeout,
                
                # SQLite
                "sqlite_journal_mode": legacy.database.sqlite.journal_mode,
                "sqlite_synchronous": legacy.database.sqlite.synchronous,
                "sqlite_cache_size": legacy.database.sqlite.cache_size,
                "sqlite_temp_store": legacy.database.sqlite.temp_store,
                "sqlite_mmap_size": legacy.database.sqlite.mmap_size,
                
                # DuckDB
                "duckdb_memory_limit": legacy.database.duckdb.memory_limit,
                "duckdb_threads": legacy.database.duckdb.threads,
                "duckdb_temp_directory": legacy.database.duckdb.temp_directory,
                "duckdb_access_mode": legacy.database.duckdb.access_mode,
                
                # PostgreSQL
                "postgresql_host": legacy.database.postgresql.host,
                "postgresql_port": legacy.database.postgresql.port,
                "postgresql_user": legacy.database.postgresql.user,
                "postgresql_password": legacy.database.postgresql.password,
                "postgresql_database_prefix": legacy.database.postgresql.database_prefix,
                "postgresql_sslmode": legacy.database.postgresql.sslmode,
                "postgresql_pool_size": legacy.database.postgresql.pool_size,
                
                # SQLAlchemy
                "sqlalchemy_echo": legacy.database.sqlalchemy.echo,
                "sqlalchemy_pool_size": legacy.database.sqlalchemy.pool_size,
                "sqlalchemy_max_overflow": legacy.database.sqlalchemy.max_overflow,
                "sqlalchemy_pool_timeout": legacy.database.sqlalchemy.pool_timeout,
                "sqlalchemy_pool_recycle": legacy.database.sqlalchemy.pool_recycle,
            },
            
            # Performance
            "performance": {
                "batch_size": legacy.performance.batch_size,
                "max_concurrent_operations": legacy.performance.max_concurrent_operations,
                "chunk_size": legacy.performance.batch_size,  # Use same value
                "max_workers": 4,  # Not in legacy
            },
            
            # Logging
            "logging": {
                "level": legacy.logging.level,
                "file": legacy.logging.file,
                "max_bytes": legacy.logging.max_bytes,
                "backup_count": legacy.logging.backup_count,
                "format": legacy.logging.format,
            },
            
            # Feature Engineering
            "feature_engineering": {
                "enabled": legacy.feature_engineering.enabled,
                
                # Type detection
                "categorical_threshold": legacy.feature_engineering.type_detection.categorical_threshold,
                "text_min_avg_length": legacy.feature_engineering.type_detection.text_min_avg_length,
                "detect_datetime": legacy.feature_engineering.type_detection.detect_datetime,
                
                # Features
                "temporal_enabled": legacy.feature_engineering.generic.temporal.enabled,
                "temporal_include_cyclical": legacy.feature_engineering.generic.temporal.include_cyclical,
                "temporal_include_lag": legacy.feature_engineering.generic.temporal.include_lag,
                
                "categorical_enabled": legacy.feature_engineering.generic.categorical.enabled,
                "categorical_max_cardinality": legacy.feature_engineering.generic.categorical.max_cardinality,
                "categorical_min_frequency": legacy.feature_engineering.generic.categorical.min_frequency,
                
                "statistical_enabled": legacy.feature_engineering.generic.statistical.enabled,
                "statistical_include_log": legacy.feature_engineering.generic.statistical.include_log,
                "statistical_include_zscore": legacy.feature_engineering.generic.statistical.include_zscore,
                "statistical_outlier_threshold": legacy.feature_engineering.generic.statistical.outlier_threshold,
                
                "text_enabled": legacy.feature_engineering.generic.text.enabled,
                "text_min_text_length": legacy.feature_engineering.generic.text.min_text_length,
                
                "binning_enabled": legacy.feature_engineering.generic.binning.enabled,
                "binning_n_bins": ",".join(map(str, legacy.feature_engineering.generic.binning.n_bins)),
            },
            
            # Export
            "export": {
                "default_format": legacy.export.default_format,
                "compression": legacy.export.compression,
                "include_metadata": legacy.export.include_metadata,
            },
            
            # Validation
            "validation": {
                "before_check_duplicates": legacy.validation.before_features.check_duplicates,
                "before_drop_duplicates": legacy.validation.before_features.drop_duplicates,
                "before_signal_detection": legacy.validation.before_features.signal_detection,
                "before_max_missing_ratio": legacy.validation.before_features.max_missing_ratio,
                "before_min_rows": legacy.validation.before_features.min_rows,
                
                "after_check_duplicates": legacy.validation.after_features.check_duplicates,
                "after_drop_duplicates": legacy.validation.after_features.drop_duplicates,
                "after_signal_detection": legacy.validation.after_features.signal_detection,
                "after_max_missing_ratio": legacy.validation.after_features.max_missing_ratio,
                "after_min_rows": legacy.validation.after_features.min_rows,
            },
            
            # CLI
            "cli": {
                "default_output_format": legacy.cli.default_output_format,
                "page_size": legacy.cli.page_size,
                "confirm_destructive": legacy.cli.confirm_destructive,
                "show_progress": legacy.cli.show_progress,
            },
            
            # Development
            "development": {
                "debug_mode": legacy.development.debug_mode,
                "profile_operations": legacy.development.profile_operations,
                "explain_queries": legacy.development.explain_queries,
            },
        }
    
    def compare_configs(self, legacy_config: LegacyMDMConfig, 
                       new_config: NewMDMConfig) -> Dict[str, Any]:
        """Compare legacy and new configurations.
        
        Args:
            legacy_config: Legacy configuration
            new_config: New configuration
            
        Returns:
            Comparison results
        """
        self.differences.clear()
        
        # Compare key values
        comparisons = [
            ("database.default_backend", 
             legacy_config.database.default_backend,
             new_config.database.default_backend),
            
            ("performance.batch_size",
             legacy_config.performance.batch_size,
             new_config.performance.batch_size),
            
            ("logging.level",
             legacy_config.logging.level,
             new_config.logging.level),
            
            ("feature_engineering.enabled",
             legacy_config.feature_engineering.enabled,
             new_config.feature_engineering.enabled),
             
            ("export.default_format",
             legacy_config.export.default_format,
             new_config.export.default_format),
        ]
        
        for key, legacy_val, new_val in comparisons:
            if legacy_val != new_val:
                self.differences.append((key, legacy_val, new_val))
        
        return {
            "identical": len(self.differences) == 0,
            "differences": self.differences,
            "difference_count": len(self.differences),
        }
    
    def validate_migration(self, legacy_path: Optional[Path] = None) -> bool:
        """Validate that migration preserves all settings.
        
        Args:
            legacy_path: Optional path to legacy config
            
        Returns:
            True if migration is valid
        """
        logger.info("Validating configuration migration")
        
        # Load legacy
        if legacy_path:
            self.legacy_manager.config_path = legacy_path
        legacy_config = self.legacy_manager.load()
        
        # Migrate
        new_config = self.migrate_from_legacy(legacy_path)
        
        # Compare
        results = self.compare_configs(legacy_config, new_config)
        
        if results["identical"]:
            logger.info("Migration validation passed - configurations are identical")
            return True
        else:
            logger.warning(f"Migration validation found {results['difference_count']} differences:")
            for key, old_val, new_val in results["differences"]:
                logger.warning(f"  {key}: {old_val} -> {new_val}")
            return False
    
    def create_migration_report(self, output_path: Path) -> Path:
        """Create detailed migration report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Path to report file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "migration_type": "legacy_to_new",
            "differences": [
                {
                    "key": key,
                    "legacy_value": str(old_val),
                    "new_value": str(new_val),
                }
                for key, old_val, new_val in self.differences
            ],
            "summary": {
                "total_differences": len(self.differences),
                "validation_passed": len(self.differences) == 0,
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Migration report saved to: {output_path}")
        return output_path


class ConfigurationValidator:
    """Validates configuration consistency."""
    
    def __init__(self):
        """Initialize validator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_config(self, config: NewMDMConfig) -> bool:
        """Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Check paths exist
        if not config.home_dir.exists():
            self.errors.append(f"Home directory does not exist: {config.home_dir}")
            
        # Check backend validity
        valid_backends = ["sqlite", "duckdb", "postgresql"]
        if config.database.default_backend not in valid_backends:
            self.errors.append(
                f"Invalid backend: {config.database.default_backend}. "
                f"Must be one of: {valid_backends}"
            )
            
        # Check numeric constraints
        if config.performance.batch_size < 100:
            self.errors.append("Batch size must be >= 100")
            
        if config.performance.max_workers < 1:
            self.errors.append("Max workers must be >= 1")
            
        # Warnings
        if config.performance.batch_size > 100000:
            self.warnings.append(
                f"Large batch size ({config.performance.batch_size}) may cause memory issues"
            )
            
        if config.database.default_backend == "sqlite" and config.performance.max_workers > 1:
            self.warnings.append(
                "SQLite with multiple workers may cause locking issues"
            )
            
        return len(self.errors) == 0
    
    def get_report(self) -> Dict[str, Any]:
        """Get validation report.
        
        Returns:
            Validation report
        """
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


def migrate_config_file(legacy_path: Path, new_path: Path) -> None:
    """Migrate configuration file from legacy to new format.
    
    Args:
        legacy_path: Path to legacy YAML file
        new_path: Path to save new YAML file
    """
    migrator = ConfigurationMigrator()
    new_config = migrator.migrate_from_legacy(legacy_path)
    new_config.to_yaml(new_path)
    logger.info(f"Migrated configuration saved to: {new_path}")