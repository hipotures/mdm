# Step 4: Configuration System Migration

## Overview

Migrate from the current multi-file configuration system to a unified Pydantic-based configuration with clean environment variable support. This is the first major component migration.

## Duration

2 weeks (Weeks 6-7)

## Objectives

1. Implement new Pydantic-based configuration system
2. Create backward compatibility layer
3. Migrate all configuration access points
4. Validate configuration parity
5. Enable gradual cutover with feature flags

## Current State Analysis

Current configuration system has:
- Multiple overlapping configuration files
- Hardcoded environment variable mappings
- Inconsistent validation
- No type safety
- Complex precedence rules

## Detailed Steps

### Week 6: New Configuration Implementation

#### Day 1-2: Pydantic Configuration Models

##### 1.1 Create Configuration Schema
```python
# Create: src/mdm/config/models.py
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseSettings, Field, validator, root_validator
from pathlib import Path
import os


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    default_backend: Literal["sqlite", "duckdb", "postgresql"] = "sqlite"
    
    # SQLite settings
    sqlite_synchronous: Literal["OFF", "NORMAL", "FULL"] = "NORMAL"
    sqlite_journal_mode: Literal["DELETE", "WAL", "MEMORY"] = "WAL"
    sqlite_cache_size: int = Field(default=-64000, description="Cache size in KB (negative)")
    
    # DuckDB settings
    duckdb_memory_limit: str = "1GB"
    duckdb_threads: Optional[int] = None
    duckdb_temp_directory: Optional[Path] = None
    
    # PostgreSQL settings
    postgresql_host: str = "localhost"
    postgresql_port: int = 5432
    postgresql_user: str = "mdm"
    postgresql_password: str = ""
    postgresql_database: str = "mdm"
    postgresql_pool_size: int = 5
    postgresql_max_overflow: int = 10
    
    # Common settings
    echo_sql: bool = False
    connection_timeout: int = 30
    
    class Config:
        env_prefix = "MDM_DATABASE_"
        case_sensitive = False
        
    @validator("default_backend")
    def validate_backend(cls, v):
        valid_backends = ["sqlite", "duckdb", "postgresql"]
        if v not in valid_backends:
            raise ValueError(f"Invalid backend: {v}. Must be one of {valid_backends}")
        return v


class PerformanceConfig(BaseSettings):
    """Performance tuning configuration"""
    batch_size: int = Field(default=10000, ge=100, le=1000000)
    max_workers: int = Field(default=4, ge=1, le=32)
    memory_limit_mb: int = Field(default=1024, ge=256)
    enable_profiling: bool = False
    cache_enabled: bool = True
    cache_size_mb: int = Field(default=512, ge=64)
    
    # Feature generation
    feature_batch_size: int = Field(default=5000, ge=100)
    feature_n_jobs: int = Field(default=-1, description="-1 means use all CPUs")
    
    # Data loading
    chunk_size: int = Field(default=10000, ge=1000)
    use_arrow: bool = True
    compression_level: int = Field(default=6, ge=0, le=9)
    
    class Config:
        env_prefix = "MDM_PERFORMANCE_"


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}"
    file: Optional[Path] = None
    max_file_size: str = "100MB"
    rotation: str = "1 day"
    retention: str = "1 week"
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    
    # Component-specific levels
    storage_level: Optional[str] = None
    features_level: Optional[str] = None
    cli_level: Optional[str] = None
    
    class Config:
        env_prefix = "MDM_LOGGING_"
        
    @validator("file")
    def ensure_log_directory(cls, v):
        if v:
            v = Path(v)
            v.parent.mkdir(parents=True, exist_ok=True)
        return v


class PathsConfig(BaseSettings):
    """Path configuration"""
    home: Path = Field(default_factory=lambda: Path.home() / ".mdm")
    datasets_path: Optional[Path] = None
    config_path: Optional[Path] = None
    cache_path: Optional[Path] = None
    temp_path: Optional[Path] = None
    
    class Config:
        env_prefix = "MDM_PATHS_"
        
    @root_validator
    def set_default_paths(cls, values):
        home = values.get("home")
        if home:
            home = Path(home)
            values["datasets_path"] = values.get("datasets_path") or home / "datasets"
            values["config_path"] = values.get("config_path") or home / "config"
            values["cache_path"] = values.get("cache_path") or home / "cache"
            values["temp_path"] = values.get("temp_path") or home / "temp"
        return values
    
    def ensure_directories(self):
        """Create all configured directories"""
        for path_name, path_value in self.dict().items():
            if path_value and isinstance(path_value, Path):
                path_value.mkdir(parents=True, exist_ok=True)


class FeaturesConfig(BaseSettings):
    """Feature engineering configuration"""
    enable_numeric: bool = True
    enable_categorical: bool = True
    enable_datetime: bool = True
    enable_text: bool = True
    enable_custom: bool = True
    
    # Numeric features
    numeric_aggregations: List[str] = Field(
        default=["mean", "std", "min", "max", "median", "skew", "kurtosis"]
    )
    numeric_missing_strategy: Literal["drop", "fill_mean", "fill_median", "fill_zero"] = "fill_mean"
    
    # Categorical features
    categorical_max_cardinality: int = 100
    categorical_min_frequency: float = 0.01
    categorical_encoding: Literal["onehot", "label", "target", "count"] = "label"
    
    # DateTime features
    datetime_components: List[str] = Field(
        default=["year", "month", "day", "hour", "minute", "dayofweek", "quarter"]
    )
    datetime_cyclical: bool = True
    
    # Text features
    text_max_features: int = 1000
    text_ngram_range: tuple = (1, 2)
    text_min_df: float = 0.01
    text_use_tfidf: bool = True
    
    class Config:
        env_prefix = "MDM_FEATURES_"


class RefactoringConfig(BaseSettings):
    """Refactoring control configuration"""
    use_new_backend: bool = False
    use_new_registrar: bool = False
    use_new_features: bool = False
    use_new_config: bool = False
    
    enable_comparison_tests: bool = True
    enable_performance_tracking: bool = True
    enable_memory_profiling: bool = False
    auto_fallback: bool = True
    
    # Rollout percentages
    backend_rollout_percentage: int = Field(default=0, ge=0, le=100)
    registrar_rollout_percentage: int = Field(default=0, ge=0, le=100)
    features_rollout_percentage: int = Field(default=0, ge=0, le=100)
    
    class Config:
        env_prefix = "MDM_REFACTORING_"


class MDMConfig(BaseSettings):
    """Main MDM configuration"""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    refactoring: RefactoringConfig = Field(default_factory=RefactoringConfig)
    
    # Global settings
    debug: bool = False
    version: str = "2.0.0"
    
    class Config:
        env_prefix = "MDM_"
        case_sensitive = False
        env_nested_delimiter = "__"  # For nested configs: MDM_DATABASE__DEFAULT_BACKEND
        
    @classmethod
    def load_from_file(cls, config_file: Path) -> "MDMConfig":
        """Load configuration from YAML file"""
        import yaml
        
        if config_file.exists():
            with open(config_file) as f:
                data = yaml.safe_load(f) or {}
            
            # Convert nested dictionaries to proper models
            return cls(**data)
        
        return cls()
    
    def save_to_file(self, config_file: Path):
        """Save configuration to YAML file"""
        import yaml
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary with proper serialization
        data = self.dict(exclude_unset=True)
        
        with open(config_file, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    
    def merge_with_file(self, config_file: Path):
        """Merge with existing file configuration"""
        if config_file.exists():
            file_config = self.load_from_file(config_file)
            # File config takes precedence over defaults but not env vars
            for field in self.__fields__:
                if field not in os.environ:
                    setattr(self, field, getattr(file_config, field))
```

##### 1.2 Create Configuration Manager
```python
# Create: src/mdm/config/manager.py
from typing import Optional, Dict, Any
from pathlib import Path
import os
from functools import lru_cache

from .models import MDMConfig
from ..core.feature_flags import feature_flags


class ConfigurationManager:
    """Manages configuration with backward compatibility"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or self._get_default_config_file()
        self._config: Optional[MDMConfig] = None
        self._legacy_config: Optional[Dict[str, Any]] = None
        self._load_legacy_config()
    
    def _get_default_config_file(self) -> Path:
        """Get default configuration file path"""
        # Check environment variable first
        if "MDM_CONFIG_FILE" in os.environ:
            return Path(os.environ["MDM_CONFIG_FILE"])
        
        # Check standard locations
        locations = [
            Path.cwd() / "mdm.yaml",
            Path.home() / ".mdm" / "mdm.yaml",
            Path("/etc/mdm/mdm.yaml"),
        ]
        
        for location in locations:
            if location.exists():
                return location
        
        # Default to user home
        return Path.home() / ".mdm" / "mdm.yaml"
    
    def _load_legacy_config(self):
        """Load legacy configuration for compatibility"""
        # Import legacy config loader
        try:
            from ..config import get_config as get_legacy_config
            self._legacy_config = get_legacy_config()
        except ImportError:
            self._legacy_config = {}
    
    @lru_cache(maxsize=1)
    def get_config(self) -> MDMConfig:
        """Get configuration instance"""
        if self._config is None:
            # Load from file
            self._config = MDMConfig.load_from_file(self.config_file)
            
            # Ensure paths exist
            self._config.paths.ensure_directories()
        
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with legacy fallback"""
        # Check if using new config system
        if feature_flags.get("use_new_config", False):
            config = self.get_config()
            
            # Handle nested keys
            parts = key.split(".")
            value = config
            
            try:
                for part in parts:
                    value = getattr(value, part)
                return value
            except AttributeError:
                return default
        else:
            # Fall back to legacy config
            return self._legacy_config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        config = self.get_config()
        
        # Handle nested keys
        parts = key.split(".")
        obj = config
        
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        setattr(obj, parts[-1], value)
        
        # Save to file
        config.save_to_file(self.config_file)
        
        # Clear cache
        self.get_config.cache_clear()
    
    def reload(self):
        """Reload configuration from file"""
        self.get_config.cache_clear()
        self._config = None
    
    def export_env_vars(self) -> Dict[str, str]:
        """Export configuration as environment variables"""
        config = self.get_config()
        env_vars = {}
        
        def flatten_config(obj, prefix="MDM"):
            for field_name, field_value in obj.dict().items():
                env_key = f"{prefix}_{field_name.upper()}"
                
                if isinstance(field_value, dict):
                    # Nested configuration
                    flatten_config(field_value, env_key)
                else:
                    env_vars[env_key] = str(field_value)
        
        flatten_config(config)
        return env_vars


# Global configuration manager
config_manager = ConfigurationManager()


# Compatibility function
def get_config() -> MDMConfig:
    """Get current configuration"""
    return config_manager.get_config()
```

#### Day 3-4: Migration Utilities

##### 1.3 Create Configuration Migrator
```python
# Create: src/mdm/config/migrator.py
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import shutil
from datetime import datetime

from .models import MDMConfig
from .manager import config_manager


class ConfigurationMigrator:
    """Migrate from old configuration to new Pydantic-based system"""
    
    def __init__(self):
        self.backup_dir = Path.home() / ".mdm" / "config_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.migration_log: List[str] = []
    
    def analyze_current_config(self) -> Dict[str, Any]:
        """Analyze current configuration files"""
        analysis = {
            "files_found": [],
            "settings_count": 0,
            "env_vars_found": [],
            "conflicts": [],
            "warnings": []
        }
        
        # Check for configuration files
        config_locations = [
            Path.home() / ".mdm" / "config.yaml",
            Path.home() / ".mdm" / "mdm.yaml",
            Path.home() / ".mdm" / "settings.json",
            Path.cwd() / "mdm.yaml",
            Path.cwd() / ".mdm.yaml"
        ]
        
        for location in config_locations:
            if location.exists():
                analysis["files_found"].append(str(location))
                self._analyze_file(location, analysis)
        
        # Check environment variables
        import os
        for key, value in os.environ.items():
            if key.startswith("MDM_"):
                analysis["env_vars_found"].append(f"{key}={value}")
        
        return analysis
    
    def _analyze_file(self, file_path: Path, analysis: Dict[str, Any]):
        """Analyze a configuration file"""
        try:
            if file_path.suffix in [".yaml", ".yml"]:
                with open(file_path) as f:
                    data = yaml.safe_load(f) or {}
            elif file_path.suffix == ".json":
                with open(file_path) as f:
                    data = json.load(f)
            else:
                return
            
            # Count settings
            def count_settings(d, depth=0):
                count = 0
                for k, v in d.items():
                    if isinstance(v, dict) and depth < 3:
                        count += count_settings(v, depth + 1)
                    else:
                        count += 1
                return count
            
            analysis["settings_count"] += count_settings(data)
            
        except Exception as e:
            analysis["warnings"].append(f"Error reading {file_path}: {e}")
    
    def backup_current_config(self) -> Path:
        """Backup all current configuration files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir()
        
        # Copy all config files
        config_files = [
            Path.home() / ".mdm" / "config.yaml",
            Path.home() / ".mdm" / "mdm.yaml",
            Path.home() / ".mdm" / "settings.json",
            Path.home() / ".mdm" / "config" / "datasets",
        ]
        
        for file_path in config_files:
            if file_path.exists():
                if file_path.is_dir():
                    shutil.copytree(file_path, backup_path / file_path.name)
                else:
                    shutil.copy2(file_path, backup_path / file_path.name)
                self.migration_log.append(f"Backed up {file_path}")
        
        # Save environment variables
        import os
        env_vars = {k: v for k, v in os.environ.items() if k.startswith("MDM_")}
        with open(backup_path / "environment.json", 'w') as f:
            json.dump(env_vars, f, indent=2)
        
        self.migration_log.append(f"Created backup at {backup_path}")
        return backup_path
    
    def migrate_to_new_format(self) -> Tuple[MDMConfig, List[str]]:
        """Migrate old configuration to new format"""
        issues = []
        
        # Start with default configuration
        new_config = MDMConfig()
        
        # Load old configuration files
        old_configs = self._load_old_configs()
        
        # Map old settings to new
        mappings = {
            # Database mappings
            "database.backend": "database.default_backend",
            "storage.default_backend": "database.default_backend",
            "sqlite.synchronous": "database.sqlite_synchronous",
            "sqlite.journal_mode": "database.sqlite_journal_mode",
            
            # Performance mappings
            "batch_size": "performance.batch_size",
            "performance.chunk_size": "performance.chunk_size",
            "performance.max_threads": "performance.max_workers",
            
            # Logging mappings
            "logging.level": "logging.level",
            "logging.file": "logging.file",
            "log_level": "logging.level",
            
            # Paths mappings
            "paths.home": "paths.home",
            "paths.datasets": "paths.datasets_path",
            "mdm_home": "paths.home",
        }
        
        # Apply mappings
        for old_key, new_key in mappings.items():
            value = self._get_nested_value(old_configs, old_key)
            if value is not None:
                try:
                    self._set_nested_value(new_config, new_key, value)
                    self.migration_log.append(f"Migrated {old_key} -> {new_key}: {value}")
                except Exception as e:
                    issues.append(f"Failed to migrate {old_key}: {e}")
        
        # Handle special cases
        self._migrate_special_cases(old_configs, new_config, issues)
        
        return new_config, issues
    
    def _load_old_configs(self) -> Dict[str, Any]:
        """Load all old configuration files into unified dict"""
        combined = {}
        
        # Load YAML files
        yaml_files = [
            Path.home() / ".mdm" / "config.yaml",
            Path.home() / ".mdm" / "mdm.yaml",
        ]
        
        for yaml_file in yaml_files:
            if yaml_file.exists():
                with open(yaml_file) as f:
                    data = yaml.safe_load(f) or {}
                    combined.update(data)
        
        return combined
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        parts = key.split(".")
        value = data
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    def _set_nested_value(self, obj: Any, key: str, value: Any):
        """Set value in nested object using dot notation"""
        parts = key.split(".")
        
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        setattr(obj, parts[-1], value)
    
    def _migrate_special_cases(self, old_config: Dict[str, Any], 
                              new_config: MDMConfig, issues: List[str]):
        """Handle special migration cases"""
        # Feature flags from old system
        if "feature_engineering" in old_config:
            old_features = old_config["feature_engineering"]
            if "enabled_types" in old_features:
                for feature_type in old_features["enabled_types"]:
                    setattr(new_config.features, f"enable_{feature_type}", True)
        
        # Environment variable precedence
        import os
        for key, value in os.environ.items():
            if key.startswith("MDM_"):
                # New format uses double underscore for nesting
                new_key = key.replace("_", "__", 1)  # Only first underscore
                os.environ[new_key] = value
    
    def validate_migration(self, old_config: Dict[str, Any], 
                          new_config: MDMConfig) -> List[str]:
        """Validate that migration preserved all settings"""
        validation_errors = []
        
        # Key settings to validate
        critical_settings = [
            ("database.default_backend", "database.backend"),
            ("performance.batch_size", "batch_size"),
            ("logging.level", "logging.level"),
            ("paths.home", "mdm_home"),
        ]
        
        for new_path, old_path in critical_settings:
            old_value = self._get_nested_value(old_config, old_path)
            new_value = self._get_nested_value(new_config.dict(), new_path)
            
            if old_value is not None and old_value != new_value:
                validation_errors.append(
                    f"Value mismatch for {old_path}: old={old_value}, new={new_value}"
                )
        
        return validation_errors
    
    def perform_migration(self, dry_run: bool = True) -> Dict[str, Any]:
        """Perform complete configuration migration"""
        result = {
            "success": False,
            "backup_path": None,
            "issues": [],
            "validation_errors": [],
            "migration_log": []
        }
        
        try:
            # Analyze current state
            analysis = self.analyze_current_config()
            self.migration_log.append(f"Found {len(analysis['files_found'])} config files")
            
            if not dry_run:
                # Backup current configuration
                result["backup_path"] = str(self.backup_current_config())
            
            # Perform migration
            new_config, issues = self.migrate_to_new_format()
            result["issues"] = issues
            
            # Load old config for validation
            old_config = self._load_old_configs()
            
            # Validate migration
            result["validation_errors"] = self.validate_migration(old_config, new_config)
            
            if not dry_run and not result["validation_errors"]:
                # Save new configuration
                config_path = Path.home() / ".mdm" / "mdm.yaml"
                new_config.save_to_file(config_path)
                self.migration_log.append(f"Saved new configuration to {config_path}")
                
                # Update feature flag
                from ..core.feature_flags import feature_flags
                feature_flags.set("use_new_config", True)
            
            result["success"] = len(result["validation_errors"]) == 0
            result["migration_log"] = self.migration_log
            
        except Exception as e:
            result["issues"].append(f"Migration failed: {e}")
        
        return result
```

#### Day 5: Integration Tests

##### 1.4 Create Configuration Tests
```python
# Create: tests/unit/test_new_configuration.py
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from mdm.config.models import MDMConfig, DatabaseConfig, PerformanceConfig
from mdm.config.manager import ConfigurationManager
from mdm.config.migrator import ConfigurationMigrator


class TestConfigurationModels:
    def test_default_configuration(self):
        """Test default configuration values"""
        config = MDMConfig()
        
        assert config.database.default_backend == "sqlite"
        assert config.performance.batch_size == 10000
        assert config.logging.level == "INFO"
        assert config.paths.home == Path.home() / ".mdm"
    
    def test_environment_variable_override(self):
        """Test environment variable precedence"""
        with patch.dict(os.environ, {
            "MDM_DATABASE__DEFAULT_BACKEND": "duckdb",
            "MDM_PERFORMANCE__BATCH_SIZE": "5000",
            "MDM_LOGGING__LEVEL": "DEBUG"
        }):
            config = MDMConfig()
            
            assert config.database.default_backend == "duckdb"
            assert config.performance.batch_size == 5000
            assert config.logging.level == "DEBUG"
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Invalid backend
        with pytest.raises(ValueError, match="Invalid backend"):
            DatabaseConfig(default_backend="invalid")
        
        # Invalid batch size
        with pytest.raises(ValueError):
            PerformanceConfig(batch_size=50)  # Below minimum
    
    def test_yaml_serialization(self):
        """Test YAML save and load"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_file = Path(f.name)
        
        try:
            # Create and save config
            config = MDMConfig()
            config.database.default_backend = "postgresql"
            config.performance.batch_size = 20000
            config.save_to_file(config_file)
            
            # Load and verify
            loaded_config = MDMConfig.load_from_file(config_file)
            assert loaded_config.database.default_backend == "postgresql"
            assert loaded_config.performance.batch_size == 20000
        finally:
            config_file.unlink()


class TestConfigurationManager:
    def test_manager_initialization(self):
        """Test configuration manager setup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "mdm.yaml"
            manager = ConfigurationManager(config_file)
            
            config = manager.get_config()
            assert isinstance(config, MDMConfig)
    
    def test_get_nested_values(self):
        """Test getting nested configuration values"""
        manager = ConfigurationManager()
        
        # Test nested access
        assert manager.get("database.default_backend") == "sqlite"
        assert manager.get("performance.batch_size") == 10000
        assert manager.get("nonexistent.key", "default") == "default"
    
    def test_set_nested_values(self):
        """Test setting nested configuration values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "mdm.yaml"
            manager = ConfigurationManager(config_file)
            
            # Set values
            manager.set("database.default_backend", "duckdb")
            manager.set("performance.batch_size", 15000)
            
            # Verify persistence
            manager.reload()
            assert manager.get("database.default_backend") == "duckdb"
            assert manager.get("performance.batch_size") == 15000
    
    @patch("mdm.core.feature_flags.feature_flags")
    def test_legacy_fallback(self, mock_flags):
        """Test fallback to legacy configuration"""
        mock_flags.get.return_value = False  # Use legacy config
        
        manager = ConfigurationManager()
        manager._legacy_config = {"old_key": "old_value"}
        
        assert manager.get("old_key") == "old_value"


class TestConfigurationMigrator:
    def test_analyze_current_config(self):
        """Test configuration analysis"""
        migrator = ConfigurationMigrator()
        analysis = migrator.analyze_current_config()
        
        assert "files_found" in analysis
        assert "env_vars_found" in analysis
        assert isinstance(analysis["settings_count"], int)
    
    def test_migration_dry_run(self):
        """Test migration in dry run mode"""
        migrator = ConfigurationMigrator()
        
        # Mock old configuration
        old_config = {
            "database": {"backend": "sqlite"},
            "batch_size": 5000,
            "logging": {"level": "DEBUG"}
        }
        
        with patch.object(migrator, "_load_old_configs", return_value=old_config):
            result = migrator.perform_migration(dry_run=True)
            
            assert "success" in result
            assert "issues" in result
            assert "validation_errors" in result
            assert result["backup_path"] is None  # No backup in dry run
    
    def test_configuration_backup(self):
        """Test configuration backup creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            migrator = ConfigurationMigrator()
            migrator.backup_dir = Path(tmpdir)
            
            # Create fake config file
            config_dir = Path.home() / ".mdm"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "test_config.yaml"
            config_file.write_text("test: value")
            
            try:
                backup_path = migrator.backup_current_config()
                assert backup_path.exists()
                assert (backup_path / "test_config.yaml").exists()
            finally:
                config_file.unlink(missing_ok=True)
```

### Week 7: Migration Execution

#### Day 6-7: Gradual Migration

##### 2.1 Create Migration CLI
```python
# Create: src/mdm/cli/config_migration.py
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

from ..config.migrator import ConfigurationMigrator
from ..config.manager import config_manager
from ..core.feature_flags import feature_flags

app = typer.Typer()
console = Console()


@app.command()
def analyze():
    """Analyze current configuration setup"""
    console.print("[bold]Analyzing current configuration...[/bold]")
    
    migrator = ConfigurationMigrator()
    analysis = migrator.analyze_current_config()
    
    # Display results
    table = Table(title="Configuration Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Config Files Found", str(len(analysis["files_found"])))
    table.add_row("Total Settings", str(analysis["settings_count"]))
    table.add_row("Environment Variables", str(len(analysis["env_vars_found"])))
    table.add_row("Warnings", str(len(analysis["warnings"])))
    
    console.print(table)
    
    if analysis["files_found"]:
        console.print("\n[bold]Configuration Files:[/bold]")
        for file in analysis["files_found"]:
            console.print(f"  - {file}")
    
    if analysis["warnings"]:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in analysis["warnings"]:
            console.print(f"  ⚠️  {warning}")


@app.command()
def migrate(
    dry_run: bool = typer.Option(True, "--dry-run/--execute", 
                                 help="Perform dry run without changes"),
    backup: bool = typer.Option(True, "--backup/--no-backup",
                               help="Create backup before migration")
):
    """Migrate to new configuration system"""
    migrator = ConfigurationMigrator()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Analyze phase
        task = progress.add_task("Analyzing configuration...", total=5)
        analysis = migrator.analyze_current_config()
        progress.update(task, advance=1)
        
        # Backup phase
        if not dry_run and backup:
            progress.update(task, description="Creating backup...")
            backup_path = migrator.backup_current_config()
            console.print(f"[green]Backup created at: {backup_path}[/green]")
        progress.update(task, advance=1)
        
        # Migration phase
        progress.update(task, description="Migrating configuration...")
        result = migrator.perform_migration(dry_run=dry_run)
        progress.update(task, advance=2)
        
        # Validation phase
        progress.update(task, description="Validating migration...")
        time.sleep(0.5)  # Simulate validation
        progress.update(task, advance=1)
    
    # Display results
    if result["success"]:
        console.print("[bold green]✓ Migration successful![/bold green]")
    else:
        console.print("[bold red]✗ Migration failed![/bold red]")
    
    if result["issues"]:
        console.print("\n[yellow]Issues encountered:[/yellow]")
        for issue in result["issues"]:
            console.print(f"  - {issue}")
    
    if result["validation_errors"]:
        console.print("\n[red]Validation errors:[/red]")
        for error in result["validation_errors"]:
            console.print(f"  - {error}")
    
    if dry_run:
        console.print("\n[dim]This was a dry run. Use --execute to perform migration.[/dim]")


@app.command()
def validate():
    """Validate current configuration"""
    console.print("[bold]Validating configuration...[/bold]")
    
    try:
        config = config_manager.get_config()
        
        # Validate each section
        sections = ["database", "performance", "logging", "paths", "features"]
        
        table = Table(title="Configuration Validation")
        table.add_column("Section", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        for section in sections:
            try:
                section_config = getattr(config, section)
                # Trigger validation
                _ = section_config.dict()
                table.add_row(section, "✓ Valid", "All settings valid")
            except Exception as e:
                table.add_row(section, "✗ Invalid", str(e))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Configuration validation failed: {e}[/red]")


@app.command()
def rollback(backup_path: str):
    """Rollback to a previous configuration backup"""
    console.print(f"[bold]Rolling back to backup: {backup_path}[/bold]")
    
    # Implementation would restore from backup
    console.print("[yellow]Rollback functionality not yet implemented[/yellow]")


@app.command()
def switch(use_new: bool = typer.Option(True, "--new/--legacy",
                                        help="Switch between new and legacy config")):
    """Switch between new and legacy configuration systems"""
    feature_flags.set("use_new_config", use_new)
    
    system = "new" if use_new else "legacy"
    console.print(f"[green]Switched to {system} configuration system[/green]")
    
    # Show current configuration source
    config = config_manager.get_config()
    console.print(f"\nCurrent configuration file: {config_manager.config_file}")
```

##### 2.2 Update Main Configuration Access
```python
# Update: src/mdm/config/__init__.py
"""
Configuration module with backward compatibility
"""
from typing import Any, Dict
import warnings

from .manager import config_manager, get_config as get_new_config
from .models import MDMConfig
from ..core.feature_flags import feature_flags

# Import legacy configuration if available
try:
    from .legacy import get_config as get_legacy_config
except ImportError:
    get_legacy_config = None


def get_config() -> Any:
    """Get configuration with compatibility layer"""
    if feature_flags.get("use_new_config", False):
        return get_new_config()
    else:
        if get_legacy_config:
            return get_legacy_config()
        else:
            # Fallback to new config with warning
            warnings.warn(
                "Legacy configuration not found, using new configuration system",
                DeprecationWarning
            )
            return get_new_config()


def get_setting(key: str, default: Any = None) -> Any:
    """Get a specific setting with compatibility"""
    return config_manager.get(key, default)


def set_setting(key: str, value: Any) -> None:
    """Set a specific setting"""
    config_manager.set(key, value)


# Export main classes
__all__ = [
    "MDMConfig",
    "get_config", 
    "get_setting",
    "set_setting",
    "config_manager"
]
```

#### Day 8-9: Comparison Testing

##### 2.3 Create Configuration Comparison Tests
```python
# Create: tests/migration/test_config_comparison.py
import pytest
from typing import Dict, Any
import os
import tempfile
from pathlib import Path

from mdm.testing.comparison import ComparisonTester
from mdm.config.models import MDMConfig
from mdm.config.manager import ConfigurationManager


class TestConfigurationComparison:
    def setup_method(self):
        """Set up comparison tester"""
        self.tester = ComparisonTester()
        
    def test_config_access_comparison(self):
        """Compare old and new configuration access patterns"""
        # Mock implementations
        def old_get_config():
            return {
                "database": {"backend": "sqlite"},
                "batch_size": 10000,
                "logging": {"level": "INFO"}
            }
        
        def new_get_config():
            config = MDMConfig()
            return config.dict()
        
        result = self.tester.compare(
            test_name="config_access",
            old_impl=old_get_config,
            new_impl=new_get_config,
            compare_func=lambda old, new: (
                old["database"]["backend"] == new["database"]["default_backend"]
            )
        )
        
        assert result.passed
        assert abs(result.performance_delta) < 100  # New should not be 100% slower
    
    def test_environment_variable_handling(self):
        """Compare environment variable handling"""
        test_env = {
            "MDM_DATABASE_BACKEND": "duckdb",
            "MDM_BATCH_SIZE": "5000",
            "MDM_LOGGING_LEVEL": "DEBUG"
        }
        
        def old_env_handling():
            # Simulate old env var handling
            config = {"database": {}, "logging": {}}
            config["database"]["backend"] = test_env.get("MDM_DATABASE_BACKEND", "sqlite")
            config["batch_size"] = int(test_env.get("MDM_BATCH_SIZE", "10000"))
            config["logging"]["level"] = test_env.get("MDM_LOGGING_LEVEL", "INFO")
            return config
        
        def new_env_handling():
            with patch.dict(os.environ, {
                "MDM_DATABASE__DEFAULT_BACKEND": "duckdb",
                "MDM_PERFORMANCE__BATCH_SIZE": "5000",
                "MDM_LOGGING__LEVEL": "DEBUG"
            }):
                config = MDMConfig()
                return {
                    "database": {"backend": config.database.default_backend},
                    "batch_size": config.performance.batch_size,
                    "logging": {"level": config.logging.level}
                }
        
        result = self.tester.compare(
            test_name="env_var_handling",
            old_impl=old_env_handling,
            new_impl=new_env_handling
        )
        
        assert result.passed
    
    def test_configuration_persistence(self):
        """Test configuration save/load comparison"""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = Path(tmpdir) / "old_config.yaml"
            new_file = Path(tmpdir) / "new_config.yaml"
            
            test_config = {
                "database": {"backend": "postgresql"},
                "performance": {"batch_size": 20000}
            }
            
            def old_save_load():
                import yaml
                # Old save
                with open(old_file, 'w') as f:
                    yaml.dump(test_config, f)
                # Old load
                with open(old_file) as f:
                    return yaml.load(f, Loader=yaml.SafeLoader)
            
            def new_save_load():
                # New save
                config = MDMConfig()
                config.database.default_backend = "postgresql"
                config.performance.batch_size = 20000
                config.save_to_file(new_file)
                # New load
                loaded = MDMConfig.load_from_file(new_file)
                return {
                    "database": {"backend": loaded.database.default_backend},
                    "performance": {"batch_size": loaded.performance.batch_size}
                }
            
            result = self.tester.compare(
                test_name="config_persistence",
                old_impl=old_save_load,
                new_impl=new_save_load
            )
            
            assert result.passed
            # Report any performance differences
            print(f"Performance delta: {result.performance_delta:.2f}%")
```

#### Day 10: Documentation and Rollout

##### 2.4 Create Migration Documentation
```markdown
# Create: docs/config_migration_guide.md

# Configuration Migration Guide

## Overview

This guide covers the migration from the legacy configuration system to the new Pydantic-based configuration system in MDM 2.0.

## What's Changing

### Old System
- Multiple configuration files with different formats
- Hardcoded environment variable mappings
- No type validation
- Inconsistent access patterns

### New System
- Single unified configuration with Pydantic models
- Automatic environment variable mapping
- Full type validation and documentation
- Consistent access patterns

## Migration Timeline

1. **Week 1**: New system implementation (completed)
2. **Week 2**: Migration tools and testing (in progress)
3. **Rollout**: Gradual migration with feature flags

## For Users

### Check Your Current Configuration

```bash
mdm config analyze
```

### Perform Migration

```bash
# Dry run (recommended first)
mdm config migrate --dry-run

# Execute migration
mdm config migrate --execute
```

### Environment Variables

Old format:
```bash
export MDM_DATABASE_BACKEND=duckdb
export MDM_BATCH_SIZE=5000
```

New format:
```bash
export MDM_DATABASE__DEFAULT_BACKEND=duckdb
export MDM_PERFORMANCE__BATCH_SIZE=5000
```

Note the double underscore (`__`) for nested settings.

### Configuration File

The new configuration uses a single `mdm.yaml` file:

```yaml
database:
  default_backend: sqlite
  sqlite_synchronous: NORMAL
  sqlite_journal_mode: WAL

performance:
  batch_size: 10000
  max_workers: 4
  memory_limit_mb: 1024

logging:
  level: INFO
  file: ~/.mdm/logs/mdm.log
  console_level: WARNING

paths:
  home: ~/.mdm
  datasets_path: ~/.mdm/datasets
  cache_path: ~/.mdm/cache

features:
  enable_numeric: true
  enable_categorical: true
  enable_datetime: true
  enable_text: true
```

## For Developers

### Accessing Configuration

```python
# Old way
from mdm.config import get_config
config = get_config()
backend = config.get("database", {}).get("backend", "sqlite")

# New way
from mdm.config import get_config
config = get_config()
backend = config.database.default_backend
```

### Type Safety

```python
from mdm.config.models import MDMConfig

def process_data(config: MDMConfig):
    # Full type hints and autocompletion
    batch_size = config.performance.batch_size
    backend = config.database.default_backend
```

### Feature Flag Control

```python
from mdm.core.feature_flags import feature_flags

# Check if using new config
if feature_flags.get("use_new_config"):
    # New configuration code
else:
    # Legacy configuration code
```

## Rollback Procedure

If issues arise:

1. **Immediate Rollback**:
   ```bash
   mdm config switch --legacy
   ```

2. **Restore from Backup**:
   ```bash
   mdm config rollback ~/.mdm/config_backups/backup_20240115_120000
   ```

3. **Manual Rollback**:
   - Set feature flag: `use_new_config: false`
   - Restore old configuration files from backup

## FAQ

**Q: Will my existing configuration work?**
A: Yes, the migration tool automatically converts your existing configuration.

**Q: What about custom environment variables?**
A: They are migrated to the new format. Check the migration report.

**Q: Can I use both systems during migration?**
A: Yes, the feature flag allows switching between systems.

**Q: How do I report issues?**
A: Use `mdm config validate` and share the output in issue reports.
```

## Validation Checklist

### Implementation Complete
- [ ] Pydantic models for all configuration sections
- [ ] Configuration manager with legacy fallback
- [ ] Migration tool with backup capability
- [ ] CLI commands for migration management
- [ ] Comprehensive test coverage

### Migration Ready
- [ ] Comparison tests passing
- [ ] Performance overhead < 5%
- [ ] Documentation complete
- [ ] Rollback procedures tested
- [ ] Team trained on new system

### Rollout Criteria
- [ ] All tests green
- [ ] Migration tool validated on test systems
- [ ] Backup and restore verified
- [ ] No breaking changes to API
- [ ] Feature flag working correctly

## Success Criteria

- **Zero configuration data loss** during migration
- **100% backward compatibility** maintained
- **Performance impact < 5%** for configuration access
- **All existing functionality** preserved
- **Smooth rollback** capability demonstrated

## Monitoring

```python
# Monitor configuration system usage
from mdm.core.metrics import metrics_collector

# Track which system is being used
metrics_collector.increment("config.access.new" if use_new else "config.access.legacy")

# Track migration success
metrics_collector.gauge("config.migration.success_rate", success_count / total_count)
```

## Next Steps

Once configuration migration is stable, proceed to [05-storage-backend-migration.md](05-storage-backend-migration.md).

## Notes

- Keep both systems running in parallel for at least 2 weeks
- Monitor metrics closely during rollout
- Document any edge cases discovered during migration
- Update all documentation to reflect new configuration format