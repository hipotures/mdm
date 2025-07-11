# Configuration System Overhaul

## Overview

The current configuration system has multiple overlapping classes, hardcoded environment variable mappings, and poor testability due to global state. This guide details the refactoring to a clean, unified configuration system.

## Current Problems

### 1. Multiple Configuration Classes
```python
# CURRENT - Confusing hierarchy
# In config.py
class MDMConfig(BaseSettings):
    # One version

# Also in config.py  
class MDMConfig(BaseModel):
    # Another version!

# In models.py
class MDMConfig:
    # Yet another!
```

### 2. Hardcoded Environment Variable Mapping
```python
# CURRENT - Brittle special cases
if len(parts) >= 2 and parts[0] == "feature" and parts[1] == "engineering":
    parts = ["feature_engineering"] + parts[2:]
elif len(parts) >= 3 and parts[1] == "default" and parts[2] == "backend":
    parts = [parts[0], "default_backend"] + parts[3:]
```

### 3. Global Singleton
```python
# CURRENT - Makes testing difficult
_config: Optional[MDMConfig] = None

def get_config() -> MDMConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
```

### 4. Path Management Chaos
```python
# CURRENT - Paths resolved in properties
@property
def base_path(self) -> Path:
    path = Path(self.paths.base_path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
```

## Target Architecture

### 1. Unified Configuration System
```python
# NEW - Single source of truth
from pydantic import BaseSettings, Field, validator
from typing import Optional, Dict, Any, List
from pathlib import Path
import os

class MDMSettings(BaseSettings):
    """Central configuration for MDM."""
    
    # Storage configuration
    storage_backend: str = Field(
        default="sqlite",
        description="Default storage backend",
        env="MDM_STORAGE_BACKEND"
    )
    
    storage_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific options"
    )
    
    # Performance settings
    batch_size: int = Field(
        default=10000,
        description="Batch size for data processing",
        env="MDM_BATCH_SIZE",
        ge=100,
        le=1000000
    )
    
    max_workers: int = Field(
        default=4,
        description="Maximum parallel workers",
        env="MDM_MAX_WORKERS",
        ge=1,
        le=32
    )
    
    # Paths configuration
    base_path: Path = Field(
        default=Path("~/.mdm"),
        description="Base directory for MDM data",
        env="MDM_BASE_PATH"
    )
    
    # Feature engineering
    feature_plugins_enabled: bool = Field(
        default=True,
        description="Enable feature plugins",
        env="MDM_FEATURE_PLUGINS_ENABLED"
    )
    
    feature_plugin_paths: List[Path] = Field(
        default_factory=lambda: [Path("~/.mdm/plugins/features")],
        description="Paths to search for feature plugins"
    )
    
    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level",
        env="MDM_LOG_LEVEL",
        regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    
    log_file: Optional[Path] = Field(
        default=None,
        description="Log file path",
        env="MDM_LOG_FILE"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Custom env var names without prefix
        fields = {
            "storage_backend": {"env": ["MDM_STORAGE_BACKEND", "STORAGE_BACKEND"]},
            "batch_size": {"env": ["MDM_BATCH_SIZE", "BATCH_SIZE"]},
        }
    
    @validator("base_path", pre=True)
    def expand_base_path(cls, v):
        """Expand user path."""
        if isinstance(v, str):
            return Path(v).expanduser()
        return v
    
    @validator("feature_plugin_paths", pre=True)
    def expand_plugin_paths(cls, v):
        """Expand plugin paths."""
        if isinstance(v, list):
            return [Path(p).expanduser() for p in v]
        return v
```

### 2. Configuration Manager
```python
# NEW - Manages configuration lifecycle
class ConfigurationManager:
    """Manages application configuration."""
    
    def __init__(self, config_file: Optional[Path] = None, env_file: Optional[Path] = None):
        self.config_file = config_file or self._default_config_file()
        self.env_file = env_file
        self._settings: Optional[MDMSettings] = None
        self._overrides: Dict[str, Any] = {}
    
    def load(self) -> MDMSettings:
        """Load configuration from all sources."""
        # 1. Load defaults
        settings = MDMSettings()
        
        # 2. Load from config file if exists
        if self.config_file.exists():
            file_config = self._load_yaml(self.config_file)
            settings = MDMSettings(**file_config)
        
        # 3. Apply environment variables (highest priority)
        settings = MDMSettings(**settings.dict())
        
        # 4. Apply runtime overrides
        if self._overrides:
            settings = MDMSettings(**{**settings.dict(), **self._overrides})
        
        self._settings = settings
        return settings
    
    def get(self) -> MDMSettings:
        """Get current settings."""
        if self._settings is None:
            self._settings = self.load()
        return self._settings
    
    def override(self, **kwargs) -> None:
        """Override settings at runtime."""
        self._overrides.update(kwargs)
        self._settings = None  # Force reload
    
    def reset(self) -> None:
        """Reset to default configuration."""
        self._overrides.clear()
        self._settings = None
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        settings = self.get()
        
        # Validate paths exist
        if not settings.base_path.exists():
            errors.append(f"Base path does not exist: {settings.base_path}")
        
        # Validate backend
        if settings.storage_backend not in ["sqlite", "postgresql", "duckdb"]:
            errors.append(f"Unknown storage backend: {settings.storage_backend}")
        
        return errors
    
    def _default_config_file(self) -> Path:
        """Get default config file path."""
        return Path.home() / ".mdm" / "mdm.yaml"
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration."""
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
```

### 3. Path Manager
```python
# NEW - Centralized path management
class PathManager:
    """Manages all MDM paths."""
    
    def __init__(self, settings: MDMSettings):
        self.settings = settings
        self._ensure_directories()
    
    @property
    def base_path(self) -> Path:
        """Get base MDM directory."""
        return self.settings.base_path
    
    @property
    def datasets_path(self) -> Path:
        """Get datasets directory."""
        return self.base_path / "datasets"
    
    @property
    def config_path(self) -> Path:
        """Get configuration directory."""
        return self.base_path / "config"
    
    @property
    def plugins_path(self) -> Path:
        """Get plugins directory."""
        return self.base_path / "plugins"
    
    @property
    def logs_path(self) -> Path:
        """Get logs directory."""
        return self.base_path / "logs"
    
    def dataset_path(self, dataset_name: str) -> Path:
        """Get path for specific dataset."""
        return self.datasets_path / dataset_name
    
    def dataset_db_path(self, dataset_name: str) -> Path:
        """Get database path for dataset."""
        backend = self.settings.storage_backend
        
        if backend == "sqlite":
            return self.dataset_path(dataset_name) / "dataset.sqlite"
        elif backend == "duckdb":
            return self.dataset_path(dataset_name) / "dataset.duckdb"
        else:
            # PostgreSQL doesn't use file path
            return self.dataset_path(dataset_name)
    
    def dataset_config_path(self, dataset_name: str) -> Path:
        """Get config path for dataset."""
        return self.config_path / "datasets" / f"{dataset_name}.yaml"
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for path in [self.base_path, self.datasets_path, self.config_path, 
                     self.plugins_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)
```

### 4. Backend-Specific Configuration
```python
# NEW - Clean backend configuration
@dataclass
class BackendConfig:
    """Base backend configuration."""
    type: str

@dataclass 
class SQLiteConfig(BackendConfig):
    """SQLite-specific configuration."""
    type: str = "sqlite"
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    cache_size: int = -64000
    temp_store: str = "MEMORY"
    mmap_size: int = 268435456

@dataclass
class PostgreSQLConfig(BackendConfig):
    """PostgreSQL-specific configuration."""
    type: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "mdm"
    user: str = "mdm_user"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10

@dataclass
class DuckDBConfig(BackendConfig):
    """DuckDB-specific configuration."""
    type: str = "duckdb"
    memory_limit: str = "1GB"
    threads: int = 4
    
class BackendConfigFactory:
    """Factory for backend configurations."""
    
    @staticmethod
    def create(backend_type: str, config_dict: Dict[str, Any]) -> BackendConfig:
        """Create backend configuration."""
        if backend_type == "sqlite":
            return SQLiteConfig(**config_dict)
        elif backend_type == "postgresql":
            return PostgreSQLConfig(**config_dict)
        elif backend_type == "duckdb":
            return DuckDBConfig(**config_dict)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
```

### 5. Dependency Injection
```python
# NEW - DI container for configuration
class Container:
    """Dependency injection container."""
    
    def __init__(self):
        self._services: Dict[type, Any] = {}
        self._factories: Dict[type, Callable] = {}
    
    def register_singleton(self, service_type: type, instance: Any) -> None:
        """Register singleton service."""
        self._services[service_type] = instance
    
    def register_factory(self, service_type: type, factory: Callable) -> None:
        """Register service factory."""
        self._factories[service_type] = factory
    
    def get(self, service_type: type) -> Any:
        """Get service instance."""
        # Check singletons
        if service_type in self._services:
            return self._services[service_type]
        
        # Check factories
        if service_type in self._factories:
            instance = self._factories[service_type]()
            return instance
        
        raise ValueError(f"Service not registered: {service_type}")

# Application bootstrap
def bootstrap_application(config_file: Optional[Path] = None) -> Container:
    """Bootstrap application with DI container."""
    container = Container()
    
    # Register configuration
    config_manager = ConfigurationManager(config_file)
    settings = config_manager.load()
    
    container.register_singleton(ConfigurationManager, config_manager)
    container.register_singleton(MDMSettings, settings)
    
    # Register path manager
    path_manager = PathManager(settings)
    container.register_singleton(PathManager, path_manager)
    
    # Register other services
    container.register_factory(
        ConnectionManager,
        lambda: ConnectionManager(
            settings.storage_backend,
            BackendConfigFactory.create(
                settings.storage_backend,
                settings.storage_options
            )
        )
    )
    
    return container
```

## Migration Strategy

### Phase 1: Create New System
1. Implement `MDMSettings` with Pydantic
2. Create `ConfigurationManager`
3. Implement `PathManager`
4. Set up DI container

### Phase 2: Parallel Run
```python
# config/adapter.py
class ConfigAdapter:
    """Adapts new config to old interface."""
    
    def __init__(self, settings: MDMSettings):
        self.settings = settings
    
    # Adapt to old interface
    @property
    def database(self):
        return {
            "default_backend": self.settings.storage_backend,
            self.settings.storage_backend: self.settings.storage_options
        }
    
    @property 
    def performance(self):
        return {
            "batch_size": self.settings.batch_size,
            "max_workers": self.settings.max_workers
        }
```

### Phase 3: Update Components
```python
# Before
class DatasetManager:
    def __init__(self):
        self.config = get_config()

# After  
class DatasetManager:
    def __init__(self, settings: MDMSettings, path_manager: PathManager):
        self.settings = settings
        self.path_manager = path_manager
```

### Phase 4: Remove Old System
1. Remove global config
2. Remove old config classes
3. Update all imports
4. Clean up adapters

## Configuration File Format

### Old Format (mdm.yaml)
```yaml
database:
  default_backend: sqlite
  sqlite:
    journal_mode: WAL
performance:
  batch_size: 10000
paths:
  base_path: ~/.mdm
```

### New Format (mdm.yaml)
```yaml
# Storage configuration
storage_backend: sqlite
storage_options:
  journal_mode: WAL
  synchronous: NORMAL
  cache_size: -64000

# Performance
batch_size: 10000
max_workers: 4

# Paths
base_path: ~/.mdm

# Features
feature_plugins_enabled: true
feature_plugin_paths:
  - ~/.mdm/plugins/features
  - ./custom_features

# Logging
log_level: INFO
log_file: ~/.mdm/logs/mdm.log
```

## Environment Variables

### Clean Mapping
```bash
# Storage
export MDM_STORAGE_BACKEND=postgresql
export MDM_STORAGE_OPTIONS='{"host": "localhost", "port": 5432}'

# Performance  
export MDM_BATCH_SIZE=50000
export MDM_MAX_WORKERS=8

# Paths
export MDM_BASE_PATH=/data/mdm

# Logging
export MDM_LOG_LEVEL=DEBUG
export MDM_LOG_FILE=/var/log/mdm.log
```

## Testing Strategy

### Unit Tests
```python
# tests/unit/config/test_settings.py
class TestMDMSettings:
    def test_default_values(self):
        settings = MDMSettings()
        assert settings.storage_backend == "sqlite"
        assert settings.batch_size == 10000
    
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MDM_STORAGE_BACKEND", "postgresql")
        monkeypatch.setenv("MDM_BATCH_SIZE", "50000")
        
        settings = MDMSettings()
        assert settings.storage_backend == "postgresql"
        assert settings.batch_size == 50000
    
    def test_validation(self):
        with pytest.raises(ValidationError):
            MDMSettings(batch_size=-1)  # Must be >= 100
```

### Integration Tests
```python
# tests/integration/config/test_manager.py
class TestConfigurationManager:
    def test_load_hierarchy(self, tmp_path):
        # Create config file
        config_file = tmp_path / "mdm.yaml"
        config_file.write_text("""
        storage_backend: duckdb
        batch_size: 20000
        """)
        
        # Set env var (higher priority)
        os.environ["MDM_BATCH_SIZE"] = "30000"
        
        manager = ConfigurationManager(config_file)
        settings = manager.load()
        
        assert settings.storage_backend == "duckdb"  # From file
        assert settings.batch_size == 30000  # From env
```

## Validation and Migrations

### Configuration Validation
```python
# config/validator.py
class ConfigValidator:
    """Validates configuration consistency."""
    
    def validate(self, settings: MDMSettings) -> List[str]:
        """Validate settings."""
        errors = []
        
        # Path validation
        if not settings.base_path.exists():
            errors.append(f"Base path does not exist: {settings.base_path}")
        
        # Backend validation
        if settings.storage_backend == "postgresql":
            if not settings.storage_options.get("password"):
                errors.append("PostgreSQL password not configured")
        
        # Feature plugin paths
        for path in settings.feature_plugin_paths:
            if not path.exists():
                errors.append(f"Plugin path does not exist: {path}")
        
        return errors
```

### Configuration Migration
```python
# config/migrator.py
class ConfigMigrator:
    """Migrates old config to new format."""
    
    def migrate(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration."""
        new_config = {}
        
        # Migrate database section
        if "database" in old_config:
            db = old_config["database"]
            new_config["storage_backend"] = db.get("default_backend", "sqlite")
            
            backend = new_config["storage_backend"]
            if backend in db:
                new_config["storage_options"] = db[backend]
        
        # Migrate performance
        if "performance" in old_config:
            perf = old_config["performance"]
            new_config["batch_size"] = perf.get("batch_size", 10000)
            new_config["max_workers"] = perf.get("max_concurrent_operations", 4)
        
        # Migrate paths
        if "paths" in old_config:
            new_config["base_path"] = old_config["paths"].get("base_path", "~/.mdm")
        
        return new_config
```

## Success Criteria

1. **Single Source**: One configuration system
2. **Type Safety**: Full Pydantic validation
3. **Testability**: No global state
4. **Flexibility**: Easy to extend
5. **Performance**: Lazy loading where appropriate
6. **Documentation**: Auto-generated from types