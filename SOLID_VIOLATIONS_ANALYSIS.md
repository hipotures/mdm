# SOLID Principles Violation Analysis for MDM

## Summary

This analysis examines key MDM files for violations of SOLID principles:
- **S**: Single Responsibility Principle
- **O**: Open/Closed Principle  
- **L**: Liskov Substitution Principle
- **I**: Interface Segregation Principle
- **D**: Dependency Inversion Principle

## 1. DatasetRegistrar (src/mdm/dataset/registrar.py)

### Violation: Single Responsibility Principle (SRP)
**Severity: HIGH**

The `DatasetRegistrar` class has too many responsibilities:
- Dataset validation
- Auto-detection of dataset structure
- Database creation
- Data loading and chunking
- Column type detection
- Feature generation
- Statistics computation
- Monitoring and metrics

**Problematic Code:**
```python
class DatasetRegistrar:
    """Handles the 12-step dataset registration process."""
    
    def __init__(self, manager: Optional[DatasetManager] = None):
        self.manager = manager or DatasetManager()
        self.feature_generator = FeatureGenerator()
        self.monitor = SimpleMonitor()
        # Too many dependencies!
```

**Why it's a violation:**
The class is responsible for the entire 12-step registration process, making it a "God Class". Changes to any step (validation, loading, feature generation, etc.) require modifying this class.

**How to fix:**
Break down into smaller, focused classes:
```python
# Separate responsibilities
class DatasetValidator:
    def validate_name(self, name: str) -> str
    def validate_path(self, path: Path) -> Path

class DatasetAutoDetector:
    def detect_structure(self, path: Path) -> Dict[str, Any]
    def discover_files(self, path: Path) -> Dict[str, Path]

class DatasetLoader:
    def load_files(self, files: Dict[str, Path], db_info: Dict) -> Dict[str, str]

class ColumnAnalyzer:
    def analyze_columns(self, db_info: Dict, tables: Dict) -> Dict[str, Dict]
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, ColumnType]

class DatasetRegistrar:
    def __init__(self, validator, detector, loader, analyzer, ...):
        # Inject dependencies
```

### Violation: Dependency Inversion Principle (DIP)
**Severity: MEDIUM**

The class directly creates concrete instances instead of depending on abstractions:

**Problematic Code:**
```python
def __init__(self, manager: Optional[DatasetManager] = None):
    self.manager = manager or DatasetManager()  # Creates concrete instance
    self.feature_generator = FeatureGenerator()  # Creates concrete instance
    self.monitor = SimpleMonitor()  # Creates concrete instance
```

**How to fix:**
```python
from abc import ABC, abstractmethod

class IDatasetManager(ABC):
    @abstractmethod
    def register_dataset(self, info: DatasetInfo) -> None: ...

class IFeatureGenerator(ABC):
    @abstractmethod
    def generate_features(self, df: pd.DataFrame, ...) -> pd.DataFrame: ...

class DatasetRegistrar:
    def __init__(self, 
                 manager: IDatasetManager,
                 feature_generator: IFeatureGenerator,
                 monitor: IMonitor):
        # Depend on abstractions, not concretions
```

## 2. StorageBackend (src/mdm/storage/base.py)

### Violation: Single Responsibility Principle (SRP)
**Severity: MEDIUM**

The `StorageBackend` class handles multiple concerns:
- Database operations
- Performance optimization
- Connection pooling
- Caching
- Query optimization
- Monitoring

**Problematic Code:**
```python
class StorageBackend(ABC):
    def __init__(self, config: dict[str, Any]):
        # Too many performance-related responsibilities
        self._query_optimizer: Optional[QueryOptimizer] = None
        self._connection_pool: Optional[ConnectionPool] = None
        self._cache_manager: Optional[CacheManager] = None
        self._batch_optimizer: Optional[BatchOptimizer] = None
        self._monitor = get_monitor()
```

**How to fix:**
Extract performance concerns into a decorator or wrapper:
```python
class StorageBackend(ABC):
    """Pure storage operations only"""
    @abstractmethod
    def create_engine(self, database_path: str) -> Engine: ...
    
    @abstractmethod
    def execute_query(self, query: str, engine: Engine) -> Any: ...

class PerformanceOptimizedBackend:
    """Wrapper that adds performance features"""
    def __init__(self, backend: StorageBackend, optimizer_config: Dict):
        self.backend = backend
        self.query_optimizer = QueryOptimizer(...)
        self.cache_manager = CacheManager(...)
```

### Violation: Interface Segregation Principle (ISP)
**Severity: LOW**

The interface includes both high-level and low-level operations that not all implementations may need:

**Problematic Code:**
```python
class StorageBackend(ABC):
    # High-level operations
    def read_table_to_dataframe(...) -> pd.DataFrame: ...
    
    # Low-level operations
    def execute_query(self, query: str, engine: Engine) -> Any: ...
    def execute_query_original(...) -> Any: ...  # Why two query methods?
```

**How to fix:**
```python
class IStorageEngine(ABC):
    """Low-level storage operations"""
    @abstractmethod
    def execute_query(self, query: str) -> Any: ...

class IDataFrameStorage(ABC):
    """High-level DataFrame operations"""
    @abstractmethod
    def read_table(self, table: str) -> pd.DataFrame: ...
    @abstractmethod
    def write_table(self, df: pd.DataFrame, table: str) -> None: ...
```

## 3. MDMClient (src/mdm/api.py)

### Violation: Single Responsibility Principle (SRP)
**Severity: HIGH**

The `MDMClient` class is a facade that does too much:
- Dataset registration
- Dataset querying
- Data loading
- Feature generation
- Statistics computation
- Time series operations
- ML framework integration
- Export operations
- Performance monitoring

**Problematic Code:**
```python
class MDMClient:
    # 40+ public methods!
    def register_dataset(...): ...
    def load_dataset_files(...): ...
    def query_dataset(...): ...
    def split_time_series(...): ...
    def prepare_for_ml(...): ...
    def create_submission(...): ...
    def process_in_chunks(...): ...
    # ... many more
```

**How to fix:**
Break into focused clients:
```python
class DatasetRegistrationClient:
    def register(self, name: str, path: str, **kwargs) -> DatasetInfo: ...

class DatasetQueryClient:
    def query(self, name: str, sql: str) -> pd.DataFrame: ...
    def load(self, name: str) -> pd.DataFrame: ...

class MLIntegrationClient:
    def prepare_for_framework(self, data: pd.DataFrame, framework: str): ...
    def create_submission(self, predictions: pd.DataFrame): ...

class MDMClient:
    """Thin facade that delegates to specialized clients"""
    def __init__(self):
        self.registration = DatasetRegistrationClient()
        self.query = DatasetQueryClient()
        self.ml = MLIntegrationClient()
```

### Violation: Open/Closed Principle (OCP)
**Severity: MEDIUM**

Adding new functionality requires modifying the class:

**Problematic Code:**
```python
def prepare_for_ml(self, name: str, framework: str = 'auto', ...):
    # Hard-coded framework handling
    if framework == 'sklearn':
        # sklearn logic
    elif framework == 'pytorch':
        # pytorch logic
    # Adding new framework requires modifying this method
```

**How to fix:**
Use strategy pattern:
```python
class MLFrameworkStrategy(ABC):
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Any: ...

class SklearnStrategy(MLFrameworkStrategy): ...
class PyTorchStrategy(MLFrameworkStrategy): ...

class MLIntegrationClient:
    def __init__(self):
        self.strategies = {
            'sklearn': SklearnStrategy(),
            'pytorch': PyTorchStrategy()
        }
    
    def register_strategy(self, name: str, strategy: MLFrameworkStrategy):
        self.strategies[name] = strategy  # Open for extension
```

## 4. FeatureGenerator (src/mdm/features/generator.py)

### Violation: Single Responsibility Principle (SRP)
**Severity: MEDIUM**

The class handles both feature generation logic and I/O operations:

**Problematic Code:**
```python
def generate_feature_tables(self, engine: Engine, ...):
    # Mixing concerns:
    # 1. Database operations (reading/writing)
    with engine.connect() as conn:
        row_count_query = sa.text(f"SELECT COUNT(*) FROM {table_name}")
    
    # 2. Progress tracking
    task = progress.add_task(...)
    
    # 3. Feature generation
    chunk_features = self.generate_features(...)
    
    # 4. Memory management
    del chunk_df
    del chunk_features
```

**How to fix:**
```python
class FeatureGenerator:
    """Pure feature generation logic"""
    def generate_features(self, df: pd.DataFrame, ...) -> pd.DataFrame: ...

class FeatureTableWriter:
    """Handles database I/O for features"""
    def write_features(self, features: pd.DataFrame, table: str): ...

class BatchFeatureProcessor:
    """Handles chunking and progress"""
    def process_in_batches(self, generator: FeatureGenerator, writer: FeatureTableWriter): ...
```

### Violation: Dependency Inversion Principle (DIP)
**Severity: LOW**

Direct file system access for custom features:

**Problematic Code:**
```python
def _load_custom_features(self, dataset_name: str) -> Optional[BaseDomainFeatures]:
    custom_features_path = (
        self.base_path / self.config.paths.custom_features_path / f"{dataset_name}.py"
    )
    if not custom_features_path.exists():  # Direct file system access
```

**How to fix:**
```python
class ICustomFeatureLoader(ABC):
    @abstractmethod
    def load_features(self, dataset_name: str) -> Optional[BaseDomainFeatures]: ...

class FileSystemFeatureLoader(ICustomFeatureLoader):
    def load_features(self, dataset_name: str) -> Optional[BaseDomainFeatures]: ...
```

## 5. ConfigManager (src/mdm/config/config.py)

### Violation: Single Responsibility Principle (SRP)
**Severity: MEDIUM**

The class handles multiple concerns:
- YAML file I/O
- Environment variable parsing
- Value type conversion
- Directory creation
- Singleton management

**Problematic Code:**
```python
class ConfigManager:
    def load(self) -> MDMConfig: ...  # Loading logic
    def save(self, config: MDMConfig) -> None: ...  # Saving logic
    def initialize_defaults(self) -> None: ...  # Directory creation
    def _apply_environment_variables(self, config_dict: dict) -> dict: ...  # Parsing
    def _convert_env_value(self, value: str) -> Any: ...  # Type conversion
```

**How to fix:**
```python
class ConfigLoader(ABC):
    @abstractmethod
    def load(self) -> Dict[str, Any]: ...

class YAMLConfigLoader(ConfigLoader): ...
class EnvironmentConfigLoader(ConfigLoader): ...

class ConfigManager:
    def __init__(self, loaders: List[ConfigLoader]):
        self.loaders = loaders
    
    def load_config(self) -> MDMConfig:
        config_dict = {}
        for loader in self.loaders:
            config_dict.update(loader.load())
        return MDMConfig(**config_dict)
```

### Violation: Open/Closed Principle (OCP)
**Severity: LOW**

Hard-coded environment variable parsing rules:

**Problematic Code:**
```python
def _apply_environment_variables(self, config_dict: dict[str, Any]) -> dict[str, Any]:
    # Hard-coded special cases
    if len(parts) >= 2 and parts[0] == "feature" and parts[1] == "engineering":
        parts = ["feature_engineering"] + parts[2:]
    # Many more hard-coded rules...
```

**How to fix:**
```python
class EnvVarParser:
    def __init__(self):
        self.rules = [
            (lambda p: p[:2] == ["feature", "engineering"], 
             lambda p: ["feature_engineering"] + p[2:]),
            # Rules are data, not code
        ]
    
    def add_rule(self, condition, transformation):
        self.rules.append((condition, transformation))  # Open for extension
```

## Overall Recommendations

1. **Extract Interfaces**: Define clear abstractions for all major components
2. **Dependency Injection**: Use constructor injection instead of creating instances
3. **Single Purpose Classes**: Break down large classes into focused components
4. **Strategy Pattern**: Replace conditional logic with pluggable strategies
5. **Decorator Pattern**: Use decorators for cross-cutting concerns (monitoring, caching)
6. **Factory Pattern**: Use factories for complex object creation
7. **Repository Pattern**: Separate data access from business logic

## Priority Fixes

1. **HIGH**: Split `DatasetRegistrar` into multiple focused classes
2. **HIGH**: Refactor `MDMClient` into specialized clients
3. **MEDIUM**: Extract performance concerns from `StorageBackend`
4. **MEDIUM**: Separate I/O from logic in `FeatureGenerator`
5. **LOW**: Improve `ConfigManager` with loader pattern