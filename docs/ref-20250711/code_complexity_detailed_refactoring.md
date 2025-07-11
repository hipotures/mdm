# Detailed Code Complexity Analysis with Refactoring Suggestions

## Critical Refactoring Needs

### 1. DatasetRegistrar._load_data_files() - URGENT REFACTORING NEEDED

**Current Issues:**
- 334 lines of code (should be <50)
- Cyclomatic complexity ~45-50
- Handles 6+ different file formats in one method
- Deeply nested if-elif chains
- Duplicate code patterns

**Suggested Refactoring - Strategy Pattern:**

```python
# Create file loader interface
class FileLoader(ABC):
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        pass
    
    @abstractmethod
    def load_in_chunks(self, file_path: Path, batch_size: int, 
                      progress: Progress) -> Iterator[pd.DataFrame]:
        pass

# Implement specific loaders
class CSVLoader(FileLoader):
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.csv', '.tsv']
    
    def load_in_chunks(self, file_path: Path, batch_size: int, 
                      progress: Progress) -> Iterator[pd.DataFrame]:
        delimiter = detect_delimiter(file_path)
        for chunk in pd.read_csv(file_path, delimiter=delimiter, 
                                chunksize=batch_size):
            yield chunk

# Refactored method
def _load_data_files(self, files: Dict[str, Path], db_info: Dict[str, Any], 
                    progress: Optional[Progress] = None) -> Dict[str, str]:
    loaders = [CSVLoader(), ParquetLoader(), JSONLoader(), 
               CompressedCSVLoader(), ExcelLoader()]
    
    for file_key, file_path in files.items():
        loader = self._get_loader(file_path, loaders)
        if loader:
            self._load_with_loader(loader, file_path, file_key, 
                                 db_info, progress)
```

### 2. DatasetRegistrar.register() - High Complexity

**Current Issues:**
- 163 lines handling 12 steps
- Mixed responsibilities
- Complex error handling

**Suggested Refactoring - Pipeline Pattern:**

```python
class RegistrationPipeline:
    def __init__(self):
        self.steps = [
            ValidateNameStep(),
            CheckExistingDatasetStep(),
            ValidatePathStep(),
            AutoDetectStep(),
            DiscoverFilesStep(),
            CreateDatabaseStep(),
            LoadDataStep(),
            AnalyzeColumnsStep(),
            DetectIDColumnsStep(),
            DetermineProblemTypeStep(),
            GenerateFeaturesStep(),
            SaveRegistrationStep()
        ]
    
    def execute(self, context: RegistrationContext) -> DatasetInfo:
        for step in self.steps:
            try:
                step.execute(context)
            except StepError as e:
                self._handle_step_error(step, e, context)
        return context.dataset_info
```

### 3. ConfigManager._apply_environment_variables() - Complex Parsing

**Current Issues:**
- 25+ hardcoded string manipulations
- Complex nested conditionals
- Difficult to maintain

**Suggested Refactoring - Configuration Mapping:**

```python
class EnvironmentVariableParser:
    # Define mapping rules
    KEY_MAPPINGS = {
        "feature_engineering": ["feature", "engineering"],
        "default_backend": ["default", "backend"],
        "batch_size": ["batch", "size"],
        "connection_timeout": ["connection", "timeout"],
        "custom_features_path": ["custom", "features", "path"],
        "n_bins": ["n", "bins"]
    }
    
    def parse_env_key(self, env_key: str) -> List[str]:
        parts = env_key[len(self.ENV_PREFIX):].lower().split("_")
        return self._apply_mappings(parts)
    
    def _apply_mappings(self, parts: List[str]) -> List[str]:
        # Use mappings to combine multi-word keys
        result = []
        i = 0
        while i < len(parts):
            # Check if current position matches any mapping
            for key, pattern in self.KEY_MAPPINGS.items():
                if self._matches_pattern(parts, i, pattern):
                    result.append(key)
                    i += len(pattern)
                    break
            else:
                result.append(parts[i])
                i += 1
        return result
```

### 4. Duplicate Code Pattern in File Loading

**Issue:** Each file type has similar loading pattern

**Suggested Refactoring - Template Method:**

```python
class BaseFileLoader(ABC):
    def load_file(self, file_path: Path, table_name: str, 
                  engine: Engine, progress: Progress) -> None:
        total_rows = self._get_total_rows(file_path)
        task = progress.add_task(f"Loading {file_path.name}", 
                               total=total_rows)
        
        first_chunk = True
        for chunk_df in self._read_chunks(file_path):
            if first_chunk:
                self._handle_first_chunk(chunk_df, table_name, engine)
                first_chunk = False
            else:
                self._append_chunk(chunk_df, table_name, engine)
            
            progress.update(task, advance=len(chunk_df))
    
    @abstractmethod
    def _get_total_rows(self, file_path: Path) -> int:
        pass
    
    @abstractmethod
    def _read_chunks(self, file_path: Path) -> Iterator[pd.DataFrame]:
        pass
```

### 5. Parameter Object Pattern for High Parameter Count Methods

**Issue:** Many methods have 6+ parameters

**Suggested Refactoring:**

```python
@dataclass
class RegistrationParams:
    name: str
    path: Path
    auto_detect: bool = True
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    target_column: Optional[str] = None
    id_columns: Optional[List[str]] = None
    problem_type: Optional[str] = None
    force: bool = False
    generate_features: bool = True

# Usage
def register_dataset(self, params: RegistrationParams) -> DatasetInfo:
    registrar = DatasetRegistrar(self.manager)
    return registrar.register(params)
```

## Code Smell Severity Rankings

### Critical (Must Fix) 🔴🔴🔴
1. **DatasetRegistrar._load_data_files()** - 334 lines, complexity ~50
2. **DatasetRegistrar.register()** - 163 lines, complexity ~30

### High (Should Fix) 🔴🔴
1. **DatasetRegistrar._detect_column_types_with_profiling()** - 176 lines
2. **ConfigManager._apply_environment_variables()** - Complex parsing logic

### Medium (Nice to Fix) 🔴
1. **DatasetRegistrar._compute_initial_statistics()** - 142 lines
2. **Methods with >6 parameters** across all classes

### Low (Consider Fixing) ⚠️
1. **Progress tracking mixed with business logic**
2. **Dynamic module loading complexity**

## Refactoring Timeline Recommendation

### Sprint 1 (Urgent)
- Extract file loaders from _load_data_files()
- Implement Strategy pattern for file handling

### Sprint 2 (High Priority)
- Refactor register() method using Pipeline pattern
- Create parameter objects for complex method signatures

### Sprint 3 (Medium Priority)
- Simplify environment variable parsing
- Extract progress tracking into decorators

### Sprint 4 (Polish)
- Add comprehensive unit tests for refactored components
- Document new patterns and architectures