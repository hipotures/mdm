# Dataset Registrar Refactoring Guide

## Overview

The `DatasetRegistrar` is currently a monolithic god class with 12 tightly coupled steps, mixing concerns and storing state in instance variables. This guide details its transformation into a flexible, testable pipeline architecture.

## Current Problems

### 1. God Class with Too Many Responsibilities
```python
# CURRENT - 1000+ lines doing everything
class DatasetRegistrar:
    def register(self, name, dataset_path, target_column, problem_type, ...):
        # Step 1: Validate dataset name
        # Step 2: Check if dataset already exists  
        # Step 3: Validate path
        # Step 4: Auto-detect dataset structure
        # Step 5: Discover data files
        # Step 6: Detect ID columns and target
        # Step 7: Create storage backend
        # Step 8: Load data files
        # Step 9: Detect column types
        # Step 10: Generate features
        # Step 11: Compute initial statistics
        # Step 12: Save configuration
```

### 2. State Management Issues
```python
# CURRENT - Instance variables for temporary state
self._detected_column_types = {}
self._detected_datetime_columns = []
self._detected_id_columns = []
self._target_column = None
```

### 3. Mixed Concerns
- Progress tracking mixed with business logic
- Database operations mixed with validation
- Feature generation embedded in registration

### 4. Poor Error Recovery
- No transaction management
- Partial state on failure
- No rollback capability

## Target Architecture

### 1. Command Pattern for Registration Steps
```python
# NEW - Each step as a command
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from enum import Enum

class StepStatus(Enum):
    """Status of a registration step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class StepResult:
    """Result of a registration step."""
    status: StepStatus
    data: Dict[str, Any]
    error: Optional[Exception] = None
    message: Optional[str] = None

@dataclass
class RegistrationContext:
    """Context passed through registration pipeline."""
    dataset_name: str
    dataset_path: Path
    target_column: Optional[str]
    problem_type: Optional[str]
    id_columns: Optional[List[str]]
    description: Optional[str]
    tags: List[str]
    force: bool
    generate_features: bool
    
    # Step results
    results: Dict[str, StepResult] = field(default_factory=dict)
    
    # Discovered data
    files: Dict[str, Path] = field(default_factory=dict)
    column_types: Dict[str, ColumnType] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class RegistrationStep(ABC):
    """Abstract base for registration steps."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Step name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Step description."""
        pass
    
    @abstractmethod
    def can_execute(self, context: RegistrationContext) -> bool:
        """Check if step can execute."""
        pass
    
    @abstractmethod
    def execute(self, context: RegistrationContext) -> StepResult:
        """Execute the step."""
        pass
    
    @abstractmethod
    def rollback(self, context: RegistrationContext) -> None:
        """Rollback step changes."""
        pass
```

### 2. Implementation of Registration Steps
```python
# NEW - Individual step implementations
class ValidateNameStep(RegistrationStep):
    """Validates dataset name."""
    
    name = "validate_name"
    description = "Validate dataset name"
    
    def can_execute(self, context: RegistrationContext) -> bool:
        return bool(context.dataset_name)
    
    def execute(self, context: RegistrationContext) -> StepResult:
        try:
            # Validation logic
            if not re.match(r'^[a-zA-Z0-9_-]+$', context.dataset_name):
                raise ValueError(
                    "Dataset name can only contain letters, numbers, underscores, and dashes"
                )
            
            return StepResult(
                status=StepStatus.COMPLETED,
                data={"validated_name": context.dataset_name}
            )
        except Exception as e:
            return StepResult(
                status=StepStatus.FAILED,
                data={},
                error=e
            )
    
    def rollback(self, context: RegistrationContext) -> None:
        # Nothing to rollback for validation
        pass

class CheckExistenceStep(RegistrationStep):
    """Checks if dataset already exists."""
    
    name = "check_existence"
    description = "Check if dataset exists"
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
    
    def can_execute(self, context: RegistrationContext) -> bool:
        return context.results.get("validate_name", StepResult(StepStatus.PENDING, {})).status == StepStatus.COMPLETED
    
    def execute(self, context: RegistrationContext) -> StepResult:
        try:
            exists = self.dataset_manager.dataset_exists(context.dataset_name)
            
            if exists and not context.force:
                raise DatasetError(f"Dataset '{context.dataset_name}' already exists")
            
            return StepResult(
                status=StepStatus.COMPLETED,
                data={"exists": exists, "will_overwrite": exists and context.force}
            )
        except Exception as e:
            return StepResult(status=StepStatus.FAILED, data={}, error=e)
    
    def rollback(self, context: RegistrationContext) -> None:
        pass

class DiscoverFilesStep(RegistrationStep):
    """Discovers data files in dataset path."""
    
    name = "discover_files"
    description = "Discover data files"
    
    def can_execute(self, context: RegistrationContext) -> bool:
        return context.dataset_path.exists()
    
    def execute(self, context: RegistrationContext) -> StepResult:
        try:
            files = self._discover_files(context.dataset_path)
            context.files = files
            
            return StepResult(
                status=StepStatus.COMPLETED,
                data={"files": {k: str(v) for k, v in files.items()}}
            )
        except Exception as e:
            return StepResult(status=StepStatus.FAILED, data={}, error=e)
    
    def _discover_files(self, path: Path) -> Dict[str, Path]:
        """Discover train/test/validation files."""
        files = {}
        
        if path.is_file():
            files["train"] = path
        else:
            # Look for standard file names
            for file_type in ["train", "test", "validation"]:
                for ext in [".csv", ".parquet", ".json"]:
                    file_path = path / f"{file_type}{ext}"
                    if file_path.exists():
                        files[file_type] = file_path
        
        return files
    
    def rollback(self, context: RegistrationContext) -> None:
        context.files.clear()

class CreateDatabaseStep(RegistrationStep):
    """Creates database for dataset."""
    
    name = "create_database"
    description = "Create dataset database"
    
    def __init__(self, backend_factory: BackendFactory, path_manager: PathManager):
        self.backend_factory = backend_factory
        self.path_manager = path_manager
    
    def can_execute(self, context: RegistrationContext) -> bool:
        return len(context.files) > 0
    
    def execute(self, context: RegistrationContext) -> StepResult:
        try:
            # Create dataset directory
            dataset_path = self.path_manager.dataset_path(context.dataset_name)
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Get database path
            db_path = self.path_manager.dataset_db_path(context.dataset_name)
            
            # Store in context
            context.metadata["database_path"] = str(db_path)
            
            return StepResult(
                status=StepStatus.COMPLETED,
                data={"database_path": str(db_path)}
            )
        except Exception as e:
            return StepResult(status=StepStatus.FAILED, data={}, error=e)
    
    def rollback(self, context: RegistrationContext) -> None:
        # Remove created database
        if "database_path" in context.metadata:
            db_path = Path(context.metadata["database_path"])
            if db_path.exists():
                db_path.unlink()
```

### 3. Registration Pipeline
```python
# NEW - Pipeline to orchestrate steps
class RegistrationPipeline:
    """Manages dataset registration pipeline."""
    
    def __init__(self, steps: List[RegistrationStep]):
        self.steps = steps
        self.hooks = {
            'before_step': [],
            'after_step': [],
            'on_error': [],
            'on_complete': []
        }
    
    def add_hook(self, event: str, hook: Callable) -> None:
        """Add event hook."""
        if event in self.hooks:
            self.hooks[event].append(hook)
    
    def execute(self, context: RegistrationContext) -> RegistrationContext:
        """Execute registration pipeline."""
        completed_steps = []
        
        try:
            for step in self.steps:
                # Before step hooks
                for hook in self.hooks['before_step']:
                    hook(step, context)
                
                # Check if step can execute
                if not step.can_execute(context):
                    result = StepResult(
                        status=StepStatus.SKIPPED,
                        data={},
                        message="Preconditions not met"
                    )
                else:
                    # Execute step
                    result = step.execute(context)
                
                # Store result
                context.results[step.name] = result
                
                # After step hooks
                for hook in self.hooks['after_step']:
                    hook(step, result, context)
                
                # Check for failure
                if result.status == StepStatus.FAILED:
                    raise PipelineError(f"Step '{step.name}' failed: {result.error}")
                
                if result.status == StepStatus.COMPLETED:
                    completed_steps.append(step)
            
            # On complete hooks
            for hook in self.hooks['on_complete']:
                hook(context)
            
            return context
            
        except Exception as e:
            # On error hooks
            for hook in self.hooks['on_error']:
                hook(e, context)
            
            # Rollback completed steps
            self._rollback(completed_steps, context)
            raise
    
    def _rollback(self, completed_steps: List[RegistrationStep], context: RegistrationContext) -> None:
        """Rollback completed steps in reverse order."""
        for step in reversed(completed_steps):
            try:
                step.rollback(context)
            except Exception as e:
                # Log rollback failure but continue
                logger.error(f"Rollback failed for step '{step.name}': {e}")
```

### 4. Progress Tracking Separated
```python
# NEW - Separate progress tracking
class ProgressTracker:
    """Tracks registration progress."""
    
    def __init__(self, console: Console):
        self.console = console
        self.progress = None
        self.task_id = None
    
    def on_pipeline_start(self, context: RegistrationContext) -> None:
        """Called when pipeline starts."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
        self.progress.start()
        
        total_steps = len(context.results)
        self.task_id = self.progress.add_task(
            f"Registering {context.dataset_name}",
            total=total_steps
        )
    
    def on_step_complete(self, step: RegistrationStep, result: StepResult, context: RegistrationContext) -> None:
        """Called after each step."""
        if self.progress and self.task_id is not None:
            self.progress.update(
                self.task_id,
                advance=1,
                description=f"Completed: {step.description}"
            )
    
    def on_pipeline_complete(self, context: RegistrationContext) -> None:
        """Called when pipeline completes."""
        if self.progress:
            self.progress.stop()
```

### 5. Refactored DatasetRegistrar
```python
# NEW - Clean registrar using pipeline
class DatasetRegistrar:
    """Manages dataset registration."""
    
    def __init__(
        self,
        pipeline: RegistrationPipeline,
        progress_tracker: Optional[ProgressTracker] = None
    ):
        self.pipeline = pipeline
        self.progress_tracker = progress_tracker
        
        # Set up progress hooks if tracker provided
        if progress_tracker:
            pipeline.add_hook('before_step', lambda s, c: progress_tracker.on_step_start(s, c))
            pipeline.add_hook('after_step', lambda s, r, c: progress_tracker.on_step_complete(s, r, c))
            pipeline.add_hook('on_complete', lambda c: progress_tracker.on_pipeline_complete(c))
    
    def register(
        self,
        name: str,
        dataset_path: Union[str, Path],
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        force: bool = False,
        generate_features: bool = True
    ) -> DatasetInfo:
        """Register a new dataset."""
        # Create context
        context = RegistrationContext(
            dataset_name=name,
            dataset_path=Path(dataset_path),
            target_column=target_column,
            problem_type=problem_type,
            id_columns=id_columns,
            description=description,
            tags=tags or [],
            force=force,
            generate_features=generate_features
        )
        
        # Execute pipeline
        context = self.pipeline.execute(context)
        
        # Build dataset info from context
        return self._build_dataset_info(context)
    
    def _build_dataset_info(self, context: RegistrationContext) -> DatasetInfo:
        """Build DatasetInfo from context."""
        return DatasetInfo(
            name=context.dataset_name,
            description=context.description,
            source=str(context.dataset_path),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tables=context.metadata.get("tables", {}),
            target_column=context.target_column,
            id_columns=context.metadata.get("id_columns", []),
            problem_type=context.problem_type,
            tags=context.tags,
            database=context.metadata.get("database", {}),
            metadata=context.metadata
        )
```

## Migration Strategy

### Phase 1: Create New Architecture
1. Implement `RegistrationStep` interface
2. Create individual step classes
3. Build `RegistrationPipeline`
4. Separate `ProgressTracker`

### Phase 2: Incremental Migration
```python
# Migrate one step at a time
class LegacyStepAdapter(RegistrationStep):
    """Adapts old method to new step interface."""
    
    def __init__(self, registrar, method_name: str):
        self.registrar = registrar
        self.method_name = method_name
    
    def execute(self, context: RegistrationContext) -> StepResult:
        # Call old method
        method = getattr(self.registrar, self.method_name)
        result = method(context.dataset_name, context.dataset_path)
        
        return StepResult(
            status=StepStatus.COMPLETED,
            data={"legacy_result": result}
        )
```

### Phase 3: Testing
```python
# Test individual steps
def test_validate_name_step():
    step = ValidateNameStep()
    context = RegistrationContext(dataset_name="test_123")
    
    result = step.execute(context)
    
    assert result.status == StepStatus.COMPLETED
    assert result.data["validated_name"] == "test_123"

# Test pipeline
def test_registration_pipeline():
    steps = [
        ValidateNameStep(),
        CheckExistenceStep(mock_manager),
        DiscoverFilesStep()
    ]
    
    pipeline = RegistrationPipeline(steps)
    context = RegistrationContext(
        dataset_name="test",
        dataset_path=Path("/tmp/test.csv")
    )
    
    result = pipeline.execute(context)
    
    assert all(r.status == StepStatus.COMPLETED for r in result.results.values())
```

## Benefits of New Architecture

### 1. Single Responsibility
Each step has one clear purpose

### 2. Testability
```python
# Easy to test individual steps
step = ValidateNameStep()
result = step.execute(context)
assert result.status == StepStatus.COMPLETED
```

### 3. Flexibility
```python
# Easy to add/remove/reorder steps
pipeline = RegistrationPipeline([
    ValidateNameStep(),
    CustomValidationStep(),  # New step
    CheckExistenceStep(),
    # Skip some steps for special case
])
```

### 4. Error Recovery
```python
# Automatic rollback on failure
try:
    pipeline.execute(context)
except PipelineError:
    # All completed steps rolled back
    pass
```

### 5. Progress Tracking
```python
# Clean separation of concerns
tracker = ProgressTracker(console)
pipeline.add_hook('after_step', tracker.on_step_complete)
```

## Custom Steps

### Adding Custom Validation
```python
class CustomValidationStep(RegistrationStep):
    """Custom validation for special requirements."""
    
    name = "custom_validation"
    description = "Validate custom requirements"
    
    def execute(self, context: RegistrationContext) -> StepResult:
        # Custom validation logic
        if context.dataset_name.startswith("temp_"):
            return StepResult(
                status=StepStatus.FAILED,
                data={},
                error=ValueError("Temporary datasets not allowed")
            )
        
        return StepResult(status=StepStatus.COMPLETED, data={})
```

### Conditional Steps
```python
class ConditionalStep(RegistrationStep):
    """Step that only runs under certain conditions."""
    
    def can_execute(self, context: RegistrationContext) -> bool:
        # Only run for CSV files
        return any(f.suffix == ".csv" for f in context.files.values())
```

## Success Criteria

1. **No God Class**: Registrar under 100 lines
2. **Testable Steps**: Each step independently testable
3. **Flexible Pipeline**: Easy to modify registration flow
4. **Error Recovery**: Full rollback capability
5. **Clean Separation**: Progress tracking separate from logic
6. **Stateless**: No instance variables for temporary state