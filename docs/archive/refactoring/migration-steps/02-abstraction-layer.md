# Step 2: Abstraction Layer Creation

## Overview

Create interfaces and adapter classes to enable parallel implementation without modifying existing code. This layer provides the foundation for gradual migration.

> ⚠️ **CRITICAL PREREQUISITE**: You MUST complete [Step 1.5: API Analysis](01.5-api-analysis.md) before starting this step. The interfaces must be based on ACTUAL usage, not idealistic design.

## Duration

2 weeks (Weeks 4-5)

## Prerequisites

- ✅ Step 1.5 (API Analysis) MUST be complete
- ✅ Have complete API usage reports for all components
- ✅ Generated interfaces from actual usage analysis
- ✅ Compatibility test suite ready

## Objectives

1. Define Protocol interfaces based on **actual API usage analysis**
2. Create adapter classes wrapping existing implementations
3. Ensure **100% method coverage** from usage analysis
4. Introduce dependency injection points
5. Update type hints throughout the codebase
6. Validate adapters match original behavior

## Design Principles

- **No Breaking Changes**: Existing code continues to work
- **Type Safety**: Use Protocol classes for compile-time checks
- **Testability**: Every adapter must be independently testable
- **Minimal Performance Impact**: Adapters should add < 1% overhead

## Detailed Steps

### Week 3: Interface Definition and Core Adapters

#### Day 1-2: Storage Backend Interfaces

##### 1.1 Create Storage Protocol
```python
# ⚠️ WARNING: This is an INCOMPLETE example!
# You MUST use the interface generated from Step 1.5 API Analysis
# See: generated_storage_interface.py from the analysis

# Create: src/mdm/interfaces/storage.py
from typing import Protocol, Any, Dict, List, Optional, runtime_checkable
from sqlalchemy.engine import Engine
import pandas as pd

@runtime_checkable
class IStorageBackend(Protocol):
    """Storage backend interface - MUST include ALL methods from usage analysis"""
    
    # ✅ Methods found in usage analysis (partial list):
    def get_engine(self, database_path: str) -> Engine:
        """Get SQLAlchemy engine - Used 11 times"""
        ...
    
    def create_table_from_dataframe(self, df: pd.DataFrame, table_name: str, 
                                   engine: Engine, if_exists: str = "fail") -> None:
        """Create table from DataFrame - Used 10 times"""
        ...
    
    def query(self, query: str) -> pd.DataFrame:
        """Execute SQL query - Used 9 times"""
        ...
    
    def read_table_to_dataframe(self, table_name: str, engine: Engine, 
                               limit: Optional[int] = None) -> pd.DataFrame:
        """Read table to DataFrame - Used 7 times"""
        ...
    
    def close_connections(self) -> None:
        """Close all connections - Used 7 times"""
        ...
    
    # ⚠️ CRITICAL: Add ALL 14 methods found in usage analysis!
    # Missing methods WILL cause runtime failures
    
    # ❌ These methods were NOT found in usage analysis:
    # def load_data() - This was an idealistic design, not actual usage
    # def save_data() - This was wishful thinking, not real usage


@runtime_checkable
class IConnectionPool(Protocol):
    """Connection pool interface for stateless backends"""
    
    def get_connection(self, dataset_name: str) -> Any:
        """Get connection from pool"""
        ...
    
    def release_connection(self, connection: Any) -> None:
        """Return connection to pool"""
        ...
    
    def close_all(self) -> None:
        """Close all connections"""
        ...
```

##### 1.2 Create Storage Adapters
```python
# Create: src/mdm/adapters/storage_adapters.py
from typing import Any, Dict, Optional
import pandas as pd
from sqlalchemy.engine import Engine

from ..interfaces.storage import IStorageBackend
from ..storage.sqlite import SQLiteBackend
from ..storage.duckdb import DuckDBBackend
from ..storage.postgresql import PostgreSQLBackend


class StorageAdapter:
    """Base adapter with common functionality"""
    
    def __init__(self, backend: Any):
        self._backend = backend
        self._call_count = {}  # For metrics
    
    def _track_call(self, method: str) -> None:
        """Track method calls for metrics"""
        self._call_count[method] = self._call_count.get(method, 0) + 1


class SQLiteAdapter(StorageAdapter, IStorageBackend):
    """Adapter for SQLite backend"""
    
    def __init__(self):
        super().__init__(SQLiteBackend())
    
    def get_engine(self) -> Engine:
        self._track_call("get_engine")
        return self._backend.get_engine()
    
    def create_dataset(self, dataset_name: str, config: Dict[str, Any]) -> None:
        self._track_call("create_dataset")
        self._backend.create_dataset(dataset_name, config)
    
    def dataset_exists(self, dataset_name: str) -> bool:
        self._track_call("dataset_exists")
        return self._backend.dataset_exists(dataset_name)
    
    def drop_dataset(self, dataset_name: str) -> None:
        self._track_call("drop_dataset")
        self._backend.drop_dataset(dataset_name)
    
    def load_data(self, dataset_name: str, table_name: str = "data") -> pd.DataFrame:
        self._track_call("load_data")
        return self._backend.load_data(dataset_name, table_name)
    
    def save_data(self, dataset_name: str, data: pd.DataFrame, 
                  table_name: str = "data", if_exists: str = "replace") -> None:
        self._track_call("save_data")
        self._backend.save_data(dataset_name, data, table_name, if_exists)
    
    def get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        self._track_call("get_metadata")
        return self._backend.get_metadata(dataset_name)
    
    def update_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        self._track_call("update_metadata")
        self._backend.update_metadata(dataset_name, metadata)
    
    def close(self) -> None:
        self._track_call("close")
        # SQLite backend doesn't have explicit close
        if hasattr(self._backend, '_engine') and self._backend._engine:
            self._backend._engine.dispose()


# Similar adapters for DuckDB and PostgreSQL
class DuckDBAdapter(StorageAdapter, IStorageBackend):
    # Implementation similar to SQLiteAdapter
    pass


class PostgreSQLAdapter(StorageAdapter, IStorageBackend):
    # Implementation similar to SQLiteAdapter
    pass
```

#### Day 3-4: Feature Engineering Interfaces

##### 1.3 Create Feature Protocol
```python
# Create: src/mdm/interfaces/features.py
from typing import Protocol, Dict, Any, List, Optional
import pandas as pd


@runtime_checkable
class IFeatureTransformer(Protocol):
    """Individual feature transformer interface"""
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit transformer to data"""
        ...
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        ...
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        ...
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names"""
        ...


@runtime_checkable
class IFeatureGenerator(Protocol):
    """Feature generation interface"""
    
    def generate_features(self, data: pd.DataFrame, 
                         config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate all features for dataset"""
        ...
    
    def generate_numeric_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate numeric features"""
        ...
    
    def generate_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate categorical features"""
        ...
    
    def generate_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate datetime features"""
        ...
    
    def generate_text_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate text features"""
        ...
```

##### 1.4 Create Feature Adapters
```python
# Create: src/mdm/adapters/feature_adapters.py
from typing import Dict, Any, List, Optional
import pandas as pd

from ..interfaces.features import IFeatureGenerator
from ..features.generator import FeatureGenerator


class FeatureGeneratorAdapter(IFeatureGenerator):
    """Adapter for existing feature generator"""
    
    def __init__(self):
        self._generator = FeatureGenerator()
        self._metrics = {
            "features_generated": 0,
            "processing_time": 0.0
        }
    
    def generate_features(self, data: pd.DataFrame, 
                         config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate all features with metrics tracking"""
        import time
        start_time = time.time()
        
        result = self._generator.generate_features(data, config)
        
        self._metrics["processing_time"] += time.time() - start_time
        self._metrics["features_generated"] += len(result.columns) - len(data.columns)
        
        return result
    
    def generate_numeric_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._generator.generate_numeric_features(data)
    
    def generate_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._generator.generate_categorical_features(data)
    
    def generate_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self._generator, 'generate_datetime_features'):
            return self._generator.generate_datetime_features(data)
        # Fallback for older versions
        return pd.DataFrame()
    
    def generate_text_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self._generator, 'generate_text_features'):
            return self._generator.generate_text_features(data)
        return pd.DataFrame()
```

#### Day 5: Dataset Management Interfaces

##### 1.5 Create Dataset Protocols
```python
# Create: src/mdm/interfaces/dataset.py
from typing import Protocol, Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd


@runtime_checkable
class IDatasetRegistrar(Protocol):
    """Dataset registration interface"""
    
    def register(self, name: str, path: str, 
                target: Optional[str] = None,
                problem_type: Optional[str] = None,
                force: bool = False) -> Dict[str, Any]:
        """Register a new dataset"""
        ...
    
    def validate_dataset_name(self, name: str) -> None:
        """Validate dataset name"""
        ...
    
    def detect_structure(self, path: str) -> Dict[str, Any]:
        """Auto-detect dataset structure"""
        ...


@runtime_checkable
class IDatasetManager(Protocol):
    """Dataset management interface"""
    
    def list_datasets(self, limit: Optional[int] = None,
                     sort_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all datasets"""
        ...
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get dataset information"""
        ...
    
    def remove_dataset(self, name: str, force: bool = False) -> None:
        """Remove dataset"""
        ...
    
    def update_dataset(self, name: str, updates: Dict[str, Any]) -> None:
        """Update dataset metadata"""
        ...
    
    def export_dataset(self, name: str, output_path: str,
                      format: str = "csv", compression: Optional[str] = None) -> str:
        """Export dataset"""
        ...
```

### Week 4: Dependency Injection and Integration

#### Day 6-7: Dependency Injection Framework

##### 2.1 Create DI Container
```python
# Create: src/mdm/core/container.py
from typing import Dict, Type, Any, Callable, Optional
from functools import lru_cache
import inspect

from ..interfaces.storage import IStorageBackend
from ..interfaces.features import IFeatureGenerator
from ..interfaces.dataset import IDatasetRegistrar, IDatasetManager


class DIContainer:
    """Simple dependency injection container"""
    
    def __init__(self):
        self._services: Dict[Type, Callable[[], Any]] = {}
        self._singletons: Dict[Type, Any] = {}
        self._config: Dict[str, Any] = {}
    
    def register(self, interface: Type, factory: Callable[[], Any], 
                 singleton: bool = False) -> None:
        """Register a service factory"""
        self._services[interface] = factory
        if singleton:
            # Mark for singleton instantiation
            self._singletons[interface] = None
    
    def get(self, interface: Type) -> Any:
        """Get service instance"""
        if interface not in self._services:
            raise ValueError(f"No service registered for {interface}")
        
        # Check if singleton
        if interface in self._singletons:
            if self._singletons[interface] is None:
                self._singletons[interface] = self._services[interface]()
            return self._singletons[interface]
        
        # Create new instance
        return self._services[interface]()
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Set configuration"""
        self._config.update(config)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)


# Global container instance
container = DIContainer()


def configure_container(config: Dict[str, Any]) -> None:
    """Configure the DI container with appropriate implementations"""
    from ..adapters.storage_adapters import SQLiteAdapter, DuckDBAdapter, PostgreSQLAdapter
    from ..adapters.feature_adapters import FeatureGeneratorAdapter
    from ..adapters.dataset_adapters import DatasetRegistrarAdapter, DatasetManagerAdapter
    
    # Configure based on settings
    backend_type = config.get("database", {}).get("default_backend", "sqlite")
    use_new_backend = config.get("refactoring", {}).get("use_new_backend", False)
    
    # Register storage backend
    if use_new_backend:
        # New implementation (to be created)
        pass
    else:
        # Use adapters for existing implementations
        if backend_type == "sqlite":
            container.register(IStorageBackend, SQLiteAdapter, singleton=True)
        elif backend_type == "duckdb":
            container.register(IStorageBackend, DuckDBAdapter, singleton=True)
        elif backend_type == "postgresql":
            container.register(IStorageBackend, PostgreSQLAdapter, singleton=True)
    
    # Register other services
    container.register(IFeatureGenerator, FeatureGeneratorAdapter, singleton=True)
    container.register(IDatasetRegistrar, DatasetRegistrarAdapter, singleton=False)
    container.register(IDatasetManager, DatasetManagerAdapter, singleton=True)
    
    # Store configuration
    container.configure(config)


# Decorator for dependency injection
def inject(func: Callable) -> Callable:
    """Decorator to inject dependencies"""
    sig = inspect.signature(func)
    
    def wrapper(*args, **kwargs):
        # Inject missing parameters
        for param_name, param in sig.parameters.items():
            if param_name not in kwargs and param.annotation != param.empty:
                # Try to inject from container
                try:
                    kwargs[param_name] = container.get(param.annotation)
                except ValueError:
                    pass  # Not a registered service
        
        return func(*args, **kwargs)
    
    return wrapper
```

##### 2.2 Update Entry Points
```python
# Update: src/mdm/cli/dataset.py
from typing import Optional
from typer import Typer, Option
from rich.console import Console

from ..core.container import inject, container
from ..interfaces.dataset import IDatasetRegistrar, IDatasetManager

app = Typer()
console = Console()


@app.command()
@inject
def register(
    name: str,
    path: str,
    target: Optional[str] = Option(None, "--target", "-t"),
    problem_type: Optional[str] = Option(None, "--problem-type", "-p"),
    force: bool = Option(False, "--force", "-f"),
    registrar: IDatasetRegistrar = None  # Will be injected
):
    """Register a new dataset with dependency injection"""
    try:
        result = registrar.register(name, path, target, problem_type, force)
        console.print(f"[green]Dataset '{name}' registered successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
```

#### Day 8-9: Type Hint Updates

##### 2.3 Update Type Hints Throughout Codebase
```python
# Update function signatures to use interfaces
# Before:
def process_dataset(backend: SQLiteBackend, generator: FeatureGenerator) -> pd.DataFrame:
    pass

# After:
def process_dataset(backend: IStorageBackend, generator: IFeatureGenerator) -> pd.DataFrame:
    pass
```

##### 2.4 Create Type Checking Script
```bash
#!/bin/bash
# scripts/check_types.sh

echo "Running type checking with mypy..."

# Check interfaces
mypy src/mdm/interfaces/ --strict

# Check adapters
mypy src/mdm/adapters/ --strict

# Check updated modules
mypy src/mdm/cli/ --strict
mypy src/mdm/core/ --strict

# Generate type coverage report
mypy src/mdm/ --html-report ./type-coverage
echo "Type coverage report generated in ./type-coverage/"
```

#### Day 10: Validation and Testing

##### 2.5 Create Adapter Validation Tests
```python
# Create: tests/unit/test_adapters.py
import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd

from mdm.interfaces.storage import IStorageBackend
from mdm.adapters.storage_adapters import SQLiteAdapter, DuckDBAdapter
from mdm.storage.sqlite import SQLiteBackend


class TestStorageAdapters:
    def test_sqlite_adapter_implements_interface(self):
        """Verify SQLiteAdapter implements IStorageBackend"""
        adapter = SQLiteAdapter()
        assert isinstance(adapter, IStorageBackend)
    
    def test_adapter_behavior_matches_original(self):
        """Ensure adapter behaves identically to original"""
        # Create test data
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        # Test with original
        original = SQLiteBackend()
        original.create_dataset("test_original", {})
        original.save_data("test_original", test_data)
        original_result = original.load_data("test_original")
        
        # Test with adapter
        adapter = SQLiteAdapter()
        adapter.create_dataset("test_adapter", {})
        adapter.save_data("test_adapter", test_data)
        adapter_result = adapter.load_data("test_adapter")
        
        # Compare results
        pd.testing.assert_frame_equal(original_result, adapter_result)
        
        # Cleanup
        original.drop_dataset("test_original")
        adapter.drop_dataset("test_adapter")
    
    def test_adapter_metrics_tracking(self):
        """Verify adapter tracks method calls"""
        adapter = SQLiteAdapter()
        
        # Make some calls
        adapter.dataset_exists("test")
        adapter.dataset_exists("test")
        adapter.get_engine()
        
        # Check metrics
        assert adapter._call_count["dataset_exists"] == 2
        assert adapter._call_count["get_engine"] == 1


class TestFeatureAdapters:
    def test_feature_adapter_compatibility(self):
        """Test feature adapter maintains compatibility"""
        from mdm.adapters.feature_adapters import FeatureGeneratorAdapter
        from mdm.features.generator import FeatureGenerator
        
        # Test data
        data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'C']
        })
        
        # Compare original and adapter results
        original = FeatureGenerator()
        adapter = FeatureGeneratorAdapter()
        
        original_features = original.generate_features(data)
        adapter_features = adapter.generate_features(data)
        
        # Should produce identical features
        assert set(original_features.columns) == set(adapter_features.columns)
```

##### 2.6 Performance Validation
```python
# Create: tests/benchmarks/test_adapter_overhead.py
import pytest
import time
import statistics
from typing import List

from mdm.storage.sqlite import SQLiteBackend
from mdm.adapters.storage_adapters import SQLiteAdapter


class TestAdapterPerformance:
    @pytest.mark.benchmark
    def test_adapter_overhead(self, benchmark_data):
        """Measure adapter overhead"""
        iterations = 100
        
        # Measure original backend
        original = SQLiteBackend()
        original_times: List[float] = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            original.dataset_exists("test")
            original_times.append(time.perf_counter() - start)
        
        # Measure adapter
        adapter = SQLiteAdapter()
        adapter_times: List[float] = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            adapter.dataset_exists("test")
            adapter_times.append(time.perf_counter() - start)
        
        # Calculate overhead
        original_avg = statistics.mean(original_times)
        adapter_avg = statistics.mean(adapter_times)
        overhead_percent = ((adapter_avg - original_avg) / original_avg) * 100
        
        print(f"Original avg: {original_avg:.6f}s")
        print(f"Adapter avg: {adapter_avg:.6f}s")
        print(f"Overhead: {overhead_percent:.2f}%")
        
        # Assert overhead is less than 1%
        assert overhead_percent < 1.0, f"Adapter overhead {overhead_percent:.2f}% exceeds 1%"
```

## Validation Checklist

### Week 3 Completion
- [ ] All interface protocols defined
- [ ] Storage adapters implemented and tested
- [ ] Feature adapters implemented and tested
- [ ] Dataset adapters implemented and tested
- [ ] All adapters pass interface compliance tests

### Week 4 Completion
- [ ] DI container implemented and tested
- [ ] All entry points updated to use DI
- [ ] Type hints updated throughout codebase
- [ ] Mypy type checking passing
- [ ] Adapter overhead < 1% verified

## Success Criteria

- **100% interface coverage** for major components
- **All adapters tested** with original implementations
- **Type safety** verified with mypy
- **Performance overhead < 1%** for all adapters
- **No breaking changes** to existing code

## Integration Guide

### Using the Abstraction Layer

```python
# Example: Using DI in new code
from mdm.core.container import container, inject
from mdm.interfaces.storage import IStorageBackend

@inject
def process_data(dataset_name: str, backend: IStorageBackend = None):
    """Function automatically receives injected backend"""
    data = backend.load_data(dataset_name)
    # Process data
    return data

# Configure container at startup
from mdm.core.container import configure_container
configure_container(config)

# Use the function - backend is automatically injected
result = process_data("my_dataset")
```

### Gradual Migration Pattern

```python
# Support both old and new patterns during migration
def get_backend(use_di: bool = False):
    if use_di:
        from mdm.core.container import container
        from mdm.interfaces.storage import IStorageBackend
        return container.get(IStorageBackend)
    else:
        from mdm.storage import get_storage_backend
        return get_storage_backend()
```

## Troubleshooting

### Issue: Circular imports with interfaces
```python
# Solution: Use TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..interfaces.storage import IStorageBackend

# Use string annotations
def process(backend: "IStorageBackend") -> None:
    pass
```

### Issue: Protocol not recognized at runtime
```python
# Solution: Use @runtime_checkable decorator
from typing import Protocol, runtime_checkable

@runtime_checkable
class IStorageBackend(Protocol):
    pass
```

### Issue: DI container not finding services
```python
# Debug with:
from mdm.core.container import container

# List all registered services
print("Registered services:", container._services.keys())

# Check configuration
print("Container config:", container._config)
```

## Next Steps

With the abstraction layer in place, proceed to [03-parallel-setup.md](03-parallel-setup.md) to set up the parallel development environment.

## Notes

- Keep interfaces minimal and focused
- Document any deviations from planned interfaces
- Monitor adapter performance throughout migration
- Use type checking to catch integration issues early