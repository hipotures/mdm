# Testing Strategy for MDM Refactoring

## Overview

This document outlines the comprehensive testing strategy for the MDM refactoring project. It covers unit testing, integration testing, migration testing, and performance benchmarking to ensure a smooth transition.

## Testing Principles

### 1. Test Pyramid
```
         /\
        /  \  E2E Tests (10%)
       /----\
      /      \ Integration Tests (30%)
     /--------\
    /          \ Unit Tests (60%)
   /____________\
```

### 2. Test Coverage Goals
- **Unit Tests**: 95%+ coverage
- **Integration Tests**: Critical paths covered
- **E2E Tests**: User scenarios covered
- **Performance**: No regression from baseline

### 3. Testing Standards
- **Fast**: Unit tests < 1ms each
- **Isolated**: No test dependencies
- **Repeatable**: Same result every run
- **Self-validating**: Clear pass/fail
- **Timely**: Written with code

## Unit Testing Strategy

### 1. Storage Backend Tests
```python
# tests/unit/storage/test_sqlite_backend.py
import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
from mdm.storage.backends.sqlite import SQLiteBackend

class TestSQLiteBackend:
    """Unit tests for SQLite backend."""
    
    @pytest.fixture
    def backend(self):
        """Create SQLite backend instance."""
        return SQLiteBackend()
    
    @pytest.fixture
    def mock_connection(self):
        """Create mock connection."""
        conn = Mock()
        conn.execute = Mock()
        conn.cursor = Mock()
        return conn
    
    def test_create_table_basic(self, backend, mock_connection):
        """Test basic table creation."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10.5, 20.5, 30.5]
        })
        
        # Mock pandas to_sql
        with patch.object(df, 'to_sql') as mock_to_sql:
            backend.create_table(mock_connection, 'test_table', df)
            
            mock_to_sql.assert_called_once_with(
                name='test_table',
                con=mock_connection,
                if_exists='replace',
                index=False
            )
    
    def test_read_table_with_columns(self, backend, mock_connection):
        """Test reading specific columns."""
        expected_df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        
        with patch('pandas.read_sql_query', return_value=expected_df) as mock_read:
            result = backend.read_table(
                mock_connection,
                'test_table',
                columns=['id', 'name']
            )
            
            mock_read.assert_called_once()
            args = mock_read.call_args[0]
            assert 'SELECT id, name FROM test_table' in args[0]
            assert result.equals(expected_df)
    
    def test_read_table_with_limit(self, backend, mock_connection):
        """Test reading with row limit."""
        with patch('pandas.read_sql_query') as mock_read:
            backend.read_table(mock_connection, 'test_table', limit=100)
            
            args = mock_read.call_args[0]
            assert 'LIMIT 100' in args[0]
    
    @pytest.mark.parametrize("table_name,expected", [
        ("users", True),
        ("test_table", True),
        ("'; DROP TABLE users; --", False),  # SQL injection attempt
    ])
    def test_table_name_validation(self, backend, table_name, expected):
        """Test table name validation."""
        if expected:
            assert backend._validate_table_name(table_name)
        else:
            with pytest.raises(ValueError):
                backend._validate_table_name(table_name)
```

### 2. Feature Transformer Tests
```python
# tests/unit/features/test_statistical_transformer.py
import pytest
import pandas as pd
import numpy as np
from mdm.features.transformers.statistical import StatisticalTransformer
from mdm.features.context import FeatureContext, ColumnType

class TestStatisticalTransformer:
    """Unit tests for statistical transformer."""
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return StatisticalTransformer()
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe."""
        return pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'text': ['a', 'b', 'c', 'd', 'e'],
            'constant': [1, 1, 1, 1, 1]
        })
    
    @pytest.fixture
    def context(self):
        """Create feature context."""
        return FeatureContext(
            dataset_name='test',
            column_types={
                'numeric': ColumnType.NUMERIC,
                'text': ColumnType.TEXT,
                'constant': ColumnType.NUMERIC
            }
        )
    
    def test_can_transform_with_numeric(self, transformer, sample_df, context):
        """Test transformer detection of numeric columns."""
        assert transformer.can_transform(sample_df, context) is True
    
    def test_can_transform_without_numeric(self, transformer, context):
        """Test transformer with no numeric columns."""
        df = pd.DataFrame({'text': ['a', 'b', 'c']})
        context.column_types = {'text': ColumnType.TEXT}
        
        assert transformer.can_transform(df, context) is False
    
    def test_zscore_transformation(self, transformer, sample_df, context):
        """Test z-score normalization."""
        result = transformer.transform(sample_df, context)
        
        # Check z-score column created
        assert 'numeric_zscore' in result.columns
        
        # Verify z-score properties
        zscore = result['numeric_zscore']
        assert np.isclose(zscore.mean(), 0, atol=1e-10)
        assert np.isclose(zscore.std(), 1, atol=1e-10)
    
    def test_log_transformation(self, transformer, context):
        """Test log transformation for positive values."""
        df = pd.DataFrame({'positive': [1, 2, 3, 4, 5]})
        context.column_types = {'positive': ColumnType.NUMERIC}
        
        result = transformer.transform(df, context)
        
        assert 'positive_log' in result.columns
        assert all(result['positive_log'] == np.log1p(df['positive']))
    
    def test_skip_log_for_negative(self, transformer, context):
        """Test log transformation skipped for negative values."""
        df = pd.DataFrame({'mixed': [-1, 0, 1, 2, 3]})
        context.column_types = {'mixed': ColumnType.NUMERIC}
        
        result = transformer.transform(df, context)
        
        assert 'mixed_log' not in result.columns
    
    def test_skip_constant_columns(self, transformer, sample_df, context):
        """Test that constant columns are skipped."""
        result = transformer.transform(sample_df, context)
        
        # Should not create zscore for constant
        assert 'constant_zscore' not in result.columns
    
    def test_feature_names_tracking(self, transformer, sample_df, context):
        """Test that created features are tracked."""
        result = transformer.transform(sample_df, context)
        feature_names = transformer.get_feature_names()
        
        # Check all created features are tracked
        for feature in feature_names:
            assert feature in result.columns
```

### 3. Configuration Tests
```python
# tests/unit/config/test_settings.py
import pytest
from pathlib import Path
from pydantic import ValidationError
from mdm.config.settings import MDMSettings

class TestMDMSettings:
    """Unit tests for configuration settings."""
    
    def test_default_values(self):
        """Test default configuration values."""
        settings = MDMSettings()
        
        assert settings.storage_backend == "sqlite"
        assert settings.batch_size == 10000
        assert settings.max_workers == 4
        assert settings.log_level == "INFO"
        assert settings.base_path == Path("~/.mdm").expanduser()
    
    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("MDM_STORAGE_BACKEND", "postgresql")
        monkeypatch.setenv("MDM_BATCH_SIZE", "50000")
        monkeypatch.setenv("MDM_LOG_LEVEL", "DEBUG")
        
        settings = MDMSettings()
        
        assert settings.storage_backend == "postgresql"
        assert settings.batch_size == 50000
        assert settings.log_level == "DEBUG"
    
    def test_validation_batch_size(self):
        """Test batch size validation."""
        # Too small
        with pytest.raises(ValidationError) as exc_info:
            MDMSettings(batch_size=50)
        assert "greater than or equal to 100" in str(exc_info.value)
        
        # Too large
        with pytest.raises(ValidationError) as exc_info:
            MDMSettings(batch_size=2000000)
        assert "less than or equal to 1000000" in str(exc_info.value)
    
    def test_validation_log_level(self):
        """Test log level validation."""
        with pytest.raises(ValidationError) as exc_info:
            MDMSettings(log_level="INVALID")
        assert "string does not match regex" in str(exc_info.value)
    
    def test_path_expansion(self):
        """Test path expansion."""
        settings = MDMSettings(base_path="~/custom/mdm")
        
        assert settings.base_path == Path.home() / "custom" / "mdm"
    
    def test_json_serialization(self):
        """Test settings can be serialized to JSON."""
        settings = MDMSettings(
            storage_backend="duckdb",
            batch_size=20000
        )
        
        json_data = settings.json()
        assert '"storage_backend": "duckdb"' in json_data
        assert '"batch_size": 20000' in json_data
```

### 4. Registration Step Tests
```python
# tests/unit/dataset/test_registration_steps.py
import pytest
from pathlib import Path
from mdm.dataset.steps import ValidateNameStep, DiscoverFilesStep
from mdm.dataset.context import RegistrationContext, StepStatus

class TestValidateNameStep:
    """Test dataset name validation step."""
    
    @pytest.fixture
    def step(self):
        return ValidateNameStep()
    
    @pytest.mark.parametrize("name,expected_status", [
        ("valid_name", StepStatus.COMPLETED),
        ("valid-name", StepStatus.COMPLETED),
        ("valid_name_123", StepStatus.COMPLETED),
        ("invalid name", StepStatus.FAILED),  # Space
        ("invalid@name", StepStatus.FAILED),  # Special char
        ("", StepStatus.FAILED),  # Empty
    ])
    def test_name_validation(self, step, name, expected_status):
        """Test various dataset names."""
        context = RegistrationContext(dataset_name=name, dataset_path=Path("/tmp"))
        
        result = step.execute(context)
        
        assert result.status == expected_status
        if expected_status == StepStatus.COMPLETED:
            assert result.data["validated_name"] == name

class TestDiscoverFilesStep:
    """Test file discovery step."""
    
    @pytest.fixture
    def step(self):
        return DiscoverFilesStep()
    
    def test_discover_single_file(self, step, tmp_path):
        """Test discovery of single file."""
        file_path = tmp_path / "data.csv"
        file_path.touch()
        
        context = RegistrationContext(
            dataset_name="test",
            dataset_path=file_path
        )
        
        result = step.execute(context)
        
        assert result.status == StepStatus.COMPLETED
        assert len(context.files) == 1
        assert context.files["train"] == file_path
    
    def test_discover_directory(self, step, tmp_path):
        """Test discovery in directory."""
        # Create standard files
        (tmp_path / "train.csv").touch()
        (tmp_path / "test.csv").touch()
        (tmp_path / "other.txt").touch()  # Should be ignored
        
        context = RegistrationContext(
            dataset_name="test",
            dataset_path=tmp_path
        )
        
        result = step.execute(context)
        
        assert result.status == StepStatus.COMPLETED
        assert len(context.files) == 2
        assert "train" in context.files
        assert "test" in context.files
        assert context.files["train"].name == "train.csv"
```

## Integration Testing Strategy

### 1. Storage Integration Tests
```python
# tests/integration/storage/test_backend_integration.py
import pytest
import pandas as pd
from mdm.storage import BackendFactory, ConnectionManager
from mdm.config import BackendConfig

class TestBackendIntegration:
    """Integration tests for storage backends."""
    
    @pytest.fixture
    def sqlite_config(self, tmp_path):
        """Create SQLite configuration."""
        return BackendConfig(
            type="sqlite",
            path=tmp_path / "test.db"
        )
    
    @pytest.fixture
    def connection_manager(self, sqlite_config):
        """Create connection manager."""
        return ConnectionManager("sqlite", sqlite_config)
    
    def test_full_crud_cycle(self, connection_manager):
        """Test complete CRUD cycle."""
        backend = BackendFactory().create("sqlite")
        test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [95.5, 87.3, 91.2]
        })
        
        with connection_manager.get_connection() as conn:
            # Create
            backend.create_table(conn, "students", test_df)
            
            # Read
            read_df = backend.read_table(conn, "students")
            assert len(read_df) == 3
            assert list(read_df.columns) == ['id', 'name', 'score']
            
            # Update (via SQL)
            backend.execute_query(
                conn,
                "UPDATE students SET score = ? WHERE id = ?",
                [99.0, 1]
            )
            
            # Verify update
            updated_df = backend.read_table(conn, "students", columns=['id', 'score'])
            assert updated_df.loc[updated_df['id'] == 1, 'score'].iloc[0] == 99.0
            
            # Delete (via SQL)
            backend.execute_query(conn, "DELETE FROM students WHERE id = ?", [3])
            
            # Verify delete
            final_df = backend.read_table(conn, "students")
            assert len(final_df) == 2
```

### 2. Feature Pipeline Integration Tests
```python
# tests/integration/features/test_pipeline_integration.py
import pytest
import pandas as pd
from mdm.features import (
    FeaturePipeline,
    StatisticalTransformer,
    TemporalTransformer,
    FeatureContext
)

class TestFeaturePipelineIntegration:
    """Integration tests for feature pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        return pd.DataFrame({
            'id': range(100),
            'amount': np.random.normal(100, 20, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'description': ['Item ' + str(i) for i in range(100)]
        })
    
    def test_multi_transformer_pipeline(self, sample_data):
        """Test pipeline with multiple transformers."""
        pipeline = FeaturePipeline()
        pipeline.add_transformer(StatisticalTransformer())
        pipeline.add_transformer(TemporalTransformer())
        
        context = FeatureContext(
            dataset_name="test",
            column_types={
                'amount': ColumnType.NUMERIC,
                'date': ColumnType.DATETIME,
                'category': ColumnType.CATEGORICAL
            }
        )
        
        result = pipeline.execute(sample_data, context)
        
        # Verify statistical features
        assert 'amount_zscore' in result.columns
        assert 'amount_percentile' in result.columns
        
        # Verify temporal features
        assert 'date_year' in result.columns
        assert 'date_month' in result.columns
        assert 'date_dayofweek' in result.columns
        
        # Verify no data loss
        assert len(result) == len(sample_data)
        assert all(col in result.columns for col in sample_data.columns)
```

### 3. Registration Integration Tests
```python
# tests/integration/dataset/test_registration_integration.py
import pytest
from pathlib import Path
from mdm.dataset import RegistrationPipeline, DatasetRegistrar
from mdm.dataset.steps import all_steps
from mdm.storage import ConnectionManager
from mdm.config import ConfigurationManager

class TestRegistrationIntegration:
    """Integration tests for dataset registration."""
    
    @pytest.fixture
    def setup_environment(self, tmp_path):
        """Set up test environment."""
        # Create config
        config_manager = ConfigurationManager()
        config_manager.override(base_path=tmp_path)
        
        # Create services
        connection_manager = ConnectionManager("sqlite", config_manager.get())
        
        # Create pipeline
        steps = all_steps(config_manager, connection_manager)
        pipeline = RegistrationPipeline(steps)
        
        # Create registrar
        registrar = DatasetRegistrar(pipeline)
        
        return registrar, tmp_path
    
    def test_end_to_end_registration(self, setup_environment, tmp_path):
        """Test complete registration flow."""
        registrar, base_path = setup_environment
        
        # Create test data
        data_file = tmp_path / "test_data.csv"
        pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'feature1': [10, 20, 30, 40, 50],
            'feature2': ['A', 'B', 'A', 'B', 'C'],
            'target': [0, 1, 0, 1, 1]
        }).to_csv(data_file, index=False)
        
        # Register dataset
        info = registrar.register(
            name="test_dataset",
            dataset_path=data_file,
            target_column="target",
            problem_type="binary_classification"
        )
        
        # Verify registration
        assert info.name == "test_dataset"
        assert info.target_column == "target"
        assert info.problem_type == "binary_classification"
        
        # Verify files created
        dataset_path = base_path / "datasets" / "test_dataset"
        assert dataset_path.exists()
        assert (dataset_path / "dataset.sqlite").exists()
        
        # Verify configuration saved
        config_path = base_path / "config" / "datasets" / "test_dataset.yaml"
        assert config_path.exists()
```

## Migration Testing

### 1. Parallel Testing
```python
# tests/migration/test_parallel_implementation.py
import pytest
from mdm.storage.legacy import LegacyBackend
from mdm.storage.backends.sqlite import SQLiteBackend
from mdm.storage.adapter import BackendAdapter

class TestParallelImplementation:
    """Test old vs new implementation."""
    
    def test_backend_compatibility(self, tmp_path):
        """Test that new backend produces same results as old."""
        # Create test data
        df = create_test_dataframe()
        
        # Old implementation
        old_backend = LegacyBackend()
        old_engine = old_backend.get_engine(str(tmp_path / "old.db"))
        old_backend.create_table_from_dataframe(df, "test", old_engine)
        old_result = old_backend.read_table_to_dataframe("test", old_engine)
        
        # New implementation
        new_backend = SQLiteBackend()
        conn_manager = ConnectionManager("sqlite", {"path": tmp_path / "new.db"})
        
        with conn_manager.get_connection() as conn:
            new_backend.create_table(conn, "test", df)
            new_result = new_backend.read_table(conn, "test")
        
        # Compare results
        assert old_result.equals(new_result)
```

### 2. Feature Flag Testing
```python
# tests/migration/test_feature_flags.py
import pytest
from mdm.config import FeatureFlags

class TestFeatureFlags:
    """Test feature flag functionality."""
    
    def test_gradual_rollout(self, monkeypatch):
        """Test gradual feature rollout."""
        # 0% rollout
        monkeypatch.setenv("MDM_FEATURE_NEW_BACKEND", "0")
        flags = FeatureFlags()
        
        assert flags.use_new_backend() is False
        
        # 100% rollout
        monkeypatch.setenv("MDM_FEATURE_NEW_BACKEND", "100")
        flags = FeatureFlags()
        
        assert flags.use_new_backend() is True
        
        # User-specific override
        monkeypatch.setenv("MDM_FEATURE_NEW_BACKEND_USERS", "alice,bob")
        flags = FeatureFlags(user="alice")
        
        assert flags.use_new_backend() is True
```

## Performance Testing

### 1. Benchmark Suite
```python
# tests/performance/test_benchmarks.py
import pytest
import time
import pandas as pd
from mdm.benchmarks import Benchmark

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create large test dataset."""
        return pd.DataFrame({
            'id': range(1000000),
            'value': np.random.randn(1000000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 1000000)
        })
    
    @pytest.mark.benchmark
    def test_registration_performance(self, benchmark, large_dataset, tmp_path):
        """Benchmark dataset registration."""
        data_file = tmp_path / "large.csv"
        large_dataset.to_csv(data_file, index=False)
        
        def register():
            registrar = create_test_registrar(tmp_path)
            registrar.register("perf_test", data_file)
        
        result = benchmark(register)
        
        # Assert performance requirements
        assert result.median < 60.0  # Should complete in < 60 seconds
    
    @pytest.mark.benchmark
    def test_feature_generation_performance(self, benchmark, large_dataset):
        """Benchmark feature generation."""
        pipeline = create_feature_pipeline()
        context = create_test_context()
        
        result = benchmark(pipeline.execute, large_dataset, context)
        
        # Performance per row
        time_per_row = result.median / len(large_dataset)
        assert time_per_row < 0.001  # < 1ms per row
```

### 2. Memory Profiling
```python
# tests/performance/test_memory.py
import pytest
from memory_profiler import profile
import tracemalloc

class TestMemoryUsage:
    """Test memory usage."""
    
    def test_batch_processing_memory(self, tmp_path):
        """Test that batch processing limits memory usage."""
        # Create large file
        large_file = create_large_csv(tmp_path, rows=5000000)
        
        # Track memory
        tracemalloc.start()
        
        # Process file
        registrar = create_test_registrar(tmp_path)
        registrar.register("memory_test", large_file)
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Should use less than 500MB for 5M rows
        assert peak / 1024 / 1024 < 500
```

## Continuous Testing

### 1. Test Automation
```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install uv
        uv pip install -e ".[dev]"
    
    - name: Run unit tests
      run: pytest tests/unit -v --cov=mdm --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    - name: Run integration tests
      run: pytest tests/integration -v

  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    
    steps:
    - uses: actions/checkout@v3
    - name: Run performance tests
      run: pytest tests/performance -v --benchmark-only
```

### 2. Test Monitoring
```python
# tests/monitoring/test_metrics.py
import pytest
from prometheus_client import Counter, Histogram

# Metrics
test_runs = Counter('mdm_test_runs_total', 'Total test runs')
test_duration = Histogram('mdm_test_duration_seconds', 'Test duration')

@pytest.fixture(autouse=True)
def track_test_metrics(request):
    """Track test metrics."""
    test_runs.inc()
    
    start_time = time.time()
    yield
    duration = time.time() - start_time
    
    test_duration.observe(duration)
```

## Test Data Management

### 1. Fixtures
```python
# tests/fixtures/datasets.py
import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture(scope="session")
def sample_datasets(tmp_path_factory):
    """Create sample datasets for testing."""
    base_path = tmp_path_factory.mktemp("datasets")
    
    datasets = {
        "small": create_small_dataset(base_path),
        "medium": create_medium_dataset(base_path),
        "large": create_large_dataset(base_path),
        "edge_cases": create_edge_case_dataset(base_path)
    }
    
    return datasets

def create_small_dataset(base_path: Path) -> Path:
    """Create small test dataset."""
    df = pd.DataFrame({
        'id': range(100),
        'feature': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    path = base_path / "small.csv"
    df.to_csv(path, index=False)
    return path
```

### 2. Test Utilities
```python
# tests/utils/helpers.py
import pandas as pd
from typing import Dict, Any

def assert_dataframes_equal(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs):
    """Assert two dataframes are equal with better error messages."""
    try:
        pd.testing.assert_frame_equal(df1, df2, **kwargs)
    except AssertionError as e:
        # Provide detailed diff
        diff = df1.compare(df2)
        raise AssertionError(f"DataFrames differ:\n{diff}\n\nOriginal error: {e}")

def create_mock_context(**kwargs) -> RegistrationContext:
    """Create mock registration context."""
    defaults = {
        "dataset_name": "test",
        "dataset_path": Path("/tmp/test.csv"),
        "target_column": None,
        "problem_type": None,
        "force": False,
        "generate_features": True
    }
    
    defaults.update(kwargs)
    return RegistrationContext(**defaults)
```

## Success Metrics

### 1. Coverage Goals
- Unit tests: 95%+ line coverage
- Integration tests: All critical paths
- Edge cases: Documented and tested
- Performance: No regression

### 2. Test Quality
- Fast: Full suite < 5 minutes
- Reliable: <1% flaky tests  
- Maintainable: Clear structure
- Documented: Purpose of each test

### 3. Continuous Improvement
- Weekly test review
- Monthly performance baseline
- Quarterly test refactoring
- Annual strategy review