# Step 1: Test Stabilization

## Overview

Fix all failing tests to create a reliable safety net for refactoring. This is the most critical step - without green tests, we cannot safely refactor.

## Duration

2 weeks (Weeks 1-2)

## Objectives

1. Achieve 100% passing tests across all test suites
2. Fix flaky tests and race conditions
3. Add missing test coverage for critical paths
4. Set up continuous test monitoring
5. Create baseline performance benchmarks

## Current State Analysis

Based on the analysis:
- **Unit Tests**: 67 failing out of 439 (15.3% failure rate)
- **Critical Failures**: 31 registration tests, 20 storage backend tests
- **Coverage**: 76% (needs improvement in feature engineering)

## Detailed Steps

### Week 1: Fix Critical Test Failures

#### Day 1-2: Storage Backend Tests

##### 1.1 Analyze Storage Backend Failures
```bash
# Run only storage backend tests with verbose output
pytest tests/unit/test_storage_* -v --tb=short > storage_test_failures.log

# Identify patterns in failures
grep -E "(FAILED|ERROR)" storage_test_failures.log | sort | uniq -c
```

##### 1.2 Fix SQLite Backend Tests
```python
# Common issue: Singleton pattern in tests
# Fix: tests/unit/test_storage_backends.py

def test_sqlite_backend_create_dataset():
    """Test SQLite dataset creation with proper cleanup"""
    backend = SQLiteBackend()
    try:
        # Ensure clean state
        if backend._engine:
            backend._engine.dispose()
            backend._engine = None
        
        # Test dataset creation
        backend.create_dataset("test_dataset", config={})
        
        # Verify creation
        assert backend.dataset_exists("test_dataset")
    finally:
        # Cleanup
        backend.drop_dataset("test_dataset")
        if backend._engine:
            backend._engine.dispose()
```

##### 1.3 Fix DuckDB Backend Tests
```python
# Common issue: Connection management
# Fix: tests/unit/test_storage_duckdb.py

@pytest.fixture
def duckdb_backend():
    """Fixture with proper connection management"""
    backend = DuckDBBackend()
    yield backend
    # Ensure all connections are closed
    if hasattr(backend, '_conn') and backend._conn:
        backend._conn.close()
        backend._conn = None
```

##### 1.4 Add PostgreSQL Backend Tests
```python
# Create: tests/unit/test_storage_postgresql.py
import pytest
from unittest.mock import patch, MagicMock
from mdm.storage.postgresql import PostgreSQLBackend

@pytest.mark.unit
class TestPostgreSQLBackend:
    @patch('mdm.storage.postgresql.create_engine')
    def test_create_dataset(self, mock_engine):
        """Test PostgreSQL dataset creation"""
        # Mock the engine
        mock_engine.return_value = MagicMock()
        
        backend = PostgreSQLBackend()
        backend.create_dataset("test_dataset", config={})
        
        # Verify SQL execution
        mock_engine.return_value.execute.assert_called()
```

#### Day 3-4: Registration Tests

##### 1.5 Fix Dataset Registrar Tests
```python
# Common issue: Missing initialization
# Fix: tests/unit/test_dataset_registrar.py

@pytest.fixture
def registrar():
    """Properly initialized registrar"""
    with patch('mdm.dataset.registrar.get_storage_backend') as mock_storage:
        mock_storage.return_value = MagicMock()
        registrar = DatasetRegistrar()
        # Initialize required attributes
        registrar._detected_datetime_columns = []
        registrar._detected_id_columns = []
        registrar._detected_target_column = None
        yield registrar
```

##### 1.6 Fix Auto-Detection Tests
```python
# Fix: tests/unit/test_auto_detect.py
def test_detect_kaggle_structure(tmp_path):
    """Test Kaggle competition structure detection"""
    # Create proper structure
    (tmp_path / "train.csv").write_text("id,feature1,target\n1,0.5,1\n")
    (tmp_path / "test.csv").write_text("id,feature1\n2,0.6\n")
    
    result = detect_dataset_structure(str(tmp_path))
    
    assert result["structure_type"] == "kaggle_competition"
    assert len(result["data_files"]) == 2
    assert result["suggested_target"] == "target"
```

#### Day 5: Integration Test Fixes

##### 1.7 Fix Integration Test Setup
```python
# Fix: tests/integration/conftest.py
@pytest.fixture(scope="session")
def test_database_dir(tmp_path_factory):
    """Create isolated test database directory"""
    test_dir = tmp_path_factory.mktemp("test_mdm")
    original_home = os.environ.get('MDM_HOME')
    os.environ['MDM_HOME'] = str(test_dir)
    yield test_dir
    # Restore original
    if original_home:
        os.environ['MDM_HOME'] = original_home
    else:
        os.environ.pop('MDM_HOME', None)
```

### Week 2: Improve Coverage and Stability

#### Day 6-7: Add Missing Test Coverage

##### 2.1 Feature Engineering Tests
```python
# Create: tests/unit/test_feature_engineering_comprehensive.py
import pytest
from mdm.features import FeatureGenerator
import pandas as pd
import numpy as np

class TestFeatureEngineeringComprehensive:
    def test_numeric_features_generation(self):
        """Test all numeric feature generation"""
        data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5, np.nan]
        })
        
        generator = FeatureGenerator()
        features = generator.generate_numeric_features(data['numeric_col'])
        
        expected_features = [
            'numeric_col_mean', 'numeric_col_std', 'numeric_col_min',
            'numeric_col_max', 'numeric_col_median', 'numeric_col_skew',
            'numeric_col_kurtosis', 'numeric_col_missing_count',
            'numeric_col_missing_ratio'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
            
    def test_categorical_features_generation(self):
        """Test categorical feature generation"""
        data = pd.DataFrame({
            'cat_col': ['A', 'B', 'A', 'C', 'B', None]
        })
        
        generator = FeatureGenerator()
        features = generator.generate_categorical_features(data['cat_col'])
        
        assert 'cat_col_nunique' in features.columns
        assert 'cat_col_mode' in features.columns
        assert 'cat_col_entropy' in features.columns
```

##### 2.2 Type Detection Tests
```python
# Create: tests/unit/test_type_detection.py
class TestTypeDetection:
    def test_datetime_detection(self):
        """Test datetime column detection"""
        data = pd.DataFrame({
            'date_str': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'timestamp': pd.date_range('2024-01-01', periods=3),
            'not_date': ['abc', 'def', 'ghi']
        })
        
        detector = TypeDetector()
        datetime_cols = detector.detect_datetime_columns(data)
        
        assert 'date_str' in datetime_cols
        assert 'timestamp' in datetime_cols
        assert 'not_date' not in datetime_cols
```

#### Day 8-9: Fix Flaky Tests

##### 2.3 Add Proper Timeouts
```python
# Fix timeout-sensitive tests
@pytest.mark.timeout(30)  # 30 seconds timeout
def test_large_dataset_registration():
    """Test with proper timeout"""
    # Test implementation
    pass
```

##### 2.4 Fix Race Conditions
```python
# Use proper synchronization
import threading
import time

def test_concurrent_access():
    """Test with proper synchronization"""
    lock = threading.Lock()
    results = []
    
    def worker(backend, dataset_name):
        with lock:
            backend.create_dataset(dataset_name, {})
            results.append(dataset_name)
    
    # Test implementation with proper cleanup
```

#### Day 10: Performance Benchmarks

##### 2.5 Create Performance Test Suite
```python
# Create: tests/benchmarks/test_performance_baseline.py
import pytest
import time
import pandas as pd
from memory_profiler import memory_usage

class TestPerformanceBaseline:
    @pytest.mark.benchmark
    def test_registration_performance(self, benchmark_data):
        """Baseline registration performance"""
        start_time = time.time()
        start_memory = memory_usage()[0]
        
        registrar = DatasetRegistrar()
        registrar.register(
            name="perf_test",
            path=benchmark_data,
            force=True
        )
        
        end_time = time.time()
        end_memory = memory_usage()[0]
        
        metrics = {
            'duration': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'rows_per_second': 10000 / (end_time - start_time)
        }
        
        # Save baseline
        save_performance_baseline(metrics)
        
        # Assert reasonable performance
        assert metrics['duration'] < 10.0  # Should complete in 10s
        assert metrics['memory_delta'] < 500  # Less than 500MB
```

### Continuous Monitoring Setup

#### 3.1 Create Test Monitor Script
```bash
#!/bin/bash
# scripts/monitor_tests.sh

while true; do
    echo "=== Test Status at $(date) ==="
    
    # Run tests and capture results
    pytest --tb=no -q > /tmp/test_results.txt 2>&1
    
    # Parse results
    PASSED=$(grep -c "passed" /tmp/test_results.txt || echo 0)
    FAILED=$(grep -c "failed" /tmp/test_results.txt || echo 0)
    ERRORS=$(grep -c "error" /tmp/test_results.txt || echo 0)
    
    # Log status
    echo "Passed: $PASSED, Failed: $FAILED, Errors: $ERRORS" >> $MDM_MIGRATION_LOG
    
    # Alert if failures increase
    if [ $FAILED -gt 0 ] || [ $ERRORS -gt 0 ]; then
        echo "⚠️  TEST FAILURES DETECTED!"
    fi
    
    sleep 3600  # Check every hour
done
```

#### 3.2 Set Up CI Integration
```yaml
# .github/workflows/test-monitor.yml
name: Migration Test Monitor

on:
  push:
    branches: [refactor-*]
  schedule:
    - cron: '0 */2 * * *'  # Every 2 hours

jobs:
  test-status:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e .
      - name: Run test suite
        run: |
          pytest -v --junitxml=test-results.xml
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: test-results.xml
```

## Validation Checklist

### Week 1 Completion
- [ ] All storage backend tests passing (SQLite, DuckDB, PostgreSQL)
- [ ] All registration tests passing
- [ ] Integration tests fixed and passing
- [ ] No flaky tests remaining

### Week 2 Completion
- [ ] Test coverage increased to 85%+
- [ ] Feature engineering tests comprehensive
- [ ] Type detection tests complete
- [ ] Performance benchmarks established
- [ ] Continuous monitoring active

## Success Criteria

- **100% test pass rate** maintained for 48 hours
- **No flaky tests** in 10 consecutive runs
- **Coverage ≥ 85%** for all modules
- **Performance benchmarks** documented
- **CI/CD pipeline** catching regressions

## Common Issues and Solutions

### Issue: Tests pass locally but fail in CI
```bash
# Debug CI environment differences
# 1. Check Python version
python --version

# 2. Check installed packages
pip freeze > local_packages.txt
# Compare with CI packages

# 3. Check environment variables
env | grep MDM > local_env.txt
```

### Issue: Database connection exhaustion
```python
# Solution: Ensure proper cleanup
@pytest.fixture(autouse=True)
def cleanup_connections():
    yield
    # Force close all connections
    from sqlalchemy import pool
    pool.clear_managers()
```

### Issue: Test isolation problems
```python
# Solution: Use proper fixtures
@pytest.fixture
def isolated_mdm_home(tmp_path, monkeypatch):
    """Completely isolated MDM environment"""
    mdm_home = tmp_path / ".mdm"
    mdm_home.mkdir()
    monkeypatch.setenv("MDM_HOME", str(mdm_home))
    yield mdm_home
```

## Daily Checklist

- [ ] Run full test suite
- [ ] Check test execution time trends
- [ ] Review any new failures
- [ ] Update test documentation
- [ ] Commit fixes with descriptive messages

## Next Steps

Once all tests are green and stable, proceed to [02-abstraction-layer.md](02-abstraction-layer.md).

## Notes

- Do not skip this step - it's the foundation for safe refactoring
- Document any workarounds needed for test fixes
- Keep performance benchmarks for comparison during migration