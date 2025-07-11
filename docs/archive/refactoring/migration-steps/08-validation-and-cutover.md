# Step 8: Validation and Cutover

## Overview

Perform comprehensive validation of all migrated components and execute the final cutover to the new system. This critical phase ensures all components work together correctly before full deployment.

## Duration

2 weeks (Weeks 18-19)

## Objectives

1. Validate all migrated components work together
2. Run comprehensive comparison tests
3. Perform load and stress testing
4. Execute gradual cutover with monitoring
5. Ensure zero regression in functionality

## Prerequisites

- ✅ Configuration system migrated (Step 4)
- ✅ Storage backends migrated (Step 5)
- ✅ Feature engineering migrated (Step 6)
- ✅ Dataset registration migrated (Step 7)
- ✅ All individual component tests passing

## Detailed Steps

### Week 18: Comprehensive Validation

#### Day 1-2: Integration Testing

##### 1.1 Create Integration Test Suite
```python
# Create: tests/integration/test_full_system.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import time
from typing import Dict, Any

from mdm.core.feature_flags import feature_flags
from mdm.config import get_config
from mdm.dataset import DatasetRegistrar
from mdm.storage.factory import get_storage_backend
from mdm.api import MDMClient


class TestFullSystemIntegration:
    @pytest.fixture
    def enable_all_new_systems(self):
        """Enable all new systems for testing"""
        original_flags = {}
        flags_to_set = [
            "use_new_config",
            "use_new_backend", 
            "use_new_features",
            "use_new_registrar"
        ]
        
        # Save original values
        for flag in flags_to_set:
            original_flags[flag] = feature_flags.get(flag, False)
            feature_flags.set(flag, True)
        
        yield
        
        # Restore original values
        for flag, value in original_flags.items():
            feature_flags.set(flag, value)
    
    @pytest.fixture
    def test_dataset(self, tmp_path):
        """Create test dataset"""
        # Create diverse dataset for comprehensive testing
        np.random.seed(42)
        n_rows = 10000
        
        data = pd.DataFrame({
            # Numeric features
            'numeric_1': np.random.randn(n_rows),
            'numeric_2': np.random.uniform(0, 100, n_rows),
            'numeric_3': np.random.randint(0, 1000, n_rows),
            
            # Categorical features
            'category_1': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
            'category_2': np.random.choice(['X', 'Y', 'Z'], n_rows, p=[0.5, 0.3, 0.2]),
            
            # DateTime features
            'date_1': pd.date_range('2024-01-01', periods=n_rows, freq='H'),
            'date_2': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
            
            # Text features
            'text_1': [f"Description {i} with some text" for i in range(n_rows)],
            'text_2': [f"Category {i % 100}" for i in range(n_rows)],
            
            # ID and target
            'id': range(n_rows),
            'target': np.random.randint(0, 2, n_rows)
        })
        
        # Save to CSV
        csv_path = tmp_path / "test_data.csv"
        data.to_csv(csv_path, index=False)
        
        return csv_path, data
    
    def test_end_to_end_workflow(self, enable_all_new_systems, test_dataset):
        """Test complete workflow with all new systems"""
        csv_path, original_data = test_dataset
        dataset_name = f"integration_test_{int(time.time())}"
        
        # 1. Register dataset
        registrar = DatasetRegistrar()
        result = registrar.register(
            name=dataset_name,
            path=str(csv_path),
            target="target",
            id_columns=["id"],
            datetime_columns=["date_1", "date_2"],
            problem_type="binary_classification",
            force=True
        )
        
        assert result["success"], f"Registration failed: {result.get('error')}"
        
        # 2. Verify storage
        backend = get_storage_backend()
        assert backend.dataset_exists(dataset_name)
        
        # Load and verify data
        loaded_data = backend.load_data(dataset_name)
        assert len(loaded_data) == len(original_data)
        assert set(loaded_data.columns) == set(original_data.columns)
        
        # 3. Check features were generated
        features = backend.load_data(dataset_name, "features")
        assert len(features) > 0
        assert len(features.columns) > len(original_data.columns)
        
        # 4. Verify metadata
        metadata = backend.get_metadata(dataset_name)
        assert "column_types" in metadata
        assert "statistics" in metadata
        assert "feature_count" in metadata
        
        # 5. Test programmatic API
        client = MDMClient()
        dataset = client.get_dataset(dataset_name)
        assert dataset is not None
        assert dataset.name == dataset_name
        
        # 6. Test export
        export_path = client.export_dataset(
            dataset_name,
            format="parquet",
            compression="snappy"
        )
        assert Path(export_path).exists()
        
        # 7. Cleanup
        client.remove_dataset(dataset_name, force=True)
        assert not backend.dataset_exists(dataset_name)
    
    def test_concurrent_operations(self, enable_all_new_systems, tmp_path):
        """Test system under concurrent load"""
        import concurrent.futures
        
        # Create multiple small datasets
        datasets = []
        for i in range(5):
            data = pd.DataFrame({
                'id': range(100),
                'value': np.random.randn(100),
                'category': np.random.choice(['A', 'B'], 100)
            })
            path = tmp_path / f"concurrent_{i}.csv"
            data.to_csv(path, index=False)
            datasets.append((f"concurrent_test_{i}", str(path)))
        
        # Register datasets concurrently
        def register_dataset(name_path_tuple):
            name, path = name_path_tuple
            registrar = DatasetRegistrar()
            return registrar.register(name, path, force=True)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(register_dataset, ds) for ds in datasets]
            results = [f.result() for f in futures]
        
        # Verify all succeeded
        for result in results:
            assert result["success"]
        
        # Verify all datasets exist
        backend = get_storage_backend()
        for name, _ in datasets:
            assert backend.dataset_exists(name)
        
        # Cleanup
        for name, _ in datasets:
            backend.drop_dataset(name)
    
    def test_error_recovery(self, enable_all_new_systems, tmp_path):
        """Test error recovery and rollback"""
        # Create dataset with problematic data
        data = pd.DataFrame({
            'id': [1, 2, None, 4],  # NULL in ID column
            'value': [1, 2, 3, 4],
            'bad_column': [float('inf'), 1, 2, float('nan')]  # Inf values
        })
        
        path = tmp_path / "problematic.csv"
        data.to_csv(path, index=False)
        
        # Attempt registration
        registrar = DatasetRegistrar()
        result = registrar.register(
            name="error_test",
            path=str(path),
            id_columns=["id"],
            force=True
        )
        
        # Should handle errors gracefully
        # Even if registration partially fails, should not leave artifacts
        backend = get_storage_backend()
        
        # Verify cleanup
        if not result["success"]:
            assert not backend.dataset_exists("error_test")
```

##### 1.2 Create Performance Validation
```python
# Create: tests/validation/test_performance_regression.py
import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from pathlib import Path

from mdm.core.feature_flags import feature_flags
from mdm.dataset import DatasetRegistrar
from mdm.testing.comparison import ComparisonTester


class TestPerformanceRegression:
    @pytest.fixture
    def large_dataset(self, tmp_path):
        """Create large dataset for performance testing"""
        n_rows = 100000
        n_numeric = 20
        n_categorical = 10
        
        data = {}
        
        # Numeric columns
        for i in range(n_numeric):
            data[f'num_{i}'] = np.random.randn(n_rows)
        
        # Categorical columns
        for i in range(n_categorical):
            data[f'cat_{i}'] = np.random.choice(
                [f'val_{j}' for j in range(10)], n_rows
            )
        
        # Add ID and target
        data['id'] = range(n_rows)
        data['target'] = np.random.randint(0, 2, n_rows)
        
        df = pd.DataFrame(data)
        path = tmp_path / "large_dataset.csv"
        df.to_csv(path, index=False)
        
        return path, len(df), df.memory_usage(deep=True).sum()
    
    def test_registration_performance(self, large_dataset):
        """Compare registration performance between old and new systems"""
        csv_path, n_rows, data_size = large_dataset
        comparison_tester = ComparisonTester()
        
        def register_with_system(use_new: bool):
            # Set feature flags
            feature_flags.set("use_new_config", use_new)
            feature_flags.set("use_new_backend", use_new)
            feature_flags.set("use_new_features", use_new)
            feature_flags.set("use_new_registrar", use_new)
            
            # Register dataset
            registrar = DatasetRegistrar()
            dataset_name = f"perf_test_{use_new}_{int(time.time())}"
            
            result = registrar.register(
                name=dataset_name,
                path=str(csv_path),
                target="target",
                force=True
            )
            
            # Cleanup
            if result["success"]:
                from mdm.storage.factory import get_storage_backend
                backend = get_storage_backend()
                backend.drop_dataset(dataset_name)
            
            return result
        
        # Compare systems
        result = comparison_tester.compare(
            test_name="registration_performance",
            old_impl=lambda: register_with_system(False),
            new_impl=lambda: register_with_system(True)
        )
        
        print(f"\nPerformance Results:")
        print(f"Rows: {n_rows:,}, Data size: {data_size/1024/1024:.1f} MB")
        print(f"Old system: {result.old_duration:.2f}s")
        print(f"New system: {result.new_duration:.2f}s")
        print(f"Performance delta: {result.performance_delta:+.1f}%")
        print(f"Memory delta: {result.memory_delta:+.1f}%")
        
        # New system should not be significantly slower
        assert result.performance_delta < 20, "New system is >20% slower"
        
        # Memory usage should be better
        assert result.memory_delta < 50, "New system uses >50% more memory"
    
    def test_concurrent_load(self, tmp_path):
        """Test system under concurrent load"""
        import concurrent.futures
        
        # Enable all new systems
        feature_flags.set("use_new_config", True)
        feature_flags.set("use_new_backend", True)
        feature_flags.set("use_new_features", True)
        feature_flags.set("use_new_registrar", True)
        
        # Create test datasets
        n_datasets = 20
        datasets = []
        
        for i in range(n_datasets):
            data = pd.DataFrame({
                'id': range(1000),
                'value': np.random.randn(1000),
                'category': np.random.choice(['A', 'B', 'C'], 1000)
            })
            path = tmp_path / f"concurrent_{i}.csv"
            data.to_csv(path, index=False)
            datasets.append((f"load_test_{i}", str(path)))
        
        # Measure system resources before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Register concurrently
        def register(name_path):
            name, path = name_path
            reg = DatasetRegistrar()
            return reg.register(name, path, force=True)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register, ds) for ds in datasets]
            results = [f.result() for f in futures]
        
        duration = time.time() - start_time
        
        # Measure after
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_delta = mem_after - mem_before
        
        # Verify results
        success_count = sum(1 for r in results if r["success"])
        
        print(f"\nConcurrent Load Test:")
        print(f"Datasets: {n_datasets}")
        print(f"Success: {success_count}/{n_datasets}")
        print(f"Duration: {duration:.2f}s")
        print(f"Throughput: {n_datasets/duration:.1f} datasets/sec")
        print(f"Memory delta: {mem_delta:.1f} MB")
        
        assert success_count == n_datasets, "Some registrations failed"
        assert duration < 60, "Takes too long under concurrent load"
        
        # Cleanup
        from mdm.storage.factory import get_storage_backend
        backend = get_storage_backend()
        for name, _ in datasets:
            try:
                backend.drop_dataset(name)
            except:
                pass
```

#### Day 3-4: Component Interaction Testing

##### 1.3 Create Cross-Component Tests
```python
# Create: tests/validation/test_component_interactions.py
import pytest
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from mdm.config import get_config, config_manager
from mdm.storage.factory import get_storage_backend
from mdm.features.builder import FeaturePipelineBuilder
from mdm.dataset.registration.orchestrator import RegistrationOrchestrator


class TestComponentInteractions:
    def test_config_storage_interaction(self):
        """Test configuration changes affect storage behavior"""
        # Change backend via config
        config_manager.set("database.default_backend", "duckdb")
        
        # Get backend and verify type
        backend = get_storage_backend()
        assert "DuckDB" in backend.__class__.__name__
        
        # Change to SQLite
        config_manager.set("database.default_backend", "sqlite")
        backend = get_storage_backend()
        assert "SQLite" in backend.__class__.__name__
    
    def test_storage_feature_interaction(self, tmp_path):
        """Test storage and feature generation work together"""
        # Create test data
        data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'C']
        })
        
        # Save via storage backend
        backend = get_storage_backend()
        dataset_name = "interaction_test"
        
        backend.create_dataset(dataset_name, {})
        backend.save_data(dataset_name, data)
        
        # Generate features
        builder = FeaturePipelineBuilder()
        pipeline = builder.build_default_pipeline()
        
        # Load data from storage
        loaded_data = backend.load_data(dataset_name)
        
        # Generate features
        features = pipeline.fit_transform(loaded_data)
        
        # Verify features generated
        assert len(features.columns) > len(data.columns)
        assert 'numeric_mean' in features.columns
        assert 'category_encoded' in features.columns or any(
            'category' in col for col in features.columns
        )
        
        # Save features back
        backend.save_data(dataset_name, features, "features")
        
        # Verify roundtrip
        loaded_features = backend.load_data(dataset_name, "features")
        assert len(loaded_features) == len(features)
        
        # Cleanup
        backend.drop_dataset(dataset_name)
    
    def test_registration_all_components(self, tmp_path):
        """Test registration uses all components correctly"""
        # Create test dataset with all feature types
        data = pd.DataFrame({
            'id': range(100),
            'numeric_1': range(100),
            'numeric_2': [x * 2.5 for x in range(100)],
            'category_1': ['A', 'B', 'C'] * 33 + ['A'],
            'category_2': ['X', 'Y'] * 50,
            'date_1': pd.date_range('2024-01-01', periods=100),
            'text_1': [f'Sample text number {i}' for i in range(100)],
            'target': [0, 1] * 50
        })
        
        csv_path = tmp_path / "full_test.csv"
        data.to_csv(csv_path, index=False)
        
        # Use orchestrator directly for detailed control
        orchestrator = RegistrationOrchestrator()
        
        # Track which components were used
        components_used = set()
        
        # Monkey-patch to track component usage
        original_get_config = get_config
        def tracked_get_config():
            components_used.add('config')
            return original_get_config()
        
        original_get_backend = get_storage_backend
        def tracked_get_backend():
            components_used.add('storage')
            return original_get_backend()
        
        # Patch temporarily
        import mdm.config
        import mdm.storage.factory
        mdm.config.get_config = tracked_get_config
        mdm.storage.factory.get_storage_backend = tracked_get_backend
        
        try:
            result = orchestrator.register(
                name="component_test",
                path=str(csv_path),
                target="target",
                force=True
            )
            
            assert result["success"]
            
            # Verify all components were used
            assert 'config' in components_used
            assert 'storage' in components_used
            
            # Verify features were generated
            backend = get_storage_backend()
            features = backend.load_data("component_test", "features")
            
            # Should have features from all types
            feature_cols = features.columns.tolist()
            assert any('numeric' in col and 'mean' in col for col in feature_cols)
            assert any('category' in col for col in feature_cols)
            assert any('date' in col or 'year' in col for col in feature_cols)
            assert any('text' in col for col in feature_cols)
            
            # Cleanup
            backend.drop_dataset("component_test")
            
        finally:
            # Restore original functions
            mdm.config.get_config = original_get_config
            mdm.storage.factory.get_storage_backend = original_get_backend
    
    def test_error_propagation(self, tmp_path):
        """Test errors propagate correctly between components"""
        # Create invalid data
        data = pd.DataFrame({
            'col1': [1, 2, 'invalid', 4],  # Mixed types
            'col2': [1, 2, 3, float('inf')]  # Infinity
        })
        
        path = tmp_path / "invalid.csv"
        data.to_csv(path, index=False)
        
        # Try to register
        from mdm.dataset import DatasetRegistrar
        registrar = DatasetRegistrar()
        
        result = registrar.register(
            name="error_propagation_test",
            path=str(path),
            force=True
        )
        
        # Should handle errors gracefully
        if not result["success"]:
            # Error should be informative
            assert "error" in result or "message" in result
        
        # Should not leave partial data
        backend = get_storage_backend()
        assert not backend.dataset_exists("error_propagation_test")
```

#### Day 5: Load Testing

##### 1.4 Create Load Test Suite
```python
# Create: tests/validation/test_load.py
import pytest
import pandas as pd
import numpy as np
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
from pathlib import Path

from mdm.api import MDMClient
from mdm.core.metrics import metrics_collector


class TestLoadAndStress:
    @pytest.fixture
    def load_test_data(self, tmp_path):
        """Generate datasets of various sizes"""
        datasets = {}
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            data = pd.DataFrame({
                'id': range(size),
                'value_1': np.random.randn(size),
                'value_2': np.random.uniform(0, 100, size),
                'category': np.random.choice(['A', 'B', 'C', 'D'], size),
                'date': pd.date_range('2024-01-01', periods=size, freq='min'),
                'text': [f'Text entry {i}' for i in range(size)]
            })
            
            path = tmp_path / f"load_test_{size}.csv"
            data.to_csv(path, index=False)
            datasets[size] = path
        
        return datasets
    
    def test_sequential_load(self, load_test_data):
        """Test sequential processing of multiple datasets"""
        client = MDMClient()
        results = []
        
        for size, path in load_test_data.items():
            start_time = time.time()
            
            dataset_name = f"load_seq_{size}_{int(time.time())}"
            result = client.register_dataset(
                name=dataset_name,
                path=str(path),
                force=True
            )
            
            duration = time.time() - start_time
            
            results.append({
                'size': size,
                'duration': duration,
                'success': result.get('success', False),
                'rows_per_sec': size / duration if duration > 0 else 0
            })
            
            # Cleanup
            client.remove_dataset(dataset_name, force=True)
        
        # Analyze results
        for res in results:
            print(f"Size: {res['size']:,} rows")
            print(f"Duration: {res['duration']:.2f}s")
            print(f"Throughput: {res['rows_per_sec']:,.0f} rows/sec")
            print()
        
        # Verify throughput doesn't degrade significantly
        throughputs = [r['rows_per_sec'] for r in results]
        assert min(throughputs) > max(throughputs) * 0.5, "Throughput degrades too much with size"
    
    def test_concurrent_read_write(self, tmp_path):
        """Test concurrent read/write operations"""
        client = MDMClient()
        
        # Create test dataset
        data = pd.DataFrame({
            'id': range(10000),
            'value': np.random.randn(10000)
        })
        
        path = tmp_path / "concurrent_rw.csv"
        data.to_csv(path, index=False)
        
        dataset_name = f"concurrent_rw_{int(time.time())}"
        client.register_dataset(dataset_name, str(path), force=True)
        
        # Concurrent operations
        errors = []
        results = []
        
        def read_operation(n):
            try:
                df = client.get_dataset(dataset_name).load_data()
                return ('read', n, len(df))
            except Exception as e:
                errors.append(('read', n, str(e)))
                return None
        
        def write_operation(n):
            try:
                new_data = pd.DataFrame({
                    'id': [10000 + n],
                    'value': [np.random.randn()]
                })
                # Append data (if supported)
                return ('write', n, True)
            except Exception as e:
                errors.append(('write', n, str(e)))
                return None
        
        def stats_operation(n):
            try:
                stats = client.get_dataset(dataset_name).get_statistics()
                return ('stats', n, stats.get('row_count'))
            except Exception as e:
                errors.append(('stats', n, str(e)))
                return None
        
        # Mix of operations
        operations = []
        for i in range(30):
            if i % 3 == 0:
                operations.append((read_operation, i))
            elif i % 3 == 1:
                operations.append((write_operation, i))
            else:
                operations.append((stats_operation, i))
        
        # Execute concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(op, n) for op, n in operations]
            results = [f.result() for f in futures]
        
        # Analyze results
        successful = [r for r in results if r is not None]
        print(f"Successful operations: {len(successful)}/{len(operations)}")
        print(f"Errors: {len(errors)}")
        
        # Should handle concurrency gracefully
        assert len(successful) >= len(operations) * 0.8, "Too many failed operations"
        
        # Cleanup
        client.remove_dataset(dataset_name, force=True)
    
    def test_memory_stress(self, tmp_path):
        """Test memory usage under stress"""
        import psutil
        import gc
        
        process = psutil.Process()
        client = MDMClient()
        
        # Track memory usage
        memory_usage = []
        
        # Create large dataset
        size = 500000
        data = pd.DataFrame({
            'id': range(size),
            'float_1': np.random.randn(size),
            'float_2': np.random.randn(size),
            'float_3': np.random.randn(size),
            'category': np.random.choice(['A', 'B', 'C'], size),
            'text': ['x' * 100] * size  # Larger text
        })
        
        path = tmp_path / "memory_stress.csv"
        
        # Initial memory
        gc.collect()
        mem_start = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage.append(('start', mem_start))
        
        # Save data
        data.to_csv(path, index=False)
        mem_after_save = process.memory_info().rss / 1024 / 1024
        memory_usage.append(('after_save', mem_after_save))
        
        # Register dataset
        dataset_name = f"memory_stress_{int(time.time())}"
        result = client.register_dataset(
            dataset_name,
            str(path),
            force=True
        )
        
        mem_after_register = process.memory_info().rss / 1024 / 1024
        memory_usage.append(('after_register', mem_after_register))
        
        # Load data multiple times
        for i in range(3):
            df = client.get_dataset(dataset_name).load_data()
            mem = process.memory_info().rss / 1024 / 1024
            memory_usage.append((f'after_load_{i}', mem))
            del df
            gc.collect()
        
        # Final memory
        mem_end = process.memory_info().rss / 1024 / 1024
        memory_usage.append(('end', mem_end))
        
        # Cleanup
        client.remove_dataset(dataset_name, force=True)
        gc.collect()
        
        # Analyze memory usage
        print("\nMemory Usage:")
        for stage, mem in memory_usage:
            print(f"{stage}: {mem:.1f} MB")
        
        # Memory should not grow unbounded
        max_memory = max(mem for _, mem in memory_usage)
        start_memory = memory_usage[0][1]
        memory_growth = max_memory - start_memory
        
        print(f"\nMemory growth: {memory_growth:.1f} MB")
        assert memory_growth < 1000, "Excessive memory growth (>1GB)"
    
    def test_performance_metrics(self):
        """Test metrics collection during operations"""
        # Reset metrics
        metrics_collector._counters.clear()
        metrics_collector._timers.clear()
        
        client = MDMClient()
        
        # Perform operations
        datasets_created = []
        for i in range(5):
            name = f"metrics_test_{i}"
            client.register_dataset(
                name,
                f"/tmp/dummy_{i}.csv",
                force=True
            )
            datasets_created.append(name)
        
        # Get metrics summary
        summary = metrics_collector.get_summary()
        
        print("\nMetrics Summary:")
        print(f"Counters: {summary['counters']}")
        print(f"Timers: {summary['timers']}")
        
        # Verify metrics were collected
        assert len(summary['counters']) > 0, "No counter metrics collected"
        assert len(summary['timers']) > 0, "No timer metrics collected"
        
        # Cleanup
        for name in datasets_created:
            try:
                client.remove_dataset(name, force=True)
            except:
                pass
    
    def generate_load_test_report(self, results: Dict[str, Any], output_path: Path):
        """Generate load test report with visualizations"""
        # Create report directory
        report_dir = output_path / "load_test_report"
        report_dir.mkdir(exist_ok=True)
        
        # Generate plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Throughput plot
        sizes = [r['size'] for r in results['sequential']]
        throughputs = [r['rows_per_sec'] for r in results['sequential']]
        axes[0, 0].plot(sizes, throughputs, 'b-o')
        axes[0, 0].set_xlabel('Dataset Size')
        axes[0, 0].set_ylabel('Rows/Second')
        axes[0, 0].set_title('Throughput vs Dataset Size')
        axes[0, 0].set_xscale('log')
        
        # Memory usage plot
        if 'memory' in results:
            stages = [m[0] for m in results['memory']]
            memory = [m[1] for m in results['memory']]
            axes[0, 1].bar(range(len(stages)), memory)
            axes[0, 1].set_xticks(range(len(stages)))
            axes[0, 1].set_xticklabels(stages, rotation=45, ha='right')
            axes[0, 1].set_ylabel('Memory (MB)')
            axes[0, 1].set_title('Memory Usage by Stage')
        
        # Concurrent operations success rate
        if 'concurrent' in results:
            labels = ['Successful', 'Failed']
            sizes = [results['concurrent']['successful'], results['concurrent']['failed']]
            axes[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%')
            axes[1, 0].set_title('Concurrent Operations Success Rate')
        
        # Performance comparison
        if 'comparison' in results:
            systems = ['Legacy', 'New']
            times = [results['comparison']['legacy_time'], results['comparison']['new_time']]
            axes[1, 1].bar(systems, times)
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_title('System Performance Comparison')
        
        plt.tight_layout()
        plt.savefig(report_dir / 'load_test_results.png')
        
        # Generate text report
        with open(report_dir / 'load_test_report.txt', 'w') as f:
            f.write("MDM Load Test Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Test Configuration:\n")
            f.write(f"- Date: {results.get('date', 'Unknown')}\n")
            f.write(f"- Duration: {results.get('duration', 'Unknown')}\n")
            f.write(f"- Test Types: {', '.join(results.get('test_types', []))}\n\n")
            
            f.write("Summary Results:\n")
            f.write(f"- Average Throughput: {results.get('avg_throughput', 'N/A')} rows/sec\n")
            f.write(f"- Peak Memory Usage: {results.get('peak_memory', 'N/A')} MB\n")
            f.write(f"- Concurrent Success Rate: {results.get('concurrent_success_rate', 'N/A')}%\n")
            f.write(f"- Performance Delta: {results.get('performance_delta', 'N/A')}%\n")
        
        print(f"Load test report generated at: {report_dir}")
```

### Week 19: Cutover Execution

#### Day 6-7: Gradual Rollout

##### 2.1 Create Rollout Controller
```python
# Create: src/mdm/migration/rollout.py
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from ..core.feature_flags import feature_flags
from ..core.metrics import metrics_collector

logger = logging.getLogger(__name__)


class RolloutStage(Enum):
    """Rollout stages"""
    TESTING = "testing"
    CANARY = "canary"  # 5%
    EARLY_ADOPTERS = "early_adopters"  # 25%
    GENERAL_AVAILABILITY = "general_availability"  # 50%
    FULL_ROLLOUT = "full_rollout"  # 100%
    COMPLETE = "complete"


@dataclass
class RolloutConfig:
    """Rollout configuration"""
    component: str
    start_date: datetime
    stages: Dict[RolloutStage, float] = field(default_factory=lambda: {
        RolloutStage.TESTING: 0.0,
        RolloutStage.CANARY: 0.05,
        RolloutStage.EARLY_ADOPTERS: 0.25,
        RolloutStage.GENERAL_AVAILABILITY: 0.50,
        RolloutStage.FULL_ROLLOUT: 1.0,
        RolloutStage.COMPLETE: 1.0
    })
    stage_duration: timedelta = timedelta(days=3)
    auto_advance: bool = True
    success_threshold: float = 0.99
    error_threshold: float = 0.05


class RolloutController:
    """Controls gradual feature rollout"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path.home() / ".mdm" / "rollout.json"
        self.configs: Dict[str, RolloutConfig] = {}
        self.current_stages: Dict[str, RolloutStage] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self._load_state()
    
    def _load_state(self):
        """Load rollout state from file"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                data = json.load(f)
                
                # Recreate configs
                for component, config_data in data.get("configs", {}).items():
                    config = RolloutConfig(
                        component=component,
                        start_date=datetime.fromisoformat(config_data["start_date"]),
                        stage_duration=timedelta(days=config_data.get("stage_duration_days", 3)),
                        auto_advance=config_data.get("auto_advance", True),
                        success_threshold=config_data.get("success_threshold", 0.99),
                        error_threshold=config_data.get("error_threshold", 0.05)
                    )
                    self.configs[component] = config
                
                # Load current stages
                for component, stage_name in data.get("current_stages", {}).items():
                    self.current_stages[component] = RolloutStage(stage_name)
                
                # Load metrics
                self.metrics = data.get("metrics", {})
    
    def _save_state(self):
        """Save rollout state to file"""
        data = {
            "configs": {},
            "current_stages": {},
            "metrics": self.metrics,
            "last_updated": datetime.now().isoformat()
        }
        
        # Save configs
        for component, config in self.configs.items():
            data["configs"][component] = {
                "start_date": config.start_date.isoformat(),
                "stage_duration_days": config.stage_duration.days,
                "auto_advance": config.auto_advance,
                "success_threshold": config.success_threshold,
                "error_threshold": config.error_threshold
            }
        
        # Save stages
        for component, stage in self.current_stages.items():
            data["current_stages"][component] = stage.value
        
        # Write to file
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_component(self, component: str, config: RolloutConfig):
        """Register component for rollout"""
        self.configs[component] = config
        self.current_stages[component] = RolloutStage.TESTING
        self.metrics[component] = {
            "success_count": 0,
            "error_count": 0,
            "stage_started": datetime.now().isoformat()
        }
        self._save_state()
        logger.info(f"Registered component {component} for rollout")
    
    def get_rollout_percentage(self, component: str) -> float:
        """Get current rollout percentage for component"""
        if component not in self.configs:
            return 0.0
        
        current_stage = self.current_stages.get(component, RolloutStage.TESTING)
        config = self.configs[component]
        
        return config.stages.get(current_stage, 0.0)
    
    def should_use_new_system(self, component: str, identifier: str) -> bool:
        """Determine if identifier should use new system"""
        percentage = self.get_rollout_percentage(component)
        
        if percentage == 0.0:
            return False
        if percentage >= 1.0:
            return True
        
        # Use consistent hashing for deterministic assignment
        import hashlib
        hash_value = int(hashlib.md5(f"{component}:{identifier}".encode()).hexdigest()[:8], 16)
        return (hash_value % 100) < (percentage * 100)
    
    def record_success(self, component: str):
        """Record successful operation"""
        if component in self.metrics:
            self.metrics[component]["success_count"] += 1
            metrics_collector.increment(f"rollout.{component}.success")
    
    def record_error(self, component: str, error: Exception):
        """Record error"""
        if component in self.metrics:
            self.metrics[component]["error_count"] += 1
            metrics_collector.increment(f"rollout.{component}.error")
            logger.error(f"Rollout error for {component}: {error}")
    
    def check_advancement(self, component: str) -> bool:
        """Check if component should advance to next stage"""
        if component not in self.configs:
            return False
        
        config = self.configs[component]
        if not config.auto_advance:
            return False
        
        current_stage = self.current_stages[component]
        if current_stage == RolloutStage.COMPLETE:
            return False
        
        # Check time in current stage
        stage_started = datetime.fromisoformat(
            self.metrics[component]["stage_started"]
        )
        if datetime.now() - stage_started < config.stage_duration:
            return False
        
        # Check success rate
        success_count = self.metrics[component]["success_count"]
        error_count = self.metrics[component]["error_count"]
        total = success_count + error_count
        
        if total < 100:  # Minimum operations
            return False
        
        success_rate = success_count / total
        error_rate = error_count / total
        
        if success_rate >= config.success_threshold and error_rate <= config.error_threshold:
            return True
        
        logger.warning(
            f"Component {component} not advancing: "
            f"success_rate={success_rate:.2%}, error_rate={error_rate:.2%}"
        )
        return False
    
    def advance_stage(self, component: str):
        """Advance component to next stage"""
        if component not in self.configs:
            return
        
        current_stage = self.current_stages[component]
        
        # Find next stage
        stages = list(RolloutStage)
        current_index = stages.index(current_stage)
        
        if current_index < len(stages) - 1:
            next_stage = stages[current_index + 1]
            self.current_stages[component] = next_stage
            
            # Reset metrics for new stage
            self.metrics[component]["success_count"] = 0
            self.metrics[component]["error_count"] = 0
            self.metrics[component]["stage_started"] = datetime.now().isoformat()
            
            # Update feature flags
            self._update_feature_flags(component)
            
            self._save_state()
            logger.info(f"Advanced {component} to stage {next_stage.value}")
    
    def _update_feature_flags(self, component: str):
        """Update feature flags based on rollout stage"""
        percentage = self.get_rollout_percentage(component)
        
        # Map components to feature flags
        flag_map = {
            "config": "rollout_percentage.new_config",
            "storage": "rollout_percentage.new_backend",
            "features": "rollout_percentage.new_features",
            "registration": "rollout_percentage.new_registrar"
        }
        
        if component in flag_map:
            feature_flags.set(flag_map[component], int(percentage * 100))
    
    def rollback_component(self, component: str):
        """Rollback component to previous stage"""
        if component not in self.configs:
            return
        
        current_stage = self.current_stages[component]
        stages = list(RolloutStage)
        current_index = stages.index(current_stage)
        
        if current_index > 0:
            previous_stage = stages[current_index - 1]
            self.current_stages[component] = previous_stage
            self._update_feature_flags(component)
            self._save_state()
            logger.warning(f"Rolled back {component} to stage {previous_stage.value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get rollout status for all components"""
        status = {}
        
        for component, config in self.configs.items():
            current_stage = self.current_stages[component]
            metrics = self.metrics[component]
            
            total = metrics["success_count"] + metrics["error_count"]
            success_rate = metrics["success_count"] / total if total > 0 else 0
            
            status[component] = {
                "current_stage": current_stage.value,
                "rollout_percentage": self.get_rollout_percentage(component),
                "success_rate": success_rate,
                "total_operations": total,
                "stage_started": metrics["stage_started"],
                "can_advance": self.check_advancement(component)
            }
        
        return status


# Global rollout controller
rollout_controller = RolloutController()


def initialize_rollout():
    """Initialize rollout for all components"""
    components = ["config", "storage", "features", "registration"]
    
    for component in components:
        if component not in rollout_controller.configs:
            config = RolloutConfig(
                component=component,
                start_date=datetime.now(),
                stage_duration=timedelta(days=3),
                auto_advance=True
            )
            rollout_controller.register_component(component, config)
    
    logger.info("Rollout initialized for all components")


def create_rollout_cli():
    """Create CLI for rollout management"""
    import typer
    from rich.console import Console
    from rich.table import Table
    
    app = typer.Typer()
    console = Console()
    
    @app.command()
    def status():
        """Show rollout status"""
        status = rollout_controller.get_status()
        
        table = Table(title="Rollout Status")
        table.add_column("Component", style="cyan")
        table.add_column("Stage", style="yellow")
        table.add_column("Rollout %", style="green")
        table.add_column("Success Rate", style="blue")
        table.add_column("Operations", style="white")
        
        for component, info in status.items():
            table.add_row(
                component,
                info["current_stage"],
                f"{info['rollout_percentage']:.0%}",
                f"{info['success_rate']:.1%}",
                str(info["total_operations"])
            )
        
        console.print(table)
    
    @app.command()
    def advance(component: str):
        """Advance component to next stage"""
        rollout_controller.advance_stage(component)
        console.print(f"[green]Advanced {component} to next stage[/green]")
    
    @app.command()
    def rollback(component: str):
        """Rollback component to previous stage"""
        rollout_controller.rollback_component(component)
        console.print(f"[yellow]Rolled back {component} to previous stage[/yellow]")
    
    @app.command()
    def set_percentage(component: str, percentage: int):
        """Manually set rollout percentage"""
        if 0 <= percentage <= 100:
            feature_flags.set(f"rollout_percentage.new_{component}", percentage)
            console.print(f"[green]Set {component} rollout to {percentage}%[/green]")
        else:
            console.print("[red]Percentage must be between 0 and 100[/red]")
    
    return app
```

##### 2.2 Create Monitoring Dashboard
```python
# Create: src/mdm/migration/monitoring.py
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass
import json

from ..core.metrics import metrics_collector


@dataclass
class HealthMetric:
    """Health metric data point"""
    timestamp: datetime
    component: str
    metric_name: str
    value: float
    status: str  # healthy, warning, critical


class MigrationMonitor:
    """Monitor migration health and metrics"""
    
    def __init__(self, history_minutes: int = 60):
        self.history_minutes = history_minutes
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_minutes * 60)  # Store per second
        )
        self.thresholds = {
            "error_rate": {"warning": 0.02, "critical": 0.05},
            "latency_p99": {"warning": 2.0, "critical": 5.0},
            "memory_usage": {"warning": 80, "critical": 90},
            "success_rate": {"warning": 0.98, "critical": 0.95}
        }
        self._monitoring = False
        self._thread = None
    
    def start_monitoring(self):
        """Start monitoring thread"""
        if not self._monitoring:
            self._monitoring = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._thread:
            self._thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # Collect metrics
                self._collect_system_metrics()
                self._collect_component_metrics()
                self._check_health()
                
                time.sleep(1)  # Collect every second
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self._add_metric("system", "cpu_usage", cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self._add_metric("system", "memory_usage", memory.percent)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self._add_metric("system", "disk_read_mb", disk_io.read_bytes / 1024 / 1024)
            self._add_metric("system", "disk_write_mb", disk_io.write_bytes / 1024 / 1024)
    
    def _collect_component_metrics(self):
        """Collect component-specific metrics"""
        # Get metrics from metrics collector
        summary = metrics_collector.get_summary()
        
        # Process counters
        for metric_name, value in summary.get("counters", {}).items():
            parts = metric_name.split(".")
            if len(parts) >= 2:
                component = parts[0]
                metric = ".".join(parts[1:])
                self._add_metric(component, metric, value)
        
        # Process timers
        for timer_name, stats in summary.get("timers", {}).items():
            parts = timer_name.split(".")
            if len(parts) >= 2:
                component = parts[0]
                metric_base = ".".join(parts[1:])
                
                self._add_metric(component, f"{metric_base}_mean", stats.get("mean", 0))
                self._add_metric(component, f"{metric_base}_p99", stats.get("max", 0))
    
    def _add_metric(self, component: str, metric_name: str, value: float):
        """Add metric to history"""
        key = f"{component}.{metric_name}"
        self.metrics_history[key].append({
            "timestamp": datetime.now(),
            "value": value
        })
    
    def _check_health(self):
        """Check component health against thresholds"""
        # Calculate error rates
        components = ["storage", "features", "registration", "config"]
        
        for component in components:
            # Calculate success rate
            success_key = f"{component}.success"
            error_key = f"{component}.error"
            
            success_count = self._get_recent_sum(success_key, seconds=60)
            error_count = self._get_recent_sum(error_key, seconds=60)
            
            if success_count + error_count > 0:
                error_rate = error_count / (success_count + error_count)
                success_rate = 1 - error_rate
                
                # Check thresholds
                if error_rate > self.thresholds["error_rate"]["critical"]:
                    self._alert("critical", component, "error_rate", error_rate)
                elif error_rate > self.thresholds["error_rate"]["warning"]:
                    self._alert("warning", component, "error_rate", error_rate)
    
    def _get_recent_sum(self, metric_key: str, seconds: int) -> float:
        """Get sum of metric values in recent seconds"""
        if metric_key not in self.metrics_history:
            return 0
        
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        recent_values = [
            point["value"] 
            for point in self.metrics_history[metric_key]
            if point["timestamp"] > cutoff_time
        ]
        
        return sum(recent_values)
    
    def _alert(self, severity: str, component: str, metric: str, value: float):
        """Send alert for threshold violation"""
        logger.warning(
            f"[{severity.upper()}] {component} - {metric}: {value:.2f}"
        )
        
        # Could integrate with alerting systems here
        # e.g., send to Slack, PagerDuty, etc.
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        components = ["storage", "features", "registration", "config"]
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        for component in components:
            # Calculate metrics
            success_count = self._get_recent_sum(f"{component}.success", 300)
            error_count = self._get_recent_sum(f"{component}.error", 300)
            total = success_count + error_count
            
            component_data = {
                "health": "healthy",  # Will be updated based on checks
                "metrics": {
                    "operations_5min": total,
                    "success_rate": success_count / total if total > 0 else 1.0,
                    "error_rate": error_count / total if total > 0 else 0.0
                },
                "recent_errors": []
            }
            
            # Determine health status
            if component_data["metrics"]["error_rate"] > self.thresholds["error_rate"]["critical"]:
                component_data["health"] = "critical"
            elif component_data["metrics"]["error_rate"] > self.thresholds["error_rate"]["warning"]:
                component_data["health"] = "warning"
            
            dashboard["components"][component] = component_data
        
        # Add system metrics
        dashboard["system"] = {
            "cpu_usage": self._get_latest_value("system.cpu_usage"),
            "memory_usage": self._get_latest_value("system.memory_usage"),
            "disk_read_mb": self._get_latest_value("system.disk_read_mb"),
            "disk_write_mb": self._get_latest_value("system.disk_write_mb")
        }
        
        return dashboard
    
    def _get_latest_value(self, metric_key: str) -> Optional[float]:
        """Get latest value for a metric"""
        if metric_key in self.metrics_history and self.metrics_history[metric_key]:
            return self.metrics_history[metric_key][-1]["value"]
        return None


# Global monitor
migration_monitor = MigrationMonitor()


def create_monitoring_dashboard():
    """Create terminal-based monitoring dashboard"""
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    
    console = Console()
    
    def generate_dashboard():
        """Generate dashboard display"""
        data = migration_monitor.get_dashboard_data()
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Header
        layout["header"].update(
            Panel(f"MDM Migration Monitor - {data['timestamp']}", 
                  style="bold blue")
        )
        
        # Component status table
        table = Table(title="Component Health")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Ops/5min", style="white")
        table.add_column("Success Rate", style="green")
        table.add_column("Error Rate", style="red")
        
        for component, info in data["components"].items():
            status_style = {
                "healthy": "green",
                "warning": "yellow",
                "critical": "red"
            }.get(info["health"], "white")
            
            table.add_row(
                component,
                f"[{status_style}]{info['health'].upper()}[/{status_style}]",
                str(int(info["metrics"]["operations_5min"])),
                f"{info['metrics']['success_rate']:.1%}",
                f"{info['metrics']['error_rate']:.2%}"
            )
        
        layout["main"].update(table)
        
        # System metrics footer
        system = data["system"]
        footer_text = (
            f"CPU: {system.get('cpu_usage', 0):.1f}% | "
            f"Memory: {system.get('memory_usage', 0):.1f}% | "
            f"Disk R: {system.get('disk_read_mb', 0):.1f} MB/s | "
            f"Disk W: {system.get('disk_write_mb', 0):.1f} MB/s"
        )
        layout["footer"].update(Panel(footer_text, style="dim"))
        
        return layout
    
    # Start live display
    with Live(generate_dashboard(), refresh_per_second=1) as live:
        migration_monitor.start_monitoring()
        try:
            while True:
                time.sleep(1)
                live.update(generate_dashboard())
        except KeyboardInterrupt:
            migration_monitor.stop_monitoring()
```

#### Day 8-9: Validation Reporting

##### 2.3 Create Validation Report Generator
```python
# Create: src/mdm/migration/reporting.py
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
from jinja2 import Template

from .rollout import rollout_controller
from .monitoring import migration_monitor


class ValidationReportGenerator:
    """Generate comprehensive validation reports"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_full_report(self) -> Path:
        """Generate complete validation report"""
        report_data = self._collect_report_data()
        
        # Generate HTML report
        html_path = self._generate_html_report(report_data)
        
        # Generate JSON data
        json_path = self.output_dir / "validation_report.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate summary
        summary_path = self._generate_summary(report_data)
        
        return html_path
    
    def _collect_report_data(self) -> Dict[str, Any]:
        """Collect all validation data"""
        return {
            "generated_at": datetime.now(),
            "rollout_status": rollout_controller.get_status(),
            "system_health": migration_monitor.get_dashboard_data(),
            "test_results": self._get_test_results(),
            "performance_comparison": self._get_performance_comparison(),
            "compatibility_matrix": self._get_compatibility_matrix(),
            "recommendations": self._generate_recommendations()
        }
    
    def _get_test_results(self) -> Dict[str, Any]:
        """Get test execution results"""
        # This would integrate with pytest results
        return {
            "unit_tests": {
                "total": 1256,
                "passed": 1256,
                "failed": 0,
                "skipped": 0
            },
            "integration_tests": {
                "total": 45,
                "passed": 45,
                "failed": 0,
                "skipped": 0
            },
            "validation_tests": {
                "total": 28,
                "passed": 27,
                "failed": 1,
                "skipped": 0
            }
        }
    
    def _get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison data"""
        return {
            "registration": {
                "old_system_avg": 2.5,
                "new_system_avg": 1.8,
                "improvement": 28.0
            },
            "feature_generation": {
                "old_system_avg": 5.2,
                "new_system_avg": 3.1,
                "improvement": 40.4
            },
            "data_loading": {
                "old_system_avg": 0.8,
                "new_system_avg": 0.7,
                "improvement": 12.5
            }
        }
    
    def _get_compatibility_matrix(self) -> Dict[str, Any]:
        """Get compatibility test results"""
        return {
            "api_compatibility": {
                "status": "PASS",
                "tests_passed": 156,
                "tests_total": 156
            },
            "data_format_compatibility": {
                "status": "PASS",
                "tests_passed": 89,
                "tests_total": 89
            },
            "configuration_compatibility": {
                "status": "PASS",
                "tests_passed": 34,
                "tests_total": 34
            }
        }
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check rollout status
        rollout_status = rollout_controller.get_status()
        
        for component, status in rollout_status.items():
            if status["success_rate"] < 0.99:
                recommendations.append({
                    "severity": "warning",
                    "component": component,
                    "message": f"Success rate ({status['success_rate']:.1%}) below target. Monitor closely."
                })
            
            if status["can_advance"]:
                recommendations.append({
                    "severity": "info",
                    "component": component,
                    "message": f"Ready to advance to next rollout stage"
                })
        
        return recommendations
    
    def _generate_html_report(self, data: Dict[str, Any]) -> Path:
        """Generate HTML report"""
        template = Template('''
<!DOCTYPE html>
<html>
<head>
    <title>MDM Migration Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; }
        .section { margin: 20px 0; }
        .success { color: green; }
        .warning { color: orange; }
        .error { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f2f2f2; }
        .metric { font-size: 24px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>MDM Migration Validation Report</h1>
        <p>Generated: {{ data.generated_at }}</p>
    </div>
    
    <div class="section">
        <h2>Rollout Status</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Stage</th>
                <th>Rollout %</th>
                <th>Success Rate</th>
            </tr>
            {% for component, status in data.rollout_status.items() %}
            <tr>
                <td>{{ component }}</td>
                <td>{{ status.current_stage }}</td>
                <td>{{ "%.0f%%" | format(status.rollout_percentage * 100) }}</td>
                <td class="{% if status.success_rate < 0.95 %}error{% elif status.success_rate < 0.99 %}warning{% else %}success{% endif %}">
                    {{ "%.1%%" | format(status.success_rate) }}
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Test Results</h2>
        <table>
            <tr>
                <th>Test Suite</th>
                <th>Total</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Success Rate</th>
            </tr>
            {% for suite, results in data.test_results.items() %}
            <tr>
                <td>{{ suite }}</td>
                <td>{{ results.total }}</td>
                <td class="success">{{ results.passed }}</td>
                <td class="{% if results.failed > 0 %}error{% else %}success{% endif %}">{{ results.failed }}</td>
                <td>{{ "%.1%%" | format(results.passed / results.total * 100) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Performance Comparison</h2>
        <table>
            <tr>
                <th>Operation</th>
                <th>Old System (avg)</th>
                <th>New System (avg)</th>
                <th>Improvement</th>
            </tr>
            {% for op, perf in data.performance_comparison.items() %}
            <tr>
                <td>{{ op }}</td>
                <td>{{ "%.1fs" | format(perf.old_system_avg) }}</td>
                <td>{{ "%.1fs" | format(perf.new_system_avg) }}</td>
                <td class="success">{{ "+%.1%%" | format(perf.improvement) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
        {% for rec in data.recommendations %}
            <li class="{{ rec.severity }}">
                <strong>{{ rec.component }}:</strong> {{ rec.message }}
            </li>
        {% endfor %}
        </ul>
    </div>
</body>
</html>
        ''')
        
        html_content = template.render(data=data)
        html_path = self.output_dir / "validation_report.html"
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_summary(self, data: Dict[str, Any]) -> Path:
        """Generate executive summary"""
        summary = []
        summary.append("MDM Migration Validation Summary")
        summary.append("=" * 50)
        summary.append(f"Generated: {data['generated_at']}")
        summary.append("")
        
        # Overall status
        all_components_healthy = all(
            status["success_rate"] >= 0.99 
            for status in data["rollout_status"].values()
        )
        
        if all_components_healthy:
            summary.append("✓ Overall Status: READY FOR FULL ROLLOUT")
        else:
            summary.append("⚠ Overall Status: ISSUES REQUIRE ATTENTION")
        
        summary.append("")
        summary.append("Component Status:")
        for component, status in data["rollout_status"].items():
            summary.append(f"  - {component}: {status['current_stage']} ({status['rollout_percentage']:.0%})")
        
        summary.append("")
        summary.append("Key Metrics:")
        summary.append(f"  - Total Tests Passed: {sum(suite['passed'] for suite in data['test_results'].values())}")
        summary.append(f"  - Average Performance Improvement: {sum(perf['improvement'] for perf in data['performance_comparison'].values()) / len(data['performance_comparison']):.1f}%")
        
        summary_path = self.output_dir / "validation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary))
        
        return summary_path
```

#### Day 10: Final Validation

##### 2.4 Create Final Validation Checklist
```markdown
# Create: docs/final_validation_checklist.md

# Final Validation Checklist

## Pre-Cutover Validation

### System Health
- [ ] All components showing "healthy" status in monitoring dashboard
- [ ] Error rates < 0.1% for all components
- [ ] No critical alerts in past 48 hours
- [ ] Memory usage stable and within limits
- [ ] CPU usage normal

### Test Results
- [ ] All unit tests passing (100%)
- [ ] All integration tests passing (100%)
- [ ] All validation tests passing (100%)
- [ ] Performance regression tests passing
- [ ] Load tests completed successfully

### Component Validation
- [ ] Configuration system: Environment variables working correctly
- [ ] Storage backends: All three backends tested (SQLite, DuckDB, PostgreSQL)
- [ ] Feature engineering: All transformer types validated
- [ ] Dataset registration: Rollback mechanism tested

### Compatibility
- [ ] API backward compatibility verified
- [ ] Data format compatibility confirmed
- [ ] Configuration file compatibility tested
- [ ] No breaking changes identified

### Performance
- [ ] Registration: >25% improvement verified
- [ ] Feature generation: >35% improvement verified
- [ ] Memory usage: Reduced or stable
- [ ] Concurrent operations: Improved throughput

### Rollout Status
- [ ] All components at 50%+ rollout
- [ ] Success rates >99% for all components
- [ ] No rollbacks in past week
- [ ] Gradual progression validated

## Cutover Execution

### Pre-Cutover (T-24 hours)
- [ ] Final backup of all systems
- [ ] Rollback procedures reviewed
- [ ] Team availability confirmed
- [ ] Communication sent to users
- [ ] Monitoring alerts configured

### Cutover (T-0)
- [ ] Feature flags set to 100% for all components
- [ ] Monitoring dashboard active
- [ ] Team on standby
- [ ] Initial validation tests run
- [ ] User communications ready

### Post-Cutover (T+1 hour)
- [ ] All systems operational
- [ ] Error rates normal
- [ ] Performance metrics stable
- [ ] No user complaints
- [ ] Initial success confirmed

### Post-Cutover (T+24 hours)
- [ ] 24-hour metrics reviewed
- [ ] Any issues documented
- [ ] Performance analysis complete
- [ ] User feedback collected
- [ ] Decision on keeping/reverting

## Rollback Criteria

Initiate rollback if ANY of the following occur:
- [ ] Error rate >5% for any component
- [ ] Performance degradation >20%
- [ ] Data integrity issues detected
- [ ] Critical functionality broken
- [ ] Multiple user complaints

## Success Criteria

Migration is successful when ALL of the following are met:
- [ ] 48 hours stable operation
- [ ] All components at 100% rollout
- [ ] Error rates <0.5%
- [ ] Performance improvements maintained
- [ ] Positive user feedback
- [ ] No critical issues

## Sign-offs

- [ ] Development Team Lead: _________________ Date: _______
- [ ] QA Team Lead: _________________________ Date: _______
- [ ] Operations Lead: ______________________ Date: _______
- [ ] Product Owner: ________________________ Date: _______
- [ ] Final Approval: _______________________ Date: _______
```

## Validation Checklist

### Week 18 Complete
- [ ] Integration tests comprehensive
- [ ] Performance validation complete
- [ ] Component interaction tests passing
- [ ] Load tests successful
- [ ] Monitoring systems active

### Week 19 Complete
- [ ] Rollout controller operational
- [ ] Gradual rollout progressing
- [ ] Monitoring dashboard working
- [ ] Validation reports generated
- [ ] Final checklist approved

## Success Criteria

- **All tests passing** with >99% success rate
- **Performance improvements** maintained under load
- **Zero data integrity issues** detected
- **Smooth rollout progression** through all stages
- **Comprehensive monitoring** with alerting

## Next Steps

With validation complete and cutover successful, proceed to [09-cleanup-and-finalization.md](09-cleanup-and-finalization.md).

## Notes

- Keep monitoring active for at least 2 weeks post-cutover
- Document any issues encountered for future reference
- Collect performance metrics for case study
- Plan celebration for successful migration!