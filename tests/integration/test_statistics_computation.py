"""Integration tests for statistics computation."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from mdm.api import MDMClient
from mdm.dataset.registrar import DatasetRegistrar
from mdm.dataset.statistics import DatasetStatistics


class TestStatisticsComputation:
    """Test statistics computation on real datasets."""

    @pytest.fixture
    def complex_dataset(self, tmp_path):
        """Create a complex dataset with various data types."""
        data_dir = tmp_path / "complex_data"
        data_dir.mkdir()
        
        # Create dataset with multiple data types
        np.random.seed(42)
        n_rows = 1000
        
        train_data = pd.DataFrame({
            'id': range(n_rows),
            'numeric_int': np.random.randint(0, 100, n_rows),
            'numeric_float': np.random.normal(50, 15, n_rows),
            'categorical_low': np.random.choice(['A', 'B', 'C'], n_rows),
            'categorical_high': [f'cat_{i%50}' for i in range(n_rows)],
            'text_data': [f'This is text sample {i}' for i in range(n_rows)],
            'datetime': pd.date_range('2024-01-01', periods=n_rows, freq='h'),
            'boolean': np.random.choice([True, False], n_rows),
            'missing_data': [np.nan if i % 10 == 0 else i for i in range(n_rows)],
            'target': np.random.choice([0, 1], n_rows)
        })
        
        # Add some duplicates
        train_data.loc[100:104] = train_data.loc[0:4].values
        
        # Save files
        train_data.to_csv(data_dir / 'train.csv', index=False)
        
        # Create smaller test set
        test_data = train_data.sample(200, random_state=42)
        test_data.to_csv(data_dir / 'test.csv', index=False)
        
        return data_dir

    def test_full_statistics_computation(self, complex_dataset, test_config):
        """Test computing full statistics on complex dataset."""
        client = MDMClient(config=test_config)
        registrar = DatasetRegistrar()
        
        # Register dataset
        dataset_info = registrar.register(
            name="stats_test",
            path=complex_dataset,
            auto_detect=True,
            target_column="target",
            id_columns=["id"]
        )
        
        # Compute full statistics
        stats = client.get_statistics("stats_test", full=True)
        
        assert stats is not None
        assert stats['dataset_name'] == "stats_test"
        assert stats['mode'] == 'full'
        
        # Check table statistics
        assert 'train' in stats['tables']
        train_stats = stats['tables']['train']
        
        # Basic counts
        assert train_stats['row_count'] == 1000
        assert train_stats['column_count'] == 10
        
        # Column statistics
        assert 'numeric_float' in train_stats['columns']
        float_stats = train_stats['columns']['numeric_float']
        assert 'mean' in float_stats
        assert 'std' in float_stats
        assert 'percentiles' in float_stats  # Full mode includes percentiles
        
        # Categorical statistics
        assert 'categorical_low' in train_stats['columns']
        cat_stats = train_stats['columns']['categorical_low']
        assert 'unique_values' in cat_stats
        assert 'cardinality' in cat_stats  # Full mode includes cardinality
        
        # Missing values
        assert train_stats['missing_values']['total_missing_cells'] > 0
        assert 'missing_data' in train_stats['missing_values']['columns_with_missing']
        
        # Correlations (full mode)
        assert 'correlations' in train_stats
        assert 'numeric_columns' in train_stats['correlations']
        
        # Data quality (full mode)
        assert 'data_quality' in train_stats
        assert train_stats['data_quality']['duplicate_rows'] == 5
        
        # Enhanced statistics (full mode)
        if 'enhanced_analysis' in train_stats:
            assert 'profile_summary' in train_stats['enhanced_analysis']
            assert 'variable_types' in train_stats['enhanced_analysis']
        
        # Summary
        assert 'summary' in stats
        assert stats['summary']['total_rows'] > 0
        assert stats['summary']['total_tables'] == 2
        
        # Cleanup
        client.remove_dataset("stats_test", force=True)

    def test_basic_statistics_computation(self, complex_dataset, test_config):
        """Test computing basic statistics (fast mode)."""
        client = MDMClient(config=test_config)
        registrar = DatasetRegistrar()
        
        # Register dataset
        registrar.register(
            name="stats_basic",
            path=complex_dataset,
            auto_detect=True
        )
        
        # Compute basic statistics
        stats = client.get_statistics("stats_basic", full=False)
        
        assert stats is not None
        assert stats['mode'] == 'basic'
        
        train_stats = stats['tables']['train']
        
        # Basic stats should not include percentiles or correlations
        float_stats = train_stats['columns']['numeric_float']
        assert 'mean' in float_stats
        assert 'percentiles' not in float_stats
        
        assert 'correlations' not in train_stats
        assert 'data_quality' not in train_stats
        
        # Cleanup
        client.remove_dataset("stats_basic", force=True)

    def test_statistics_with_edge_cases(self, tmp_path, test_config):
        """Test statistics computation with edge cases."""
        data_dir = tmp_path / "edge_cases"
        data_dir.mkdir()
        
        # Create dataset with edge cases
        edge_data = pd.DataFrame({
            'all_null': [np.nan] * 100,
            'all_same': [42] * 100,
            'all_unique': range(100),
            'infinite_values': [float('inf') if i % 20 == 0 else i for i in range(100)],
            'mixed_types': ['text' if i % 2 == 0 else i for i in range(100)],
            'empty_string': [''] * 100,
            'very_long_text': ['x' * 1000] * 100
        })
        
        edge_data.to_csv(data_dir / 'data.csv', index=False)
        
        client = MDMClient(config=test_config)
        registrar = DatasetRegistrar()
        
        # Register dataset
        registrar.register(
            name="edge_stats",
            path=data_dir,
            auto_detect=True
        )
        
        # Compute statistics
        stats = client.get_statistics("edge_stats", full=True)
        
        assert stats is not None
        data_stats = stats['tables']['data']
        
        # Check handling of edge cases
        assert data_stats['columns']['all_null']['null_percentage'] == 100.0
        assert data_stats['columns']['all_same']['unique_count'] == 1
        assert data_stats['columns']['all_unique']['unique_count'] == 100
        
        # Cleanup
        client.remove_dataset("edge_stats", force=True)

    def test_statistics_persistence(self, complex_dataset, test_config):
        """Test saving and loading statistics."""
        stats_computer = DatasetStatistics()
        registrar = DatasetRegistrar()
        
        # Register dataset
        registrar.register(
            name="stats_persist",
            path=complex_dataset,
            auto_detect=True
        )
        
        # Compute and save statistics
        stats1 = stats_computer.compute_statistics("stats_persist", full=True, save=True)
        
        # Load saved statistics
        stats2 = stats_computer.load_statistics("stats_persist")
        
        assert stats2 is not None
        assert stats2['dataset_name'] == stats1['dataset_name']
        assert stats2['mode'] == stats1['mode']
        assert len(stats2['tables']) == len(stats1['tables'])
        
        # Cleanup
        client = MDMClient(config=test_config)
        client.remove_dataset("stats_persist", force=True)

    def test_statistics_performance(self, tmp_path, test_config):
        """Test statistics computation performance on larger dataset."""
        import time
        
        data_dir = tmp_path / "large_data"
        data_dir.mkdir()
        
        # Create larger dataset
        n_rows = 50000
        large_data = pd.DataFrame({
            'id': range(n_rows),
            'col1': np.random.randn(n_rows),
            'col2': np.random.randn(n_rows),
            'col3': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
            'col4': pd.date_range('2020-01-01', periods=n_rows, freq='min')
        })
        
        large_data.to_csv(data_dir / 'data.csv', index=False)
        
        client = MDMClient(config=test_config)
        registrar = DatasetRegistrar()
        
        # Register dataset
        registrar.register(
            name="stats_perf",
            path=data_dir,
            auto_detect=True
        )
        
        # Time basic statistics
        start = time.time()
        basic_stats = client.get_statistics("stats_perf", full=False)
        basic_time = time.time() - start
        
        # Time full statistics
        start = time.time()
        full_stats = client.get_statistics("stats_perf", full=True)
        full_time = time.time() - start
        
        assert basic_stats is not None
        assert full_stats is not None
        
        # Basic should be faster than full
        assert basic_time < full_time
        
        # Both should complete in reasonable time
        assert basic_time < 5  # 5 seconds for basic
        assert full_time < 30  # 30 seconds for full
        
        # Cleanup
        client.remove_dataset("stats_perf", force=True)

    def test_statistics_error_recovery(self, tmp_path, test_config):
        """Test statistics computation error recovery."""
        data_dir = tmp_path / "corrupt_data"
        data_dir.mkdir()
        
        # Create dataset with potential issues
        corrupt_data = pd.DataFrame({
            'id': range(100),
            'problematic': ['corrupt' if i == 50 else i for i in range(100)]
        })
        
        corrupt_data.to_csv(data_dir / 'data.csv', index=False)
        
        client = MDMClient(config=test_config)
        registrar = DatasetRegistrar()
        
        # Register dataset
        registrar.register(
            name="stats_error",
            path=data_dir,
            auto_detect=True
        )
        
        # Should handle errors gracefully
        stats = client.get_statistics("stats_error")
        assert stats is not None
        
        # Should at least return basic structure
        assert 'dataset_name' in stats
        assert 'tables' in stats
        
        # Cleanup
        client.remove_dataset("stats_error", force=True)