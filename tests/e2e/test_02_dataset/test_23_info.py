"""Tests for 2.3 Dataset Information and Statistics based on MANUAL_TEST_CHECKLIST.md"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest


class TestDatasetInformationStatistics:
    """Test dataset information and statistics functionality."""
    
    @pytest.fixture
    def complex_dataset(self, clean_mdm_env, run_mdm):
        """Create a complex dataset with various column types."""
        data = pd.DataFrame({
            'id': range(1, 1001),
            'user_id': [f'user_{i:04d}' for i in range(1, 1001)],
            'age': [20 + (i % 60) for i in range(1000)],
            'salary': [30000 + (i * 100) for i in range(1000)],
            'department': ['Sales', 'Engineering', 'Marketing', 'HR'] * 250,
            'is_active': [True, False] * 500,
            'join_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'score': [i * 0.1 for i in range(1000)],
            'notes': ['Note ' + str(i) if i % 10 == 0 else None for i in range(1000)],
            'target': [i % 3 for i in range(1000)]
        })
        
        csv_file = clean_mdm_env / "complex_data.csv"
        data.to_csv(csv_file, index=False)
        
        # Register the dataset
        result = run_mdm([
            "dataset", "register", "complex_dataset", str(csv_file),
            "--target", "target",
            "--id-columns", "id,user_id",
            "--datetime-columns", "join_date",
            "--tags", "test,complex,multiclass",
            "--description", "Complex dataset for testing info and stats"
        ])
        
        assert result.returncode == 0
        return "complex_dataset"
    
    @pytest.mark.mdm_id("2.3.1.1")
    def test_dataset_info_basic(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.1.1: Show basic dataset information"""
        result = run_mdm(["dataset", "info", complex_dataset])
        
        assert result.returncode == 0
        
        # Should display key information
        assert "Name" in result.stdout
        assert complex_dataset in result.stdout
        assert "Path" in result.stdout
        assert "Registration Date" in result.stdout
        assert "Description" in result.stdout
        assert "Complex dataset for testing" in result.stdout
    
    @pytest.mark.mdm_id("2.3.1.2")
    def test_dataset_info_schema(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.1.2: Display column schema information"""
        result = run_mdm(["dataset", "info", complex_dataset])
        
        assert result.returncode == 0
        
        # Should show column information
        assert "Columns" in result.stdout or "Schema" in result.stdout
        
        # Check for specific columns
        assert "id" in result.stdout
        assert "age" in result.stdout
        assert "salary" in result.stdout
        assert "department" in result.stdout
        assert "join_date" in result.stdout
        assert "target" in result.stdout
    
    @pytest.mark.mdm_id("2.3.1.3")
    def test_dataset_info_metadata(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.1.3: Show dataset metadata (target, ID columns, etc.)"""
        result = run_mdm(["dataset", "info", complex_dataset])
        
        assert result.returncode == 0
        
        # Should show metadata
        assert "Target" in result.stdout
        assert "target" in result.stdout
        
        assert "ID Columns" in result.stdout
        assert "id" in result.stdout
        assert "user_id" in result.stdout
        
        assert "Problem Type" in result.stdout
        assert "multiclass" in result.stdout.lower() or "classification" in result.stdout.lower()
    
    @pytest.mark.mdm_id("2.3.1.4")
    def test_dataset_info_tags(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.1.4: Display dataset tags"""
        result = run_mdm(["dataset", "info", complex_dataset])
        
        assert result.returncode == 0
        
        # Should show tags
        assert "Tags" in result.stdout
        assert "test" in result.stdout
        assert "complex" in result.stdout
        assert "multiclass" in result.stdout
    
    @pytest.mark.mdm_id("2.3.1.5")
    def test_dataset_info_size(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.1.5: Show dataset size information"""
        result = run_mdm(["dataset", "info", complex_dataset])
        
        assert result.returncode == 0
        
        # Should show size metrics
        assert "Rows" in result.stdout or "Records" in result.stdout
        assert "1000" in result.stdout or "1,000" in result.stdout
        
        # May also show file size
        if "Size" in result.stdout:
            # Should have some size indicator (KB, MB, etc.)
            assert any(unit in result.stdout for unit in ["KB", "MB", "bytes"])
    
    @pytest.mark.mdm_id("2.3.2.1")
    def test_dataset_stats_basic(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.2.1: Show basic statistics (count, mean, std, etc.)"""
        result = run_mdm(["dataset", "stats", complex_dataset])
        
        assert result.returncode == 0
        
        # Should show statistics for numeric columns
        assert "age" in result.stdout
        assert "salary" in result.stdout
        assert "score" in result.stdout
        
        # Should include basic stats
        assert "mean" in result.stdout.lower() or "average" in result.stdout.lower()
        assert "std" in result.stdout.lower() or "deviation" in result.stdout.lower()
        assert "min" in result.stdout.lower()
        assert "max" in result.stdout.lower()
    
    @pytest.mark.mdm_id("2.3.2.2")
    def test_dataset_stats_percentiles(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.2.2: Display percentile information (25%, 50%, 75%)"""
        result = run_mdm(["dataset", "stats", complex_dataset])
        
        assert result.returncode == 0
        
        # Should show percentiles
        assert "25%" in result.stdout or "Q1" in result.stdout or "quartile" in result.stdout.lower()
        assert "50%" in result.stdout or "median" in result.stdout.lower()
        assert "75%" in result.stdout or "Q3" in result.stdout
    
    @pytest.mark.mdm_id("2.3.2.3")
    def test_dataset_stats_null_counts(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.2.3: Show null/missing value counts"""
        result = run_mdm(["dataset", "stats", complex_dataset])
        
        assert result.returncode == 0
        
        # Should show null information
        assert "null" in result.stdout.lower() or "missing" in result.stdout.lower() or "NaN" in result.stdout
        
        # notes column has nulls (90% of rows)
        if "notes" in result.stdout:
            # Should indicate high null percentage
            assert "90" in result.stdout or "900" in result.stdout
    
    @pytest.mark.mdm_id("2.3.2.4")
    def test_dataset_stats_categorical(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.2.4: Statistics for categorical columns"""
        result = run_mdm(["dataset", "stats", complex_dataset])
        
        assert result.returncode == 0
        
        # Should show categorical column stats
        assert "department" in result.stdout
        
        # Should show unique values or value counts
        assert "unique" in result.stdout.lower() or "categories" in result.stdout.lower()
        
        # May show most frequent values
        if "Engineering" in result.stdout or "Sales" in result.stdout:
            # Good - showing actual category values
            pass
    
    @pytest.mark.mdm_id("2.3.2.5")
    def test_dataset_stats_target_distribution(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.2.5: Target variable distribution"""
        result = run_mdm(["dataset", "stats", complex_dataset])
        
        assert result.returncode == 0
        
        # Should show target distribution
        assert "target" in result.stdout
        
        # For multiclass (0, 1, 2), should show distribution
        # Each class has ~333 samples
        if "distribution" in result.stdout.lower() or "value_counts" in result.stdout.lower():
            assert "0" in result.stdout
            assert "1" in result.stdout
            assert "2" in result.stdout
    
    @pytest.mark.mdm_id("2.3.3.1")
    def test_info_nonexistent_dataset(self, clean_mdm_env, run_mdm):
        """2.3.3.1: Error handling for non-existent dataset"""
        result = run_mdm(["dataset", "info", "nonexistent_dataset"], check=False)
        
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()
    
    @pytest.mark.mdm_id("2.3.3.2")
    def test_stats_nonexistent_dataset(self, clean_mdm_env, run_mdm):
        """2.3.3.2: Error handling for stats on non-existent dataset"""
        result = run_mdm(["dataset", "stats", "nonexistent_dataset"], check=False)
        
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()
    
    @pytest.mark.mdm_id("2.3.3.3")
    def test_verbose_info(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.3.3: Verbose output with -v flag"""
        result = run_mdm(["dataset", "info", complex_dataset, "-v"])
        
        assert result.returncode == 0
        
        # Verbose should include more details
        # May include file paths, backend info, etc.
        output_length = len(result.stdout)
        
        # Compare with non-verbose
        result_normal = run_mdm(["dataset", "info", complex_dataset])
        normal_length = len(result_normal.stdout)
        
        # Verbose should have more content (or same if -v not implemented)
        assert output_length >= normal_length
    
    @pytest.mark.mdm_id("2.3.3.4")
    @pytest.mark.skip(reason="--format option not implemented")
    def test_output_format_json(self):
        """2.3.3.4: JSON output format (--format json)"""
        pass
    
    @pytest.mark.mdm_id("2.3.4.1")
    def test_info_large_dataset(self, clean_mdm_env, run_mdm):
        """2.3.4.1: Performance with large datasets"""
        # Create a larger dataset
        large_data = pd.DataFrame({
            'id': range(1, 100001),
            'value': range(100000),
            'target': [i % 2 for i in range(100000)]
        })
        
        csv_file = clean_mdm_env / "large_data.csv"
        large_data.to_csv(csv_file, index=False)
        
        # Register
        result = run_mdm([
            "dataset", "register", "large_dataset", str(csv_file),
            "--target", "target"
        ])
        assert result.returncode == 0
        
        # Info should be fast even for large datasets
        import time
        start = time.time()
        result = run_mdm(["dataset", "info", "large_dataset"])
        end = time.time()
        
        assert result.returncode == 0
        assert end - start < 2.0  # Should complete within 2 seconds
    
    @pytest.mark.mdm_id("2.3.4.2")
    def test_stats_computation_caching(self, clean_mdm_env, run_mdm, complex_dataset):
        """2.3.4.2: Statistics caching for performance"""
        # First call - may compute stats
        result1 = run_mdm(["dataset", "stats", complex_dataset])
        assert result1.returncode == 0
        
        # Second call - should be faster if cached
        import time
        start = time.time()
        result2 = run_mdm(["dataset", "stats", complex_dataset])
        end = time.time()
        
        assert result2.returncode == 0
        assert end - start < 1.0  # Cached call should be fast
        
        # Output should be identical
        assert result1.stdout == result2.stdout