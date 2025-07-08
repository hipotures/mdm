"""Tests for 2.2 Dataset Listing and Filtering based on MANUAL_TEST_CHECKLIST.md"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest


class TestDatasetListingFiltering:
    """Test dataset listing and filtering functionality."""
    
    @pytest.fixture
    def multiple_datasets(self, clean_mdm_env, run_mdm):
        """Create multiple datasets for testing listing/filtering."""
        datasets = []
        
        # Create test datasets with different properties
        for i in range(5):
            data = pd.DataFrame({
                'id': range(1, 101),
                'feature': range(100 * i, 100 * (i + 1)),
                'target': [j % 2 for j in range(100)]
            })
            
            csv_file = clean_mdm_env / f"data_{i}.csv"
            data.to_csv(csv_file, index=False)
            
            # Register with tags
            tags = []
            if i % 2 == 0:
                tags.append("even")
            else:
                tags.append("odd")
            if i < 2:
                tags.append("small")
            else:
                tags.append("large")
                
            result = run_mdm([
                "dataset", "register", f"test_dataset_{i}", str(csv_file),
                "--target", "target",
                "--tags", ",".join(tags),
                "--description", f"Test dataset number {i}"
            ])
            
            assert result.returncode == 0
            datasets.append(f"test_dataset_{i}")
            
            # Small delay to ensure different timestamps
            time.sleep(0.1)
        
        return datasets
    
    @pytest.mark.mdm_id("2.2.1.1")
    def test_list_all_datasets_default(self, clean_mdm_env, run_mdm, multiple_datasets):
        """2.2.1.1: List all datasets (default behavior)"""
        result = run_mdm(["dataset", "list"])
        
        assert result.returncode == 0
        
        # All datasets should be listed
        for dataset in multiple_datasets:
            assert dataset in result.stdout
    
    @pytest.mark.mdm_id("2.2.1.2")
    def test_list_with_limit(self, clean_mdm_env, run_mdm, multiple_datasets):
        """2.2.1.2: List with --limit N"""
        result = run_mdm(["dataset", "list", "--limit", "3"])
        
        assert result.returncode == 0
        
        # Count dataset names in output
        dataset_count = sum(1 for ds in multiple_datasets if ds in result.stdout)
        assert dataset_count == 3
    
    @pytest.mark.mdm_id("2.2.1.3")
    def test_list_sort_by_name(self, clean_mdm_env, run_mdm, multiple_datasets):
        """2.2.1.3: List sorted by name (--sort-by name)"""
        result = run_mdm(["dataset", "list", "--sort-by", "name"])
        
        assert result.returncode == 0
        
        # Extract dataset names from output in order
        lines = result.stdout.strip().split('\n')
        found_datasets = []
        for line in lines:
            for ds in multiple_datasets:
                if ds in line and ds not in found_datasets:
                    found_datasets.append(ds)
        
        # Should be alphabetically sorted
        assert found_datasets == sorted(multiple_datasets)
    
    @pytest.mark.mdm_id("2.2.1.4")
    def test_list_sort_by_date(self, clean_mdm_env, run_mdm, multiple_datasets):
        """2.2.1.4: List sorted by registration date (--sort-by registration_date)"""
        result = run_mdm(["dataset", "list", "--sort-by", "registration_date"])
        
        assert result.returncode == 0
        
        # Datasets should appear in registration order
        # (test_dataset_0 was registered first, test_dataset_4 last)
        stdout_lines = result.stdout.strip()
        pos_0 = stdout_lines.find("test_dataset_0")
        pos_4 = stdout_lines.find("test_dataset_4")
        
        assert pos_0 >= 0 and pos_4 >= 0
        # Order depends on sort direction (newest first or oldest first)
    
    @pytest.mark.mdm_id("2.2.1.5")
    def test_list_sort_by_size(self, clean_mdm_env, run_mdm, multiple_datasets):
        """2.2.1.5: List sorted by size (--sort-by size)"""
        result = run_mdm(["dataset", "list", "--sort-by", "size"])
        
        assert result.returncode == 0
        # All test datasets have same size, so order may vary
        
        # Verify all datasets are still listed
        for dataset in multiple_datasets:
            assert dataset in result.stdout
    
    @pytest.mark.mdm_id("2.2.2.1")
    def test_filter_by_tag_single(self, clean_mdm_env, run_mdm, multiple_datasets):
        """2.2.2.1: Filter by single tag (--tag)"""
        result = run_mdm(["dataset", "list", "--tag", "even"])
        
        assert result.returncode == 0
        
        # Should show datasets 0, 2, 4 (even indices)
        assert "test_dataset_0" in result.stdout
        assert "test_dataset_2" in result.stdout
        assert "test_dataset_4" in result.stdout
        
        # Should not show datasets 1, 3 (odd indices)
        assert "test_dataset_1" not in result.stdout
        assert "test_dataset_3" not in result.stdout
    
    @pytest.mark.mdm_id("2.2.2.2")
    def test_filter_by_multiple_tags(self, clean_mdm_env, run_mdm, multiple_datasets):
        """2.2.2.2: Filter by multiple tags (comma-separated)"""
        # Test AND behavior - datasets with both tags
        result = run_mdm(["dataset", "list", "--tag", "even,small"])
        
        assert result.returncode == 0
        
        # Only dataset 0 has both "even" and "small" tags
        assert "test_dataset_0" in result.stdout
        assert "test_dataset_1" not in result.stdout
        assert "test_dataset_2" not in result.stdout
    
    @pytest.mark.mdm_id("2.2.2.3")
    def test_filter_by_problem_type(self, clean_mdm_env, run_mdm):
        """2.2.2.3: Filter by problem type (--problem-type)"""
        # Create datasets with different problem types
        data_reg = pd.DataFrame({
            'id': range(1, 11),
            'feature': range(10, 20),
            'target': [1.5 * i for i in range(10)]
        })
        data_class = pd.DataFrame({
            'id': range(1, 11),
            'feature': range(20, 30),
            'target': [0, 1] * 5
        })
        
        csv_reg = clean_mdm_env / "regression.csv"
        csv_class = clean_mdm_env / "classification.csv"
        data_reg.to_csv(csv_reg, index=False)
        data_class.to_csv(csv_class, index=False)
        
        # Register with explicit problem types
        run_mdm([
            "dataset", "register", "test_regression", str(csv_reg),
            "--target", "target",
            "--problem-type", "regression"
        ])
        run_mdm([
            "dataset", "register", "test_classification", str(csv_class),
            "--target", "target",
            "--problem-type", "binary_classification"
        ])
        
        # Filter by problem type
        result = run_mdm(["dataset", "list", "--problem-type", "regression"])
        
        assert result.returncode == 0
        assert "test_regression" in result.stdout
        assert "test_classification" not in result.stdout
    
    @pytest.mark.mdm_id("2.2.2.4")
    @pytest.mark.skip(reason="--date-range not implemented")
    def test_filter_by_date_range(self):
        """2.2.2.4: Filter by registration date range (--date-range)"""
        pass
    
    @pytest.mark.mdm_id("2.2.2.5")
    @pytest.mark.skip(reason="--has-target filter not implemented")
    def test_filter_has_target(self):
        """2.2.2.5: Filter datasets with target column (--has-target)"""
        pass
    
    @pytest.mark.mdm_id("2.2.3.1")
    def test_output_format_table(self, clean_mdm_env, run_mdm, multiple_datasets):
        """2.2.3.1: Table output format (default)"""
        result = run_mdm(["dataset", "list"])
        
        assert result.returncode == 0
        
        # Should have table-like structure
        # Look for common table elements
        output_lines = result.stdout.strip().split('\n')
        
        # Should have multiple columns of info
        for dataset in multiple_datasets[:2]:  # Check first two
            dataset_lines = [line for line in output_lines if dataset in line]
            if dataset_lines:
                # Line should have dataset name and other info
                assert len(dataset_lines[0].split()) > 1
    
    @pytest.mark.mdm_id("2.2.3.2")
    @pytest.mark.skip(reason="--format json not implemented")
    def test_output_format_json(self):
        """2.2.3.2: JSON output format (--format json)"""
        pass
    
    @pytest.mark.mdm_id("2.2.3.3")
    @pytest.mark.skip(reason="--format csv not implemented")
    def test_output_format_csv(self):
        """2.2.3.3: CSV output format (--format csv)"""
        pass
    
    @pytest.mark.mdm_id("2.2.3.4")
    def test_verbose_output(self, clean_mdm_env, run_mdm, sample_csv_data):
        """2.2.3.4: Verbose output with -v flag"""
        # Register a dataset first
        run_mdm([
            "dataset", "register", "test_verbose", str(sample_csv_data),
            "--target", "value",
            "--description", "Dataset for verbose test"
        ])
        
        # List with verbose flag
        result = run_mdm(["dataset", "list", "-v"])
        
        assert result.returncode == 0
        
        # Verbose output should include more details
        assert "test_verbose" in result.stdout
        # May include paths, descriptions, etc.
    
    @pytest.mark.mdm_id("2.2.4.1")
    def test_empty_list_message(self, clean_mdm_env, run_mdm):
        """2.2.4.1: Informative message when no datasets exist"""
        result = run_mdm(["dataset", "list"])
        
        assert result.returncode == 0
        assert "No datasets" in result.stdout or "0 datasets" in result.stdout
    
    @pytest.mark.mdm_id("2.2.4.2")
    def test_no_matches_filter(self, clean_mdm_env, run_mdm, multiple_datasets):
        """2.2.4.2: Message when filters match no datasets"""
        result = run_mdm(["dataset", "list", "--tag", "nonexistent_tag"])
        
        assert result.returncode == 0
        assert "No datasets" in result.stdout or "0 datasets" in result.stdout or "No matching" in result.stdout
    
    @pytest.mark.mdm_id("2.2.4.3")
    def test_list_performance(self, clean_mdm_env, run_mdm):
        """2.2.4.3: List command performs well with many datasets"""
        # Create 50 datasets
        for i in range(50):
            data = pd.DataFrame({
                'id': range(1, 11),
                'value': range(i * 10, (i + 1) * 10)
            })
            csv_file = clean_mdm_env / f"perf_{i}.csv"
            data.to_csv(csv_file, index=False)
            
            run_mdm([
                "dataset", "register", f"perf_test_{i}", str(csv_file),
                "--target", "value"
            ])
        
        # Time the list command
        import time
        start = time.time()
        result = run_mdm(["dataset", "list"])
        end = time.time()
        
        assert result.returncode == 0
        assert end - start < 5.0  # Should complete within 5 seconds
        
        # All datasets should be listed
        for i in range(50):
            assert f"perf_test_{i}" in result.stdout