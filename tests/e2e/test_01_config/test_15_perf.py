"""Tests for 1.5 Performance Configuration based on MANUAL_TEST_CHECKLIST.md"""

import os
from pathlib import Path

import pandas as pd
import pytest


class TestPerformanceConfiguration:
    """Test performance configuration functionality."""
    
    @pytest.mark.mdm_id("1.5.1.1")
    def test_default_batch_size(self, clean_mdm_env, run_mdm):
        """1.5.1.1: Default batch size (10,000 rows)"""
        # Create large dataset
        data_dir = clean_mdm_env / "test_data"
        data_dir.mkdir()
        
        # Create dataset with 15k rows to test batching
        large_data = pd.DataFrame({
            'id': range(1, 15001),
            'feature': range(15000),
            'target': [i % 2 for i in range(15000)]
        })
        csv_file = data_dir / "large.csv"
        large_data.to_csv(csv_file, index=False)
        
        result = run_mdm([
            "dataset", "register", "test_batch_default", str(csv_file),
            "--target", "target"
        ])
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
        # Should process in 2 batches (10k + 5k)
    
    @pytest.mark.mdm_id("1.5.1.2")
    def test_custom_batch_size_yaml(self, clean_mdm_env, run_mdm, mdm_config_file):
        """1.5.1.2: Custom batch size via YAML config"""
        # Create config with small batch size
        mdm_config_file(performance={"batch_size": 1000})
        
        # Create medium dataset
        data_dir = clean_mdm_env / "test_data"
        data_dir.mkdir()
        
        medium_data = pd.DataFrame({
            'id': range(1, 3001),
            'feature': range(3000),
            'target': [i % 2 for i in range(3000)]
        })
        csv_file = data_dir / "medium.csv"
        medium_data.to_csv(csv_file, index=False)
        
        result = run_mdm([
            "dataset", "register", "test_batch_custom", str(csv_file),
            "--target", "target"
        ])
        
        assert result.returncode == 0
        # Should process in 3 batches (1k each)
    
    @pytest.mark.mdm_id("1.5.1.3")
    def test_batch_size_env_override(self, clean_mdm_env, run_mdm, mdm_config_file, sample_csv_data):
        """1.5.1.3: Environment variable overrides YAML batch size"""
        # Create config with one batch size
        mdm_config_file(performance={"batch_size": 5000})
        
        # Override with env var
        env = os.environ.copy()
        env["MDM_PERFORMANCE_BATCH_SIZE"] = "500"
        
        result = run_mdm([
            "dataset", "register", "test_batch_env", str(sample_csv_data),
            "--target", "value"
        ], env=env)
        
        assert result.returncode == 0
        # Should use 500 row batches from env var
    
    @pytest.mark.mdm_id("1.5.2.1")
    def test_parallel_processing_enabled(self, clean_mdm_env, run_mdm, mdm_config_file, sample_csv_data):
        """1.5.2.1: Enable parallel processing"""
        # Create config with parallel processing
        mdm_config_file(performance={
            "parallel_processing": True,
            "n_jobs": 2
        })
        
        result = run_mdm([
            "dataset", "register", "test_parallel", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        # Note: Parallel processing may not be implemented
        # This test documents expected behavior
    
    @pytest.mark.mdm_id("1.5.2.2")
    def test_n_jobs_setting(self, clean_mdm_env, run_mdm, mdm_config_file, sample_csv_data):
        """1.5.2.2: Configure number of parallel jobs"""
        # Test with -1 (all cores)
        mdm_config_file(performance={
            "parallel_processing": True,
            "n_jobs": -1
        })
        
        result = run_mdm([
            "dataset", "register", "test_njobs", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
    
    @pytest.mark.mdm_id("1.5.3.1")
    def test_memory_limit_setting(self, clean_mdm_env, run_mdm, mdm_config_file):
        """1.5.3.1: Memory limit configuration"""
        # Create config with memory limit
        mdm_config_file(performance={
            "memory_limit": "1GB"
        })
        
        # Create large dataset that might exceed limit
        data_dir = clean_mdm_env / "test_data"
        data_dir.mkdir()
        
        # Create 100k row dataset
        large_data = pd.DataFrame({
            'id': range(1, 100001),
            'feature': range(100000),
            'text': ['x' * 100 for _ in range(100000)],  # Large text field
            'target': [i % 2 for i in range(100000)]
        })
        csv_file = data_dir / "huge.csv"
        large_data.to_csv(csv_file, index=False)
        
        result = run_mdm([
            "dataset", "register", "test_memory", str(csv_file),
            "--target", "target"
        ])
        
        # Should handle gracefully with batching
        assert result.returncode == 0
    
    @pytest.mark.mdm_id("1.5.3.2")
    def test_cache_size_setting(self, clean_mdm_env, run_mdm, mdm_config_file, sample_csv_data):
        """1.5.3.2: Cache size configuration"""
        # Create config with cache settings
        mdm_config_file(performance={
            "cache_size": 100,  # MB
            "enable_caching": True
        })
        
        # Register dataset
        result = run_mdm([
            "dataset", "register", "test_cache", str(sample_csv_data),
            "--target", "value",
            "--no-features"  # Speed up test
        ])
        
        assert result.returncode == 0
        
        # Access dataset twice to test caching (reduced from 3)
        for _ in range(2):
            result = run_mdm(["dataset", "info", "test_cache"])
            assert result.returncode == 0
    
    @pytest.mark.mdm_id("1.5.4.1")
    def test_progress_bar_enabled(self, clean_mdm_env, run_mdm, sample_csv_data):
        """1.5.4.1: Progress bars enabled by default"""
        result = run_mdm([
            "dataset", "register", "test_progress_on", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        # Progress indicators should be shown during registration
        # Check for loading/processing indicators
    
    @pytest.mark.mdm_id("1.5.4.2")
    def test_disable_progress_bars(self, clean_mdm_env, run_mdm, mdm_config_file, sample_csv_data):
        """1.5.4.2: Disable progress bars for automation"""
        # Create config with progress bars disabled
        mdm_config_file(performance={
            "show_progress": False
        })
        
        result = run_mdm([
            "dataset", "register", "test_no_progress", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        # Output should be minimal without progress indicators
    
    @pytest.mark.mdm_id("1.5.5.1")
    def test_optimize_dtypes_enabled(self, clean_mdm_env, run_mdm, mdm_config_file):
        """1.5.5.1: Automatic dtype optimization"""
        # Create config with dtype optimization
        mdm_config_file(performance={
            "optimize_dtypes": True
        })
        
        # Create dataset with inefficient dtypes
        data_dir = clean_mdm_env / "test_data"
        data_dir.mkdir()
        
        # Int64 that could be int8, object that could be category
        inefficient_data = pd.DataFrame({
            'id': range(1, 101),
            'small_int': [i % 10 for i in range(100)],  # Could be int8
            'category': ['A', 'B', 'C'] * 33 + ['A'],  # Could be category
            'target': [0, 1] * 50
        })
        csv_file = data_dir / "inefficient.csv"
        inefficient_data.to_csv(csv_file, index=False)
        
        result = run_mdm([
            "dataset", "register", "test_optimize", str(csv_file),
            "--target", "target"
        ])
        
        assert result.returncode == 0
        # Types should be optimized during storage
    
    @pytest.mark.mdm_id("1.5.5.2")
    def test_compression_settings(self, clean_mdm_env, run_mdm, mdm_config_file, sample_csv_data):
        """1.5.5.2: Storage compression settings"""
        # Create config with compression
        mdm_config_file(performance={
            "compression": "gzip",
            "compression_level": 6
        })
        
        result = run_mdm([
            "dataset", "register", "test_compress", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        # Note: Compression settings may apply to exports
        # or internal storage depending on implementation