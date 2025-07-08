"""Tests for 2.1 Dataset Registration based on MANUAL_TEST_CHECKLIST.md"""

import os
from pathlib import Path

import pandas as pd
import pytest


class TestDatasetRegistration:
    """Test dataset registration functionality."""
    
    @pytest.mark.mdm_id("2.1.1.1")
    def test_register_single_csv(self, clean_mdm_env, run_mdm, sample_csv_data):
        """2.1.1.1: Register single CSV file"""
        result = run_mdm([
            "dataset", "register", "test_single_csv", str(sample_csv_data)
        ])
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
        
        # Verify dataset was created
        assert (clean_mdm_env / "datasets" / "test_single_csv").exists()
        assert (clean_mdm_env / "config" / "datasets" / "test_single_csv.yaml").exists()
    
    @pytest.mark.mdm_id("2.1.1.2")
    def test_register_directory_with_csv(self, clean_mdm_env, run_mdm):
        """2.1.1.2: Register directory containing train.csv"""
        # Create directory with train.csv
        data_dir = clean_mdm_env / "test_data"
        data_dir.mkdir()
        
        train_data = pd.DataFrame({
            'id': range(1, 101),
            'feature1': range(100),
            'target': [i % 2 for i in range(100)]
        })
        train_data.to_csv(data_dir / "train.csv", index=False)
        
        result = run_mdm([
            "dataset", "register", "test_dir_csv", str(data_dir),
            "--target", "target"
        ])
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
    
    @pytest.mark.mdm_id("2.1.1.3")
    def test_register_kaggle_structure(self, clean_mdm_env, run_mdm, kaggle_dataset_structure):
        """2.1.1.3: Register Kaggle-style dataset (train/test/sample_submission)"""
        result = run_mdm([
            "dataset", "register", "test_kaggle", str(kaggle_dataset_structure)
        ])
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
        
        # Should auto-detect target from sample_submission
        # Verify in the output
        assert "target" in result.stdout.lower()
    
    @pytest.mark.mdm_id("2.1.1.4")
    @pytest.mark.skip(reason="Parquet support to be tested separately")
    def test_register_parquet_file(self):
        """2.1.1.4: Register Parquet file"""
        pass
    
    @pytest.mark.mdm_id("2.1.1.5")
    @pytest.mark.skip(reason="JSON support to be tested separately")
    def test_register_json_file(self):
        """2.1.1.5: Register JSON file"""
        pass
    
    @pytest.mark.mdm_id("2.1.1.6")
    @pytest.mark.skip(reason="Compressed file support needs verification")
    def test_register_compressed_csv(self):
        """2.1.1.6: Register compressed CSV (.csv.gz)"""
        pass
    
    @pytest.mark.mdm_id("2.1.2.1")
    def test_auto_detect_delimiter(self, clean_mdm_env, run_mdm):
        """2.1.2.1: Auto-detect delimiter for CSV"""
        # Create TSV file
        data_dir = clean_mdm_env / "test_data"
        data_dir.mkdir()
        
        tsv_data = pd.DataFrame({
            'id': range(1, 11),
            'value': range(10, 20)
        })
        tsv_file = data_dir / "data.tsv"
        tsv_data.to_csv(tsv_file, sep='\t', index=False)
        
        result = run_mdm([
            "dataset", "register", "test_tsv", str(tsv_file),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
    
    @pytest.mark.mdm_id("2.1.2.2")
    def test_auto_detect_id_columns(self, clean_mdm_env, run_mdm, sample_csv_data):
        """2.1.2.2: Auto-detect ID columns"""
        result = run_mdm([
            "dataset", "register", "test_id_detect", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        # Should detect 'id' column as ID
        assert "ID Columns" in result.stdout
        assert "id" in result.stdout
    
    @pytest.mark.mdm_id("2.1.2.3")
    def test_auto_detect_target_kaggle(self, clean_mdm_env, run_mdm, kaggle_dataset_structure):
        """2.1.2.3: Auto-detect target from sample_submission.csv"""
        result = run_mdm([
            "dataset", "register", "test_target_detect", str(kaggle_dataset_structure)
        ])
        
        assert result.returncode == 0
        assert "Target" in result.stdout
        assert "target" in result.stdout.lower()
    
    @pytest.mark.mdm_id("2.1.2.4")
    def test_auto_detect_problem_type(self, clean_mdm_env, run_mdm):
        """2.1.2.4: Auto-detect problem type"""
        data_dir = clean_mdm_env / "test_data"
        data_dir.mkdir()
        
        # Binary classification
        binary_data = pd.DataFrame({
            'id': range(1, 101),
            'feature': range(100),
            'target': [0, 1] * 50
        })
        binary_file = data_dir / "binary.csv"
        binary_data.to_csv(binary_file, index=False)
        
        result = run_mdm([
            "dataset", "register", "test_binary", str(binary_file),
            "--target", "target"
        ])
        
        assert result.returncode == 0
        assert "binary_classification" in result.stdout
    
    @pytest.mark.mdm_id("2.1.3.1")
    def test_specify_target_column(self, clean_mdm_env, run_mdm, sample_csv_data):
        """2.1.3.1: Specify target column with --target"""
        result = run_mdm([
            "dataset", "register", "test_target_spec", str(sample_csv_data),
            "--target", "value"
        ])
        
        assert result.returncode == 0
        assert "Target" in result.stdout
        assert "value" in result.stdout
    
    @pytest.mark.mdm_id("2.1.3.2")
    def test_specify_problem_type(self, clean_mdm_env, run_mdm, sample_csv_data):
        """2.1.3.2: Specify problem type with --problem-type"""
        result = run_mdm([
            "dataset", "register", "test_prob_type", str(sample_csv_data),
            "--target", "value",
            "--problem-type", "regression"
        ])
        
        assert result.returncode == 0
        assert "Problem Type" in result.stdout
        assert "regression" in result.stdout
    
    @pytest.mark.mdm_id("2.1.3.3")
    def test_specify_id_columns(self, clean_mdm_env, run_mdm, sample_csv_data):
        """2.1.3.3: Specify ID columns with --id-columns"""
        result = run_mdm([
            "dataset", "register", "test_id_spec", str(sample_csv_data),
            "--target", "value",
            "--id-columns", "id,feature2"
        ])
        
        assert result.returncode == 0
        assert "ID Columns" in result.stdout
        assert "id" in result.stdout
        assert "feature2" in result.stdout
    
    @pytest.mark.mdm_id("2.1.3.4")
    def test_specify_datetime_columns(self, clean_mdm_env, run_mdm, sample_csv_data):
        """2.1.3.4: Force datetime parsing with --datetime-columns"""
        result = run_mdm([
            "dataset", "register", "test_datetime", str(sample_csv_data),
            "--target", "value",
            "--datetime-columns", "date"
        ])
        
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
    
    @pytest.mark.mdm_id("2.1.3.5")
    @pytest.mark.skip(reason="--time-column has known issues")
    def test_specify_time_column(self):
        """2.1.3.5: Specify time column with --time-column"""
        pass
    
    @pytest.mark.mdm_id("2.1.3.6")
    @pytest.mark.skip(reason="--group-column has known issues")
    def test_specify_group_column(self):
        """2.1.3.6: Specify group column with --group-column"""
        pass
    
    @pytest.mark.mdm_id("2.1.4.1")
    def test_register_with_description(self, clean_mdm_env, run_mdm, sample_csv_data):
        """2.1.4.1: Register with --description text"""
        description = "Test dataset for unit testing"
        result = run_mdm([
            "dataset", "register", "test_desc", str(sample_csv_data),
            "--target", "value",
            "--description", description
        ])
        
        assert result.returncode == 0
        
        # Verify description is saved
        config_file = clean_mdm_env / "config" / "datasets" / "test_desc.yaml"
        assert config_file.exists()
        config_text = config_file.read_text()
        assert description in config_text
    
    @pytest.mark.mdm_id("2.1.4.2")
    @pytest.mark.skip(reason="--source option not implemented")
    def test_register_with_source(self):
        """2.1.4.2: Register with --source specification"""
        pass
    
    @pytest.mark.mdm_id("2.1.4.3")
    def test_register_with_tags(self, clean_mdm_env, run_mdm, sample_csv_data):
        """2.1.4.3: Register with --tags (comma-separated)"""
        result = run_mdm([
            "dataset", "register", "test_tags", str(sample_csv_data),
            "--target", "value",
            "--tags", "test,sample,regression"
        ])
        
        assert result.returncode == 0
        
        # Verify tags are saved
        config_file = clean_mdm_env / "config" / "datasets" / "test_tags.yaml"
        config_text = config_file.read_text()
        assert "test" in config_text
        assert "sample" in config_text
        assert "regression" in config_text
    
    @pytest.mark.mdm_id("2.1.5.1")
    def test_force_flag_overwrites(self, clean_mdm_env, run_mdm, sample_csv_data):
        """2.1.5.1: Test --force flag to overwrite existing"""
        # Register once
        result = run_mdm([
            "dataset", "register", "test_force", str(sample_csv_data),
            "--target", "value"
        ])
        assert result.returncode == 0
        
        # Register again without force - should fail
        result = run_mdm([
            "dataset", "register", "test_force", str(sample_csv_data),
            "--target", "value"
        ], check=False)
        assert result.returncode != 0
        assert "already exists" in result.stderr
        
        # Register with force - should succeed
        result = run_mdm([
            "dataset", "register", "test_force", str(sample_csv_data),
            "--target", "value",
            "--force"
        ])
        assert result.returncode == 0
        assert "registered successfully" in result.stdout
    
    @pytest.mark.mdm_id("2.1.5.2")
    @pytest.mark.skip(reason="--no-auto flag functionality unclear")
    def test_no_auto_flag(self):
        """2.1.5.2: Test --no-auto flag with manual settings"""
        pass
    
    @pytest.mark.mdm_id("2.1.5.3")
    @pytest.mark.skip(reason="--skip-analysis option not implemented")
    def test_skip_analysis_flag(self):
        """2.1.5.3: Test --skip-analysis for faster registration"""
        pass
    
    @pytest.mark.mdm_id("2.1.5.4")
    @pytest.mark.skip(reason="--dry-run option not needed per requirements")
    def test_dry_run_flag(self):
        """2.1.5.4: Test --dry-run for preview without saving"""
        pass