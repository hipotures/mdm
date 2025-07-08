"""Test to verify MDM_HOME_DIR isolation works correctly."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


class TestIsolation:
    """Test MDM environment isolation."""
    
    def test_mdm_home_dir_isolation(self, clean_mdm_env, run_mdm):
        """Verify that MDM_HOME_DIR properly isolates environments."""
        # Create test CSV first
        csv_path = clean_mdm_env / "test.csv"
        csv_path.write_text("id,value\n1,10\n2,20\n")
        
        # Register dataset in test environment
        dataset_name = "isolation_test"
        result = run_mdm([
            "dataset", "register", dataset_name,
            str(csv_path),
            "--target", "value"
        ])
        assert result.returncode == 0
        
        # List datasets in test environment
        result = run_mdm(["dataset", "list"])
        # Name might be truncated in table, so check for partial match
        assert "isolation" in result.stdout
        
        # Now test with different MDM_HOME_DIR
        other_env = Path(f"/tmp/mdm_other_{os.getpid()}")
        other_env.mkdir(exist_ok=True)
        (other_env / "datasets").mkdir(exist_ok=True)
        (other_env / "config" / "datasets").mkdir(parents=True, exist_ok=True)
        
        # List datasets in different environment - should be empty
        result = subprocess.run(
            [sys.executable, "-m", "mdm.cli.main", "dataset", "list"],
            env={"MDM_HOME_DIR": str(other_env)},
            capture_output=True,
            text=True
        )
        
        # Should NOT see the dataset from the other environment
        assert dataset_name not in result.stdout
        assert "No datasets" in result.stdout or result.stdout.count("│") <= 5  # Empty table
        
        # Cleanup
        import shutil
        if other_env.exists():
            shutil.rmtree(other_env)
    
    def test_parallel_test_isolation(self, clean_mdm_env):
        """Verify that parallel tests get different environments."""
        # Each test should get a unique directory
        assert str(clean_mdm_env).startswith("/tmp/mdm_test_")
        
        # Directory should be empty (only structure dirs)
        datasets_dir = clean_mdm_env / "datasets"
        assert datasets_dir.exists()
        assert len(list(datasets_dir.iterdir())) == 0
        
        # Config should be empty
        config_dir = clean_mdm_env / "config" / "datasets"
        assert config_dir.exists()
        assert len(list(config_dir.iterdir())) == 0