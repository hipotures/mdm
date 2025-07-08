"""Shared fixtures for end-to-end tests with isolated environment in /tmp."""

import os
import shutil
import uuid
from pathlib import Path
import subprocess
import sys

import pytest
import pandas as pd


# Register custom marker
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "mdm_id(test_id): mark test with MDM test ID from MANUAL_TEST_CHECKLIST.md"
    )
    
    # SAFETY CHECK: Prevent tests from accidentally destroying user data
    if "MDM_HOME_DIR" not in os.environ:
        # Force tests to use a safe temporary directory
        test_safety_dir = Path(f"/tmp/mdm_test_safety_{uuid.uuid4().hex[:8]}")
        test_safety_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MDM_HOME_DIR"] = str(test_safety_dir)
        print(f"\n⚠️  SAFETY: MDM_HOME_DIR not set, using temporary directory: {test_safety_dir}")
        print("   This prevents tests from accidentally modifying ~/.mdm\n")


@pytest.fixture(scope="session")
def test_home():
    """Create a temporary MDM home directory in /tmp for the entire test session."""
    test_dir = Path(f"/tmp/mdm_test_{uuid.uuid4().hex[:8]}")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable - MDM uses MDM_HOME_DIR
    original_home = os.environ.get("MDM_HOME_DIR")
    os.environ["MDM_HOME_DIR"] = str(test_dir)
    
    yield test_dir
    
    # Cleanup
    if original_home:
        os.environ["MDM_HOME_DIR"] = original_home
    else:
        os.environ.pop("MDM_HOME_DIR", None)
    
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def clean_mdm_env():
    """Clean MDM environment before each test - each test gets its own directory."""
    # Import here to avoid circular imports
    from mdm.core.config import reset_config
    from mdm.config import reset_config_manager
    
    # SAFETY: Check and ensure safe test environment
    current_mdm_home = os.environ.get("MDM_HOME_DIR", "")
    home_mdm = str(Path.home() / ".mdm")
    
    # If MDM_HOME_DIR is not set or points to production directory, force safe directory
    if not current_mdm_home or current_mdm_home == home_mdm or not current_mdm_home.startswith("/tmp"):
        # Create a safety directory
        safety_dir = Path(f"/tmp/mdm_test_safety_{os.getpid()}_{uuid.uuid4().hex[:4]}")
        safety_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MDM_HOME_DIR"] = str(safety_dir)
        
        if not current_mdm_home:
            print(f"\n⚠️  SAFETY: MDM_HOME_DIR not set, using temporary: {safety_dir}")
        else:
            print(f"\n⚠️  SAFETY: MDM_HOME_DIR was unsafe ({current_mdm_home}), using: {safety_dir}")
        print("   This prevents tests from accidentally modifying production data.\n")
    
    # Reset MDM config to force reload with new environment
    reset_config()
    reset_config_manager()
    
    # Create unique test directory for THIS test
    test_env_dir = Path(f"/tmp/mdm_test_{uuid.uuid4().hex[:8]}")
    test_env_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original environment
    original_env = {}
    for key in list(os.environ.keys()):
        if key.startswith("MDM_"):
            original_env[key] = os.environ[key]
            del os.environ[key]
    
    # Set MDM_HOME_DIR to unique directory for this test
    os.environ["MDM_HOME_DIR"] = str(test_env_dir)
    
    # Reset config again to pick up new environment
    reset_config()
    reset_config_manager()
    
    # Create required directories
    for subdir in ["datasets", "config/datasets", "logs", "config"]:
        path = test_env_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
    
    yield test_env_dir
    
    # Cleanup - restore original environment
    for key in list(os.environ.keys()):
        if key.startswith("MDM_"):
            del os.environ[key]
    
    for key, value in original_env.items():
        os.environ[key] = value
    
    # Reset config to original state
    reset_config()
    reset_config_manager()
    
    # Remove test directory
    if test_env_dir.exists():
        shutil.rmtree(test_env_dir)


@pytest.fixture
def mdm_config_file(clean_mdm_env):
    """Create mdm.yaml config file in test environment."""
    import yaml
    config_file = clean_mdm_env / "mdm.yaml"
    
    def _create_config(content=None, **kwargs):
        """Create config from either string content or keyword arguments."""
        if content is not None:
            # If content is a string, write it directly
            if isinstance(content, str):
                config_file.write_text(content)
            else:
                # If content is a dict, convert to YAML
                config_file.write_text(yaml.dump(content))
        elif kwargs:
            # Convert kwargs to YAML directly
            config_file.write_text(yaml.dump(kwargs))
        return config_file
    
    return _create_config


@pytest.fixture
def sample_csv_data(clean_mdm_env):
    """Create sample CSV data for testing."""
    data_dir = clean_mdm_env / "test_data"
    data_dir.mkdir(exist_ok=True)
    
    # Create sample data
    data = {
        'id': range(1, 101),
        'feature1': [i * 2 for i in range(100)],
        'feature2': ['A' if i % 2 == 0 else 'B' for i in range(100)],
        'value': [i * 1.5 + 10 for i in range(100)],
        'category': ['cat1', 'cat2', 'cat3'] * 33 + ['cat1'],
        'date': pd.date_range('2023-01-01', periods=100, freq='D')
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = data_dir / "sample_data.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def kaggle_dataset_structure(clean_mdm_env):
    """Create Kaggle-style dataset structure."""
    kaggle_dir = clean_mdm_env / "kaggle_dataset"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Train data
    train_data = pd.DataFrame({
        'id': range(1, 101),
        'feature1': [i * 2 for i in range(100)],
        'feature2': ['A' if i % 2 == 0 else 'B' for i in range(100)],
        'target': [i % 3 for i in range(100)]
    })
    train_data.to_csv(kaggle_dir / "train.csv", index=False)
    
    # Test data (without target)
    test_data = pd.DataFrame({
        'id': range(101, 151),
        'feature1': [i * 2 for i in range(50)],
        'feature2': ['A' if i % 2 == 0 else 'B' for i in range(50)]
    })
    test_data.to_csv(kaggle_dir / "test.csv", index=False)
    
    # Sample submission
    submission = pd.DataFrame({
        'id': range(101, 151),
        'target': [0] * 50
    })
    submission.to_csv(kaggle_dir / "sample_submission.csv", index=False)
    
    return kaggle_dir


@pytest.fixture
def run_mdm():
    """Run MDM command and capture output."""
    def _run_command(args, check=True, env=None):
        """
        Run MDM command with arguments.
        
        Args:
            args: List of command arguments (without 'mdm' prefix)
            check: Whether to check return code
            env: Additional environment variables
            
        Returns:
            CompletedProcess object with stdout, stderr, and returncode
        """
        cmd = [sys.executable, "-m", "mdm.cli.main"] + args
        
        # Merge environment variables
        cmd_env = os.environ.copy()
        if env:
            cmd_env.update(env)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=cmd_env,
            check=False
        )
        
        if check and result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        
        return result
    
    return _run_command


@pytest.fixture
def verify_dataset_registered(run_mdm):
    """Verify that a dataset is registered."""
    def _verify(dataset_name):
        result = run_mdm(["dataset", "list"])
        return dataset_name in result.stdout
    
    return _verify


# Test markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "mdm_id(id): Mark test with MDM checklist ID (e.g., 1.1.1)"
    )
    config.addinivalue_line(
        "markers", "slow: Mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_db(backend): Mark test as requiring specific database backend"
    )