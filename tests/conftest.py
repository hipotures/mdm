"""Pytest configuration and fixtures."""

import shutil
import tempfile
from pathlib import Path

import pytest

from mdm.config import get_config
from mdm.dataset.manager import DatasetManager
from mdm.core import configure_services


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration."""
    config = get_config()
    config.paths.datasets_path = str(temp_dir / "datasets")
    config.paths.configs_path = str(temp_dir / "configs")
    config.logging.file = str(temp_dir / "logs" / "mdm.log")
    
    # Create directories
    Path(config.paths.datasets_path).mkdir(parents=True, exist_ok=True)
    Path(config.paths.configs_path).mkdir(parents=True, exist_ok=True)
    Path(config.logging.file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure DI container
    configure_services(config.model_dump())
    
    return config


@pytest.fixture
def dataset_manager(test_config):
    """Create a dataset manager for testing."""
    return DatasetManager()


@pytest.fixture
def sample_data(temp_dir):
    """Create sample CSV data for testing."""
    import pandas as pd
    
    # Create sample data
    data = {
        'id': range(1, 101),
        'feature1': [i * 2 for i in range(100)],
        'feature2': ['A' if i % 2 == 0 else 'B' for i in range(100)],
        'target': [i % 3 for i in range(100)]
    }
    df = pd.DataFrame(data)
    
    # Save to CSV files with standard names
    train_path = temp_dir / "train.csv"
    df.to_csv(train_path, index=False)
    
    # Also create a test file (optional)
    test_df = df.sample(frac=0.2, random_state=42)
    test_path = temp_dir / "test.csv"
    test_df.to_csv(test_path, index=False)
    
    return train_path