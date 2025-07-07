"""Pytest configuration and fixtures."""

import shutil
import tempfile
from pathlib import Path

import pytest

from mdm.config import get_config
from mdm.dataset.manager import DatasetManager


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
    config.storage.datasets_path = temp_dir / "datasets"
    config.storage.configs_path = temp_dir / "configs"
    config.logs.file_path = temp_dir / "logs" / "mdm.log"
    
    # Create directories
    config.storage.datasets_path.mkdir(parents=True, exist_ok=True)
    config.storage.configs_path.mkdir(parents=True, exist_ok=True)
    config.logs.file_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    # Save to CSV
    data_path = temp_dir / "sample_data.csv"
    df.to_csv(data_path, index=False)
    
    return data_path