"""Integration tests for dataset lifecycle."""

import pandas as pd
import pytest

from mdm.api import MDMClient
from mdm.dataset.registrar import DatasetRegistrar


class TestDatasetLifecycle:
    """Test complete dataset lifecycle."""

    def test_full_lifecycle(self, sample_data, test_config):
        """Test registering, accessing, and removing a dataset."""
        # Initialize client
        client = MDMClient(config=test_config)
        registrar = DatasetRegistrar()
        
        # Register dataset
        dataset_info = registrar.register(
            name="test_lifecycle",
            path=sample_data.parent,
            auto_detect=True,
            description="Test dataset for lifecycle testing",
            tags=["test", "integration"],
        )
        
        assert dataset_info.name == "test_lifecycle"
        assert dataset_info.description == "Test dataset for lifecycle testing"
        assert "test" in dataset_info.tags
        
        # Get dataset info
        retrieved_info = client.get_dataset("test_lifecycle")
        assert retrieved_info is not None
        assert retrieved_info.name == dataset_info.name
        
        # Check dataset exists
        assert client.dataset_exists("test_lifecycle") is True
        
        # List datasets
        datasets = client.list_datasets()
        dataset_names = [d.name for d in datasets]
        assert "test_lifecycle" in dataset_names
        
        # Load data
        train_df, test_df = client.load_dataset_files("test_lifecycle")
        assert isinstance(train_df, pd.DataFrame)
        assert len(train_df) > 0
        
        # Update metadata
        updated_info = client.update_dataset(
            "test_lifecycle",
            description="Updated description",
            tags=["test", "integration", "updated"],
        )
        assert updated_info.description == "Updated description"
        assert "updated" in updated_info.tags
        
        # Get statistics
        stats = client.get_statistics("test_lifecycle")
        assert "tables" in stats
        assert "train" in stats["tables"]
        
        # Export dataset
        export_paths = client.export_dataset(
            "test_lifecycle",
            output_dir=str(test_config.storage.datasets_path / "exports"),
            format="csv",
        )
        assert len(export_paths) > 0
        
        # Remove dataset
        client.remove_dataset("test_lifecycle", force=True)
        assert client.dataset_exists("test_lifecycle") is False

    def test_case_insensitive_access(self, sample_data, test_config):
        """Test case-insensitive dataset access."""
        client = MDMClient(config=test_config)
        registrar = DatasetRegistrar()
        
        # Register with mixed case
        registrar.register(
            name="TestDataset",
            path=sample_data.parent,
            auto_detect=True,
        )
        
        # Access with different cases
        assert client.dataset_exists("TestDataset") is True
        assert client.dataset_exists("testdataset") is True
        assert client.dataset_exists("TESTDATASET") is True
        
        # Get info with different cases
        info1 = client.get_dataset("TestDataset")
        info2 = client.get_dataset("testdataset")
        assert info1.name == info2.name
        
        # Cleanup
        client.remove_dataset("TESTDATASET", force=True)

    def test_duplicate_registration(self, sample_data, test_config):
        """Test handling of duplicate dataset registration."""
        registrar = DatasetRegistrar()
        
        # First registration
        dataset_info = registrar.register(
            name="duplicate_test",
            path=sample_data.parent,
            auto_detect=True,
        )
        assert dataset_info is not None
        
        # Attempt duplicate registration without force
        with pytest.raises(Exception):
            registrar.register(
                name="duplicate_test",
                path=sample_data.parent,
                auto_detect=True,
            )
        
        # Force re-registration
        dataset_info2 = registrar.register(
            name="duplicate_test",
            path=sample_data.parent,
            auto_detect=True,
            force=True,
        )
        assert dataset_info2 is not None
        
        # Cleanup
        client = MDMClient(config=test_config)
        client.remove_dataset("duplicate_test", force=True)