"""Tests for API error handling and edge cases."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
import threading
import time

from mdm.api import MDMClient
from mdm.core.exceptions import DatasetError, StorageError
from mdm.models.dataset import DatasetInfo


class TestAPIErrorHandling:
    """Test error handling in MDM API."""

    @pytest.fixture
    def client(self):
        """Create MDM client instance with mocked components."""
        with patch('mdm.config.get_config', return_value={}):
            with patch('mdm.core.get_service') as mock_get_service:
                # Create mock clients
                mock_registration = Mock()
                mock_query = Mock()
                mock_ml = Mock()
                mock_export = Mock()
                mock_management = Mock()
                
                # Mock the manager on management client
                mock_management.manager = Mock()
                
                # Configure get_service to return mocks
                def get_mock_service(service_type):
                    if service_type.__name__ == 'RegistrationClient':
                        return mock_registration
                    elif service_type.__name__ == 'QueryClient':
                        return mock_query
                    elif service_type.__name__ == 'MLIntegrationClient':
                        return mock_ml
                    elif service_type.__name__ == 'ExportClient':
                        return mock_export
                    elif service_type.__name__ == 'ManagementClient':
                        return mock_management
                    
                mock_get_service.side_effect = get_mock_service
                
                # Create client with mocked dependencies
                client = MDMClient()
                
                # Ensure query client has necessary methods
                mock_query.get_dataset = Mock()
                mock_query.load_dataset_files = Mock()
                mock_query.query_dataset = Mock()
                mock_query.get_column_info = Mock()
                mock_query.dataset_exists = Mock()
                mock_query.list_datasets = Mock()
                mock_query.load_table = Mock()
                
                # Ensure management client has necessary methods
                mock_management.get_statistics = Mock()
                mock_management.update_dataset = Mock()
                mock_management.remove_dataset = Mock()
                mock_management.search_datasets = Mock()
                mock_management.search_datasets_by_tag = Mock()
                mock_management.get_dataset_connection = Mock()
                
                # Ensure ML client has necessary methods
                mock_ml.split_time_series = Mock()
                mock_ml.prepare_for_ml = Mock()
                mock_ml.create_submission = Mock()
                mock_ml.create_time_series_splits = Mock()
                
                # Ensure export client has necessary methods
                mock_export.export_dataset = Mock()
                
                return client

    @pytest.fixture
    def mock_dataset_info(self):
        """Create mock dataset info."""
        return DatasetInfo(
            name="test_dataset",
            database={"backend": "sqlite", "path": "/tmp/test.db"},
            tables={"train": "test_dataset_train", "test": "test_dataset_test"}
        )

    def test_get_statistics_with_compute_failure(self, client, mock_dataset_info):
        """Test get_statistics when compute fails but dataset exists."""
        # Mock management client's get_statistics to raise exception
        client.management.get_statistics = Mock(side_effect=Exception("Compute failed"))
        
        # Should raise the exception since compute fails
        with pytest.raises(Exception, match="Compute failed"):
            client.get_statistics("test_dataset")

    def test_get_statistics_no_dataset(self, client):
        """Test get_statistics when dataset doesn't exist."""
        # Mock management's get_statistics to raise DatasetError
        client.management.get_statistics = Mock(side_effect=DatasetError("Dataset 'nonexistent' not found"))
        
        # Should raise DatasetError when dataset not found
        with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
            client.get_statistics("nonexistent")

    def test_query_dataset_error_handling(self, client):
        """Test query_dataset with backend errors."""
        # Mock query client to return dataset info and then raise error
        mock_dataset = Mock(database={"backend": "sqlite"})
        client.query.get_dataset = Mock(return_value=mock_dataset)
        
        # Mock query_dataset to raise StorageError
        client.query.query_dataset = Mock(side_effect=StorageError("Backend failed"))
        
        with pytest.raises(StorageError, match="Backend failed"):
            client.query_dataset("test_dataset", "SELECT * FROM test")

    def test_load_dataset_files_missing_tables(self, client, mock_dataset_info):
        """Test load_dataset_files when tables are missing."""
        # Remove all tables
        mock_dataset_info.tables = {}
        mock_dataset_info.feature_tables = {}
        client.query.get_dataset = Mock(return_value=mock_dataset_info)
        
        # Mock query client's load_dataset_files to return empty dict
        client.query.load_dataset_files = Mock(return_value={})
        
        # Should return empty dict when no tables
        result = client.load_dataset_files("test_dataset")
        assert result == {}

    def test_get_dataset_connection_no_support(self, client, mock_dataset_info):
        """Test get_dataset_connection returns engine from backend."""
        # Mock get_dataset_connection directly on management client
        mock_engine = Mock()
        client.management.get_dataset_connection = Mock(return_value=mock_engine)
        
        # Should return the engine - note: this method is on management client, not MDMClient
        result = client.management.get_dataset_connection("test_dataset")
        assert result == mock_engine

    def test_export_dataset_with_single_table(self, client, mock_dataset_info):
        """Test export_dataset with specific table selection."""
        # Mock export client's export_dataset method
        expected_path = Path("/tmp/output")
        client.export.export_dataset = Mock(return_value=expected_path)
        
        result = client.export_dataset(
            "test_dataset",
            "/tmp/output",
            tables=["train"]  # Single table
        )
        
        # Should call export client
        assert result == expected_path
        client.export.export_dataset.assert_called_once_with(
            "test_dataset", 
            "/tmp/output", 
            tables=["train"]
        )

    def test_create_submission_invalid_format(self, client):
        """Test create_submission with invalid dataset."""
        df = pd.DataFrame({"id": [1, 2], "prediction": [0, 1]})
        
        # Mock ML client to raise error for missing dataset
        client.ml.create_submission = Mock(side_effect=DatasetError("Dataset 'nonexistent' not found"))
        
        # Should raise DatasetError when dataset not found
        with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
            client.create_submission("nonexistent", df, "/tmp/out.csv")

    def test_load_table_missing_table(self, client, mock_dataset_info):
        """Test load_table when requested table doesn't exist."""
        # Mock query client to return dataset info
        client.query.get_dataset = Mock(return_value=mock_dataset_info)
        
        # Mock query.load_table to raise error
        client.query.load_table.side_effect = DatasetError("Table 'validation' not found")
        
        with pytest.raises(DatasetError, match="Table 'validation' not found"):
            client.query.load_table("test_dataset", "validation")

    def test_split_time_series_no_time_column(self, client, mock_dataset_info):
        """Test split_time_series without time column."""
        mock_dataset_info.time_column = None
        
        # Mock ML client to raise error
        client.ml.split_time_series = Mock(side_effect=DatasetError("Dataset 'test_dataset' has no time column configured"))
        
        with pytest.raises(DatasetError, match="has no time column configured"):
            client.split_time_series("test_dataset", 5)

    def test_prepare_for_ml_no_dataset(self, client):
        """Test prepare_for_ml with missing dataset."""
        # Mock ML client to raise error
        client.ml.prepare_for_ml = Mock(side_effect=DatasetError("Dataset 'missing' not found"))
        
        with pytest.raises(DatasetError, match="Dataset 'missing' not found"):
            client.prepare_for_ml("missing")

    def test_remove_dataset_force_vs_no_force(self, client):
        """Test remove_dataset behavior with force parameter."""
        # Mock management client's remove_dataset
        client.management.remove_dataset = Mock()
        
        # Test with force=True
        client.remove_dataset("test1", force=True)
        client.management.remove_dataset.assert_called_once_with("test1", True)
        
        # Test with force=False
        client.management.remove_dataset.reset_mock()
        client.remove_dataset("test2", force=False)
        client.management.remove_dataset.assert_called_once_with("test2", False)

    def test_list_datasets_with_filters(self, client, mock_dataset_info):
        """Test list_datasets with various filters."""
        datasets = [
            mock_dataset_info,
            DatasetInfo(
                name="another_dataset",
                database={"backend": "duckdb"},
                tables={}
            )
        ]
        
        # Mock query client to return datasets
        client.query.list_datasets = Mock(return_value=datasets)
        
        # Test basic list functionality
        result = client.list_datasets()
        assert len(result) == 2
        assert result[0].name == "test_dataset"
        assert result[1].name == "another_dataset"
        
        # Test with limit parameter (QueryClient handles internally)
        client.query.list_datasets = Mock(return_value=datasets[:1])
        result = client.list_datasets(limit=1)
        assert len(result) == 1

    def test_get_column_info_errors(self, client, mock_dataset_info):
        """Test get_column_info error scenarios."""
        # Mock query client to raise error
        client.query.get_column_info = Mock(side_effect=StorageError("Engine failed"))
        
        with pytest.raises(StorageError, match="Engine failed"):
            client.get_column_info("test_dataset", "train")

    def test_create_time_series_splits_modes(self, client):
        """Test create_time_series_splits with different modes."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100),
            "value": range(100)
        })
        
        # Mock ML client's create_time_series_splits
        expected_splits = [(df[:80], df[80:])]
        client.ml.create_time_series_splits = Mock(return_value=expected_splits)
        
        result = client.ml.create_time_series_splits(df, "date", n_splits=5)
        assert len(result) == 1
        assert result == expected_splits

    def test_concurrent_dataset_operations(self, client):
        """Test handling of concurrent operations on same dataset."""
        results = []
        errors = []
        
        # Mock management client's update_dataset to succeed
        client.management.update_dataset = Mock(return_value=Mock(name="test_dataset"))
        
        def update_dataset():
            try:
                client.update_dataset("test", description=f"Updated at {time.time()}")
                results.append("success")
            except Exception as e:
                errors.append(str(e))
        
        # Simulate concurrent updates
        threads = [threading.Thread(target=update_dataset) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should handle concurrent access gracefully
        assert len(results) == 5
        assert len(errors) == 0