"""Tests for API error handling - simplified version."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import pandas as pd

from mdm.api import MDMClient
from mdm.core.exceptions import DatasetError, StorageError
from mdm.models.dataset import DatasetInfo


class TestAPIErrorHandling:
    """Test error handling in MDM API - simplified tests."""

    @pytest.fixture
    def client(self):
        """Create MDM client instance with mocked components."""
        with patch('mdm.config.get_config', return_value={}):
            with patch('mdm.core.get_service') as mock_get_service:
                # Create mocks for each client type
                mock_services = {
                    'RegistrationClient': Mock(),
                    'QueryClient': Mock(),
                    'MLIntegrationClient': Mock(),
                    'ExportClient': Mock(),
                    'ManagementClient': Mock()
                }
                
                def get_service_mock(service_type):
                    return mock_services.get(service_type.__name__, Mock())
                
                mock_get_service.side_effect = get_service_mock
                return MDMClient()

    @pytest.fixture
    def mock_dataset_info(self):
        """Create mock dataset info."""
        return DatasetInfo(
            name="test_dataset",
            database={"backend": "sqlite", "path": "/tmp/test.db"},
            tables={"train": "test_dataset_train", "test": "test_dataset_test"}
        )

    def test_get_statistics_error_propagation(self, client):
        """Test that get_statistics propagates errors from management client."""
        # Make management client raise an error
        client.management.get_statistics.side_effect = DatasetError("Dataset not found")
        
        with pytest.raises(DatasetError, match="Dataset not found"):
            client.get_statistics("nonexistent")

    def test_query_dataset_delegates_to_query_client(self, client):
        """Test that query_dataset delegates to query client."""
        expected_result = pd.DataFrame({"col": [1, 2, 3]})
        client.query.query_dataset.return_value = expected_result
        
        result = client.query_dataset("test_dataset", "SELECT * FROM test")
        
        assert result.equals(expected_result)
        client.query.query_dataset.assert_called_once_with("test_dataset", "SELECT * FROM test")

    def test_export_dataset_delegates_to_export_client(self, client):
        """Test that export_dataset delegates to export client."""
        expected_path = Path("/tmp/output")
        client.export.export_dataset.return_value = expected_path
        
        result = client.export_dataset("test_dataset", "/tmp/output", format="csv")
        
        assert result == expected_path
        client.export.export_dataset.assert_called_once()

    def test_remove_dataset_delegates_to_management_client(self, client):
        """Test that remove_dataset delegates to management client."""
        client.remove_dataset("test_dataset", force=True)
        
        client.management.remove_dataset.assert_called_once_with("test_dataset", True)

    def test_list_datasets_delegates_to_query_client(self, client, mock_dataset_info):
        """Test that list_datasets delegates to query client."""
        client.query.list_datasets.return_value = [mock_dataset_info]
        
        result = client.list_datasets()
        
        assert len(result) == 1
        assert result[0].name == "test_dataset"

    def test_prepare_for_ml_delegates_to_ml_client(self, client):
        """Test that prepare_for_ml delegates to ML client."""
        expected_data = (pd.DataFrame(), pd.Series(), pd.DataFrame())
        client.ml.prepare_for_ml.return_value = expected_data
        
        result = client.prepare_for_ml("test_dataset")
        
        assert result == expected_data
        client.ml.prepare_for_ml.assert_called_once_with("test_dataset", "auto")

    def test_update_dataset_delegates_to_management_client(self, client, mock_dataset_info):
        """Test that update_dataset delegates to management client."""
        client.management.update_dataset.return_value = mock_dataset_info
        
        result = client.update_dataset("test_dataset", description="New description")
        
        assert result == mock_dataset_info
        client.management.update_dataset.assert_called_once_with(
            "test_dataset", 
            description="New description"
        )

    def test_dataset_exists_delegates_to_query_client(self, client):
        """Test that dataset_exists delegates to query client."""
        client.query.dataset_exists.return_value = True
        
        result = client.dataset_exists("test_dataset")
        
        assert result is True
        client.query.dataset_exists.assert_called_once_with("test_dataset")

    def test_get_column_info_delegates_to_query_client(self, client):
        """Test that get_column_info delegates to query client."""
        expected_columns = [{"name": "id", "type": "INTEGER"}]
        client.query.get_column_info.return_value = expected_columns
        
        result = client.get_column_info("test_dataset", "train")
        
        assert result == expected_columns
        client.query.get_column_info.assert_called_once_with("test_dataset", "train")

    def test_split_time_series_delegates_to_ml_client(self, client):
        """Test that split_time_series delegates to ML client."""
        expected_splits = [(pd.DataFrame(), pd.DataFrame())]
        client.ml.split_time_series.return_value = expected_splits
        
        result = client.split_time_series("test_dataset", n_splits=5)
        
        assert result == expected_splits
        client.ml.split_time_series.assert_called_once()