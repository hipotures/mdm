"""Tests for API error handling and edge cases."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from mdm.api import MDMClient
from mdm.core.exceptions import DatasetError, StorageError
from mdm.models.dataset import DatasetInfo


class TestAPIErrorHandling:
    """Test error handling in MDM API."""

    @pytest.fixture
    def client(self):
        """Create MDM client instance."""
        with patch('mdm.api.DatasetManager'):
            return MDMClient()

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
        # Mock get_dataset to return dataset info
        client.get_dataset = Mock(return_value=mock_dataset_info)
        
        # Mock manager.get_statistics to return None (no saved stats)
        client.manager.get_statistics = Mock(return_value=None)
        
        # Mock DatasetStatistics to raise exception
        with patch('mdm.dataset.statistics.DatasetStatistics') as mock_stats:
            mock_stats.return_value.compute_statistics.side_effect = Exception("Compute failed")
            
            # Should return minimal structure from dataset info
            result = client.get_statistics("test_dataset")
            
            assert result is not None
            assert result['dataset_name'] == "test_dataset"
            assert 'tables' in result
            assert 'train' in result['tables']
            assert 'test' in result['tables']
            assert result['summary']['total_rows'] == 0

    def test_get_statistics_no_dataset(self, client):
        """Test get_statistics when dataset doesn't exist."""
        client.get_dataset = Mock(return_value=None)
        client.manager.get_statistics = Mock(return_value=None)
        
        with patch('mdm.dataset.statistics.DatasetStatistics') as mock_stats:
            mock_stats.return_value.compute_statistics.side_effect = Exception("Not found")
            
            result = client.get_statistics("nonexistent")
            assert result is None

    def test_query_dataset_error_handling(self, client):
        """Test query_dataset with backend errors."""
        client.manager.get_backend = Mock(side_effect=StorageError("Backend failed"))
        
        with pytest.raises(StorageError):
            client.query_dataset("test_dataset", "SELECT * FROM test")

    def test_load_dataset_files_missing_tables(self, client, mock_dataset_info):
        """Test load_dataset_files when tables are missing."""
        # Remove all tables
        mock_dataset_info.tables = {}
        client.get_dataset = Mock(return_value=mock_dataset_info)
        
        with pytest.raises(ValueError, match="has no train or data table"):
            client.load_dataset_files("test_dataset")

    def test_get_dataset_connection_no_support(self, client):
        """Test get_dataset_connection with backend that doesn't support connections."""
        mock_backend = Mock(spec=[])  # No get_connection or connection attribute
        client.manager.get_backend = Mock(return_value=mock_backend)
        
        with pytest.raises(NotImplementedError):
            client.get_dataset_connection("test_dataset")

    def test_export_dataset_with_single_table(self, client, mock_dataset_info):
        """Test export_dataset with specific table selection."""
        with patch('mdm.dataset.operations.ExportOperation') as mock_export:
            mock_export.return_value.execute.return_value = ["/tmp/export.csv"]
            
            result = client.export_dataset(
                "test_dataset",
                "/tmp/output",
                tables=["train"]  # Single table
            )
            
            # Should pass table_name instead of tables list
            mock_export.return_value.execute.assert_called_once()
            call_args = mock_export.return_value.execute.call_args
            assert call_args.kwargs['table'] == "train"

    def test_create_submission_invalid_format(self, client):
        """Test create_submission with invalid format."""
        df = pd.DataFrame({"id": [1, 2], "prediction": [0, 1]})
        
        with pytest.raises(ValueError, match="Unknown submission format"):
            client.create_submission(df, "/tmp/out.csv", format="invalid")

    def test_load_table_missing_table(self, client, mock_dataset_info):
        """Test load_table when requested table doesn't exist."""
        client.get_dataset = Mock(return_value=mock_dataset_info)
        
        with pytest.raises(ValueError, match="Table 'validation' not found"):
            client.load_table("test_dataset", "validation")

    def test_split_time_series_no_time_column(self, client, mock_dataset_info):
        """Test split_time_series without time column."""
        mock_dataset_info.time_column = None
        client.get_dataset = Mock(return_value=mock_dataset_info)
        
        with pytest.raises(ValueError, match="No time column specified"):
            client.split_time_series("test_dataset", 0.2)

    def test_prepare_for_ml_no_dataset(self, client):
        """Test prepare_for_ml with missing dataset."""
        client.get_dataset = Mock(return_value=None)
        
        with pytest.raises(ValueError, match="Dataset 'missing' not found"):
            client.prepare_for_ml("missing")

    def test_remove_dataset_force_vs_no_force(self, client):
        """Test remove_dataset behavior with force parameter."""
        # Test with force=True
        client.remove_dataset("test1", force=True)
        client.manager.remove_dataset.assert_called_once_with("test1")
        
        # Test with force=False
        client.manager.reset_mock()
        client.remove_dataset("test2", force=False)
        client.manager.delete_dataset.assert_called_once_with("test2", force=False)

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
        client.manager.list_datasets = Mock(return_value=datasets)
        
        # Test backend filter
        result = client.list_datasets(filter_backend="sqlite")
        assert len(result) == 1
        assert result[0].name == "test_dataset"
        
        # Test custom filter function
        result = client.list_datasets(
            filter_func=lambda d: len(d.tables) > 0
        )
        assert len(result) == 1
        
        # Test sorting
        result = client.list_datasets(sort_by="-name")
        assert result[0].name == "test_dataset"  # Reverse alphabetical
        
        # Test limit
        result = client.list_datasets(limit=1)
        assert len(result) == 1

    def test_get_column_info_errors(self, client, mock_dataset_info):
        """Test get_column_info error scenarios."""
        client.get_dataset = Mock(return_value=mock_dataset_info)
        
        # Test with missing table
        with pytest.raises(DatasetError, match="Table 'missing' not found"):
            client.get_column_info("test_dataset", "missing")
        
        # Test with backend error
        mock_backend = Mock()
        mock_backend.get_engine.side_effect = StorageError("Engine failed")
        client.manager.get_backend = Mock(return_value=mock_backend)
        
        with pytest.raises(StorageError):
            client.get_column_info("test_dataset")

    def test_create_time_series_splits_modes(self, client):
        """Test create_time_series_splits with different modes."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100),
            "value": range(100)
        })
        
        # Test with n_splits
        with patch('mdm.utils.time_series.TimeSeriesSplitter') as mock_splitter:
            mock_splitter.return_value.split_by_folds.return_value = [
                {"train": df[:80], "test": df[80:]}
            ]
            
            result = client.create_time_series_splits(df, "date", n_splits=5)
            assert len(result) == 1
            mock_splitter.return_value.split_by_folds.assert_called_once()
        
        # Test without n_splits (single split)
        with patch('mdm.utils.time_series.TimeSeriesSplitter') as mock_splitter:
            mock_splitter.return_value.split_by_time.return_value = {
                "train": df[:80], "test": df[80:]
            }
            
            result = client.create_time_series_splits(df, "date", n_splits=None)
            assert len(result) == 1
            mock_splitter.return_value.split_by_time.assert_called_once()

    def test_concurrent_dataset_operations(self, client):
        """Test handling of concurrent operations on same dataset."""
        import threading
        import time
        
        results = []
        errors = []
        
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
        assert len(errors) == 0 or all("not found" in e for e in errors)