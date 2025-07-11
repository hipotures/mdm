"""Comprehensive unit tests for MDM API."""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from mdm.api import MDMClient
from mdm.core.exceptions import DatasetError
from mdm.models.dataset import DatasetInfo, DatasetStatistics
from mdm.models.enums import ProblemType


class TestMDMClientComplete:
    """Comprehensive test coverage for MDMClient."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.database.default_backend = "sqlite"
        config.paths.datasets_path = "datasets/"
        config.paths.configs_path = "config/datasets/"
        return config

    @pytest.fixture
    def mock_manager(self):
        """Create mock DatasetManager."""
        manager = Mock()
        manager.get_dataset.return_value = None
        manager.list_datasets.return_value = []
        manager.update_dataset.return_value = None
        manager.search_datasets.return_value = []
        manager.search_datasets_by_tag.return_value = []
        return manager

    @pytest.fixture
    def client(self, mock_config, mock_manager):
        """Create MDMClient instance."""
        with patch('mdm.api.mdm_client.get_config', return_value=mock_config):
            with patch('mdm.core.get_service') as mock_get_service:
                # Create mock clients
                mock_registration = Mock()
                mock_query = Mock()
                mock_ml = Mock()
                mock_export = Mock()
                mock_management = Mock()
                
                # Configure query client to wrap mock_manager methods
                def query_get_dataset(name):
                    try:
                        return mock_manager.get_dataset(name)
                    except DatasetError:
                        return None
                mock_query.get_dataset = query_get_dataset
                mock_query.list_datasets = mock_manager.list_datasets
                
                # Configure management client to use mock_manager  
                mock_management.update_dataset = mock_manager.update_dataset
                mock_management.remove_dataset = mock_manager.remove_dataset
                
                # search_datasets needs to add default search_fields
                def management_search_datasets(pattern):
                    return mock_manager.search_datasets(pattern, ["name", "description", "tags"])
                mock_management.search_datasets = management_search_datasets
                
                mock_management.search_datasets_by_tag = mock_manager.search_datasets_by_tag
                
                # get_statistics needs custom handling
                def management_get_statistics(name, full=False):
                    return mock_manager.get_statistics(name, full)
                mock_management.get_statistics = management_get_statistics
                
                # Configure mock_get_service to return appropriate clients
                def get_service_side_effect(cls):
                    if cls.__name__ == 'RegistrationClient':
                        return mock_registration
                    elif cls.__name__ == 'QueryClient':
                        return mock_query
                    elif cls.__name__ == 'MLIntegrationClient':
                        return mock_ml
                    elif cls.__name__ == 'ExportClient':
                        return mock_export
                    elif cls.__name__ == 'ManagementClient':
                        return mock_management
                    return Mock()
                
                mock_get_service.side_effect = get_service_side_effect
                return MDMClient()

    @pytest.fixture
    def sample_dataset_info(self):
        """Create sample DatasetInfo."""
        return DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            source_path="/data/test",
            database={"backend": "sqlite", "path": "test.db"},
            tables={"train": "test_dataset_train"},
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            target_column="target",
            id_columns=["id"],
            tags=["test"],
            created_at=datetime.now(timezone.utc),
            last_updated_at=datetime.now(timezone.utc)
        )

    def test_init_with_default_config(self, mock_config):
        """Test initialization with default config."""
        with patch('mdm.api.mdm_client.get_config', return_value=mock_config):
            with patch('mdm.core.get_service') as mock_get_service:
                # Mock the get_service function to return mocked clients
                mock_get_service.side_effect = lambda cls: Mock()
                
                client = MDMClient()
                assert client.config == mock_config
                assert client.registration is not None
                assert client.query is not None
                assert client.ml is not None
                assert client.export is not None
                assert client.management is not None
                
                # Verify that get_service was called for each client
                assert mock_get_service.call_count == 5

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        custom_config = Mock()
        with patch('mdm.core.get_service') as mock_get_service:
            # Mock the get_service function to return mocked clients
            mock_get_service.side_effect = lambda cls: Mock()
            
            client = MDMClient(config=custom_config)
            assert client.config == custom_config
            assert client.registration is not None
            assert client.query is not None
            assert client.ml is not None
            assert client.export is not None
            assert client.management is not None

    def test_register_dataset_success(self, client, sample_dataset_info, tmp_path):
        """Test successful dataset registration."""
        # Create test data
        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "train.csv").write_text("id,feature,target\n1,0.5,1\n")
        
        # Mock the registration client's register_dataset method
        client.registration.register_dataset.return_value = sample_dataset_info
        
        result = client.register_dataset(
            name="test_dataset",
            dataset_path=str(data_path),
            target_column="target",
            id_columns=["id"],
            problem_type="binary_classification",
            description="Test dataset",
            tags=["test"],
            auto_analyze=True,
            force=False
        )
        
        assert result == sample_dataset_info
        client.registration.register_dataset.assert_called_once()
        
        # Check call arguments
        call_args = client.registration.register_dataset.call_args
        assert call_args.args[0] == "test_dataset"
        assert call_args.args[1] == str(data_path)
        assert call_args.kwargs['target_column'] == "target"
        assert call_args.kwargs['id_columns'] == ["id"]
        assert call_args.kwargs['force'] is False

    def test_register_dataset_with_kwargs(self, client, sample_dataset_info, tmp_path):
        """Test dataset registration with additional kwargs."""
        data_path = tmp_path / "data"
        data_path.mkdir()
        
        # Mock the registration client's register_dataset method
        client.registration.register_dataset.return_value = sample_dataset_info
        
        result = client.register_dataset(
            name="test_dataset",
            dataset_path=str(data_path),
            custom_param="custom_value",
            another_param=123
        )
        
        # Check that kwargs are passed through
        call_kwargs = client.registration.register_dataset.call_args.kwargs
        assert call_kwargs['custom_param'] == "custom_value"
        assert call_kwargs['another_param'] == 123

    def test_register_dataset_path_conversion(self, client, sample_dataset_info):
        """Test that string path is passed correctly."""
        # Mock the registration client's register_dataset method
        client.registration.register_dataset.return_value = sample_dataset_info
        
        client.register_dataset(
            name="test_dataset",
            dataset_path="/data/test"
        )
        
        call_args = client.registration.register_dataset.call_args
        # In the new architecture, path conversion happens inside RegistrationClient
        assert call_args.args[1] == "/data/test"

    def test_get_dataset_found(self, client, mock_manager, sample_dataset_info):
        """Test getting existing dataset."""
        mock_manager.get_dataset.return_value = sample_dataset_info
        
        result = client.get_dataset("test_dataset")
        
        assert result == sample_dataset_info
        mock_manager.get_dataset.assert_called_once_with("test_dataset")

    def test_get_dataset_not_found(self, client, mock_manager):
        """Test getting non-existent dataset."""
        mock_manager.get_dataset.side_effect = DatasetError("Not found")
        
        result = client.get_dataset("nonexistent")
        
        assert result is None
        mock_manager.get_dataset.assert_called_once_with("nonexistent")

    def test_list_datasets(self, client, mock_manager, sample_dataset_info):
        """Test listing datasets."""
        datasets = [sample_dataset_info, sample_dataset_info.model_copy()]
        mock_manager.list_datasets.return_value = datasets
        
        result = client.list_datasets(limit=10, sort_by="name")
        
        assert result == datasets
        mock_manager.list_datasets.assert_called_once()

    def test_list_datasets_empty(self, client, mock_manager):
        """Test listing datasets when none exist."""
        mock_manager.list_datasets.return_value = []
        
        result = client.list_datasets()
        
        assert result == []
        mock_manager.list_datasets.assert_called_once()

    def test_remove_dataset_success(self, client, mock_manager):
        """Test removing dataset successfully."""
        mock_manager.remove_dataset.return_value = None
        
        client.remove_dataset("test_dataset", force=True)
        
        mock_manager.remove_dataset.assert_called_once_with("test_dataset", True)

    def test_remove_dataset_not_found(self, client, mock_manager):
        """Test removing non-existent dataset."""
        mock_manager.remove_dataset.side_effect = DatasetError("Not found")
        
        with pytest.raises(DatasetError, match="Not found"):
            client.remove_dataset("nonexistent")

    def test_update_dataset_success(self, client, mock_manager, sample_dataset_info):
        """Test updating dataset."""
        updated_info = sample_dataset_info.model_copy()
        updated_info.description = "Updated description"
        mock_manager.update_dataset.return_value = updated_info
        
        result = client.update_dataset(
            "test_dataset",
            description="Updated description",
            tags=["test", "updated"]
        )
        
        assert result == updated_info
        mock_manager.update_dataset.assert_called_once_with(
            "test_dataset",
            description="Updated description",
            tags=["test", "updated"]
        )

    def test_search_datasets_simple(self, client, mock_manager, sample_dataset_info):
        """Test searching datasets with simple query."""
        mock_manager.search_datasets.return_value = [sample_dataset_info]
        
        result = client.search_datasets("test")
        
        assert len(result) == 1
        assert result[0] == sample_dataset_info
        mock_manager.search_datasets.assert_called_once_with("test", ["name", "description", "tags"])

    def test_search_datasets_advanced(self, client, mock_manager):
        """Test searching datasets with advanced options."""
        mock_manager.search_datasets.return_value = []
        
        result = client.search_datasets("TEST", deep=True, case_sensitive=True)
        
        assert result == []
        # Management client ignores deep and case_sensitive params currently
        mock_manager.search_datasets.assert_called_once_with("TEST", ["name", "description", "tags"])

    def test_search_datasets_by_tag(self, client, mock_manager, sample_dataset_info):
        """Test searching datasets by tag."""
        mock_manager.search_datasets_by_tag.return_value = [sample_dataset_info]
        
        result = client.search_datasets_by_tag("test")
        
        assert len(result) == 1
        assert result[0] == sample_dataset_info
        mock_manager.search_datasets_by_tag.assert_called_once_with("test")

    def test_get_statistics_success(self, client, mock_manager):
        """Test getting dataset statistics."""
        expected_stats = {
            'dataset_name': 'test_dataset',
            'tables': {
                'train': {'row_count': 1000}
            },
            'summary': {
                'total_rows': 1000,
                'total_columns': 10
            }
        }
        
        mock_manager.get_statistics.return_value = expected_stats
        
        result = client.get_statistics("test_dataset")
        
        assert result == expected_stats
        mock_manager.get_statistics.assert_called_once_with("test_dataset", False)

    def test_get_statistics_not_found(self, client, mock_manager):
        """Test getting statistics for non-existent dataset."""
        mock_manager.get_statistics.return_value = None
        
        result = client.get_statistics("nonexistent")
        
        assert result is None
        mock_manager.get_statistics.assert_called_once_with("nonexistent", False)

    def test_load_dataset_pandas_success(self, client, tmp_path):
        """Test loading dataset as pandas DataFrame."""
        # Create expected DataFrame
        expected_df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature': [0.1, 0.2, 0.3],
            'target': [0, 1, 0]
        })
        
        # Mock query client's load_dataset method
        client.query.load_dataset.return_value = expected_df
        
        df = client.load_dataset("test_dataset", table="train")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['id', 'feature', 'target']
        client.query.load_dataset.assert_called_once_with("test_dataset", table="train")

    def test_load_dataset_table_not_found(self, client):
        """Test loading non-existent table."""
        # Mock query client to raise error
        client.query.load_dataset.side_effect = DatasetError("Table 'test' not found")
        
        with pytest.raises(DatasetError, match="Table 'test' not found"):
            client.load_dataset("test_dataset", table="test")

    def test_load_dataset_files_success(self, client, tmp_path):
        """Test loading dataset files directly."""
        # Create expected data
        train_df = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        test_df = pd.DataFrame({'id': [3, 4], 'value': [30, 40]})
        
        expected_result = {
            'train': train_df,
            'test': test_df
        }
        
        # Mock query client's load_dataset_files method
        client.query.load_dataset_files.return_value = expected_result
        
        result = client.load_dataset_files("test_dataset")
        
        assert 'train' in result
        assert 'test' in result
        assert len(result['train']) == 2
        assert len(result['test']) == 2
        assert list(result['train'].columns) == ['id', 'value']
        
        client.query.load_dataset_files.assert_called_once_with("test_dataset", True, None)

    def test_export_dataset_csv(self, client, tmp_path):
        """Test exporting dataset to CSV."""
        # Mock export client's export_dataset method
        expected_result = [str(tmp_path / "export.csv")]
        client.export.export_dataset.return_value = expected_result
        
        result = client.export_dataset(
            "test_dataset",
            output_dir=str(tmp_path),
            format="csv",
            compression="gzip"
        )
        
        assert len(result) == 1
        assert result[0] == str(tmp_path / "export.csv")
        
        client.export.export_dataset.assert_called_once_with(
            "test_dataset",
            str(tmp_path),
            format="csv",
            compression="gzip"
        )

    def test_get_column_info(self, client):
        """Test getting column information."""
        column_info = {
            'id': {'dtype': 'int64', 'nullable': False},
            'feature': {'dtype': 'float64', 'nullable': True}
        }
        
        # Mock query client's get_column_info method
        client.query.get_column_info.return_value = column_info
        
        result = client.get_column_info("test_dataset", table="train")
        
        assert 'id' in result
        assert 'feature' in result
        assert result['id']['dtype'] == 'int64'
        assert result['feature']['dtype'] == 'float64'
        
        client.query.get_column_info.assert_called_once_with("test_dataset", "train")

    def test_create_submission_kaggle(self, client, tmp_path):
        """Test creating Kaggle submission."""
        predictions = pd.DataFrame({
            'id': [1, 2, 3],
            'prediction': [0.8, 0.2, 0.6]
        })
        
        submission_file = tmp_path / "submission.csv"
        
        # Mock ml client's create_submission method
        client.ml.create_submission.return_value = str(submission_file)
        
        result = client.create_submission(
            "test_dataset",
            predictions,
            str(submission_file),
            format="kaggle"
        )
        
        assert result == str(submission_file)
        client.ml.create_submission.assert_called_once_with(
            "test_dataset",
            predictions,
            str(submission_file),
            format="kaggle"
        )

    def test_get_framework_adapter_sklearn(self, client):
        """Test getting sklearn adapter."""
        mock_adapter = Mock()
        client.ml.get_framework_adapter.return_value = mock_adapter
        
        adapter = client.get_framework_adapter("sklearn")
        
        assert adapter == mock_adapter
        client.ml.get_framework_adapter.assert_called_once_with("sklearn")

    def test_process_in_chunks(self, client):
        """Test chunk processing."""
        data = pd.DataFrame({'id': range(1000), 'value': range(1000)})
        
        def process_chunk(chunk):
            return chunk['value'].sum()
        
        with patch('mdm.utils.performance.ChunkProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_dataframe.return_value = [sum(range(1000))]
            mock_processor_class.return_value = mock_processor
            
            result = client.process_in_chunks(
                data,
                process_chunk,
                chunk_size=100
            )
            
            assert result == [sum(range(1000))]
            mock_processor.process_dataframe.assert_called_once_with(data, process_chunk, show_progress=False)

    def test_monitor_performance(self, client):
        """Test performance monitoring."""
        # The monitor_performance method returns a context manager that yields
        # the PerformanceMonitor instance. We need to test that it works correctly.
        with client.monitor_performance() as monitor:
            # Do some work
            assert monitor is not None
            assert hasattr(monitor, 'metrics')
            assert hasattr(monitor, 'start_time')
            
        # Check that the monitor was created and has the expected attributes
        assert client._performance_monitor is not None
        assert hasattr(client._performance_monitor, 'metrics')

    def test_create_time_series_splits(self, client):
        """Test time series splitting."""
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'value': range(100)
        })
        
        # Expected split result - split_by_folds returns dicts with 'train' and 'test' keys
        expected_folds = [
            {'train': data[:80], 'test': data[80:]}
        ]
        
        with patch('mdm.utils.time_series.TimeSeriesSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter.split_by_folds.return_value = expected_folds
            mock_splitter_class.return_value = mock_splitter
            
            result = client.create_time_series_splits(
                data,
                time_column='date',
                n_splits=1
            )
            
            # Result should be list of tuples
            assert len(result) == 1
            assert isinstance(result[0], tuple)
            assert len(result[0]) == 2
            assert result[0][0].equals(data[:80])  # train
            assert result[0][1].equals(data[80:])  # test
            
            mock_splitter.split_by_folds.assert_called_once_with(data, n_folds=1, gap_days=0)