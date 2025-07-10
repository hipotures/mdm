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
        with patch('mdm.api.get_config', return_value=mock_config):
            with patch('mdm.api.DatasetManager', return_value=mock_manager):
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
        with patch('mdm.api.get_config', return_value=mock_config):
            with patch('mdm.api.DatasetManager') as mock_dm:
                client = MDMClient()
                assert client.config == mock_config
                assert client.manager is not None
                mock_dm.assert_called_once()

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        custom_config = Mock()
        with patch('mdm.api.DatasetManager') as mock_dm:
            client = MDMClient(config=custom_config)
            assert client.config == custom_config
            mock_dm.assert_called_once()

    def test_register_dataset_success(self, client, sample_dataset_info, tmp_path):
        """Test successful dataset registration."""
        # Create test data
        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "train.csv").write_text("id,feature,target\n1,0.5,1\n")
        
        with patch('mdm.api.DatasetRegistrar') as mock_registrar:
            mock_instance = Mock()
            mock_instance.register.return_value = sample_dataset_info
            mock_registrar.return_value = mock_instance
            
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
            mock_registrar.assert_called_once_with(client.manager)
            mock_instance.register.assert_called_once()
            
            # Check call arguments
            call_args = mock_instance.register.call_args
            assert call_args.kwargs['name'] == "test_dataset"
            assert call_args.kwargs['path'] == data_path
            assert call_args.kwargs['target_column'] == "target"
            assert call_args.kwargs['id_columns'] == ["id"]
            assert call_args.kwargs['force'] is False

    def test_register_dataset_with_kwargs(self, client, sample_dataset_info, tmp_path):
        """Test dataset registration with additional kwargs."""
        data_path = tmp_path / "data"
        data_path.mkdir()
        
        with patch('mdm.api.DatasetRegistrar') as mock_registrar:
            mock_instance = Mock()
            mock_instance.register.return_value = sample_dataset_info
            mock_registrar.return_value = mock_instance
            
            result = client.register_dataset(
                name="test_dataset",
                dataset_path=str(data_path),
                custom_param="custom_value",
                another_param=123
            )
            
            # Check that kwargs are passed through
            call_kwargs = mock_instance.register.call_args.kwargs
            assert call_kwargs['custom_param'] == "custom_value"
            assert call_kwargs['another_param'] == 123

    def test_register_dataset_path_conversion(self, client, sample_dataset_info):
        """Test that string path is converted to Path object."""
        with patch('mdm.api.DatasetRegistrar') as mock_registrar:
            mock_instance = Mock()
            mock_instance.register.return_value = sample_dataset_info
            mock_registrar.return_value = mock_instance
            
            client.register_dataset(
                name="test_dataset",
                dataset_path="/data/test"
            )
            
            call_args = mock_instance.register.call_args
            assert isinstance(call_args.kwargs['path'], Path)
            assert str(call_args.kwargs['path']) == "/data/test"

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
        mock_manager.delete_dataset.return_value = None
        
        client.remove_dataset("test_dataset", force=True)
        
        mock_manager.delete_dataset.assert_called_once_with("test_dataset", force=True)

    def test_remove_dataset_not_found(self, client, mock_manager):
        """Test removing non-existent dataset."""
        mock_manager.delete_dataset.side_effect = DatasetError("Not found")
        
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
            {"description": "Updated description", "tags": ["test", "updated"]}
        )

    def test_search_datasets_simple(self, client, mock_manager, sample_dataset_info):
        """Test searching datasets with simple query."""
        mock_manager.search_datasets.return_value = [sample_dataset_info]
        
        result = client.search_datasets("test")
        
        assert len(result) == 1
        assert result[0] == sample_dataset_info
        mock_manager.search_datasets.assert_called_once_with("test", deep=False, case_sensitive=False)

    def test_search_datasets_advanced(self, client, mock_manager):
        """Test searching datasets with advanced options."""
        mock_manager.search_datasets.return_value = []
        
        result = client.search_datasets("TEST", deep=True, case_sensitive=True)
        
        assert result == []
        mock_manager.search_datasets.assert_called_once_with("TEST", deep=True, case_sensitive=True)

    def test_search_datasets_by_tag(self, client, mock_manager, sample_dataset_info):
        """Test searching datasets by tag."""
        mock_manager.search_datasets_by_tag.return_value = [sample_dataset_info]
        
        result = client.search_datasets_by_tag("test")
        
        assert len(result) == 1
        assert result[0] == sample_dataset_info
        mock_manager.search_datasets_by_tag.assert_called_once_with("test")

    def test_get_statistics_success(self, client, mock_manager):
        """Test getting dataset statistics."""
        stats = DatasetStatistics(
            row_count=1000,
            column_count=10,
            memory_usage_mb=50.0
        )
        mock_manager.get_statistics.return_value = stats
        
        result = client.get_statistics("test_dataset")
        
        assert result == stats
        mock_manager.get_statistics.assert_called_once_with("test_dataset")

    def test_get_statistics_not_found(self, client, mock_manager):
        """Test getting statistics for non-existent dataset."""
        mock_manager.get_statistics.return_value = None
        
        result = client.get_statistics("nonexistent")
        
        assert result is None
        mock_manager.get_statistics.assert_called_once_with("nonexistent")

    def test_load_dataset_pandas_success(self, client, mock_manager, tmp_path):
        """Test loading dataset as pandas DataFrame."""
        # Create mock backend
        mock_backend = Mock()
        mock_backend.query.return_value = pd.DataFrame({
            'id': [1, 2, 3],
            'feature': [0.1, 0.2, 0.3],
            'target': [0, 1, 0]
        })
        
        # Mock manager methods
        mock_manager.get_dataset.return_value = Mock(
            tables={'train': 'test_dataset_train'},
            database={'backend': 'sqlite', 'path': str(tmp_path / 'test.db')}
        )
        
        with patch.object(client, 'manager', mock_manager):
            mock_manager.get_backend.return_value = mock_backend
            
            df = client.load_dataset("test_dataset", table="train")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert list(df.columns) == ['id', 'feature', 'target']
            mock_backend.query.assert_called_once_with("SELECT * FROM test_dataset_train")

    def test_load_dataset_table_not_found(self, client, mock_manager):
        """Test loading non-existent table."""
        mock_manager.get_dataset.return_value = Mock(
            tables={'train': 'test_dataset_train'}
        )
        
        with pytest.raises(DatasetError, match="Table 'test' not found"):
            client.load_dataset("test_dataset", table="test")

    def test_load_dataset_files_success(self, client, mock_manager, tmp_path):
        """Test loading dataset files directly."""
        # Create test files
        train_file = tmp_path / "train.csv"
        test_file = tmp_path / "test.csv"
        
        train_df = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        test_df = pd.DataFrame({'id': [3, 4], 'value': [30, 40]})
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        mock_manager.get_dataset.return_value = Mock(
            source_path=str(tmp_path),
            tables={'train': 'train', 'test': 'test'}
        )
        
        # Mock discover_data_files
        with patch('mdm.dataset.auto_detect.discover_data_files') as mock_discover:
            mock_discover.return_value = {
                'train': train_file,
                'test': test_file
            }
            
            # Mock backend to return dataframes
            mock_backend = Mock()
            mock_backend.get_engine.return_value = Mock()
            mock_backend.read_table_to_dataframe.side_effect = [train_df, test_df]
            
            with patch.object(client, 'manager', mock_manager):
                mock_manager.get_backend.return_value = mock_backend
                
                train_loaded, test_loaded = client.load_dataset_files("test_dataset")
                
                assert len(train_loaded) == 2
                assert len(test_loaded) == 2
                assert list(train_loaded.columns) == ['id', 'value']

    def test_export_dataset_csv(self, client, mock_manager, tmp_path):
        """Test exporting dataset to CSV."""
        mock_manager.get_dataset.return_value = Mock(name="test_dataset")
        
        # Mock ExportOperation
        with patch('mdm.dataset.operations.ExportOperation') as mock_export_class:
            mock_export = Mock()
            mock_export.execute.return_value = [tmp_path / "export.csv"]
            mock_export_class.return_value = mock_export
            
            result = client.export_dataset(
                "test_dataset",
                output_dir=str(tmp_path),
                format="csv",
                compression="gzip"
            )
            
            assert len(result) == 1
            assert result[0] == str(tmp_path / "export.csv")
            
            mock_export.execute.assert_called_once_with(
                name="test_dataset",
                format="csv",
                output_dir=Path(str(tmp_path)),
                table=None,
                compression="gzip"
            )

    def test_get_column_info(self, client, mock_manager):
        """Test getting column information."""
        column_info = {
            'id': {'dtype': 'int64', 'nullable': False},
            'feature': {'dtype': 'float64', 'nullable': True}
        }
        
        mock_manager.get_dataset.return_value = Mock(
            tables={'train': 'test_dataset_train'}
        )
        
        # Mock backend and column info retrieval
        mock_backend = Mock()
        mock_backend.get_columns.return_value = ['id', 'feature']
        mock_backend.analyze_column.side_effect = lambda col, table, engine: column_info[col]
        
        with patch('mdm.api.BackendFactory.create', return_value=mock_backend):
            with patch.object(client, 'manager', mock_manager):
                mock_manager.get_backend.return_value = mock_backend
                
                result = client.get_column_info("test_dataset", table="train")
                
                assert 'id' in result
                assert 'feature' in result
                assert result['id']['dtype'] == 'int64'

    def test_create_submission_kaggle(self, client, tmp_path):
        """Test creating Kaggle submission."""
        predictions = pd.DataFrame({
            'id': [1, 2, 3],
            'prediction': [0.8, 0.2, 0.6]
        })
        
        submission_file = tmp_path / "submission.csv"
        
        # Mock SubmissionCreator
        with patch('mdm.utils.integration.SubmissionCreator') as mock_creator_class:
            mock_creator = Mock()
            mock_creator.create_kaggle_submission.return_value = submission_file
            mock_creator_class.return_value = mock_creator
            
            result = client.create_submission(
                predictions,
                output_path=str(submission_file),
                format="kaggle"
            )
            
            assert result == submission_file
            mock_creator.create_kaggle_submission.assert_called_once()

    def test_get_framework_adapter_sklearn(self, client):
        """Test getting sklearn adapter."""
        with patch('mdm.utils.integration.MLFrameworkAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter_class.return_value = mock_adapter
            
            adapter = client.get_framework_adapter("sklearn")
            
            assert adapter == mock_adapter
            mock_adapter_class.assert_called_once_with("sklearn", config=client.config)

    def test_process_in_chunks(self, client):
        """Test chunk processing."""
        data = pd.DataFrame({'id': range(1000), 'value': range(1000)})
        
        def process_chunk(chunk):
            return chunk['value'].sum()
        
        with patch('mdm.utils.performance.ChunkProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process.return_value = [sum(range(1000))]
            mock_processor_class.return_value = mock_processor
            
            result = client.process_in_chunks(
                data,
                process_chunk,
                chunk_size=100
            )
            
            assert result == [sum(range(1000))]
            mock_processor.process.assert_called_once()

    def test_monitor_performance(self, client):
        """Test performance monitoring."""
        with patch('mdm.utils.performance.PerformanceMonitor') as mock_monitor_class:
            mock_monitor = Mock()
            mock_monitor.__enter__ = Mock(return_value=mock_monitor)
            mock_monitor.__exit__ = Mock(return_value=None)
            mock_monitor.get_report.return_value = {"memory_usage": 100}
            mock_monitor_class.return_value = mock_monitor
            
            with client.monitor_performance() as monitor:
                # Do some work
                pass
            
            assert monitor == mock_monitor
            mock_monitor.__enter__.assert_called_once()
            mock_monitor.__exit__.assert_called_once()

    def test_create_time_series_splits(self, client):
        """Test time series splitting."""
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'value': range(100)
        })
        
        with patch('mdm.utils.time_series.TimeSeriesSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter.split.return_value = [(data[:80], data[80:])]
            mock_splitter_class.return_value = mock_splitter
            
            result = client.create_time_series_splits(
                data,
                time_column='date',
                n_splits=1
            )
            
            assert len(result) == 1
            assert len(result[0][0]) == 80  # Train
            assert len(result[0][1]) == 20  # Test