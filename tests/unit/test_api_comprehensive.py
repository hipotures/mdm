"""Comprehensive unit tests for MDM API."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path

from mdm.api import MDMClient
from mdm.core.exceptions import DatasetError
from mdm.models.dataset import DatasetInfo
from mdm.models.enums import ProblemType


class TestMDMClientInit:
    """Test MDMClient initialization."""
    
    @patch('mdm.api.get_config')
    @patch('mdm.api.DatasetManager')
    def test_init_default(self, mock_manager_class, mock_get_config):
        """Test default initialization."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        client = MDMClient()
        
        mock_get_config.assert_called_once()
        mock_manager_class.assert_called_once()
        assert client.config == mock_config
        assert client.manager == mock_manager_class.return_value
    
    @patch('mdm.api.DatasetManager')
    def test_init_with_config(self, mock_manager_class):
        """Test initialization with custom config."""
        custom_config = Mock()
        client = MDMClient(config=custom_config)
        
        assert client.config == custom_config
        mock_manager_class.assert_called_once()


class TestMDMClientDatasetOperations:
    """Test MDMClient dataset operations."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.api.DatasetManager') as mock_manager_class:
            client = MDMClient()
            return client
    
    @patch('mdm.api.DatasetRegistrar')
    def test_register_dataset_minimal(self, mock_registrar_class, client):
        """Test minimal dataset registration."""
        mock_registrar = Mock()
        mock_registrar_class.return_value = mock_registrar
        expected_info = Mock(spec=DatasetInfo)
        mock_registrar.register.return_value = expected_info
        
        result = client.register_dataset('test_dataset', '/data/test')
        
        mock_registrar.register.assert_called_once_with(
            name='test_dataset',
            path=Path('/data/test'),
            auto_detect=True,
            description=None,
            tags=None,
            target_column=None,
            problem_type=None,
            id_columns=None,
            force=False
        )
        assert result == expected_info
    
    @patch('mdm.api.DatasetRegistrar')
    def test_register_dataset_full(self, mock_registrar_class, client):
        """Test full dataset registration with all parameters."""
        mock_registrar = Mock()
        mock_registrar_class.return_value = mock_registrar
        expected_info = Mock(spec=DatasetInfo)
        mock_registrar.register.return_value = expected_info
        
        result = client.register_dataset(
            name='test_dataset',
            dataset_path='/data/test',
            auto_analyze=False,
            description='Test dataset',
            tags=['test', 'sample'],
            target_column='target',
            problem_type='classification',
            id_columns=['id1', 'id2'],
            force=True
        )
        
        mock_registrar.register.assert_called_once_with(
            name='test_dataset',
            path=Path('/data/test'),
            auto_detect=False,
            description='Test dataset',
            tags=['test', 'sample'],
            target_column='target',
            problem_type='classification',
            id_columns=['id1', 'id2'],
            force=True
        )
        assert result == expected_info
    
    def test_list_datasets(self, client):
        """Test listing datasets."""
        mock_dataset1 = Mock(spec=DatasetInfo)
        mock_dataset1.name = 'dataset1'
        mock_dataset2 = Mock(spec=DatasetInfo)
        mock_dataset2.name = 'dataset2'
        expected_datasets = [mock_dataset2, mock_dataset1]  # Out of order
        client.manager.list_datasets.return_value = expected_datasets
        
        result = client.list_datasets(limit=10, sort_by='name')
        
        # Manager's list_datasets is called without parameters
        client.manager.list_datasets.assert_called_once_with()
        # Result should be sorted by name and limited
        assert len(result) == 2
        assert result[0].name == 'dataset1'
        assert result[1].name == 'dataset2'
    
    def test_get_dataset_exists(self, client):
        """Test getting existing dataset."""
        expected_info = Mock(spec=DatasetInfo)
        client.manager.get_dataset.return_value = expected_info
        
        result = client.get_dataset('test_dataset')
        
        client.manager.get_dataset.assert_called_once_with('test_dataset')
        assert result == expected_info
    
    def test_get_dataset_not_exists(self, client):
        """Test getting non-existent dataset."""
        client.manager.get_dataset.side_effect = DatasetError("Dataset not found")
        
        result = client.get_dataset('nonexistent')
        
        assert result is None
    
    def test_remove_dataset(self, client):
        """Test removing dataset."""
        client.remove_dataset('test_dataset', force=True)
        
        client.manager.remove_dataset.assert_called_once_with('test_dataset')
    
    def test_update_dataset(self, client):
        """Test updating dataset metadata."""
        client.update_dataset(
            'test_dataset',
            description='Updated description',
            tags=['new', 'tags']
        )
        
        client.manager.update_dataset.assert_called_once_with(
            'test_dataset',
            {
                'description': 'Updated description',
                'tags': ['new', 'tags']
            }
        )


class TestMDMClientDataLoading:
    """Test MDMClient data loading methods."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.api.DatasetManager') as mock_manager_class:
            client = MDMClient()
            return client
    
    def test_load_dataset_files_success(self, client):
        """Test loading dataset files (train and test)."""
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table', 'test': 'test_table'}
        client.get_dataset = Mock(return_value=dataset_info)
        
        mock_backend = Mock()
        mock_engine = Mock()
        train_df = pd.DataFrame({'col1': [1, 2, 3]})
        test_df = pd.DataFrame({'col1': [4, 5, 6]})
        mock_backend.get_engine.return_value = mock_engine
        mock_backend.read_table_to_dataframe.side_effect = [train_df, test_df]
        client.manager.get_backend.return_value = mock_backend
        
        result_train, result_test = client.load_dataset_files('test_dataset')
        
        assert mock_backend.read_table_to_dataframe.call_count == 2
        pd.testing.assert_frame_equal(result_train, train_df)
        pd.testing.assert_frame_equal(result_test, test_df)
    
    def test_load_dataset_files_no_test(self, client):
        """Test loading dataset files without test set."""
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table'}
        client.get_dataset = Mock(return_value=dataset_info)
        
        mock_backend = Mock()
        mock_engine = Mock()
        train_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_backend.get_engine.return_value = mock_engine
        mock_backend.read_table_to_dataframe.return_value = train_df
        client.manager.get_backend.return_value = mock_backend
        
        result_train, result_test = client.load_dataset_files('test_dataset')
        
        pd.testing.assert_frame_equal(result_train, train_df)
        assert result_test is None
    
    def test_load_table(self, client):
        """Test loading specific table."""
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table', 'test': 'test_table'}
        client.get_dataset = Mock(return_value=dataset_info)
        
        mock_backend = Mock()
        mock_engine = Mock()
        expected_df = pd.DataFrame({'col1': [4, 5, 6]})
        mock_backend.get_engine.return_value = mock_engine
        mock_backend.read_table_to_dataframe.return_value = expected_df
        client.manager.get_backend.return_value = mock_backend
        
        result = client.load_table('test_dataset', 'test')
        
        mock_backend.read_table_to_dataframe.assert_called_once_with('test_table', mock_engine)
        pd.testing.assert_frame_equal(result, expected_df)


class TestMDMClientStatisticsAndExport:
    """Test MDMClient statistics and export methods."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.api.DatasetManager') as mock_manager_class:
            client = MDMClient()
            return client
    
    @patch('mdm.dataset.operations.ExportOperation')
    def test_export_dataset(self, mock_export_class, client):
        """Test exporting dataset."""
        mock_export = Mock()
        mock_export_class.return_value = mock_export
        expected_files = [Path('/tmp/export/train.csv')]
        mock_export.execute.return_value = expected_files
        
        result = client.export_dataset(
            'test_dataset',
            output_dir='/tmp/export',
            format='csv',
            compression='gzip'
        )
        
        mock_export.execute.assert_called_once_with(
            name='test_dataset',
            format='csv',
            output_dir=Path('/tmp/export'),
            table=None,
            compression='gzip'
        )
        assert result == ['/tmp/export/train.csv']
    
    @patch('mdm.dataset.statistics.DatasetStatistics')
    def test_get_statistics(self, mock_stats_class, client):
        """Test getting dataset statistics."""
        mock_stats = Mock()
        mock_stats_class.return_value = mock_stats
        expected_stats = {'rows': 1000, 'columns': 10}
        mock_stats.compute_statistics.return_value = expected_stats
        
        result = client.get_statistics('test_dataset', full=True)
        
        mock_stats.compute_statistics.assert_called_once_with('test_dataset', full=True, save=False)
        assert result == expected_stats


class TestMDMClientTimeSeries:
    """Test MDMClient time series methods."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.api.DatasetManager') as mock_manager_class:
            client = MDMClient()
            return client
    
    @patch('mdm.api.TimeSeriesSplitter')
    def test_split_time_series_basic(self, mock_splitter_class, client):
        """Test basic time series split."""
        # Setup dataset info
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table'}
        dataset_info.time_column = 'date'
        dataset_info.group_column = None
        client.get_dataset = Mock(return_value=dataset_info)
        
        # Setup load_dataset_files
        full_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'value': range(100)
        })
        client.load_dataset_files = Mock(return_value=(full_df, None))
        
        # Setup splitter
        mock_splitter = Mock()
        train_df = full_df.iloc[:80]
        test_df = full_df.iloc[80:]
        result_dict = {'train': train_df, 'val': None, 'test': test_df}
        mock_splitter.split_by_time.return_value = result_dict
        mock_splitter_class.return_value = mock_splitter
        
        result = client.split_time_series('test_dataset', test_size=20)
        
        client.get_dataset.assert_called_once_with('test_dataset')
        client.load_dataset_files.assert_called_once_with('test_dataset')
        mock_splitter_class.assert_called_once_with('date', None)
        mock_splitter.split_by_time.assert_called_once_with(full_df, 20, None)
        assert result == result_dict
    
    @patch('mdm.api.TimeSeriesSplitter')
    def test_split_time_series_with_validation(self, mock_splitter_class, client):
        """Test time series split with validation set."""
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table'}
        dataset_info.time_column = 'date'
        dataset_info.group_column = 'group'
        client.get_dataset = Mock(return_value=dataset_info)
        
        full_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'group': ['A'] * 50 + ['B'] * 50,
            'value': range(100)
        })
        client.load_dataset_files = Mock(return_value=(full_df, None))
        
        mock_splitter = Mock()
        train_df = full_df.iloc[:60]
        val_df = full_df.iloc[60:80]
        test_df = full_df.iloc[80:]
        result_dict = {'train': train_df, 'val': val_df, 'test': test_df}
        mock_splitter.split_by_time.return_value = result_dict
        mock_splitter_class.return_value = mock_splitter
        
        result = client.split_time_series('test_dataset', test_size=20, validation_size=20)
        
        mock_splitter_class.assert_called_once_with('date', 'group')
        mock_splitter.split_by_time.assert_called_once_with(full_df, 20, 20)
        assert result == result_dict
    
    def test_split_time_series_no_time_column(self, client):
        """Test time series split without time column."""
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.time_column = None
        client.get_dataset = Mock(return_value=dataset_info)
        
        with pytest.raises(ValueError, match="No time column specified"):
            client.split_time_series('test_dataset', test_size=20)


class TestMDMClientQueryOperations:
    """Test MDMClient query operations."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.api.DatasetManager') as mock_manager_class:
            client = MDMClient()
            return client
    
    @patch('mdm.storage.factory.BackendFactory')
    def test_query_dataset(self, mock_factory, client):
        """Test querying dataset."""
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table'}
        client.manager.get_dataset.return_value = dataset_info
        
        mock_backend = Mock()
        expected_df = pd.DataFrame({'count': [42]})
        mock_backend.execute_query.return_value = expected_df
        client.manager.get_backend.return_value = mock_backend
        
        result = client.query_dataset(
            'test_dataset',
            "SELECT COUNT(*) as count FROM train_table"
        )
        
        mock_backend.execute_query.assert_called_once_with(
            "SELECT COUNT(*) as count FROM train_table"
        )
        pd.testing.assert_frame_equal(result, expected_df)