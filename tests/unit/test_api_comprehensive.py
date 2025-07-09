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
            with patch('mdm.dataset.registrar.DatasetRegistrar') as mock_registrar_class:
                client = MDMClient()
                return client
    
    def test_register_dataset_minimal(self, client):
        """Test minimal dataset registration."""
        expected_info = Mock(spec=DatasetInfo)
        client.registrar.register.return_value = expected_info
        
        result = client.register_dataset('test_dataset', Path('/data/test'))
        
        client.registrar.register.assert_called_once_with(
            name='test_dataset',
            path=Path('/data/test'),
            auto_detect=True,
            description=None,
            tags=None
        )
        assert result == expected_info
    
    def test_register_dataset_full(self, client):
        """Test full dataset registration with all parameters."""
        expected_info = Mock(spec=DatasetInfo)
        client.registrar.register.return_value = expected_info
        
        result = client.register_dataset(
            name='test_dataset',
            path=Path('/data/test'),
            auto_detect=False,
            description='Test dataset',
            tags=['test', 'sample'],
            target_column='target',
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            id_columns=['id1', 'id2'],
            time_column='date',
            group_column='group'
        )
        
        client.registrar.register.assert_called_once_with(
            name='test_dataset',
            path=Path('/data/test'),
            auto_detect=False,
            description='Test dataset',
            tags=['test', 'sample'],
            target_column='target',
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            id_columns=['id1', 'id2'],
            time_column='date',
            group_column='group'
        )
        assert result == expected_info
    
    def test_list_datasets(self, client):
        """Test listing datasets."""
        expected_datasets = [Mock(), Mock()]
        client.manager.list_datasets.return_value = expected_datasets
        
        result = client.list_datasets(limit=10, sort_by='name')
        
        client.manager.list_datasets.assert_called_once_with(
            limit=10,
            sort_by='name',
            filter_backend=None
        )
        assert result == expected_datasets
    
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
        
        client.manager.remove_dataset.assert_called_once_with('test_dataset', force=True)
    
    def test_update_dataset(self, client):
        """Test updating dataset metadata."""
        client.update_dataset(
            'test_dataset',
            description='Updated description',
            tags=['new', 'tags']
        )
        
        client.manager.update_dataset_metadata.assert_called_once_with(
            'test_dataset',
            description='Updated description',
            tags=['new', 'tags']
        )
    
            limit=5
        )
        
        client.manager.search_datasets.assert_called_once_with(
            query='test',
            deep=True,
            pattern=True,
            case_sensitive=False,
            tag='sample',
            limit=5
        )
        assert result == expected_results


class TestMDMClientDataLoading:
    """Test MDMClient data loading methods."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.api.DatasetManager') as mock_manager_class:
            with patch('mdm.dataset.registrar.DatasetRegistrar') as mock_registrar_class:
                client = MDMClient()
                return client
    
    @patch('mdm.storage.factory.BackendFactory')
    def test_load_dataset_success(self, mock_factory, client):
        """Test successful dataset loading."""
        # Setup dataset info
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table'}
        client.manager.get_dataset.return_value = dataset_info
        
        # Setup backend
        mock_backend = Mock()
        expected_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_backend.read_table.return_value = expected_df
        mock_factory.create.return_value = mock_backend
        
        result = client.load_dataset('test_dataset')
        
        client.manager.get_dataset.assert_called_once_with('test_dataset')
        mock_factory.create.assert_called_once_with(
            'sqlite',
            {'backend': 'sqlite', 'path': '/tmp/test.db'}
        )
        mock_backend.read_table.assert_called_once_with('train_table')
        pd.testing.assert_frame_equal(result, expected_df)
    
    @patch('mdm.storage.factory.BackendFactory')
    def test_load_dataset_with_table(self, mock_factory, client):
        """Test loading specific table."""
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table', 'test': 'test_table'}
        client.manager.get_dataset.return_value = dataset_info
        
        mock_backend = Mock()
        expected_df = pd.DataFrame({'col1': [4, 5, 6]})
        mock_backend.read_table.return_value = expected_df
        mock_factory.create.return_value = mock_backend
        
        result = client.load_dataset('test_dataset', table='test')
        
        mock_backend.read_table.assert_called_once_with('test_table')
        pd.testing.assert_frame_equal(result, expected_df)
    
    def test_load_dataset_not_found(self, client):
        """Test loading non-existent dataset."""
        client.manager.get_dataset.side_effect = DatasetError("Dataset not found")
        
        with pytest.raises(DatasetError, match="Dataset not found"):
            client.load_dataset('nonexistent')
    
    @patch('mdm.storage.factory.BackendFactory')
    def test_load_dataset_files_success(self, mock_factory, client):
        """Test loading dataset files (train and test)."""
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table', 'test': 'test_table'}
        client.manager.get_dataset.return_value = dataset_info
        
        mock_backend = Mock()
        train_df = pd.DataFrame({'col1': [1, 2, 3]})
        test_df = pd.DataFrame({'col1': [4, 5, 6]})
        mock_backend.read_table.side_effect = [train_df, test_df]
        mock_factory.create.return_value = mock_backend
        
        result_train, result_test = client.load_dataset_files('test_dataset')
        
        assert mock_backend.read_table.call_count == 2
        pd.testing.assert_frame_equal(result_train, train_df)
        pd.testing.assert_frame_equal(result_test, test_df)
    
    @patch('mdm.storage.factory.BackendFactory')
    def test_load_dataset_files_no_test(self, mock_factory, client):
        """Test loading dataset files without test set."""
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table'}
        client.manager.get_dataset.return_value = dataset_info
        
        mock_backend = Mock()
        train_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_backend.read_table.return_value = train_df
        mock_factory.create.return_value = mock_backend
        
        result_train, result_test = client.load_dataset_files('test_dataset')
        
        pd.testing.assert_frame_equal(result_train, train_df)
        assert result_test is None


class TestMDMClientStatisticsAndExport:
    """Test MDMClient statistics and export methods."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.api.DatasetManager') as mock_manager_class:
            with patch('mdm.dataset.registrar.DatasetRegistrar') as mock_registrar_class:
                client = MDMClient()
                return client
    
        """Test exporting dataset."""
        expected_files = [Path('/tmp/export/train.csv')]
        client.manager.export_dataset.return_value = expected_files
        
        result = client.export_dataset(
            'test_dataset',
            output_dir=Path('/tmp/export'),
            format='csv',
            compression='gzip'
        )
        
        client.manager.export_dataset.assert_called_once_with(
            'test_dataset',
            output_dir=Path('/tmp/export'),
            format='csv',
            compression='gzip',
            table=None,
            metadata_only=False,
            no_header=False
        )
        assert result == expected_files


class TestMDMClientTimeSeries:
    """Test MDMClient time series methods."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.api.DatasetManager') as mock_manager_class:
            with patch('mdm.dataset.registrar.DatasetRegistrar') as mock_registrar_class:
                client = MDMClient()
                return client
    
    @patch('mdm.storage.factory.BackendFactory')
    @patch('mdm.api.TimeSeriesSplitter')
    def test_split_time_series_basic(self, mock_splitter_class, mock_factory, client):
        """Test basic time series split."""
        # Setup dataset info
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table'}
        dataset_info.time_column = 'date'
        dataset_info.group_column = None
        client.manager.get_dataset.return_value = dataset_info
        
        # Setup backend
        mock_backend = Mock()
        full_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'value': range(100)
        })
        mock_backend.read_table.return_value = full_df
        mock_factory.create.return_value = mock_backend
        
        # Setup splitter
        mock_splitter = Mock()
        train_df = full_df.iloc[:80]
        test_df = full_df.iloc[80:]
        mock_splitter.split_by_days.return_value = (train_df, None, test_df)
        mock_splitter_class.return_value = mock_splitter
        
        result = client.split_time_series('test_dataset')
        
        assert 'train' in result
        assert 'test' in result
        assert result['val'] is None
        pd.testing.assert_frame_equal(result['train'], train_df)
        pd.testing.assert_frame_equal(result['test'], test_df)
    
    @patch('mdm.storage.factory.BackendFactory')
    @patch('mdm.api.TimeSeriesSplitter')
    def test_split_time_series_with_validation(self, mock_splitter_class, mock_factory, client):
        """Test time series split with validation set."""
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        dataset_info.tables = {'train': 'train_table'}
        dataset_info.time_column = 'date'
        dataset_info.group_column = 'group'
        client.manager.get_dataset.return_value = dataset_info
        
        mock_backend = Mock()
        full_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'group': ['A'] * 50 + ['B'] * 50,
            'value': range(100)
        })
        mock_backend.read_table.return_value = full_df
        mock_factory.create.return_value = mock_backend
        
        mock_splitter = Mock()
        train_df = full_df.iloc[:60]
        val_df = full_df.iloc[60:80]
        test_df = full_df.iloc[80:]
        mock_splitter.split_by_days.return_value = (train_df, val_df, test_df)
        mock_splitter_class.return_value = mock_splitter
        
        result = client.split_time_series('test_dataset', test_days=20, val_days=20)
        
        mock_splitter.split_by_days.assert_called_once_with(full_df, 20, 20)
        assert all(k in result for k in ['train', 'val', 'test'])
        pd.testing.assert_frame_equal(result['train'], train_df)
        pd.testing.assert_frame_equal(result['val'], val_df)
        pd.testing.assert_frame_equal(result['test'], test_df)
    
    def test_split_time_series_no_time_column(self, client):
        """Test time series split without time column."""
        dataset_info = Mock(spec=DatasetInfo)
        dataset_info.time_column = None
        client.manager.get_dataset.return_value = dataset_info
        
        with pytest.raises(DatasetError, match="does not have a time column"):
            client.split_time_series('test_dataset')


class TestMDMClientDataframeOperations:
    """Test MDMClient dataframe operations."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.api.DatasetManager') as mock_manager_class:
            with patch('mdm.dataset.registrar.DatasetRegistrar') as mock_registrar_class:
                client = MDMClient()
                return client
    

class TestMDMClientQueryOperations:
    """Test MDMClient query operations."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.api.DatasetManager') as mock_manager_class:
            with patch('mdm.dataset.registrar.DatasetRegistrar') as mock_registrar_class:
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
        mock_backend.query.return_value = expected_df
        mock_factory.create.return_value = mock_backend
        
        result = client.query_dataset(
            'test_dataset',
            "SELECT COUNT(*) as count FROM train_table",
            params={'param1': 'value1'}
        )
        
        mock_backend.query.assert_called_once_with(
            "SELECT COUNT(*) as count FROM train_table",
            params={'param1': 'value1'}
        )
        pd.testing.assert_frame_equal(result, expected_df)
    
    @patch('mdm.storage.factory.BackendFactory')
            "UPDATE train_table SET col1 = 1"
        )
        
        mock_backend.execute.assert_called_once_with(
            "UPDATE train_table SET col1 = 1"
        )


class TestMDMClientTableInfo:
    """Test MDMClient table information methods."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.api.DatasetManager') as mock_manager_class:
            with patch('mdm.dataset.registrar.DatasetRegistrar') as mock_registrar_class:
                client = MDMClient()
                return client
    
    @patch('mdm.storage.factory.BackendFactory')
                {'name': 'col1', 'type': 'INTEGER'},
                {'name': 'col2', 'type': 'TEXT'}
            ]
        }
        mock_backend.get_table_info.return_value = expected_info
        mock_factory.create.return_value = mock_backend
        
        result = client.get_table_info('test_dataset', 'train')
        
        mock_backend.get_table_info.assert_called_once_with('train_table')
        assert result == expected_info
    
    @patch('mdm.storage.factory.BackendFactory')
        result = client.list_tables('test_dataset')
        
        mock_backend.list_tables.assert_called_once()
        assert result == expected_tables