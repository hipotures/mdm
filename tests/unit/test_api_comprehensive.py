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
    
    @patch('mdm.api.mdm_client.get_config')
    @patch('mdm.core.get_service')
    def test_init_default(self, mock_get_service, mock_get_config):
        """Test default initialization."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        # Mock the specialized clients
        mock_get_service.side_effect = lambda cls: Mock()
        
        client = MDMClient()
        
        mock_get_config.assert_called_once()
        assert client.config == mock_config
        assert client.registration is not None
        assert client.query is not None
        assert client.ml is not None
        assert client.export is not None
        assert client.management is not None
    
    @patch('mdm.core.get_service')
    def test_init_with_config(self, mock_get_service):
        """Test initialization with custom config."""
        custom_config = Mock()
        mock_get_service.side_effect = lambda cls: Mock()
        
        client = MDMClient(config=custom_config)
        
        assert client.config == custom_config
        assert client.registration is not None


class TestMDMClientDatasetOperations:
    """Test MDMClient dataset operations."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.core.get_service') as mock_get_service:
            # Create mock specialized clients
            mock_registration = Mock()
            mock_query = Mock()
            mock_ml = Mock()
            mock_export = Mock()
            mock_management = Mock()
            
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
            client = MDMClient()
            return client
    
    def test_register_dataset_minimal(self, client):
        """Test minimal dataset registration."""
        expected_info = Mock(spec=DatasetInfo)
        client.registration.register_dataset.return_value = expected_info
        
        result = client.register_dataset('test_dataset', '/data/test')
        
        client.registration.register_dataset.assert_called_once_with(
            'test_dataset', 
            '/data/test'
        )
        assert result == expected_info
    
    def test_register_dataset_full(self, client):
        """Test full dataset registration with all parameters."""
        expected_info = Mock(spec=DatasetInfo)
        client.registration.register_dataset.return_value = expected_info
        
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
        
        client.registration.register_dataset.assert_called_once_with(
            'test_dataset',
            '/data/test',
            auto_analyze=False,
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
        client.query.list_datasets.return_value = expected_datasets
        
        result = client.list_datasets(limit=10, sort_by='name')
        
        # Query's list_datasets is called
        client.query.list_datasets.assert_called_once()
        # Result should be the same as returned by query client
        assert result == expected_datasets
    
    def test_get_dataset_exists(self, client):
        """Test getting existing dataset."""
        expected_info = Mock(spec=DatasetInfo)
        client.query.get_dataset.return_value = expected_info
        
        result = client.get_dataset('test_dataset')
        
        client.query.get_dataset.assert_called_once_with('test_dataset')
        assert result == expected_info
    
    def test_get_dataset_not_exists(self, client):
        """Test getting non-existent dataset."""
        client.query.get_dataset.side_effect = DatasetError("Dataset not found")
        
        with pytest.raises(DatasetError, match="Dataset not found"):
            client.get_dataset('nonexistent')
    
    def test_remove_dataset(self, client):
        """Test removing dataset."""
        client.remove_dataset('test_dataset', force=True)
        
        client.management.remove_dataset.assert_called_once_with('test_dataset', True)
    
    def test_update_dataset(self, client):
        """Test updating dataset metadata."""
        client.update_dataset(
            'test_dataset',
            description='Updated description',
            tags=['new', 'tags']
        )
        
        client.management.update_dataset.assert_called_once_with(
            'test_dataset',
            description='Updated description',
            tags=['new', 'tags']
        )


class TestMDMClientDataLoading:
    """Test MDMClient data loading methods."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.core.get_service') as mock_get_service:
            # Create mock specialized clients
            mock_registration = Mock()
            mock_query = Mock()
            mock_ml = Mock()
            mock_export = Mock()
            mock_management = Mock()
            
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
            client = MDMClient()
            return client
    
    def test_load_dataset_files_success(self, client):
        """Test loading dataset files (train and test)."""
        train_df = pd.DataFrame({'col1': [1, 2, 3]})
        test_df = pd.DataFrame({'col1': [4, 5, 6]})
        
        # The load_dataset_files method returns a dictionary, not a tuple
        client.query.load_dataset_files.return_value = {'train': train_df, 'test': test_df}
        
        result = client.load_dataset_files('test_dataset')
        
        client.query.load_dataset_files.assert_called_once_with('test_dataset', True, None)
        assert 'train' in result
        assert 'test' in result
        pd.testing.assert_frame_equal(result['train'], train_df)
        pd.testing.assert_frame_equal(result['test'], test_df)
    
    def test_load_dataset_files_no_test(self, client):
        """Test loading dataset files without test set."""
        train_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # The load_dataset_files method returns a dictionary with only train
        client.query.load_dataset_files.return_value = {'train': train_df}
        
        result = client.load_dataset_files('test_dataset')
        
        client.query.load_dataset_files.assert_called_once_with('test_dataset', True, None)
        assert 'train' in result
        assert 'test' not in result
        pd.testing.assert_frame_equal(result['train'], train_df)
    
    def test_load_table(self, client):
        """Test loading specific table."""
        expected_df = pd.DataFrame({'col1': [4, 5, 6]})
        
        # The MDMClient has load_dataset method with table parameter
        client.query.load_dataset.return_value = expected_df
        
        result = client.load_dataset('test_dataset', table='test')
        
        client.query.load_dataset.assert_called_once_with('test_dataset', table='test')
        pd.testing.assert_frame_equal(result, expected_df)


class TestMDMClientStatisticsAndExport:
    """Test MDMClient statistics and export methods."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.core.get_service') as mock_get_service:
            # Create mock specialized clients
            mock_registration = Mock()
            mock_query = Mock()
            mock_ml = Mock()
            mock_export = Mock()
            mock_management = Mock()
            
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
            client = MDMClient()
            return client
    
    def test_export_dataset(self, client):
        """Test exporting dataset."""
        expected_files = ['/tmp/export/train.csv']
        client.export.export_dataset.return_value = expected_files
        
        result = client.export_dataset(
            'test_dataset',
            output_dir='/tmp/export',
            format='csv',
            compression='gzip'
        )
        
        client.export.export_dataset.assert_called_once_with(
            'test_dataset',
            '/tmp/export',
            format='csv',
            compression='gzip'
        )
        assert result == expected_files
    
    def test_get_statistics(self, client):
        """Test getting dataset statistics."""
        expected_stats = {'rows': 1000, 'columns': 10}
        client.management.get_statistics.return_value = expected_stats
        
        result = client.get_statistics('test_dataset', full=True)
        
        client.management.get_statistics.assert_called_once_with('test_dataset', True)
        assert result == expected_stats


class TestMDMClientTimeSeries:
    """Test MDMClient time series methods."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.core.get_service') as mock_get_service:
            # Create mock specialized clients
            mock_registration = Mock()
            mock_query = Mock()
            mock_ml = Mock()
            mock_export = Mock()
            mock_management = Mock()
            
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
            client = MDMClient()
            return client
    
    def test_split_time_series_basic(self, client):
        """Test basic time series split."""
        # Setup expected splits - split_time_series returns list of tuples
        train_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=80),
            'value': range(80)
        })
        test_df = pd.DataFrame({
            'date': pd.date_range('2023-03-22', periods=20),
            'value': range(80, 100)
        })
        expected_splits = [(train_df, test_df)]
        
        client.ml.split_time_series.return_value = expected_splits
        
        result = client.split_time_series('test_dataset', test_size=0.2)
        
        client.ml.split_time_series.assert_called_once_with(
            'test_dataset', 
            5,  # default n_splits
            0.2,  # test_size
            0,  # default gap
            'expanding'  # default strategy
        )
        assert result == expected_splits
    
    def test_split_time_series_with_validation(self, client):
        """Test time series split with validation set."""
        # Setup expected splits - with validation, we still get list of tuples
        train_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=60),
            'group': ['A'] * 30 + ['B'] * 30,
            'value': range(60)
        })
        test_df = pd.DataFrame({
            'date': pd.date_range('2023-03-02', periods=40),
            'group': ['A'] * 20 + ['B'] * 20,
            'value': range(60, 100)
        })
        expected_splits = [(train_df, test_df)]
        
        client.ml.split_time_series.return_value = expected_splits
        
        result = client.split_time_series('test_dataset', n_splits=1, test_size=0.4, gap=5)
        
        client.ml.split_time_series.assert_called_once_with(
            'test_dataset',
            1,  # n_splits
            0.4,  # test_size
            5,  # gap
            'expanding'  # strategy
        )
        assert result == expected_splits
    
    def test_split_time_series_no_time_column(self, client):
        """Test time series split without time column."""
        # The ML client should raise an error when dataset has no time column
        client.ml.split_time_series.side_effect = ValueError("No time column specified")
        
        with pytest.raises(ValueError, match="No time column specified"):
            client.split_time_series('test_dataset', test_size=0.2)


class TestMDMClientQueryOperations:
    """Test MDMClient query operations."""
    
    @pytest.fixture
    def client(self):
        """Create client with mocked dependencies."""
        with patch('mdm.core.get_service') as mock_get_service:
            # Create mock specialized clients
            mock_registration = Mock()
            mock_query = Mock()
            mock_ml = Mock()
            mock_export = Mock()
            mock_management = Mock()
            
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
            client = MDMClient()
            return client
    
    def test_query_dataset(self, client):
        """Test querying dataset."""
        expected_df = pd.DataFrame({'count': [42]})
        client.query.query_dataset.return_value = expected_df
        
        result = client.query_dataset(
            'test_dataset',
            "SELECT COUNT(*) as count FROM train_table"
        )
        
        client.query.query_dataset.assert_called_once_with(
            'test_dataset',
            "SELECT COUNT(*) as count FROM train_table"
        )
        pd.testing.assert_frame_equal(result, expected_df)