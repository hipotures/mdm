"""Simple unit tests for MDM API to improve coverage."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path

from mdm.api import MDMClient
from mdm.core.exceptions import DatasetError
from mdm.models.dataset import DatasetInfo
from mdm.models.enums import ProblemType


class TestMDMClientBasic:
    """Test MDMClient basic functionality."""
    
    @patch('mdm.api.mdm_client.get_config')
    @patch('mdm.core.get_service')
    def test_init_default(self, mock_get_service, mock_get_config):
        """Test default initialization."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        # Mock the specialized clients
        mock_get_service.side_effect = lambda cls: Mock()
        
        client = MDMClient()
        
        assert client.config == mock_config
        assert client.registration is not None
        assert client.query is not None
        assert client.ml is not None
        assert client.export is not None
        assert client.management is not None
    
    def test_init_with_config(self):
        """Test initialization with config."""
        mock_config = Mock()
        with patch('mdm.core.get_service') as mock_get_service:
            mock_get_service.side_effect = lambda cls: Mock()
            client = MDMClient(config=mock_config)
            assert client.config == mock_config
    
    @patch('mdm.core.get_service')
    def test_dataset_exists(self, mock_get_service):
        """Test checking if dataset exists."""
        # Setup mock query client
        mock_query = Mock()
        mock_dataset = Mock(spec=DatasetInfo)
        mock_query.get_dataset.return_value = mock_dataset
        
        # Configure mock_get_service to return appropriate clients
        def get_service_side_effect(cls):
            if cls.__name__ == 'QueryClient':
                return mock_query
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        # MDMClient doesn't have dataset_exists method, we use get_dataset
        result = client.get_dataset('test_dataset')
        
        assert result == mock_dataset
        mock_query.get_dataset.assert_called_once_with('test_dataset')
    
    @patch('mdm.core.get_service')
    def test_get_dataset(self, mock_get_service):
        """Test getting dataset info."""
        # Setup mock query client
        mock_query = Mock()
        mock_info = Mock(spec=DatasetInfo)
        mock_query.get_dataset.return_value = mock_info
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'QueryClient':
                return mock_query
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        result = client.get_dataset('test_dataset')
        
        assert result == mock_info
        mock_query.get_dataset.assert_called_once_with('test_dataset')
    
    @patch('mdm.core.get_service')
    def test_list_datasets_no_filter(self, mock_get_service):
        """Test listing datasets without filter."""
        # Setup mock query client
        mock_query = Mock()
        mock_datasets = [Mock(), Mock()]
        mock_query.list_datasets.return_value = mock_datasets
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'QueryClient':
                return mock_query
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        result = client.list_datasets()
        
        assert result == mock_datasets
        mock_query.list_datasets.assert_called_once()
    
    @patch('mdm.core.get_service')
    def test_list_datasets_with_filter(self, mock_get_service):
        """Test listing datasets with filter."""
        # Setup mock query client
        mock_query = Mock()
        mock_dataset1 = Mock()
        mock_dataset1.name = 'test1'
        mock_dataset2 = Mock() 
        mock_dataset2.name = 'test2'
        mock_query.list_datasets.return_value = [mock_dataset1, mock_dataset2]
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'QueryClient':
                return mock_query
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        # MDMClient.list_datasets doesn't support filter_func, filtering must be done after
        result = client.list_datasets()
        filtered_result = [d for d in result if d.name == 'test1']
        
        assert len(filtered_result) == 1
        assert filtered_result[0] == mock_dataset1
    
    @patch('mdm.core.get_service')
    def test_remove_dataset(self, mock_get_service):
        """Test removing dataset."""
        # Setup mock management client
        mock_management = Mock()
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'ManagementClient':
                return mock_management
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        client.remove_dataset('test_dataset', force=True)
        
        mock_management.remove_dataset.assert_called_once_with('test_dataset', True)


class TestMDMClientRegistration:
    """Test dataset registration."""
    
    @patch('mdm.core.get_service')
    def test_register_dataset_path_not_exists(self, mock_get_service):
        """Test registering dataset with non-existent path."""
        # Setup mock registration client
        mock_registration = Mock()
        mock_registration.register_dataset.side_effect = DatasetError("Path does not exist")
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'RegistrationClient':
                return mock_registration
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        
        with pytest.raises(DatasetError, match="Path does not exist"):
            client.register_dataset('test', '/nonexistent/path')
    
    @patch('mdm.core.get_service')
    def test_register_dataset_success(self, mock_get_service):
        """Test successful dataset registration."""
        # Setup mock registration client
        mock_registration = Mock()
        mock_info = Mock(spec=DatasetInfo)
        mock_registration.register_dataset.return_value = mock_info
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'RegistrationClient':
                return mock_registration
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        result = client.register_dataset(
            name='test_dataset',
            dataset_path='/data/test',
            target_column='target',
            description='Test dataset'
        )
        
        assert result == mock_info
        mock_registration.register_dataset.assert_called_once()


class TestMDMClientDataOperations:
    """Test data loading and query operations."""
    
    @patch('mdm.core.get_service')
    def test_load_dataset_train(self, mock_get_service):
        """Test loading train dataset."""
        # Setup mock query client
        mock_query = Mock()
        expected_df = pd.DataFrame({'col1': [1, 2, 3]})
        # load_dataset_files returns a dictionary, not a tuple
        mock_query.load_dataset_files.return_value = {'train': expected_df}
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'QueryClient':
                return mock_query
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        result = client.load_dataset_files('test_dataset')
        
        assert 'train' in result
        pd.testing.assert_frame_equal(result['train'], expected_df)
        assert 'test' not in result
    
    @patch('mdm.core.get_service')
    def test_load_dataset_specific_table(self, mock_get_service):
        """Test loading specific table."""
        # Setup mock query client
        mock_query = Mock()
        expected_df = pd.DataFrame({'col1': [4, 5, 6]})
        # MDMClient doesn't have load_table, it has load_dataset with table parameter
        mock_query.load_dataset.return_value = expected_df
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'QueryClient':
                return mock_query
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        result = client.load_dataset('test_dataset', table='test')
        
        pd.testing.assert_frame_equal(result, expected_df)
        mock_query.load_dataset.assert_called_once_with('test_dataset', table='test')
    
    @patch('mdm.core.get_service')
    def test_load_dataset_not_found(self, mock_get_service):
        """Test loading non-existent dataset."""
        # Setup mock query client
        mock_query = Mock()
        mock_query.load_dataset_files.side_effect = DatasetError("Dataset 'nonexistent' not found")
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'QueryClient':
                return mock_query
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        
        with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
            client.load_dataset_files('nonexistent')
    
    @patch('mdm.core.get_service')
    def test_query(self, mock_get_service):
        """Test executing query."""
        # Setup mock query client
        mock_query = Mock()
        expected_df = pd.DataFrame({'count': [42]})
        mock_query.query_dataset.return_value = expected_df
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'QueryClient':
                return mock_query
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        result = client.query_dataset('test_dataset', "SELECT COUNT(*) FROM train")
        
        pd.testing.assert_frame_equal(result, expected_df)
        mock_query.query_dataset.assert_called_once_with('test_dataset', "SELECT COUNT(*) FROM train")


class TestMDMClientUtils:
    """Test utility methods."""
    
    @patch('mdm.core.get_service')
    def test_get_stats(self, mock_get_service):
        """Test getting dataset statistics."""
        # Setup mock management client
        mock_management = Mock()
        expected_stats = {
            'dataset_name': 'test_dataset', 
            'tables': {'train': {'row_count': 1000}},
            'summary': {'total_rows': 1000, 'total_columns': 10}
        }
        mock_management.get_statistics.return_value = expected_stats
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'ManagementClient':
                return mock_management
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        result = client.get_statistics('test_dataset')
        
        assert result == expected_stats
        mock_management.get_statistics.assert_called_once_with('test_dataset', False)
    
    @patch('mdm.core.get_service')
    def test_export_dataset(self, mock_get_service):
        """Test exporting dataset."""
        # Setup mock export client
        mock_export = Mock()
        mock_export.export_dataset.return_value = ['/tmp/export/train.parquet']
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'ExportClient':
                return mock_export
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        result = client.export_dataset('test_dataset', '/tmp/export', format='parquet')
        
        assert result == ['/tmp/export/train.parquet']
        mock_export.export_dataset.assert_called_once_with('test_dataset', '/tmp/export', format='parquet')
    
    @patch('mdm.core.get_service')
    def test_update_metadata(self, mock_get_service):
        """Test updating dataset metadata."""
        # Setup mock management client
        mock_management = Mock()
        mock_management.update_dataset.return_value = Mock()
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'ManagementClient':
                return mock_management
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        client.update_dataset(
            'test_dataset',
            description='Updated description',
            tags=['new', 'tags']
        )
        
        mock_management.update_dataset.assert_called_once_with(
            'test_dataset',
            description='Updated description',
            tags=['new', 'tags']
        )


class TestMDMClientIntegration:
    """Test integration utilities."""
    
    @patch('mdm.core.get_service')
    def test_to_sklearn(self, mock_get_service):
        """Test converting to sklearn format."""
        # Setup mock ML client
        mock_ml = Mock()
        expected_result = {'X_train': Mock(), 'y_train': Mock()}
        mock_ml.prepare_for_ml.return_value = expected_result
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'MLIntegrationClient':
                return mock_ml
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        
        # Call prepare_for_ml
        result = client.prepare_for_ml('test_dataset', framework='sklearn')
        
        assert result == expected_result
        mock_ml.prepare_for_ml.assert_called_once_with('test_dataset', 'sklearn')
    
    @patch('mdm.core.get_service')
    def test_to_tensorflow(self, mock_get_service):
        """Test converting to TensorFlow format."""
        # Setup mock ML client
        mock_ml = Mock()
        expected_result = {'train_ds': Mock(), 'test_ds': Mock()}
        mock_ml.prepare_for_ml.return_value = expected_result
        
        # Configure mock_get_service
        def get_service_side_effect(cls):
            if cls.__name__ == 'MLIntegrationClient':
                return mock_ml
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        client = MDMClient()
        
        # Call prepare_for_ml
        result = client.prepare_for_ml('test_dataset', framework='tensorflow')
        
        assert result == expected_result
        mock_ml.prepare_for_ml.assert_called_once_with('test_dataset', 'tensorflow')
