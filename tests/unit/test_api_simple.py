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
    
    @patch('mdm.api.get_config')
    @patch('mdm.api.DatasetManager')
    def test_init_default(self, mock_manager_class, mock_get_config):
        """Test default initialization."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        client = MDMClient()
        
        assert client.config == mock_config
        assert client.manager == mock_manager_class.return_value
    
    def test_init_with_config(self):
        """Test initialization with config."""
        mock_config = Mock()
        with patch('mdm.api.DatasetManager') as mock_manager:
            client = MDMClient(config=mock_config)
            assert client.config == mock_config
    
    @patch('mdm.api.DatasetManager')
    def test_dataset_exists(self, mock_manager_class):
        """Test checking if dataset exists."""
        mock_manager = mock_manager_class.return_value
        mock_manager.dataset_exists.return_value = True
        
        client = MDMClient()
        result = client.dataset_exists('test_dataset')
        
        assert result is True
        mock_manager.dataset_exists.assert_called_once_with('test_dataset')
    
    @patch('mdm.api.DatasetManager')
    def test_get_dataset(self, mock_manager_class):
        """Test getting dataset info."""
        mock_manager = mock_manager_class.return_value
        mock_info = Mock(spec=DatasetInfo)
        mock_manager.get_dataset.return_value = mock_info
        
        client = MDMClient()
        result = client.get_dataset('test_dataset')
        
        assert result == mock_info
        mock_manager.get_dataset.assert_called_once_with('test_dataset')
    
    @patch('mdm.api.DatasetManager')
    def test_list_datasets_no_filter(self, mock_manager_class):
        """Test listing datasets without filter."""
        mock_manager = mock_manager_class.return_value
        mock_datasets = [Mock(), Mock()]
        mock_manager.list_datasets.return_value = mock_datasets
        
        client = MDMClient()
        result = client.list_datasets()
        
        assert result == mock_datasets
        mock_manager.list_datasets.assert_called_once()
    
    @patch('mdm.api.DatasetManager')
    def test_list_datasets_with_filter(self, mock_manager_class):
        """Test listing datasets with filter."""
        mock_manager = mock_manager_class.return_value
        mock_dataset1 = Mock()
        mock_dataset1.name = 'test1'
        mock_dataset2 = Mock() 
        mock_dataset2.name = 'test2'
        mock_manager.list_datasets.return_value = [mock_dataset1, mock_dataset2]
        
        client = MDMClient()
        # Filter to only get datasets with 'test1' in name
        result = client.list_datasets(filter_func=lambda d: d.name == 'test1')
        
        assert len(result) == 1
        assert result[0] == mock_dataset1
    
    @patch('mdm.api.DatasetManager')
    def test_remove_dataset(self, mock_manager_class):
        """Test removing dataset."""
        mock_manager = mock_manager_class.return_value
        
        client = MDMClient()
        client.remove_dataset('test_dataset', force=True)
        
        mock_manager.remove_dataset.assert_called_once_with('test_dataset', force=True)


class TestMDMClientRegistration:
    """Test dataset registration."""
    
    @patch('mdm.api.DatasetManager')
    @patch('mdm.api.Path')
    def test_register_dataset_path_not_exists(self, mock_path_class, mock_manager_class):
        """Test registering dataset with non-existent path."""
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_path_class.return_value = mock_path
        
        client = MDMClient()
        
        with pytest.raises(ValueError, match="Path does not exist"):
            client.register_dataset('test', '/nonexistent/path')
    
    @patch('mdm.api.DatasetRegistrar')
    @patch('mdm.api.DatasetManager')
    @patch('mdm.api.Path')
    def test_register_dataset_success(self, mock_path_class, mock_manager_class, mock_registrar_class):
        """Test successful dataset registration."""
        # Setup path
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path_class.return_value = mock_path
        
        # Setup registrar
        mock_registrar = Mock()
        mock_info = Mock(spec=DatasetInfo)
        mock_registrar.register.return_value = mock_info
        mock_registrar_class.return_value = mock_registrar
        
        client = MDMClient()
        result = client.register_dataset(
            name='test_dataset',
            dataset_path='/data/test',
            target_column='target',
            description='Test dataset'
        )
        
        assert result == mock_info
        mock_registrar.register.assert_called_once()


class TestMDMClientDataOperations:
    """Test data loading and query operations."""
    
    @patch('mdm.api.BackendFactory')
    @patch('mdm.api.DatasetManager')
    def test_load_dataset_train(self, mock_manager_class, mock_factory_class):
        """Test loading train dataset."""
        # Setup manager
        mock_manager = mock_manager_class.return_value
        mock_info = Mock()
        mock_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_info.tables = {'train': 'train_table'}
        mock_manager.get_dataset.return_value = mock_info
        
        # Setup backend
        mock_backend = Mock()
        mock_backend.get_engine.return_value = Mock()
        expected_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_backend.read_table.return_value = expected_df
        mock_factory_class.create.return_value = mock_backend
        
        client = MDMClient()
        result = client.load_dataset('test_dataset')
        
        pd.testing.assert_frame_equal(result, expected_df)
        mock_backend.read_table.assert_called_once_with('train_table')
    
    @patch('mdm.api.BackendFactory')
    @patch('mdm.api.DatasetManager')
    def test_load_dataset_specific_table(self, mock_manager_class, mock_factory_class):
        """Test loading specific table."""
        mock_manager = mock_manager_class.return_value
        mock_info = Mock()
        mock_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_info.tables = {'train': 'train_table', 'test': 'test_table'}
        mock_manager.get_dataset.return_value = mock_info
        
        mock_backend = Mock()
        mock_backend.get_engine.return_value = Mock()
        expected_df = pd.DataFrame({'col1': [4, 5, 6]})
        mock_backend.read_table.return_value = expected_df
        mock_factory_class.create.return_value = mock_backend
        
        client = MDMClient()
        result = client.load_dataset('test_dataset', table='test')
        
        pd.testing.assert_frame_equal(result, expected_df)
        mock_backend.read_table.assert_called_once_with('test_table')
    
    @patch('mdm.api.DatasetManager')
    def test_load_dataset_not_found(self, mock_manager_class):
        """Test loading non-existent dataset."""
        mock_manager = mock_manager_class.return_value
        mock_manager.get_dataset.return_value = None
        
        client = MDMClient()
        
        with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
            client.load_dataset('nonexistent')
    
    @patch('mdm.api.BackendFactory')
    @patch('mdm.api.DatasetManager')
    def test_query(self, mock_manager_class, mock_factory_class):
        """Test executing query."""
        mock_manager = mock_manager_class.return_value
        mock_info = Mock()
        mock_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_manager.get_dataset.return_value = mock_info
        
        mock_backend = Mock()
        mock_backend.get_engine.return_value = Mock()
        expected_df = pd.DataFrame({'count': [42]})
        mock_backend.query.return_value = expected_df
        mock_factory_class.create.return_value = mock_backend
        
        client = MDMClient()
        result = client.query('test_dataset', "SELECT COUNT(*) FROM train")
        
        pd.testing.assert_frame_equal(result, expected_df)
        mock_backend.query.assert_called_once_with("SELECT COUNT(*) FROM train")


class TestMDMClientUtils:
    """Test utility methods."""
    
    @patch('mdm.api.DatasetManager')
    def test_get_stats(self, mock_manager_class):
        """Test getting dataset statistics."""
        mock_manager = mock_manager_class.return_value
        expected_stats = {'rows': 1000, 'columns': 10}
        mock_manager.get_dataset_stats.return_value = expected_stats
        
        client = MDMClient()
        result = client.get_stats('test_dataset')
        
        assert result == expected_stats
        mock_manager.get_dataset_stats.assert_called_once_with('test_dataset')
    
    @patch('mdm.api.DatasetManager')
    def test_export_dataset(self, mock_manager_class):
        """Test exporting dataset."""
        mock_manager = mock_manager_class.return_value
        
        client = MDMClient()
        client.export_dataset('test_dataset', '/tmp/export', format='parquet')
        
        mock_manager.export_dataset.assert_called_once_with(
            'test_dataset',
            Path('/tmp/export'),
            format='parquet'
        )
    
    @patch('mdm.api.DatasetManager')
    def test_update_metadata(self, mock_manager_class):
        """Test updating dataset metadata."""
        mock_manager = mock_manager_class.return_value
        
        client = MDMClient()
        client.update_metadata(
            'test_dataset',
            description='Updated description',
            tags=['new', 'tags']
        )
        
        mock_manager.update_dataset_metadata.assert_called_once_with(
            'test_dataset',
            description='Updated description',
            tags=['new', 'tags']
        )


class TestMDMClientIntegration:
    """Test integration utilities."""
    
    @patch('mdm.api.MLFrameworkAdapter')
    @patch('mdm.api.DatasetManager')
    def test_to_sklearn(self, mock_manager_class, mock_adapter_class):
        """Test converting to sklearn format."""
        mock_adapter = mock_adapter_class.return_value
        expected_result = (Mock(), Mock())  # X, y
        mock_adapter.prepare_for_sklearn.return_value = expected_result
        
        client = MDMClient()
        with patch.object(client, 'load_dataset') as mock_load:
            mock_df = pd.DataFrame({'feature': [1, 2], 'target': [0, 1]})
            mock_load.return_value = mock_df
            
            with patch.object(client, 'get_dataset') as mock_get:
                mock_info = Mock()
                mock_info.target_column = 'target'
                mock_get.return_value = mock_info
                
                result = client.to_sklearn('test_dataset')
        
        assert result == expected_result
        mock_adapter.prepare_for_sklearn.assert_called_once()
    
    @patch('mdm.api.MLFrameworkAdapter')
    @patch('mdm.api.DatasetManager')
    def test_to_tensorflow(self, mock_manager_class, mock_adapter_class):
        """Test converting to TensorFlow format."""
        mock_adapter = mock_adapter_class.return_value
        expected_ds = Mock()  # tf.data.Dataset
        mock_adapter.prepare_for_tensorflow.return_value = expected_ds
        
        client = MDMClient()
        with patch.object(client, 'load_dataset') as mock_load:
            mock_df = pd.DataFrame({'feature': [1, 2], 'target': [0, 1]})
            mock_load.return_value = mock_df
            
            with patch.object(client, 'get_dataset') as mock_get:
                mock_info = Mock()
                mock_info.target_column = 'target'
                mock_get.return_value = mock_info
                
                result = client.to_tensorflow('test_dataset', batch_size=32)
        
        assert result == expected_ds
        mock_adapter.prepare_for_tensorflow.assert_called_once()
    
    @patch('mdm.api.SubmissionCreator')
    @patch('mdm.api.DatasetManager')
    def test_create_submission(self, mock_manager_class, mock_creator_class):
        """Test creating submission file."""
        mock_creator = mock_creator_class.return_value
        
        client = MDMClient()
        predictions = pd.Series([0, 1, 0, 1])
        
        with patch.object(client, 'get_dataset') as mock_get:
            mock_info = Mock()
            mock_info.problem_type = 'binary_classification'
            mock_get.return_value = mock_info
            
            client.create_submission('test_dataset', predictions, '/tmp/submission.csv')
        
        mock_creator.create_submission.assert_called_once()