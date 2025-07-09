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
    
    @patch('mdm.api.DatasetRegistrar')
    @patch('mdm.api.DatasetManager')
    def test_register_dataset_path_not_exists(self, mock_manager_class, mock_registrar_class):
        """Test registering dataset with non-existent path."""
        mock_manager = mock_manager_class.return_value
        mock_manager.dataset_exists.return_value = False
        mock_manager.validate_dataset_name.side_effect = lambda x: x.lower()
        
        mock_registrar = mock_registrar_class.return_value
        mock_registrar.register.side_effect = DatasetError("Path does not exist")
        
        client = MDMClient()
        
        with pytest.raises(DatasetError, match="Path does not exist"):
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
        client.get_dataset = Mock(return_value=mock_info)
        client.manager.get_backend.return_value = mock_backend
        mock_backend.read_table_to_dataframe.return_value = expected_df
        
        train_df, test_df = client.load_dataset_files('test_dataset')
        
        pd.testing.assert_frame_equal(train_df, expected_df)
        assert test_df is None
    
    @patch('mdm.api.DatasetManager')
    def test_load_dataset_specific_table(self, mock_manager_class):
        """Test loading specific table."""
        client = MDMClient()
        
        # Setup mocks
        mock_info = Mock()
        mock_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_info.tables = {'train': 'train_table', 'test': 'test_table'}
        client.get_dataset = Mock(return_value=mock_info)
        
        mock_backend = Mock()
        mock_engine = Mock()
        mock_backend.get_engine.return_value = mock_engine
        expected_df = pd.DataFrame({'col1': [4, 5, 6]})
        mock_backend.read_table_to_dataframe.return_value = expected_df
        client.manager.get_backend.return_value = mock_backend
        
        # load_dataset_files doesn't take table parameter, use load_table instead
        result = client.load_table('test_dataset', 'test')
        
        pd.testing.assert_frame_equal(result, expected_df)
        
        # Verify
        client.get_dataset.assert_called_once_with('test_dataset')
        client.manager.get_backend.assert_called_once_with('test_dataset')
        mock_backend.get_engine.assert_called_once_with('/tmp/test.db')
        mock_backend.read_table_to_dataframe.assert_called_once_with('test_table', mock_engine)
    
    @patch('mdm.api.DatasetManager')
    def test_load_dataset_not_found(self, mock_manager_class):
        """Test loading non-existent dataset."""
        mock_manager = mock_manager_class.return_value
        mock_manager.get_dataset.return_value = None
        
        client = MDMClient()
        
        client.get_dataset = Mock(return_value=None)
        
        with pytest.raises(ValueError, match="Dataset 'nonexistent' not found"):
            client.load_dataset_files('nonexistent')
    
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
        mock_backend.execute_query.return_value = expected_df
        client.manager.get_backend.return_value = mock_backend
        
        result = client.query_dataset('test_dataset', "SELECT COUNT(*) FROM train")
        
        pd.testing.assert_frame_equal(result, expected_df)
        mock_backend.execute_query.assert_called_once_with("SELECT COUNT(*) FROM train")


class TestMDMClientUtils:
    """Test utility methods."""
    
    @patch('mdm.api.DatasetManager')
    def test_get_stats(self, mock_manager_class):
        """Test getting dataset statistics."""
        mock_manager = mock_manager_class.return_value
        expected_stats = {'rows': 1000, 'columns': 10}
        mock_manager.get_dataset_stats.return_value = expected_stats
        
        client = MDMClient()
        # get_stats doesn't exist, use get_statistics
        with patch('mdm.dataset.operations.StatsOperation') as mock_stats_op:
            mock_stats = mock_stats_op.return_value
            mock_stats.execute.return_value = expected_stats
            
            result = client.get_statistics('test_dataset')
            
            assert result == expected_stats
            mock_stats.execute.assert_called_once_with('test_dataset', full=False)
    
    @patch('mdm.api.DatasetManager')
    def test_export_dataset(self, mock_manager_class):
        """Test exporting dataset."""
        mock_manager = mock_manager_class.return_value
        
        client = MDMClient()
        with patch('mdm.dataset.operations.ExportOperation') as mock_export_op:
            mock_export = mock_export_op.return_value
            mock_export.execute.return_value = [Path('/tmp/export/train.parquet')]
            
            result = client.export_dataset('test_dataset', '/tmp/export', format='parquet')
            
            mock_export.execute.assert_called_once()
            assert result == ['/tmp/export/train.parquet']
    
    @patch('mdm.api.DatasetManager')
    def test_update_metadata(self, mock_manager_class):
        """Test updating dataset metadata."""
        mock_manager = mock_manager_class.return_value
        
        client = MDMClient()
        # update_metadata doesn't exist, use update_dataset
        mock_manager.update_dataset.return_value = Mock()
        
        client.update_dataset(
            'test_dataset',
            description='Updated description',
            tags=['new', 'tags']
        )
        
        mock_manager.update_dataset.assert_called_once_with(
            'test_dataset',
            {
                'description': 'Updated description',
                'tags': ['new', 'tags']
            }
        )


class TestMDMClientIntegration:
    """Test integration utilities."""
    
    @patch('mdm.api.MLFrameworkAdapter')
    @patch('mdm.api.DatasetManager')
    def test_to_sklearn(self, mock_manager_class, mock_adapter_class):
        """Test converting to sklearn format."""
        mock_adapter = mock_adapter_class.return_value
        expected_result = {'X_train': Mock(), 'y_train': Mock()}
        mock_adapter.prepare_data.return_value = expected_result
        
        client = MDMClient()
        
        # Setup dataset info
        mock_info = Mock()
        mock_info.target_column = 'target'
        mock_info.id_columns = ['id']
        client.get_dataset = Mock(return_value=mock_info)
        
        # Setup load_dataset_files
        train_df = pd.DataFrame({'feature': [1, 2], 'target': [0, 1]})
        test_df = pd.DataFrame({'feature': [3, 4], 'target': [1, 0]})
        client.load_dataset_files = Mock(return_value=(train_df, test_df))
        
        # Call prepare_for_ml
        result = client.prepare_for_ml('test_dataset', framework='sklearn', sample_size=100)
        
        assert result == expected_result
        client.get_dataset.assert_called_once_with('test_dataset')
        client.load_dataset_files.assert_called_once_with('test_dataset', 100)
        mock_adapter_class.assert_called_once_with('sklearn')
        mock_adapter.prepare_data.assert_called_once_with(
            train_df, test_df, 'target', ['id']
        )
    
    @patch('mdm.api.MLFrameworkAdapter')
    @patch('mdm.api.DatasetManager')
    def test_to_tensorflow(self, mock_manager_class, mock_adapter_class):
        """Test converting to TensorFlow format."""
        mock_adapter = mock_adapter_class.return_value
        expected_result = {'train_ds': Mock(), 'test_ds': Mock()}
        mock_adapter.prepare_data.return_value = expected_result
        
        client = MDMClient()
        
        # Setup dataset info
        mock_info = Mock()
        mock_info.target_column = 'target'
        mock_info.id_columns = None
        client.get_dataset = Mock(return_value=mock_info)
        
        # Setup load_dataset_files
        train_df = pd.DataFrame({'feature': [1, 2], 'target': [0, 1]})
        test_df = pd.DataFrame({'feature': [3, 4], 'target': [1, 0]})
        client.load_dataset_files = Mock(return_value=(train_df, test_df))
        
        # Call prepare_for_ml
        result = client.prepare_for_ml('test_dataset', framework='tensorflow')
        
        assert result == expected_result
        mock_adapter_class.assert_called_once_with('tensorflow')
        mock_adapter.prepare_data.assert_called_once_with(
            train_df, test_df, 'target', None
        )
    
    @patch('mdm.api.SubmissionCreator')
    @patch('mdm.api.DatasetManager')
    def test_create_submission(self, mock_manager_class, mock_creator_class):
        """Test creating submission file."""
        client = MDMClient()
        predictions = pd.Series([0, 1, 0, 1])
        
        # Setup mock SubmissionCreator
        mock_creator = mock_creator_class.return_value
        mock_creator.create_submission.return_value = '/tmp/submission.csv'
        
        result = client.create_submission('test_dataset', predictions, '/tmp/submission.csv')
        
        assert result == '/tmp/submission.csv'
        mock_creator_class.assert_called_once_with(client.manager)
        mock_creator.create_submission.assert_called_once_with(
            'test_dataset', predictions, '/tmp/submission.csv'
        )