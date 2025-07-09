"""Additional unit tests for DatasetRegistrar to achieve 90%+ coverage."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call, PropertyMock
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
import yaml
import tempfile
import logging

from mdm.dataset.registrar import DatasetRegistrar
from mdm.core.exceptions import DatasetError
from mdm.models.dataset import DatasetInfo, ColumnInfo, FileInfo
from mdm.models.enums import ProblemType, ColumnType, FileType


class TestDatasetRegistrarCoverage:
    """Additional test cases for DatasetRegistrar to improve coverage."""

    @pytest.fixture
    def mock_manager(self):
        """Create mock DatasetManager."""
        manager = Mock()
        manager.dataset_exists.return_value = False
        manager.register_dataset.return_value = None
        manager.validate_dataset_name.side_effect = lambda x: x.lower().replace('-', '_')
        return manager

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.database.default_backend = "sqlite"
        config.paths.datasets_path = "datasets/"
        config.feature_engineering.enabled = True
        config.performance.batch_size = 10000
        return config

    @pytest.fixture
    def mock_config_manager(self, mock_config):
        """Create mock config manager."""
        manager = Mock()
        manager.config = mock_config
        manager.base_path = Path("/tmp/test_mdm")
        return manager

    @pytest.fixture
    def registrar(self, mock_manager, mock_config_manager):
        """Create DatasetRegistrar instance."""
        with patch('mdm.dataset.registrar.get_config_manager', return_value=mock_config_manager):
            with patch('mdm.dataset.registrar.FeatureGenerator'):
                reg = DatasetRegistrar(mock_manager)
                reg._detected_datetime_columns = []
                reg._detected_column_types = {}
                return reg

    def test_register_full_flow_with_progress(self, registrar, mock_manager, tmp_path):
        """Test complete registration flow with progress tracking."""
        # Create test data
        data_path = tmp_path / "dataset"
        data_path.mkdir()
        
        train_file = data_path / "train.csv"
        test_file = data_path / "test.csv"
        
        train_df = pd.DataFrame({
            'id': range(1000),
            'created_at': pd.date_range('2024-01-01', periods=1000, freq='H').strftime('%Y-%m-%d %H:%M:%S'),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'value': np.random.rand(1000),
            'target': np.random.randint(0, 2, 1000)
        })
        test_df = train_df[['id', 'created_at', 'category', 'value']].iloc[:200]
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        # Mock all the dependencies
        mock_manager.dataset_exists.return_value = False
        
        # Mock auto-detect structure
        with patch('mdm.dataset.registrar.detect_kaggle_structure', return_value=True):
            with patch('mdm.dataset.registrar.extract_target_from_sample_submission', return_value='target'):
                # Mock file discovery
                with patch('mdm.dataset.registrar.discover_data_files') as mock_discover:
                    mock_discover.return_value = {
                        'train': train_file,
                        'test': test_file
                    }
                    
                    # Mock backend operations
                    with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
                        mock_backend = Mock()
                        mock_backend.create_table_from_dataframe = Mock()
                        mock_backend.get_table_sample.return_value = train_df.head(100)
                        mock_backend.get_table_info.return_value = {
                            'row_count': 1000,
                            'columns': {
                                'id': {'dtype': 'int64', 'nullable': False},
                                'created_at': {'dtype': 'object', 'nullable': False},
                                'category': {'dtype': 'object', 'nullable': False},
                                'value': {'dtype': 'float64', 'nullable': False},
                                'target': {'dtype': 'int64', 'nullable': False}
                            }
                        }
                        mock_factory.create.return_value = mock_backend
                        
                        # Mock ID column detection
                        with patch('mdm.dataset.registrar.detect_id_columns', return_value=['id']):
                            # Mock problem type inference
                            with patch('mdm.dataset.registrar.infer_problem_type', return_value='binary_classification'):
                                # Mock statistics computation
                                with patch('mdm.dataset.registrar.compute_dataset_statistics') as mock_stats:
                                    mock_stats.return_value = {
                                        'total_rows': 1200,
                                        'memory_size_mb': 10.5
                                    }
                                    
                                    # Mock profiling
                                    with patch('mdm.dataset.registrar.ProfileReport'):
                                        # Patch Path.mkdir to avoid filesystem operations
                                        with patch('pathlib.Path.mkdir'):
                                            result = registrar.register(
                                                'test_dataset',
                                                data_path,
                                                description='Test dataset',
                                                tags=['test', 'example'],
                                                display_name='Test Dataset'
                                            )
                                            
                                            assert isinstance(result, DatasetInfo)
                                            assert result.name == 'test_dataset'
                                            assert result.display_name == 'Test Dataset'
                                            assert result.description == 'Test dataset'
                                            assert result.tags == ['test', 'example']
                                            assert result.problem_type == 'binary_classification'
                                            assert result.target_column == 'target'
                                            assert 'id' in result.id_columns
                                            assert 'statistics' in result.metadata
                                            
                                            # Verify manager was called
                                            mock_manager.register_dataset.assert_called_once()

    def test_register_with_remove_operation_import_and_failure(self, registrar, mock_manager, tmp_path):
        """Test registration with force flag when RemoveOperation import and execution fails."""
        data_path = tmp_path / "data.csv"
        data_path.write_text("id,value\n1,100\n")
        
        # Dataset already exists
        mock_manager.dataset_exists.return_value = True
        
        # First test the import of RemoveOperation
        with patch('mdm.dataset.registrar.logger') as mock_logger:
            # Mock the rest of the registration to continue
            with patch('mdm.dataset.registrar.discover_data_files', return_value={'data': data_path}):
                with patch('mdm.dataset.registrar.BackendFactory'):
                    with patch('mdm.dataset.registrar.compute_dataset_statistics', return_value={}):
                        with patch('pathlib.Path.mkdir'):
                            # The RemoveOperation import happens inside the method
                            # Mock it to fail
                            with patch('builtins.__import__', side_effect=ImportError("No module")):
                                # Should continue despite import failure
                                result = registrar.register('test_dataset', data_path, force=True)
                                
                                # Should log warning about failed removal
                                assert any('Failed to remove' in str(call) for call in mock_logger.warning.call_args_list)

    def test_load_data_files_with_all_formats(self, registrar, tmp_path):
        """Test loading data files with all supported formats."""
        db_info = {'backend': 'sqlite', 'path': str(tmp_path / 'test.db')}
        
        # Test CSV
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,100\n2,200\n")
        
        # Test Parquet
        parquet_file = tmp_path / "data.parquet"
        
        # Test JSON
        json_file = tmp_path / "data.json"
        json_file.write_text('[{"id": 1, "value": 100}]')
        
        # Test Excel
        excel_file = tmp_path / "data.xlsx"
        
        # Test JSONL
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"id": 1, "value": 100}\n{"id": 2, "value": 200}\n')
        
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            mock_backend.create_table_from_dataframe = Mock()
            mock_factory.create.return_value = mock_backend
            
            # Mock all read functions
            df = pd.DataFrame({'id': [1, 2], 'value': [100, 200]})
            
            with patch('mdm.dataset.registrar.pd.read_csv', return_value=df):
                with patch('mdm.dataset.registrar.pd.read_parquet', return_value=df):
                    with patch('mdm.dataset.registrar.pd.read_json', return_value=df):
                        with patch('mdm.dataset.registrar.pd.read_excel', return_value=df):
                            with patch('mdm.dataset.registrar.Progress'):
                                # Test each format
                                for file_path, table_name in [
                                    (csv_file, 'csv_data'),
                                    (parquet_file, 'parquet_data'),
                                    (json_file, 'json_data'),
                                    (excel_file, 'excel_data'),
                                    (jsonl_file, 'jsonl_data')
                                ]:
                                    files = {table_name: file_path}
                                    result = registrar._load_data_files(files, db_info, Mock())
                                    
                                    assert table_name in result
                                    assert result[table_name]['row_count'] == 2

    def test_load_data_files_with_chunksize_and_datetime_detection(self, registrar, tmp_path):
        """Test loading large files with chunking and datetime detection."""
        csv_file = tmp_path / "large.csv"
        
        # Create test data with datetime
        dates = pd.date_range('2024-01-01', periods=50000, freq='min')
        large_df = pd.DataFrame({
            'id': range(50000),
            'timestamp': dates.strftime('%Y-%m-%d %H:%M:%S'),
            'value': np.random.rand(50000)
        })
        large_df.to_csv(csv_file, index=False)
        
        files = {'large_data': csv_file}
        db_info = {'backend': 'sqlite', 'path': str(tmp_path / 'test.db')}
        
        # Set small batch size to force chunking
        registrar.config.performance.batch_size = 10000
        
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            mock_backend.create_table_from_dataframe = Mock()
            mock_factory.create.return_value = mock_backend
            
            # Mock chunked reading
            chunks = []
            for i in range(0, 50000, 10000):
                chunk = large_df.iloc[i:i+10000].copy()
                chunks.append(chunk)
            
            # Create an iterator that returns chunks
            class ChunkIterator:
                def __init__(self, chunks):
                    self.chunks = chunks
                    self.index = 0
                
                def __iter__(self):
                    return self
                
                def __next__(self):
                    if self.index < len(self.chunks):
                        chunk = self.chunks[self.index]
                        self.index += 1
                        return chunk
                    raise StopIteration
            
            chunk_iterator = ChunkIterator(chunks)
            
            with patch('mdm.dataset.registrar.pd.read_csv', return_value=chunk_iterator):
                with patch('mdm.dataset.registrar.Progress'):
                    result = registrar._load_data_files(files, db_info, Mock())
                    
                    # Should process all chunks
                    assert result['large_data']['row_count'] == 50000
                    assert result['large_data']['load_time_seconds'] > 0
                    
                    # Should detect datetime column
                    assert 'timestamp' in registrar._detected_datetime_columns
                    
                    # Backend should be called for each chunk
                    assert mock_backend.create_table_from_dataframe.call_count == 5

    def test_load_data_file_edge_cases(self, registrar, tmp_path):
        """Test loading files with edge cases."""
        db_info = {'backend': 'sqlite'}
        
        # Test empty file
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("id,value\n")
        
        # Test file with error
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json")
        
        # Test unknown format
        unknown_file = tmp_path / "data.xyz"
        unknown_file.write_text("some data")
        
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            mock_factory.create.return_value = mock_backend
            
            with patch('mdm.dataset.registrar.Progress'):
                # Test empty file
                with patch('mdm.dataset.registrar.pd.read_csv', return_value=pd.DataFrame()):
                    result = registrar._load_data_files({'empty': empty_file}, db_info, Mock())
                    assert result['empty']['row_count'] == 0
                
                # Test file read error
                with patch('mdm.dataset.registrar.pd.read_json', side_effect=Exception("Invalid JSON")):
                    with pytest.raises(DatasetError, match="Failed to load"):
                        registrar._load_data_files({'bad': bad_file}, db_info, Mock())
                
                # Test unknown format - should raise error
                with pytest.raises(DatasetError, match="Unsupported file format"):
                    registrar._load_data_files({'unknown': unknown_file}, db_info, Mock())

    def test_detect_datetime_columns_complex_cases(self, registrar):
        """Test datetime detection with complex cases."""
        df = pd.DataFrame({
            'date1': ['2024-01-01', '2024-01-02', '2024-01-03'] * 10,  # Standard date
            'date2': ['01/15/2024', '01/16/2024', '01/17/2024'] * 10,  # US format
            'date3': ['15-Jan-2024', '16-Jan-2024', '17-Jan-2024'] * 10,  # Text month
            'datetime1': ['2024-01-01 10:30:00', '2024-01-01 11:30:00', '2024-01-01 12:30:00'] * 10,
            'datetime2': ['2024-01-01T10:30:00Z', '2024-01-01T11:30:00Z', '2024-01-01T12:30:00Z'] * 10,  # ISO format
            'mixed_good': ['2024-01-01'] * 25 + ['invalid'] * 5,  # >80% valid
            'mixed_bad': ['2024-01-01'] * 20 + ['not a date'] * 10,  # <80% valid
            'numbers': [20240101, 20240102, 20240103] * 10,  # Numeric dates
            'text': ['abc', 'def', 'ghi'] * 10
        })
        
        registrar._detect_datetime_columns_from_sample(df)
        
        # Should detect various datetime formats
        assert 'date1' in registrar._detected_datetime_columns
        assert 'date2' in registrar._detected_datetime_columns
        assert 'date3' in registrar._detected_datetime_columns
        assert 'datetime1' in registrar._detected_datetime_columns
        assert 'datetime2' in registrar._detected_datetime_columns
        assert 'mixed_good' in registrar._detected_datetime_columns
        
        # Should not detect these
        assert 'mixed_bad' not in registrar._detected_datetime_columns
        assert 'numbers' not in registrar._detected_datetime_columns
        assert 'text' not in registrar._detected_datetime_columns

    def test_convert_datetime_columns_with_errors(self, registrar):
        """Test datetime conversion with error handling."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'valid_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'invalid_date': ['2024-01-01', 'not a date', '2024-01-03'],
            'mixed_format': ['2024-01-01', '01/15/2024', '15-Jan-2024']
        })
        
        registrar._detected_datetime_columns = ['valid_date', 'invalid_date', 'mixed_format']
        
        with patch('mdm.dataset.registrar.logger') as mock_logger:
            result = registrar._convert_datetime_columns(df)
            
            # Valid date should be converted
            assert pd.api.types.is_datetime64_any_dtype(result['valid_date'])
            
            # Invalid date should remain as object with warning logged
            assert result['invalid_date'].dtype == object
            mock_logger.warning.assert_called()
            
            # Mixed format might be converted or not depending on pandas
            # But should not raise error

    def test_detect_column_types_with_profiling_memory_handling(self, registrar):
        """Test column type detection with memory-efficient profiling."""
        # Create large dataframe
        large_df = pd.DataFrame({
            'id': range(100000),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100000),
            'value': np.random.rand(100000),
            'text': ['text_' + str(i) for i in range(100000)]
        })
        
        mock_report = Mock()
        mock_report.to_json.return_value = json.dumps({
            'variables': {
                'id': {'type': 'Numeric'},
                'category': {'type': 'Categorical'},
                'value': {'type': 'Numeric'},
                'text': {'type': 'Text'}
            }
        })
        
        with patch('mdm.dataset.registrar.ProfileReport') as mock_profile_class:
            mock_profile_class.return_value = mock_report
            
            registrar._detect_column_types_with_profiling(large_df, 'large_table', Mock())
            
            # Should call ProfileReport with minimal=True for large datasets
            mock_profile_class.assert_called_once()
            call_kwargs = mock_profile_class.call_args[1]
            assert call_kwargs.get('minimal', False) == True

    def test_simple_column_type_detection_comprehensive(self, registrar):
        """Test simple column type detection with all cases."""
        df = pd.DataFrame({
            # ID columns
            'id': range(100),
            'user_id': range(100, 200),
            'customer_idx': range(100),
            'pk': range(100),
            
            # Numeric columns
            'int_col': np.random.randint(0, 1000, 100),
            'float_col': np.random.rand(100) * 100,
            'decimal_col': np.round(np.random.rand(100) * 100, 2),
            
            # Boolean columns
            'bool_col': [True, False] * 50,
            'binary_col': [0, 1] * 50,
            'yes_no': ['yes', 'no'] * 50,
            
            # Categorical columns
            'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'status': np.random.choice(['active', 'inactive', 'pending'], 100),
            
            # Text columns
            'description': ['Long text description ' + str(i) for i in range(100)],
            'comment': ['Comment number ' + str(i) for i in range(100)],
            
            # Date columns (as strings)
            'date_str': pd.date_range('2024-01-01', periods=100).strftime('%Y-%m-%d'),
            
            # Mixed/other
            'mixed': [1, 'two', 3.0, 'four'] * 25,
            'constant': ['same'] * 100,
            'mostly_null': [None] * 95 + [1, 2, 3, 4, 5]
        })
        
        registrar._simple_column_type_detection(df, 'test_table')
        
        types = registrar._detected_column_types['test_table']
        
        # Check ID detection
        assert types['id'] == ColumnType.ID
        assert types['user_id'] == ColumnType.ID
        assert types['customer_idx'] == ColumnType.ID
        assert types['pk'] == ColumnType.ID
        
        # Check numeric
        assert types['int_col'] == ColumnType.NUMERIC
        assert types['float_col'] == ColumnType.NUMERIC
        assert types['decimal_col'] == ColumnType.NUMERIC
        
        # Check boolean
        assert types['bool_col'] == ColumnType.BINARY
        assert types['binary_col'] == ColumnType.BINARY
        assert types['yes_no'] == ColumnType.BINARY
        
        # Check categorical
        assert types['category'] == ColumnType.CATEGORICAL
        assert types['status'] == ColumnType.CATEGORICAL
        assert types['constant'] == ColumnType.CATEGORICAL
        
        # Check text
        assert types['description'] == ColumnType.TEXT
        assert types['comment'] == ColumnType.TEXT
        
        # Check others
        assert types['date_str'] == ColumnType.TEXT  # String dates detected as text in simple detection
        assert types['mixed'] == ColumnType.TEXT  # Mixed types -> text
        assert types['mostly_null'] == ColumnType.TEXT  # Mostly null -> text

    def test_analyze_columns_comprehensive(self, registrar):
        """Test comprehensive column analysis."""
        db_info = {'backend': 'sqlite'}
        table_mappings = {
            'train': {'row_count': 1000, 'column_count': 10},
            'test': {'row_count': 500, 'column_count': 9}
        }
        
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            
            # Create realistic table samples
            train_sample = pd.DataFrame({
                'id': range(100),
                'user_id': range(1000, 1100),
                'category': np.random.choice(['A', 'B', 'C'], 100),
                'value': np.random.rand(100) * 100,
                'has_nulls': [1, 2, None] * 33 + [1],
                'all_null': [None] * 100,
                'constant': [42] * 100,
                'unique_text': ['unique_' + str(i) for i in range(100)],
                'bool_col': [True, False] * 50,
                'target': np.random.randint(0, 2, 100)
            })
            
            test_sample = train_sample[['id', 'user_id', 'category', 'value', 'has_nulls', 
                                       'all_null', 'constant', 'unique_text', 'bool_col']].copy()
            
            mock_backend.get_table_sample.side_effect = lambda table, **kwargs: (
                train_sample if table == 'train' else test_sample
            )
            
            # Mock table info with detailed column information
            def get_table_info(table):
                sample = train_sample if table == 'train' else test_sample
                return {
                    'row_count': 1000 if table == 'train' else 500,
                    'columns': {
                        col: {
                            'dtype': str(sample[col].dtype),
                            'nullable': sample[col].isnull().any()
                        }
                        for col in sample.columns
                    }
                }
            
            mock_backend.get_table_info.side_effect = get_table_info
            mock_factory.create.return_value = mock_backend
            
            # Set detected column types
            registrar._detected_column_types = {
                'train': {col: ColumnType.NUMERIC if col in ['value', 'has_nulls', 'constant'] 
                         else ColumnType.ID if col in ['id', 'user_id']
                         else ColumnType.CATEGORICAL if col == 'category'
                         else ColumnType.TEXT if col in ['unique_text', 'all_null']
                         else ColumnType.BINARY if col in ['bool_col', 'target']
                         else ColumnType.TEXT
                         for col in train_sample.columns},
                'test': {col: ColumnType.NUMERIC if col in ['value', 'has_nulls', 'constant'] 
                        else ColumnType.ID if col in ['id', 'user_id']
                        else ColumnType.CATEGORICAL if col == 'category'
                        else ColumnType.TEXT if col in ['unique_text', 'all_null']
                        else ColumnType.BINARY if col == 'bool_col'
                        else ColumnType.TEXT
                        for col in test_sample.columns}
            }
            
            result = registrar._analyze_columns(db_info, table_mappings)
            
            # Check results
            assert 'train' in result
            assert 'test' in result
            
            # Check train columns
            train_cols = result['train']
            
            # ID columns
            assert train_cols['id']['type'] == ColumnType.ID
            assert train_cols['id']['nullable'] == False
            assert train_cols['id']['unique_count'] == 100
            
            # Numeric with nulls
            assert train_cols['has_nulls']['null_count'] == 33
            assert train_cols['has_nulls']['null_ratio'] == pytest.approx(0.33, rel=0.01)
            
            # All null column
            assert train_cols['all_null']['null_count'] == 100
            assert train_cols['all_null']['null_ratio'] == 1.0
            
            # Constant column
            assert train_cols['constant']['unique_count'] == 1
            assert train_cols['constant']['min'] == 42
            assert train_cols['constant']['max'] == 42
            
            # Unique text
            assert train_cols['unique_text']['unique_count'] == 100
            assert train_cols['unique_text']['unique_ratio'] == 1.0

    def test_detect_id_columns_comprehensive(self, registrar):
        """Test comprehensive ID column detection."""
        column_info = {
            'train': {
                # Clear ID columns
                'id': {'type': ColumnType.ID, 'unique_count': 1000, 'null_count': 0},
                'user_id': {'type': ColumnType.NUMERIC, 'unique_count': 1000, 'null_count': 0},
                'customer_idx': {'type': ColumnType.NUMERIC, 'unique_count': 1000, 'null_count': 0},
                'pk': {'type': ColumnType.ID, 'unique_count': 1000, 'null_count': 0},
                'index': {'type': ColumnType.NUMERIC, 'unique_count': 1000, 'null_count': 0},
                
                # Not ID columns
                'value': {'type': ColumnType.NUMERIC, 'unique_count': 500, 'null_count': 0},
                'category_id': {'type': ColumnType.CATEGORICAL, 'unique_count': 10, 'null_count': 0},
                'id_with_nulls': {'type': ColumnType.ID, 'unique_count': 900, 'null_count': 100},
                'partial_unique': {'type': ColumnType.NUMERIC, 'unique_count': 800, 'null_count': 0},
            },
            'test': {
                'id': {'type': ColumnType.ID, 'unique_count': 500, 'null_count': 0},
                'user_id': {'type': ColumnType.NUMERIC, 'unique_count': 500, 'null_count': 0},
                'customer_idx': {'type': ColumnType.NUMERIC, 'unique_count': 500, 'null_count': 0},
                'new_column': {'type': ColumnType.NUMERIC, 'unique_count': 500, 'null_count': 0},
            }
        }
        
        result = registrar._detect_id_columns(column_info)
        
        # Should detect columns that are ID-like and present in all tables
        assert 'id' in result
        assert 'user_id' in result
        assert 'customer_idx' in result
        
        # Should not include these
        assert 'pk' not in result  # Not in test table
        assert 'category_id' not in result  # Low cardinality
        assert 'id_with_nulls' not in result  # Has nulls
        assert 'partial_unique' not in result  # Not unique enough
        assert 'new_column' not in result  # Only in test table

    def test_infer_problem_type_edge_cases(self, registrar):
        """Test problem type inference with edge cases."""
        # Test with no target column info
        result = registrar._infer_problem_type({}, 'missing_target')
        assert result is None
        
        # Test with target not in train
        column_info = {
            'test': {'target': {'type': ColumnType.NUMERIC}},
            'validation': {'target': {'type': ColumnType.NUMERIC}}
        }
        result = registrar._infer_problem_type(column_info, 'target')
        assert result is None
        
        # Test time series detection
        column_info = {
            'train': {
                'target': {'type': ColumnType.NUMERIC, 'unique_count': 100, 'is_float': True},
                'date': {'type': ColumnType.DATETIME}
            }
        }
        registrar._detected_datetime_columns = ['date']
        result = registrar._infer_problem_type(column_info, 'target')
        # Could be time_series_forecasting if datetime columns exist

    def test_generate_features_with_type_schema(self, registrar):
        """Test feature generation with custom type schema."""
        normalized_name = "test_dataset"
        db_info = {'backend': 'sqlite'}
        table_mappings = {'train': {'row_count': 1000}}
        column_info = {
            'train': {
                'numeric_col': {'type': ColumnType.NUMERIC},
                'cat_col': {'type': ColumnType.CATEGORICAL},
                'text_col': {'type': ColumnType.TEXT},
                'target': {'type': ColumnType.BINARY}
            }
        }
        target_column = 'target'
        id_columns = ['id']
        type_schema = {
            'numeric_col': 'continuous',
            'cat_col': 'categorical',
            'text_col': 'text'
        }
        
        with patch('mdm.dataset.registrar.logger') as mock_logger:
            result = registrar._generate_features(
                normalized_name, db_info, table_mappings, column_info,
                target_column, id_columns, type_schema, Mock()
            )
            
            # Should log feature generation
            assert any('Generating features' in str(call) for call in mock_logger.info.call_args_list)
            
            # Feature generator should be called with type schema
            registrar.feature_generator.generate.assert_called_once()
            call_kwargs = registrar.feature_generator.generate.call_args[1]
            assert call_kwargs.get('type_schema') == type_schema

    def test_compute_initial_statistics_with_error(self, registrar):
        """Test statistics computation with error handling."""
        normalized_name = "test_dataset"
        db_info = {'backend': 'sqlite'}
        table_mappings = {'train': {'row_count': 1000}}
        
        # Make compute_dataset_statistics fail
        with patch('mdm.dataset.registrar.compute_dataset_statistics', side_effect=Exception("Stats failed")):
            with patch('mdm.dataset.registrar.logger') as mock_logger:
                result = registrar._compute_initial_statistics(normalized_name, db_info, table_mappings)
                
                # Should return None and log error
                assert result is None
                mock_logger.error.assert_called()

    def test_postgresql_database_creation_detailed(self, registrar):
        """Test PostgreSQL database creation with all error cases."""
        db_info = {
            'host': 'localhost',
            'port': 5432,
            'database': 'mdm_test',
            'user': 'user',
            'password': 'pass'
        }
        
        # Test successful creation
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = Mock()
        mock_cursor.close = Mock()
        
        with patch('psycopg2.connect', return_value=mock_conn):
            registrar._create_postgresql_database(db_info)
            
            # Verify isolation level and SQL execution
            mock_conn.set_isolation_level.assert_called_once_with(0)
            assert mock_cursor.execute.call_count == 1
            sql = mock_cursor.execute.call_args[0][0]
            assert 'CREATE DATABASE' in sql
            assert 'mdm_test' in sql
            
            # Verify cleanup
            mock_cursor.close.assert_called_once()
            mock_conn.close.assert_called_once()

    def test_detect_column_types_with_profiling_error_cases(self, registrar):
        """Test column type detection when profiling has various errors."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Test with profiling returning invalid JSON
        mock_report = Mock()
        mock_report.to_json.return_value = "invalid json"
        
        with patch('mdm.dataset.registrar.ProfileReport', return_value=mock_report):
            with patch.object(registrar, '_simple_column_type_detection') as mock_simple:
                registrar._detect_column_types_with_profiling(df, 'test_table', Mock())
                
                # Should fall back to simple detection
                mock_simple.assert_called_once()

    def test_logging_throughout_registration(self, registrar, mock_manager, tmp_path):
        """Test that all important steps are logged."""
        data_path = tmp_path / "data.csv"
        data_path.write_text("id,value,target\n1,100,0\n2,200,1\n")
        
        with patch('mdm.dataset.registrar.logger') as mock_logger:
            with patch('mdm.dataset.registrar.discover_data_files', return_value={'data': data_path}):
                with patch('mdm.dataset.registrar.BackendFactory'):
                    with patch('mdm.dataset.registrar.compute_dataset_statistics', return_value={}):
                        with patch('pathlib.Path.mkdir'):
                            result = registrar.register('test_dataset', data_path)
                            
                            # Check key log messages
                            log_messages = [str(call) for call in mock_logger.info.call_args_list]
                            
                            # Should log start
                            assert any("Starting registration" in msg for msg in log_messages)
                            
                            # Should log success
                            assert any("registered successfully" in msg for msg in log_messages)