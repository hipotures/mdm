"""Final comprehensive unit tests for DatasetRegistrar to achieve 90%+ coverage."""

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
from sqlalchemy.exc import ProgrammingError

from mdm.dataset.registrar import DatasetRegistrar
from mdm.core.exceptions import DatasetError
from mdm.models.dataset import DatasetInfo, ColumnInfo, FileInfo
from mdm.models.enums import ProblemType, ColumnType, FileType


class TestDatasetRegistrarFinal:
    """Final comprehensive test cases for DatasetRegistrar to achieve 90%+ coverage."""

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
        
        # SQLite config
        sqlite_config = Mock()
        sqlite_config.model_dump.return_value = {
            'journal_mode': 'WAL',
            'synchronous': 'NORMAL'
        }
        config.database.sqlite = sqlite_config
        
        # PostgreSQL config
        pg_config = Mock()
        pg_config.host = "localhost"
        pg_config.port = 5432
        pg_config.user = "test_user"
        pg_config.password = "test_pass"
        pg_config.database = "test_db"
        pg_config.model_dump.return_value = {}
        config.database.postgresql = pg_config
        
        # DuckDB config
        duckdb_config = Mock()
        duckdb_config.model_dump.return_value = {}
        config.database.duckdb = duckdb_config
        
        # Other configs
        config.paths.datasets_path = "datasets/"
        config.feature_engineering.enabled = True
        config.performance.batch_size = 10000
        
        return config

    @pytest.fixture
    def mock_config_manager(self, mock_config):
        """Create mock config manager."""
        manager = Mock()
        manager.config = mock_config
        manager.base_path = Path("/home/user/.mdm")
        return manager

    @pytest.fixture
    def mock_feature_generator(self):
        """Create mock FeatureGenerator."""
        generator = Mock()
        generator.generate.return_value = None
        return generator

    @pytest.fixture
    def registrar(self, mock_manager, mock_config_manager, mock_feature_generator):
        """Create DatasetRegistrar instance."""
        with patch('mdm.dataset.registrar.get_config_manager', return_value=mock_config_manager):
            with patch('mdm.dataset.registrar.FeatureGenerator', return_value=mock_feature_generator):
                reg = DatasetRegistrar(mock_manager)
                # Initialize detection attributes
                reg._detected_datetime_columns = []
                reg._detected_column_types = {}
                return reg

    def test_init_with_manager(self, mock_manager, mock_config_manager, mock_feature_generator):
        """Test initialization with provided manager."""
        with patch('mdm.dataset.registrar.get_config_manager', return_value=mock_config_manager):
            with patch('mdm.dataset.registrar.FeatureGenerator', return_value=mock_feature_generator):
                registrar = DatasetRegistrar(mock_manager)
                assert registrar.manager == mock_manager
                assert registrar.config == mock_config_manager.config
                assert registrar.base_path == mock_config_manager.base_path
                assert registrar.feature_generator == mock_feature_generator

    def test_init_without_manager(self, mock_config_manager, mock_feature_generator):
        """Test initialization without manager."""
        with patch('mdm.dataset.registrar.get_config_manager', return_value=mock_config_manager):
            with patch('mdm.dataset.registrar.DatasetManager') as mock_manager_class:
                with patch('mdm.dataset.registrar.FeatureGenerator', return_value=mock_feature_generator):
                    registrar = DatasetRegistrar()
                    mock_manager_class.assert_called_once()

    def test_validate_name(self, registrar, mock_manager):
        """Test name validation."""
        mock_manager.validate_dataset_name.return_value = "test_dataset"
        result = registrar._validate_name("Test-Dataset")
        assert result == "test_dataset"
        mock_manager.validate_dataset_name.assert_called_once_with("Test-Dataset")

    def test_validate_path_file(self, registrar, tmp_path):
        """Test path validation for file."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("id,value\n1,100\n")
        
        result = registrar._validate_path(test_file)
        assert result == test_file.resolve()

    def test_validate_path_directory(self, registrar, tmp_path):
        """Test path validation for directory."""
        test_dir = tmp_path / "dataset"
        test_dir.mkdir()
        
        result = registrar._validate_path(test_dir)
        assert result == test_dir.resolve()

    def test_validate_path_not_exists(self, registrar):
        """Test path validation for non-existent path."""
        with pytest.raises(DatasetError, match="Path does not exist"):
            registrar._validate_path(Path("/non/existent/path"))

    def test_auto_detect_kaggle(self, registrar, tmp_path):
        """Test auto-detection of Kaggle structure."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        submission_file = dataset_dir / "sample_submission.csv"
        submission_file.write_text("id,prediction\n1,0\n")
        
        with patch('mdm.dataset.registrar.detect_kaggle_structure', return_value=True):
            with patch('mdm.dataset.registrar.extract_target_from_sample_submission', return_value='prediction'):
                result = registrar._auto_detect(dataset_dir)
                
                assert result['structure'] == 'kaggle'
                assert result['target_column'] == 'prediction'

    def test_auto_detect_non_kaggle(self, registrar, tmp_path):
        """Test auto-detection for non-Kaggle structure."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        
        with patch('mdm.dataset.registrar.detect_kaggle_structure', return_value=False):
            result = registrar._auto_detect(dataset_dir)
            assert result == {}

    def test_discover_files_single_file(self, registrar, tmp_path):
        """Test file discovery for single file."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("id,value\n1,100\n")
        
        result = registrar._discover_files(data_file, {})
        assert result == {'data': data_file}

    def test_discover_files_directory(self, registrar, tmp_path):
        """Test file discovery for directory."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        
        files = {
            'train': dataset_dir / "train.csv",
            'test': dataset_dir / "test.csv"
        }
        for name, path in files.items():
            path.write_text("id,value\n1,100\n")
        
        with patch('mdm.dataset.registrar.discover_data_files', return_value=files):
            result = registrar._discover_files(dataset_dir, {})
            assert result == files

    def test_discover_files_no_files(self, registrar, tmp_path):
        """Test file discovery with no data files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with patch('mdm.dataset.registrar.discover_data_files', return_value={}):
            with pytest.raises(DatasetError, match="No data files found"):
                registrar._discover_files(empty_dir, {})

    def test_discover_files_kaggle_validation(self, registrar, tmp_path):
        """Test Kaggle validation during file discovery."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        
        test_df = pd.DataFrame({'id': [1, 2], 'feature': [0.5, 0.6]})
        files = {
            'test': dataset_dir / "test.csv",
            'sample_submission': dataset_dir / "sample_submission.csv"
        }
        
        detected_info = {'structure': 'kaggle'}
        
        with patch('mdm.dataset.registrar.discover_data_files', return_value=files):
            with patch('mdm.dataset.registrar.pd.read_csv', return_value=test_df):
                with patch('mdm.dataset.registrar.validate_kaggle_submission_format', 
                          return_value=(True, None)):
                    result = registrar._discover_files(dataset_dir, detected_info)
                    assert result == files

    def test_discover_files_kaggle_validation_warning(self, registrar, tmp_path):
        """Test Kaggle validation warning."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        
        test_df = pd.DataFrame({'id': [1, 2], 'feature': [0.5, 0.6]})
        files = {
            'test': dataset_dir / "test.csv",
            'sample_submission': dataset_dir / "sample_submission.csv"
        }
        
        detected_info = {'structure': 'kaggle'}
        
        with patch('mdm.dataset.registrar.discover_data_files', return_value=files):
            with patch('mdm.dataset.registrar.pd.read_csv', return_value=test_df):
                with patch('mdm.dataset.registrar.validate_kaggle_submission_format', 
                          return_value=(False, "ID mismatch")):
                    with patch('mdm.dataset.registrar.logger') as mock_logger:
                        result = registrar._discover_files(dataset_dir, detected_info)
                        mock_logger.warning.assert_called_once()
                        assert result == files

    def test_create_database_sqlite(self, registrar, mock_config_manager):
        """Test SQLite database creation."""
        mock_config_manager.config.database.default_backend = "sqlite"
        
        with patch('pathlib.Path.mkdir'):
            result = registrar._create_database("test_dataset")
        
            assert result['backend'] == 'sqlite'
            assert 'path' in result
            assert 'test_dataset.sqlite' in result['path']
            assert result['journal_mode'] == 'WAL'
            assert result['synchronous'] == 'NORMAL'

    def test_create_database_duckdb(self, registrar, mock_config_manager):
        """Test DuckDB database creation."""
        mock_config_manager.config.database.default_backend = "duckdb"
        
        with patch('pathlib.Path.mkdir'):
            result = registrar._create_database("test_dataset")
        
            assert result['backend'] == 'duckdb'
            assert 'path' in result
            assert 'test_dataset.duckdb' in result['path']

    def test_create_database_postgresql(self, registrar, mock_config_manager):
        """Test PostgreSQL database creation."""
        mock_config_manager.config.database.default_backend = "postgresql"
        
        with patch.object(registrar, '_create_postgresql_database'):
            result = registrar._create_database("test_dataset")
            
            assert result['backend'] == 'postgresql'
            assert result['host'] == 'localhost'
            assert result['port'] == 5432
            assert result['database'] == 'mdm_test_dataset'
            assert result['user'] == 'test_user'
            assert result['password'] == 'test_pass'
            
            registrar._create_postgresql_database.assert_called_once()

    def test_create_postgresql_database_success(self, registrar):
        """Test successful PostgreSQL database creation."""
        db_info = {
            'host': 'localhost',
            'port': 5432,
            'database': 'mdm_test',
            'user': 'user',
            'password': 'pass'
        }
        
        mock_conn = Mock()
        mock_cursor = Mock()
        # Make cursor support context manager
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        # Mock the database check
        mock_cursor.fetchone.return_value = None  # Database doesn't exist
        
        with patch('psycopg2.connect', return_value=mock_conn):
            registrar._create_postgresql_database(db_info)
            
            mock_conn.set_isolation_level.assert_called_once_with(0)
            assert mock_cursor.execute.call_count == 2  # Check exists + create
            mock_conn.close.assert_called_once()

    def test_create_postgresql_database_already_exists(self, registrar):
        """Test PostgreSQL database creation when already exists."""
        db_info = {
            'host': 'localhost',
            'port': 5432,
            'database': 'mdm_test',
            'user': 'user',
            'password': 'pass'
        }
        
        mock_conn = Mock()
        mock_cursor = Mock()
        # Make cursor support context manager
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        # Mock the database check - database exists
        mock_cursor.fetchone.return_value = (1,)  # Database exists
        
        with patch('psycopg2.connect', return_value=mock_conn):
            registrar._create_postgresql_database(db_info)
            
            # Should only check, not create
            assert mock_cursor.execute.call_count == 1
            mock_conn.close.assert_called_once()

    def test_create_postgresql_database_import_error(self, registrar):
        """Test PostgreSQL database creation with import error."""
        db_info = {'database': 'test'}
        
        # Mock import error
        with patch('builtins.__import__', side_effect=ImportError("No module named 'psycopg2'")):
            with pytest.raises(DatasetError, match="Failed to create PostgreSQL database"):
                registrar._create_postgresql_database(db_info)

    def test_load_data_files_basic(self, registrar, tmp_path):
        """Test basic data file loading."""
        # Create test CSV
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            'id': [1, 2],
            'value': [10.5, 20.5]
        })
        df.to_csv(csv_file, index=False)
        
        files = {'data': csv_file}
        db_info = {'backend': 'sqlite', 'path': str(tmp_path / 'test.db')}
        
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            mock_backend.database_exists.return_value = False
            mock_backend.create_database = Mock()
            mock_engine = Mock()
            mock_backend.get_engine.return_value = mock_engine
            mock_backend.create_table_from_dataframe = Mock()
            mock_backend.query = Mock(return_value=pd.DataFrame({'count': [2]}))
            mock_factory.create.return_value = mock_backend
            
            # Mock detect_delimiter
            with patch('mdm.dataset.registrar.detect_delimiter', return_value=','):
                result = registrar._load_data_files(files, db_info)
                
                # Check result is table mappings
                assert 'data' in result
                assert result['data'] == 'data'  # table name mapping
                
                # Verify backend operations
                mock_backend.create_database.assert_called_once()
                assert mock_backend.create_table_from_dataframe.called

    def test_load_data_files_with_datetime(self, registrar, tmp_path):
        """Test loading files with datetime detection."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            'id': [1, 2],
            'date': ['2024-01-01', '2024-01-02']
        })
        df.to_csv(csv_file, index=False)
        
        files = {'data': csv_file}
        db_info = {'backend': 'sqlite', 'path': str(tmp_path / 'test.db')}
        
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            mock_backend.database_exists.return_value = True
            mock_engine = Mock()
            mock_backend.get_engine.return_value = mock_engine
            mock_backend.create_table_from_dataframe = Mock()
            mock_backend.query = Mock(return_value=pd.DataFrame({'count': [2]}))
            mock_factory.create.return_value = mock_backend
            
            # Mock detect_delimiter
            with patch('mdm.dataset.registrar.detect_delimiter', return_value=','):
                # The method reads the file, so no need to mock pd.read_csv
                result = registrar._load_data_files(files, db_info)
                
                # Should have detected datetime columns internally
                assert 'data' in result
                assert result['data'] == 'data'

    def test_convert_datetime_columns(self, registrar):
        """Test datetime column conversion."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00'],
            'text': ['a', 'b', 'c']
        })
        
        registrar._detected_datetime_columns = ['date', 'timestamp']
        
        result = registrar._convert_datetime_columns(df)
        
        # Check that datetime columns were converted
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
        # Other columns should remain unchanged
        assert result['text'].dtype == object

    def test_detect_datetime_columns_from_sample(self, registrar):
        """Test datetime column detection from sample."""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00'],
            'created_at': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'not_date': ['abc', 'def', 'ghi'],
            'mixed': ['2024-01-01', 'not a date', '2024-01-03']  # Less than 80% success
        })
        
        registrar._detect_datetime_columns_from_sample(df)
        
        assert 'date' in registrar._detected_datetime_columns
        assert 'timestamp' in registrar._detected_datetime_columns
        assert 'created_at' in registrar._detected_datetime_columns
        assert 'not_date' not in registrar._detected_datetime_columns
        assert 'mixed' not in registrar._detected_datetime_columns

    def test_detect_and_store_column_types_with_profiling(self, registrar):
        """Test column type detection with profiling."""
        df = pd.DataFrame({
            'id': range(100),
            'category': ['A', 'B', 'C'] * 33 + ['A'],
            'numeric': np.random.rand(100),
            'text': ['text ' + str(i) for i in range(100)],
            'binary': [0, 1] * 50
        })
        
        # Mock profiling report
        mock_report = Mock()
        mock_report.to_json.return_value = json.dumps({
            'variables': {
                'id': {'type': 'Numeric'},
                'category': {'type': 'Categorical'},
                'numeric': {'type': 'Numeric'},
                'text': {'type': 'Text'},
                'binary': {'type': 'Boolean'}
            }
        })
        
        with patch('mdm.dataset.registrar.ProfileReport', return_value=mock_report):
            registrar._detect_and_store_column_types(df, 'test_table')
            
            assert registrar._detected_column_types['test_table']['id'] == ColumnType.ID
            assert registrar._detected_column_types['test_table']['category'] == ColumnType.CATEGORICAL
            assert registrar._detected_column_types['test_table']['numeric'] == ColumnType.NUMERIC
            assert registrar._detected_column_types['test_table']['text'] == ColumnType.TEXT
            assert registrar._detected_column_types['test_table']['binary'] == ColumnType.BINARY

    def test_detect_and_store_column_types_fallback(self, registrar):
        """Test column type detection with fallback to simple method."""
        df = pd.DataFrame({
            'id': range(10),
            'value': np.random.rand(10)
        })
        
        # Make ProfileReport fail
        with patch('mdm.dataset.registrar.ProfileReport', side_effect=Exception("Profile failed")):
            with patch.object(registrar, '_simple_column_type_detection') as mock_simple:
                registrar._detect_and_store_column_types(df, 'test_table')
                
                # Should fall back to simple detection
                mock_simple.assert_called_once_with(df, 'test_table')

    def test_simple_column_type_detection(self, registrar):
        """Test simple column type detection."""
        df = pd.DataFrame({
            'id': range(100),
            'float_col': np.random.rand(100),
            'int_col': np.random.randint(0, 100, 100),
            'text_col': ['text'] * 100,
            'bool_col': [True, False] * 50,
            'cat_col': ['A', 'B', 'C'] * 33 + ['A']
        })
        
        registrar._simple_column_type_detection(df, 'test_table')
        
        types = registrar._detected_column_types['test_table']
        assert types['id'] == ColumnType.ID
        assert types['float_col'] == ColumnType.NUMERIC
        assert types['int_col'] == ColumnType.NUMERIC
        assert types['text_col'] == ColumnType.TEXT
        assert types['bool_col'] == ColumnType.BINARY
        assert types['cat_col'] == ColumnType.CATEGORICAL

    def test_analyze_columns(self, registrar):
        """Test column analysis."""
        db_info = {'backend': 'sqlite'}
        table_mappings = {
            'train': {'row_count': 100, 'column_count': 5}
        }
        
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            
            # Mock table data
            df = pd.DataFrame({
                'id': range(100),
                'category': ['A', 'B', 'C'] * 33 + ['A'],
                'numeric': np.random.rand(100),
                'with_nulls': [1, 2, None] * 33 + [1]
            })
            mock_backend.get_table_sample.return_value = df
            
            mock_factory.create.return_value = mock_backend
            
            # Set detected column types
            registrar._detected_column_types = {
                'train': {
                    'id': ColumnType.ID,
                    'category': ColumnType.CATEGORICAL,
                    'numeric': ColumnType.NUMERIC,
                    'with_nulls': ColumnType.NUMERIC
                }
            }
            
            result = registrar._analyze_columns(db_info, table_mappings)
            
            assert 'train' in result
            assert len(result['train']) == 4
            
            # Check column details
            id_info = result['train']['id']
            assert id_info['type'] == ColumnType.ID
            assert id_info['nullable'] == False
            assert id_info['unique_count'] == 100

    def test_detect_id_columns(self, registrar):
        """Test ID column detection."""
        column_info = {
            'train': {
                'id': {'type': ColumnType.ID, 'unique_count': 100},
                'user_id': {'type': ColumnType.NUMERIC, 'unique_count': 100},
                'value': {'type': ColumnType.NUMERIC, 'unique_count': 50}
            },
            'test': {
                'id': {'type': ColumnType.ID, 'unique_count': 50},
                'user_id': {'type': ColumnType.NUMERIC, 'unique_count': 50}
            }
        }
        
        result = registrar._detect_id_columns(column_info)
        
        # Should detect common ID columns
        assert 'id' in result
        assert 'user_id' in result

    def test_infer_problem_type_binary(self, registrar):
        """Test binary classification problem type inference."""
        column_info = {
            'train': {
                'columns': ['target', 'feature1'],
                'sample_data': {
                    'target': [0, 1, 0, 1, 1, 0, 1, 0],
                    'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                }
            }
        }
        
        result = registrar._infer_problem_type(column_info, 'target')
        assert result == ProblemType.BINARY_CLASSIFICATION

    def test_infer_problem_type_multiclass(self, registrar):
        """Test multiclass classification problem type inference."""
        column_info = {
            'train': {
                'columns': ['target', 'feature1'],
                'sample_data': {
                    'target': ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C'],
                    'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                }
            }
        }
        
        result = registrar._infer_problem_type(column_info, 'target')
        assert result == ProblemType.MULTICLASS_CLASSIFICATION

    def test_infer_problem_type_regression(self, registrar):
        """Test regression problem type inference."""
        column_info = {
            'train': {
                'columns': ['target', 'feature1'],
                'sample_data': {
                    'target': [1.5, 2.7, 3.9, 4.2, 5.6, 6.1, 7.3, 8.8, 9.9, 10.2],
                    'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
                }
            }
        }
        
        result = registrar._infer_problem_type(column_info, 'target')
        assert result == ProblemType.REGRESSION

    def test_generate_features_enabled(self, registrar, mock_feature_generator):
        """Test feature generation when enabled."""
        normalized_name = "test_dataset"
        db_info = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        table_mappings = {'train': 'train_table'}
        column_info = {
            'train': {
                'columns': ['id', 'feature1', 'target'],
                'sample_data': {
                    'id': [1, 2, 3],
                    'feature1': [1.0, 2.0, 3.0],
                    'target': [0, 1, 0]
                }
            }
        }
        target_column = 'target'
        id_columns = ['id']
        
        # Mock the backend and engine
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            mock_engine = Mock()
            mock_backend.get_engine.return_value = mock_engine
            mock_backend.close_connections = Mock()
            mock_factory.create.return_value = mock_backend
            
            # Mock _detect_column_types_with_profiling
            with patch.object(registrar, '_detect_column_types_with_profiling') as mock_detect:
                mock_detect.return_value = {
                    'id': ColumnType.NUMERIC,
                    'feature1': ColumnType.NUMERIC,
                    'target': ColumnType.NUMERIC
                }
                
                # Call the method
                registrar._generate_features(
                    normalized_name, db_info, table_mappings, column_info,
                    target_column, id_columns, None, Mock()
                )
        
        # Feature generator should be called
        mock_feature_generator.generate_feature_tables.assert_called_once()

    def test_compute_initial_statistics(self, registrar):
        """Test initial statistics computation."""
        normalized_name = "test_dataset"
        db_info = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        table_mappings = {'train': 'train_table', 'test': 'test_table'}
        
        # Mock BackendFactory and backend
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            mock_factory.create.return_value = mock_backend
            
            # Mock compute_dataset_statistics (imported in the method)
            with patch('mdm.dataset.registrar.compute_dataset_statistics') as mock_compute:
                mock_compute.return_value = {
                    'total_rows': 1500,
                    'memory_size_mb': 50.5
                }
                
                result = registrar._compute_initial_statistics(normalized_name, db_info, table_mappings)
                
                assert result is not None
                assert result['total_rows'] == 1500
                assert result['memory_size_mb'] == 50.5

    def test_register_success(self, registrar, mock_manager, tmp_path):
        """Test successful dataset registration."""
        # Create test data
        data_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            'id': range(100),
            'feature': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })
        df.to_csv(data_path, index=False)
        
        # Mock all dependencies
        mock_manager.dataset_exists.return_value = False
        
        # Mock file discovery
        with patch('mdm.dataset.registrar.discover_data_files', return_value={'data': data_path}):
            # Mock backend operations
            with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
                mock_backend = Mock()
                mock_backend.create_table_from_dataframe = Mock()
                mock_backend.get_table_sample.return_value = df
                mock_factory.create.return_value = mock_backend
                
                # Mock other operations
                with patch('mdm.dataset.registrar.detect_id_columns', return_value=['id']):
                    with patch('mdm.dataset.registrar.infer_problem_type', return_value='binary_classification'):
                        with patch('mdm.dataset.registrar.compute_dataset_statistics', return_value={}):
                            with patch('mdm.dataset.registrar.ProfileReport'):
                                result = registrar.register(
                                    'test_dataset',
                                    data_path,
                                    description='Test dataset',
                                    tags=['test']
                                )
                                
                                assert isinstance(result, DatasetInfo)
                                assert result.name == 'test_dataset'
                                assert result.description == 'Test dataset'
                                assert result.tags == ['test']
                                mock_manager.register_dataset.assert_called_once()

    def test_register_with_force(self, registrar, mock_manager, tmp_path):
        """Test registration with force flag."""
        data_path = tmp_path / "data.csv"
        data_path.write_text("id,value\n1,100\n")
        
        # Dataset already exists
        mock_manager.dataset_exists.return_value = True
        
        with patch('mdm.dataset.registrar.RemoveOperation') as mock_remove:
            mock_remove_instance = Mock()
            mock_remove.return_value = mock_remove_instance
            
            # Mock rest of registration
            with patch('mdm.dataset.registrar.discover_data_files', return_value={'data': data_path}):
                with patch('mdm.dataset.registrar.BackendFactory'):
                    with patch('mdm.dataset.registrar.ProfileReport'):
                        with patch('mdm.dataset.registrar.compute_dataset_statistics'):
                            registrar.register('test_dataset', data_path, force=True)
                            
                            # Should remove existing dataset
                            mock_remove_instance.execute.assert_called_once_with(
                                'test_dataset', force=True, dry_run=False
                            )

    def test_register_dataset_exists_no_force(self, registrar, mock_manager, tmp_path):
        """Test registration fails when dataset exists without force."""
        data_path = tmp_path / "data.csv"
        data_path.write_text("id,value\n1,100\n")
        
        mock_manager.dataset_exists.return_value = True
        
        with pytest.raises(DatasetError, match="already exists"):
            registrar.register('test_dataset', data_path)

    def test_register_with_manual_metadata(self, registrar, mock_manager, tmp_path):
        """Test registration with manually provided metadata."""
        data_path = tmp_path / "data.csv"
        data_path.write_text("id,feature,target\n1,0.5,1\n")
        
        mock_manager.dataset_exists.return_value = False
        
        # Mock dependencies
        with patch('mdm.dataset.registrar.discover_data_files', return_value={'data': data_path}):
            with patch('mdm.dataset.registrar.BackendFactory'):
                with patch('mdm.dataset.registrar.compute_dataset_statistics'):
                    result = registrar.register(
                        'test_dataset',
                        data_path,
                        auto_detect=False,  # Disable auto-detection
                        target_column='target',
                        problem_type='binary_classification',
                        id_columns=['id'],
                        generate_features=False  # Disable feature generation
                    )
                    
                    assert result.target_column == 'target'
                    assert result.problem_type == 'binary_classification'
                    assert result.id_columns == ['id']

    def test_load_data_files_large_dataset(self, registrar, tmp_path):
        """Test loading large dataset with chunking."""
        csv_file = tmp_path / "large.csv"
        
        # Create chunks
        chunks = []
        for i in range(5):
            chunk = pd.DataFrame({
                'id': range(i*10000, (i+1)*10000),
                'value': np.random.rand(10000)
            })
            chunks.append(chunk)
        
        files = {'data': csv_file}
        db_info = {'backend': 'sqlite'}
        
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            mock_factory.create.return_value = mock_backend
            
            # Mock chunked reading
            with patch('mdm.dataset.registrar.pd.read_csv') as mock_read:
                # First call returns iterator
                mock_iterator = Mock()
                mock_iterator.__iter__ = Mock(return_value=iter(chunks))
                mock_read.return_value = mock_iterator
                
                with patch.object(registrar, '_detect_and_store_column_types'):
                    progress = Mock()
                    result = registrar._load_data_files(files, db_info, progress)
                    
                    # Should process all chunks
                    assert result['data']['row_count'] == 50000
                    assert mock_backend.create_table_from_dataframe.call_count == 5

    def test_error_handling_remove_fails(self, registrar, mock_manager, tmp_path):
        """Test handling when removing existing dataset fails."""
        data_path = tmp_path / "data.csv"
        data_path.write_text("id,value\n1,100\n")
        
        mock_manager.dataset_exists.return_value = True
        
        with patch('mdm.dataset.registrar.RemoveOperation') as mock_remove:
            mock_remove_instance = Mock()
            mock_remove_instance.execute.side_effect = Exception("Remove failed")
            mock_remove.return_value = mock_remove_instance
            
            with patch('mdm.dataset.registrar.logger') as mock_logger:
                # Mock rest of registration
                with patch('mdm.dataset.registrar.discover_data_files', return_value={'data': data_path}):
                    with patch('mdm.dataset.registrar.BackendFactory'):
                        with patch('mdm.dataset.registrar.compute_dataset_statistics'):
                            # Should continue despite remove failure
                            registrar.register('test_dataset', data_path, force=True)
                            
                            # Should log warning
                            mock_logger.warning.assert_called()

    def test__detect_column_types_with_profiling_edge_cases(self, registrar):
        """Test column type detection with edge cases."""
        df = pd.DataFrame({
            'mixed_numeric': [1, 2, '3', 4, 5],  # Mixed types
            'all_null': [None] * 5,
            'single_value': [1] * 5,
            'date_like': ['2024-01-01'] * 5,
            'bool_as_int': [0, 1, 0, 1, 0]
        })
        
        mock_report = Mock()
        mock_report.to_json.return_value = json.dumps({
            'variables': {
                'mixed_numeric': {'type': 'Unsupported'},
                'all_null': {'type': 'Unsupported'},
                'single_value': {'type': 'Constant'},
                'date_like': {'type': 'DateTime'},
                'bool_as_int': {'type': 'Boolean'}
            }
        })
        
        with patch('mdm.dataset.registrar.ProfileReport', return_value=mock_report):
            registrar._detect_column_types_with_profiling(df, 'test_table', Mock())
            
            types = registrar._detected_column_types['test_table']
            # Unsupported types should fall back to TEXT
            assert types['mixed_numeric'] == ColumnType.TEXT
            assert types['all_null'] == ColumnType.TEXT
            # Constant should be CATEGORICAL
            assert types['single_value'] == ColumnType.CATEGORICAL
            # DateTime should be DATETIME
            assert types['date_like'] == ColumnType.DATETIME
            # Boolean should be BINARY
            assert types['bool_as_int'] == ColumnType.BINARY