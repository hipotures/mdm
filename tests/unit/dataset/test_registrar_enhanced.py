"""Enhanced unit tests for DatasetRegistrar to achieve 90%+ coverage."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
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


class TestDatasetRegistrarEnhanced:
    """Enhanced test cases for DatasetRegistrar achieving 90%+ coverage."""

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
        config.database.sqlite = Mock()
        config.database.sqlite.model_dump.return_value = {
            'journal_mode': 'WAL',
            'synchronous': 'NORMAL'
        }
        config.database.postgresql = Mock()
        config.database.postgresql.host = "localhost"
        config.database.postgresql.port = 5432
        config.database.postgresql.user = "test_user"
        config.database.postgresql.password = "test_pass"
        config.database.postgresql.database = "test_db"
        config.paths.datasets_path = "datasets/"
        config.features.enable_at_registration = True
        config.datasets.generate_features = True
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
                return DatasetRegistrar(mock_manager)

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
                    assert registrar.manager is not None

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

        # Mock all the registration steps
        mock_manager.dataset_exists.return_value = False
        
        # Initialize detected_datetime_columns
        registrar._detected_datetime_columns = []
        
        with patch.multiple(
            'mdm.dataset.registrar.DatasetRegistrar',
            _validate_path=Mock(return_value=data_path),
            _auto_detect=Mock(return_value={'target_column': 'target'}),
            _discover_files=Mock(return_value={'data': data_path}),
            _create_database=Mock(return_value={'backend': 'sqlite', 'path': str(tmp_path / 'test.db')}),
            _load_data_files=Mock(return_value={'data': 'data'}),
            _analyze_columns=Mock(return_value={}),
            _detect_id_columns=Mock(return_value=[]),
            _infer_problem_type=Mock(return_value='binary_classification'),
            _generate_features=Mock(return_value={}),
            _compute_initial_statistics=Mock(return_value={})
        ):
            result = registrar.register('test_dataset', data_path)
            
            assert isinstance(result, DatasetInfo)
            assert result.name == 'test_dataset'
            mock_manager.register_dataset.assert_called_once()

    def test_register_with_force(self, registrar, mock_manager, tmp_path):
        """Test registration with force flag."""
        data_path = tmp_path / "data.csv"
        data_path.write_text("id,value\n1,100\n")
        
        # Dataset already exists
        mock_manager.dataset_exists.return_value = True
        
        with patch('mdm.dataset.operations.RemoveOperation') as mock_remove:
            mock_remove_instance = Mock()
            mock_remove.return_value = mock_remove_instance
            
            with patch.multiple(
                'mdm.dataset.registrar.DatasetRegistrar',
                _validate_path=Mock(return_value=data_path),
                _auto_detect=Mock(return_value={}),
                _discover_files=Mock(return_value={'data': data_path}),
                _create_database=Mock(return_value={'backend': 'sqlite', 'path': str(tmp_path / 'test.db')}),
                _load_data_files=Mock(return_value={'train': 'test_dataset_train'}),
                _analyze_columns=Mock(return_value={}),
                _detect_id_columns=Mock(return_value=[]),
                _infer_problem_type=Mock(return_value=None),
                _generate_features=Mock(return_value={}),
                _compute_initial_statistics=Mock(return_value={})
            ):
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

    def test_validate_name(self, registrar, mock_manager):
        """Test name validation."""
        mock_manager.validate_dataset_name.side_effect = lambda x: x.lower()
        
        result = registrar._validate_name("TestDataset")
        assert result == "testdataset"
        mock_manager.validate_dataset_name.assert_called_once_with("TestDataset")

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
        # Create Kaggle structure
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "train.csv").write_text("id,feature,target\n1,0.5,1\n")
        (dataset_dir / "test.csv").write_text("id,feature\n2,0.6\n")
        
        submission_file = dataset_dir / "sample_submission.csv"
        submission_file.write_text("id,prediction\n2,0\n")
        
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
        
        # Create test and submission files
        test_file = dataset_dir / "test.csv"
        test_file.write_text("id,feature\n1,0.5\n")
        
        submission_file = dataset_dir / "sample_submission.csv"
        submission_file.write_text("id,prediction\n1,0\n")
        
        files = {
            'test': test_file,
            'sample_submission': submission_file
        }
        
        detected_info = {'structure': 'kaggle'}
        
        with patch('mdm.dataset.registrar.discover_data_files', return_value=files):
            with patch('mdm.dataset.registrar.validate_kaggle_submission_format', return_value=(True, None)):
                result = registrar._discover_files(dataset_dir, detected_info)
                assert result == files

    def test_create_database_sqlite(self, registrar, mock_config_manager, tmp_path):
        """Test SQLite database creation."""
        mock_config_manager.config.database.default_backend = "sqlite"
        mock_config_manager.base_path = tmp_path
        registrar.base_path = tmp_path
        
        result = registrar._create_database("test_dataset")
        
        assert result['backend'] == 'sqlite'
        assert 'path' in result
        assert 'test_dataset' in result['path']
        assert result['journal_mode'] == 'WAL'
        assert result['synchronous'] == 'NORMAL'

    def test_create_database_duckdb(self, registrar, mock_config_manager, tmp_path):
        """Test DuckDB database creation."""
        mock_config_manager.config.database.default_backend = "duckdb"
        mock_config_manager.config.database.duckdb = Mock()
        mock_config_manager.config.database.duckdb.model_dump.return_value = {}
        mock_config_manager.base_path = tmp_path
        registrar.base_path = tmp_path
        
        result = registrar._create_database("test_dataset")
        
        assert result['backend'] == 'duckdb'
        assert 'path' in result
        assert result['path'].endswith('.duckdb')

    def test_create_database_postgresql(self, registrar, mock_config_manager):
        """Test PostgreSQL database creation."""
        mock_config_manager.config.database.default_backend = "postgresql"
        mock_config_manager.config.database.postgresql.model_dump.return_value = {}
        
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
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = Mock()
        mock_cursor.fetchone.return_value = None
        
        with patch('psycopg2.connect', return_value=mock_conn):
            registrar._create_postgresql_database(db_info)
            
            mock_conn.set_isolation_level.assert_called_once_with(0)
            mock_cursor.execute.assert_called()
            mock_conn.close.assert_called()

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
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = True  # Database already exists
        
        with patch('psycopg2.connect', return_value=mock_conn):
            with patch('mdm.dataset.registrar.logger'):
                # Should not raise
                registrar._create_postgresql_database(db_info)

    def test_create_postgresql_database_import_error(self, registrar):
        """Test PostgreSQL database creation with import error."""
        db_info = {'database': 'test'}
        
        with patch.dict('sys.modules', {'psycopg2': None}):
            with pytest.raises(DatasetError, match="Failed to create PostgreSQL database"):
                registrar._create_postgresql_database(db_info)

    def test_load_data_basic(self, registrar, tmp_path):
        """Test basic data loading."""
        # Create test files
        files = {}
        for name in ['train', 'test']:
            csv_file = tmp_path / f"{name}.csv"
            df = pd.DataFrame({
                'id': range(10),
                'value': np.random.rand(10)
            })
            df.to_csv(csv_file, index=False)
            files[name] = csv_file
        
        db_info = {'backend': 'sqlite', 'path': str(tmp_path / 'test.db')}
        
        with patch('mdm.dataset.registrar.detect_delimiter', return_value=','):
            with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
                mock_backend = Mock()
                mock_backend.database_exists.return_value = False
                mock_backend.create_database.return_value = None
                mock_backend.get_engine.return_value = Mock()
                mock_backend.create_table_from_dataframe.return_value = None
                mock_backend.query.return_value = pd.DataFrame({'count': [10]})
                mock_backend.close_connections.return_value = None
                mock_factory.create.return_value = mock_backend
                
                result = registrar._load_data_files(files, db_info, progress=None)
                
                assert 'train' in result
                assert 'test' in result
                assert result['train'] == 'train'
                assert result['test'] == 'test'
                assert mock_backend.create_table_from_dataframe.call_count == 2

    def test_infer_metadata(self, registrar):
        """Test metadata inference through methods that exist."""
        # Test detect_id_columns functionality
        column_info = {
            'train': {
                'columns': {
                    'id': 'INTEGER',
                    'user_id': 'INTEGER', 
                    'feature1': 'REAL',
                    'target': 'INTEGER'
                },
                'sample_data': {
                    'id': list(range(100)),
                    'user_id': list(range(100, 200)),
                    'feature1': list(np.random.rand(100)),
                    'target': [0, 1] * 50
                }
            }
        }
        
        id_columns = registrar._detect_id_columns(column_info)
        assert 'id' in id_columns
        assert 'user_id' in id_columns
        
        # Test problem type inference
        problem_type = registrar._infer_problem_type(column_info, 'target')
        assert problem_type == 'binary_classification'

    def test_generate_features_enabled(self, registrar, mock_feature_generator):
        """Test feature generation when enabled."""
        registrar.config.datasets.generate_features = True
        
        db_info = {'backend': 'sqlite', 'path': 'test.db'}
        table_mappings = {'train': 'test_dataset_train', 'test': 'test_dataset_test'}
        column_info = {
            'id': {'dtype': 'int64'},
            'feature1': {'dtype': 'float64'},
            'target': {'dtype': 'int64'}
        }
        target_column = 'target'
        id_columns = ['id']
        
        # Mock the feature generator to avoid actual feature generation
        registrar.feature_generator = mock_feature_generator
        mock_feature_generator.generate_feature_tables.return_value = {'train_features': 'test_dataset_train_features'}
        
        # Mock the backend and dependencies
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            mock_backend.get_engine.return_value = Mock()
            mock_backend.close_connections.return_value = None
            mock_factory.create.return_value = mock_backend
            
            # Mock the column type detection method
            with patch.object(registrar, '_detect_column_types_with_profiling') as mock_detect:
                mock_detect.return_value = {
                    'id': {'type': 'numeric'},
                    'feature1': {'type': 'numeric'},
                    'target': {'type': 'categorical'}
                }
                
                result = registrar._generate_features(
                    'test_dataset', 
                    db_info, 
                    table_mappings,
                    column_info,
                    target_column,
                    id_columns
                )
        
        mock_feature_generator.generate_feature_tables.assert_called_once()

    def test_generate_features_disabled(self, registrar, mock_feature_generator):
        """Test feature generation when disabled."""
        registrar.config.datasets.generate_features = False
        
        db_info = {'backend': 'sqlite', 'path': 'test.db'}
        table_mappings = {'train': 'test_dataset_train'}
        column_info = {'id': {'dtype': 'int64'}}
        target_column = None
        id_columns = ['id']
        
        # Mock the feature generator
        registrar.feature_generator = mock_feature_generator
        
        result = registrar._generate_features(
            'test_dataset', 
            db_info, 
            table_mappings,
            column_info,
            target_column,
            id_columns
        )
        
        # Should not call generator
        mock_feature_generator.generate.assert_not_called()

    def test_compute_statistics(self, registrar):
        """Test statistics computation."""
        db_info = {'backend': 'sqlite', 'path': 'test.db'}
        
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            
            # Mock table statistics
            mock_backend.get_tables.return_value = ['train', 'test']
            mock_backend.get_row_count.side_effect = lambda table: 1000 if table == 'train' else 500
            mock_backend.get_columns.side_effect = lambda table: ['id', 'feature', 'target'] if table == 'train' else ['id', 'feature']
            
            # Mock column analysis
            mock_backend.analyze_column.return_value = {
                'dtype': 'int64',
                'null_count': 0,
                'unique_count': 100,
                'min': 0,
                'max': 99,
                'mean': 49.5,
                'std': 28.87
            }
            
            mock_factory.create.return_value = mock_backend
            
            # Mock query for row counts
            mock_backend.query.side_effect = [
                pd.DataFrame({'count': [1000]}),  # train
                pd.DataFrame({'count': [500]})    # test
            ]
            
            result = registrar._compute_initial_statistics('test_dataset', db_info, {'train': 'train', 'test': 'test'})
            
            assert 'row_count' in result
            assert 'memory_size_bytes' in result
            assert 'computed_at' in result
            assert result['row_count'] == 1500

    def test_load_with_progress(self, registrar, tmp_path):
        """Test data loading with progress tracking."""
        csv_file = tmp_path / "large.csv"
        
        # Create large dataset
        df = pd.DataFrame({
            'id': range(100000),
            'value': np.random.rand(100000)
        })
        df.to_csv(csv_file, index=False)
        
        files = {'data': csv_file}
        db_info = {'backend': 'sqlite', 'path': str(tmp_path / 'test.db')}
        
        with patch('mdm.dataset.registrar.detect_delimiter', return_value=','):
            with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
                mock_backend = Mock()
                mock_backend.database_exists.return_value = False
                mock_backend.create_database.return_value = None
                mock_backend.get_engine.return_value = Mock()
                mock_backend.create_table_from_dataframe.return_value = None
                mock_backend.query.return_value = pd.DataFrame({'count': [100000]})
                mock_backend.close_connections.return_value = None
                mock_factory.create.return_value = mock_backend
                
                with patch('mdm.dataset.registrar.Progress'):
                    result = registrar._load_data_files(files, db_info, progress=None)
                    
                    assert result['data'] == 'data'
                    assert mock_backend.create_table_from_dataframe.call_count >= 1

    def test_error_handling_remove_existing_fails(self, registrar, mock_manager, tmp_path):
        """Test error handling when removing existing dataset fails."""
        data_path = tmp_path / "data.csv"
        data_path.write_text("id,value\n1,100\n")
        
        # Initialize detected_datetime_columns
        registrar._detected_datetime_columns = []
        
        mock_manager.dataset_exists.return_value = True
        
        with patch('mdm.dataset.operations.RemoveOperation') as mock_remove:
            mock_remove_instance = Mock()
            mock_remove_instance.execute.side_effect = Exception("Remove failed")
            mock_remove.return_value = mock_remove_instance
            
            with patch.multiple(
                'mdm.dataset.registrar.DatasetRegistrar',
                _validate_path=Mock(return_value=data_path),
                _auto_detect=Mock(return_value={}),
                _discover_files=Mock(return_value={'data': data_path}),
                _create_database=Mock(return_value={'backend': 'sqlite'}),
                _load_data_files=Mock(return_value={'data': 'data'}),
                _analyze_columns=Mock(return_value={}),
                _detect_id_columns=Mock(return_value=[]),
                _infer_problem_type=Mock(return_value=None),
                _generate_features=Mock(return_value={}),
                _compute_initial_statistics=Mock(return_value={})
            ):
                with patch('mdm.dataset.registrar.logger'):
                    # Should continue despite remove failure
                    registrar.register('test_dataset', data_path, force=True)

    def test_full_integration_sqlite(self, registrar, mock_manager, tmp_path):
        """Test full registration flow with SQLite."""
        # Create realistic dataset
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        
        train_df = pd.DataFrame({
            'id': range(1000),
            'created_at': pd.date_range('2024-01-01', periods=1000, freq='h'),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'value': np.random.rand(1000),
            'target': np.random.randint(0, 2, 1000)
        })
        test_df = train_df[['id', 'created_at', 'category', 'value']].iloc[:200]
        
        train_df.to_csv(dataset_dir / 'train.csv', index=False)
        test_df.to_csv(dataset_dir / 'test.csv', index=False)
        
        # Mock file discovery
        with patch('mdm.dataset.registrar.discover_data_files') as mock_discover:
            mock_discover.return_value = {
                'train': dataset_dir / 'train.csv',
                'test': dataset_dir / 'test.csv'
            }
            
            # Mock backend operations
            with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
                mock_backend = Mock()
                
                # Load file returns info
                mock_backend.load_file.side_effect = lambda path, table, **kwargs: {
                    'row_count': 1000 if 'train' in str(path) else 200,
                    'column_count': 5 if 'train' in str(path) else 4
                }
                
                # Column info
                mock_backend.get_columns.side_effect = lambda table: (
                    ['id', 'created_at', 'category', 'value', 'target'] if table == 'train'
                    else ['id', 'created_at', 'category', 'value']
                )
                
                # Sample data for inference
                mock_backend.read_table.return_value = train_df.head(100)
                
                # Table stats
                mock_backend.get_tables.return_value = ['train', 'test']
                mock_backend.get_row_count.side_effect = lambda t: 1000 if t == 'train' else 200
                mock_backend.analyze_column.return_value = {
                    'dtype': 'int64',
                    'null_count': 0,
                    'unique_count': 100
                }
                
                mock_factory.create.return_value = mock_backend
                
                # Mock inference functions
                with patch('mdm.dataset.registrar.detect_id_columns', return_value=['id']):
                    with patch('mdm.dataset.registrar.infer_problem_type', return_value='binary_classification'):
                        # Mock internal methods to avoid actual file operations
                        with patch.object(registrar, '_create_database') as mock_create_db:
                            mock_create_db.return_value = {
                                'backend': 'sqlite',
                                'path': str(tmp_path / 'test_dataset.sqlite')
                            }
                            
                            with patch.object(registrar, '_load_data_files') as mock_load:
                                mock_load.return_value = {
                                    'train': 'test_dataset_train',
                                    'test': 'test_dataset_test'
                                }
                                
                                with patch.object(registrar, '_analyze_columns') as mock_analyze:
                                    mock_analyze.return_value = {
                                        'train': {
                                            'columns': {'id': 'INTEGER', 'target': 'INTEGER'},
                                            'sample_data': train_df.head(100)
                                        }
                                    }
                                    
                                    with patch.object(registrar, '_compute_initial_statistics') as mock_stats:
                                        mock_stats.return_value = {'row_count': 1200}
                                        
                                        # Mock _infer_problem_type
                                        with patch.object(registrar, '_infer_problem_type') as mock_infer:
                                            mock_infer.return_value = 'binary_classification'
                                            
                                            result = registrar.register(
                                                'test_dataset',
                                                dataset_dir,
                                                description='Test dataset',
                                                tags=['test', 'example'],
                                                target_column='target'
                                            )
                        
                        assert result.name == 'test_dataset'
                        assert result.description == 'Test dataset'
                        assert result.tags == ['test', 'example']
                        assert result.problem_type == ProblemType.BINARY_CLASSIFICATION
                        mock_manager.register_dataset.assert_called_once()

    def test_logging_messages(self, registrar, mock_manager, tmp_path):
        """Test that appropriate log messages are generated."""
        data_path = tmp_path / "data.csv"
        data_path.write_text("id,value\n1,100\n")
        
        # Initialize detected_datetime_columns
        registrar._detected_datetime_columns = []
        
        with patch('mdm.dataset.registrar.logger') as mock_logger:
            with patch.multiple(
                'mdm.dataset.registrar.DatasetRegistrar',
                _validate_path=Mock(return_value=data_path),
                _auto_detect=Mock(return_value={}),
                _discover_files=Mock(return_value={'data': data_path}),
                _create_database=Mock(return_value={'backend': 'sqlite', 'path': str(tmp_path / 'test.db')}),
                _load_data_files=Mock(return_value={'data': 'data'}),
                _analyze_columns=Mock(return_value={}),
                _detect_id_columns=Mock(return_value=[]),
                _infer_problem_type=Mock(return_value=None),
                _generate_features=Mock(return_value={}),
                _compute_initial_statistics=Mock(return_value={})
            ):
                registrar.register('test_dataset', data_path)
                
                # Check key log messages
                mock_logger.info.assert_any_call("Starting registration for dataset 'test_dataset'")
                mock_logger.info.assert_any_call("Dataset 'test_dataset' registered successfully")

    def test_validate_kaggle_with_error(self, registrar, tmp_path):
        """Test Kaggle validation with error."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        
        test_file = dataset_dir / "test.csv"
        test_file.write_text("id,feature\n1,0.5\n")
        
        submission_file = dataset_dir / "sample_submission.csv"
        submission_file.write_text("wrong_id,prediction\n2,0\n")  # Wrong ID column
        
        files = {
            'test': test_file,
            'sample_submission': submission_file
        }
        
        detected_info = {'structure': 'kaggle'}
        
        with patch('mdm.dataset.registrar.discover_data_files', return_value=files):
            with patch('mdm.dataset.registrar.validate_kaggle_submission_format', 
                      return_value=(False, "ID columns don't match")):
                with patch('mdm.dataset.registrar.logger') as mock_logger:
                    result = registrar._discover_files(dataset_dir, detected_info)
                    
                    # Should log warning but continue
                    mock_logger.warning.assert_called()
                    assert result == files