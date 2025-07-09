"""Unit tests for DatasetRegistrar."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import pandas as pd
import yaml

from mdm.dataset.registrar import DatasetRegistrar
from mdm.core.exceptions import DatasetError
from mdm.models.dataset import DatasetInfo
from mdm.models.enums import ProblemType, ColumnType, FileType


class TestDatasetRegistrar:
    """Test cases for DatasetRegistrar."""

    @pytest.fixture
    def mock_manager(self):
        """Create mock DatasetManager."""
        manager = Mock()
        manager.dataset_exists.return_value = False
        manager.save_dataset.return_value = None
        return manager

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.database.default_backend = "sqlite"
        config.features.enable_at_registration = True
        config.performance.batch_size = 1000
        config.performance.max_concurrent_operations = 4
        return config

    @pytest.fixture
    def mock_feature_generator(self):
        """Create mock FeatureGenerator."""
        generator = Mock()
        generator.generate_features.return_value = {}
        return generator

    @pytest.fixture
    def registrar(self, mock_manager, mock_config, mock_feature_generator):
        """Create DatasetRegistrar instance."""
        with patch('mdm.config.get_config_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.config = mock_config
            mock_config_manager.base_path = Path("/test")
            mock_get_config.return_value = mock_config_manager
            
            with patch('mdm.dataset.registrar.FeatureGenerator', return_value=mock_feature_generator):
                registrar = DatasetRegistrar(manager=mock_manager)
                registrar.feature_generator = mock_feature_generator
                return registrar

    def test_register_simple_dataset(self, registrar, mock_manager):
        """Test basic dataset registration."""
        # Arrange
        path = Path("/data/test_dataset")
        
        with patch.object(registrar, '_validate_name', return_value="test_dataset"):
            with patch.object(registrar, '_validate_path', return_value=path):
                with patch.object(registrar, '_auto_detect', return_value={}):
                    with patch.object(registrar, '_discover_files', return_value={'train': Path("/data/train.csv")}):
                        with patch.object(registrar, '_detect_target_and_ids', return_value=('target', ['id'])):
                            with patch.object(registrar, '_create_storage', return_value=Mock()) as mock_storage:
                                with patch.object(registrar, '_load_data') as mock_load:
                                    with patch.object(registrar, '_detect_column_types', return_value={}):
                                        with patch.object(registrar, '_generate_features', return_value={}):
                                            with patch.object(registrar, '_compute_statistics', return_value={}):
                                                with patch.object(registrar, '_save_configuration') as mock_save:
                                                    # Act
                                                    result = registrar.register(
                                                        name="test_dataset",
                                                        path=path,
                                                        description="Test dataset"
                                                    )

        # Assert
        assert isinstance(result, DatasetInfo)
        assert result.name == "test_dataset"
        mock_manager.dataset_exists.assert_called_once_with("test_dataset")
        mock_save.assert_called_once()

    def test_register_dataset_already_exists(self, registrar, mock_manager):
        """Test error when dataset already exists."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        
        # Act & Assert
        with pytest.raises(DatasetError, match="already exists"):
            registrar.register("existing_dataset", Path("/data"))

    def test_register_dataset_with_force(self, registrar, mock_manager):
        """Test registration with force flag removes existing dataset."""
        # Arrange
        mock_manager.dataset_exists.return_value = True
        path = Path("/data/test_dataset")
        
        with patch('mdm.dataset.operations.RemoveOperation') as mock_remove_class:
            mock_remove_op = Mock()
            mock_remove_class.return_value = mock_remove_op
            
            with patch.object(registrar, '_validate_name', return_value="test_dataset"):
                with patch.object(registrar, '_validate_path', return_value=path):
                    with patch.object(registrar, '_auto_detect', return_value={}):
                        with patch.object(registrar, '_discover_files', return_value={'train': Path("/data/train.csv")}):
                            with patch.object(registrar, '_complete_registration', return_value=Mock()):
                                # Act
                                registrar.register(
                                    name="test_dataset",
                                    path=path,
                                    force=True
                                )

        # Assert
        mock_remove_op.execute.assert_called_once_with("test_dataset", force=True, dry_run=False)

    def test_validate_name(self, registrar):
        """Test name validation."""
        # Test valid names
        assert registrar._validate_name("valid_name") == "valid_name"
        assert registrar._validate_name("Name-123") == "name-123"
        assert registrar._validate_name("UPPERCASE") == "uppercase"
        
        # Test invalid names
        with pytest.raises(DatasetError, match="at least 3 characters"):
            registrar._validate_name("ab")
        
        with pytest.raises(DatasetError, match="Invalid characters"):
            registrar._validate_name("name with spaces")
        
        with pytest.raises(DatasetError, match="Invalid characters"):
            registrar._validate_name("name@special")

    def test_validate_path(self, registrar):
        """Test path validation."""
        # Test existing directory
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.is_dir', return_value=True):
                result = registrar._validate_path("/existing/dir")
                assert result == Path("/existing/dir")
        
        # Test existing file
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.is_dir', return_value=False):
                result = registrar._validate_path("/existing/file.csv")
                assert result == Path("/existing/file.csv")
        
        # Test non-existing path
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(DatasetError, match="does not exist"):
                registrar._validate_path("/non/existing")

    @patch('mdm.dataset.registrar.detect_kaggle_structure')
    @patch('mdm.dataset.registrar.extract_target_from_sample_submission')
    @patch('mdm.dataset.registrar.infer_problem_type')
    def test_auto_detect_kaggle(self, mock_infer, mock_extract, mock_detect_kaggle, registrar):
        """Test auto-detection for Kaggle datasets."""
        # Arrange
        path = Path("/data/kaggle")
        mock_detect_kaggle.return_value = {
            'is_kaggle': True,
            'train_file': Path("/data/kaggle/train.csv"),
            'test_file': Path("/data/kaggle/test.csv"),
            'submission_file': Path("/data/kaggle/sample_submission.csv")
        }
        mock_extract.return_value = "target"
        mock_infer.return_value = ProblemType.CLASSIFICATION
        
        # Act
        result = registrar._auto_detect(path)
        
        # Assert
        assert result['structure'] == 'kaggle'
        assert result['target_column'] == 'target'
        assert result['problem_type'] == ProblemType.CLASSIFICATION
        assert result['files'] == {
            'train': Path("/data/kaggle/train.csv"),
            'test': Path("/data/kaggle/test.csv"),
            'submission': Path("/data/kaggle/sample_submission.csv")
        }

    @patch('mdm.dataset.registrar.discover_data_files')
    def test_discover_files(self, mock_discover, registrar):
        """Test file discovery."""
        # Arrange
        path = Path("/data")
        mock_discover.return_value = [
            Path("/data/train.csv"),
            Path("/data/test.csv")
        ]
        
        # Act
        result = registrar._discover_files(path, {})
        
        # Assert
        assert 'train' in result
        assert 'test' in result
        assert result['train'] == Path("/data/train.csv")
        assert result['test'] == Path("/data/test.csv")

    @patch('mdm.dataset.registrar.detect_id_columns')
    def test_detect_target_and_ids(self, mock_detect_ids, registrar):
        """Test target and ID column detection."""
        # Arrange
        files = {'train': Path("/data/train.csv")}
        sample_df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature1': [10, 20, 30],
            'target': [0, 1, 0]
        })
        
        mock_detect_ids.return_value = ['id']
        
        with patch('pandas.read_csv', return_value=sample_df):
            # Act
            target, ids = registrar._detect_target_and_ids(
                files, 
                target_column='target',
                id_columns=None
            )
        
        # Assert
        assert target == 'target'
        assert ids == ['id']

    def test_create_storage(self, registrar):
        """Test storage backend creation."""
        # Arrange
        with patch('mdm.dataset.registrar.BackendFactory') as mock_factory:
            mock_backend = Mock()
            mock_factory.create.return_value = mock_backend
            
            # Act
            result = registrar._create_storage("test_dataset")
            
            # Assert
            mock_factory.create.assert_called_once_with(
                "sqlite",
                {"type": "sqlite", "dataset_name": "test_dataset"}
            )
            assert result == mock_backend

    @patch('pandas.read_csv')
    def test_load_data_batch_processing(self, mock_read_csv, registrar):
        """Test data loading with batch processing."""
        # Arrange
        files = {
            'train': Path("/data/train.csv"),
            'test': Path("/data/test.csv")
        }
        backend = Mock()
        
        # Create sample data
        train_df = pd.DataFrame({
            'id': range(2500),
            'feature': range(2500),
            'target': [0, 1] * 1250
        })
        test_df = pd.DataFrame({
            'id': range(1000),
            'feature': range(1000)
        })
        
        mock_read_csv.side_effect = [train_df, test_df]
        
        # Act
        with patch('mdm.dataset.registrar.Progress'):
            result = registrar._load_data(
                files,
                backend,
                target_column='target',
                id_columns=['id']
            )
        
        # Assert
        assert result['train']['rows'] == 2500
        assert result['train']['columns'] == 3
        assert result['test']['rows'] == 1000
        assert result['test']['columns'] == 2
        
        # Check batch processing was used (3 batches for train, 1 for test)
        assert backend.create_table.call_count == 4

    def test_detect_column_types(self, registrar):
        """Test column type detection."""
        # Arrange
        backend = Mock()
        tables = {'train': 'train_table'}
        
        sample_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'date_col': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'bool_col': [True, False, True]
        })
        
        backend.read_table.return_value = sample_df
        
        # Act
        with patch('mdm.dataset.registrar.ProfileReport') as mock_profile:
            mock_report = Mock()
            mock_report.to_json.return_value = '{"variables": {}}'
            mock_profile.return_value = mock_report
            
            result = registrar._detect_column_types(backend, tables, 'target', ['id'])
        
        # Assert
        assert 'train' in result
        mock_profile.assert_called_once()

    def test_save_configuration(self, registrar, mock_manager):
        """Test configuration saving."""
        # Arrange
        dataset_info = DatasetInfo(
            name="test_dataset",
            problem_type="binary_classification",
            target_column="target",
            id_columns=["id"],
            tables={"train": "train_table"},
            shape=(1000, 10),
            database={"backend": "sqlite"}
        )
        
        # Act
        registrar._save_configuration(dataset_info)
        
        # Assert
        mock_manager.save_dataset.assert_called_once_with(dataset_info)

    def test_complete_registration_flow(self, registrar):
        """Test complete registration flow with mocked steps."""
        # Arrange
        path = Path("/data/test")
        
        # Mock all internal methods
        with patch.multiple(
            registrar,
            _validate_name=Mock(return_value="test_dataset"),
            _validate_path=Mock(return_value=path),
            _auto_detect=Mock(return_value={'problem_type': ProblemType.REGRESSION}),
            _discover_files=Mock(return_value={'train': Path("/data/train.csv")}),
            _detect_target_and_ids=Mock(return_value=('price', ['id'])),
            _create_storage=Mock(return_value=Mock()),
            _load_data=Mock(return_value={'train': {'rows': 1000, 'columns': 10}}),
            _detect_column_types=Mock(return_value={}),
            _generate_features=Mock(return_value={}),
            _compute_statistics=Mock(return_value={'row_count': 1000}),
            _save_configuration=Mock()
        ):
            # Act
            result = registrar.register(
                name="test_dataset",
                path=path,
                tags=["test", "regression"]
            )
        
        # Assert
        assert result.name == "test_dataset"
        assert result.problem_type == "regression"
        assert result.target_column == "price"
        assert result.tags == ["test", "regression"]