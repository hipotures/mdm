"""Unit tests for FeatureEngine."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import pandas as pd

from mdm.features.engine import FeatureEngine


class TestFeatureEngine:
    """Test cases for FeatureEngine."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.features.enable_at_registration = True
        config.performance.batch_size = 1000
        return config

    @pytest.fixture
    def mock_signal_detector(self):
        """Mock signal detector."""
        detector = Mock()
        detector.detect_signals.return_value = {}
        return detector

    @pytest.fixture
    def feature_engine(self, mock_config, mock_signal_detector):
        """Create FeatureEngine instance."""
        with patch('mdm.features.engine.get_config', return_value=mock_config):
            with patch('mdm.features.engine.SignalDetector', return_value=mock_signal_detector):
                engine = FeatureEngine()
                engine.signal_detector = mock_signal_detector
                return engine

    @pytest.fixture
    def mock_backend(self):
        """Mock storage backend."""
        backend = Mock()
        return backend

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame({
            'id': range(100),
            'feature1': range(100),
            'feature2': ['A', 'B'] * 50,
            'target': [0, 1] * 50
        })

    def test_generate_features_single_table(self, feature_engine, mock_backend, sample_dataframe):
        """Test feature generation for single table."""
        # Arrange
        tables = {'train': 'train_table'}
        
        with patch.object(feature_engine, '_generate_table_features') as mock_generate:
            mock_generate.return_value = {
                'feature_table': 'train_features',
                'feature_count': 10,
                'original_columns': 4
            }
            
            # Act
            result = feature_engine.generate_features(
                dataset_name="test_dataset",
                backend=mock_backend,
                tables=tables,
                target_column="target",
                id_columns=["id"]
            )
        
        # Assert
        assert 'train' in result
        assert result['train']['feature_table'] == 'train_features'
        mock_generate.assert_called_once_with(
            dataset_name="test_dataset",
            backend=mock_backend,
            table_name="train_table",
            table_type="train",
            target_column="target",
            id_columns=["id"]
        )

    def test_generate_features_multiple_tables(self, feature_engine, mock_backend):
        """Test feature generation for multiple tables."""
        # Arrange
        tables = {
            'train': 'train_table',
            'test': 'test_table',
            'submission': 'submission_table'  # Should be skipped
        }
        
        with patch.object(feature_engine, '_generate_table_features') as mock_generate:
            mock_generate.return_value = {'feature_table': 'features', 'feature_count': 5}
            
            # Act
            result = feature_engine.generate_features(
                dataset_name="test_dataset",
                backend=mock_backend,
                tables=tables
            )
        
        # Assert
        assert len(result) == 2  # train and test only
        assert 'train' in result
        assert 'test' in result
        assert 'submission' not in result
        assert mock_generate.call_count == 2

    def test_generate_features_error_handling(self, feature_engine, mock_backend):
        """Test error handling during feature generation."""
        # Arrange
        tables = {'train': 'train_table', 'test': 'test_table'}
        
        with patch.object(feature_engine, '_generate_table_features') as mock_generate:
            # First call succeeds, second fails
            mock_generate.side_effect = [
                {'feature_table': 'train_features', 'feature_count': 10},
                Exception("Feature generation failed")
            ]
            
            with patch('mdm.features.engine.logger') as mock_logger:
                # Act
                result = feature_engine.generate_features(
                    dataset_name="test_dataset",
                    backend=mock_backend,
                    tables=tables
                )
        
        # Assert
        assert len(result) == 1  # Only train succeeded
        assert 'train' in result
        assert 'test' not in result
        mock_logger.error.assert_called_once()

    @patch('mdm.features.engine.Progress')
    def test_generate_features_progress_tracking(self, mock_progress_class, feature_engine, mock_backend):
        """Test progress tracking during feature generation."""
        # Arrange
        mock_progress = MagicMock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress
        
        tables = {'train': 'train_table'}
        
        with patch.object(feature_engine, '_generate_table_features', return_value={}):
            # Act
            feature_engine.generate_features(
                dataset_name="test_dataset",
                backend=mock_backend,
                tables=tables
            )
        
        # Assert
        mock_progress.add_task.assert_called_once()
        mock_progress.update.assert_called()

    def test_generate_table_features_basic(self, feature_engine, mock_backend, sample_dataframe):
        """Test table feature generation."""
        # Arrange
        mock_backend.read_table.return_value = sample_dataframe
        
        # Mock transformers
        mock_transformers = {
            'temporal': Mock(),
            'categorical': Mock(),
            'statistical': Mock(),
            'text': Mock()
        }
        
        for name, transformer in mock_transformers.items():
            transformer.can_handle.return_value = name in ['categorical', 'statistical']
            transformer.generate_features.return_value = pd.DataFrame({
                f'{name}_feature1': range(100),
                f'{name}_feature2': range(100)
            })
        
        feature_engine.generic_transformers = mock_transformers
        
        with patch.object(feature_engine, '_load_custom_transformer', return_value=None):
            with patch.object(feature_engine, '_save_feature_table') as mock_save:
                # Act
                result = feature_engine._generate_table_features(
                    dataset_name="test_dataset",
                    backend=mock_backend,
                    table_name="train_table",
                    table_type="train",
                    target_column="target",
                    id_columns=["id"]
                )
        
        # Assert
        assert 'feature_table' in result
        assert 'feature_count' in result
        assert 'original_columns' in result
        mock_save.assert_called_once()

    def test_generate_table_features_batch_processing(self, feature_engine, mock_backend):
        """Test batch processing for large tables."""
        # Arrange
        # Create large DataFrame that requires batching
        large_df = pd.DataFrame({
            'id': range(2500),
            'feature': range(2500)
        })
        mock_backend.read_table.return_value = large_df
        mock_backend.get_row_count.return_value = 2500
        
        feature_engine.config.performance.batch_size = 1000
        
        # Mock transformers
        mock_transformer = Mock()
        mock_transformer.can_handle.return_value = True
        mock_transformer.generate_features.return_value = pd.DataFrame()
        
        feature_engine.generic_transformers = {'statistical': mock_transformer}
        
        with patch.object(feature_engine, '_load_custom_transformer', return_value=None):
            with patch.object(feature_engine, '_save_feature_table'):
                # Act
                result = feature_engine._generate_table_features(
                    dataset_name="test_dataset",
                    backend=mock_backend,
                    table_name="train_table",
                    table_type="train"
                )
        
        # Assert
        # Should process in 3 batches (1000, 1000, 500)
        assert mock_transformer.generate_features.call_count >= 2

    def test_load_custom_transformer_exists(self, feature_engine):
        """Test loading existing custom transformer."""
        # Arrange
        dataset_path = Path("/test/config/custom_features/test_dataset.py")
        
        mock_spec = Mock()
        mock_module = Mock()
        mock_transformer = Mock()
        mock_module.CustomTransformer = Mock(return_value=mock_transformer)
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('importlib.util.spec_from_file_location', return_value=mock_spec):
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    # Act
                    result = feature_engine._load_custom_transformer("test_dataset")
        
        # Assert
        assert result == mock_transformer

    def test_load_custom_transformer_not_exists(self, feature_engine):
        """Test loading custom transformer when file doesn't exist."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=False):
            # Act
            result = feature_engine._load_custom_transformer("test_dataset")
        
        # Assert
        assert result is None

    def test_save_feature_table(self, feature_engine, mock_backend):
        """Test saving feature table."""
        # Arrange
        feature_df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature1': [10, 20, 30],
            'feature2': ['A', 'B', 'C']
        })
        
        # Act
        feature_engine._save_feature_table(
            backend=mock_backend,
            feature_df=feature_df,
            table_name="train_features",
            id_columns=["id"]
        )
        
        # Assert
        mock_backend.create_table.assert_called_once_with(
            table_name="train_features",
            df=feature_df,
            if_exists="replace"
        )

    def test_generate_table_features_no_features(self, feature_engine, mock_backend):
        """Test when no features are generated."""
        # Arrange
        df = pd.DataFrame({'id': [1, 2, 3], 'target': [0, 1, 0]})
        mock_backend.read_table.return_value = df
        
        # All transformers say they can't handle
        for transformer in feature_engine.generic_transformers.values():
            transformer.can_handle.return_value = False
        
        with patch.object(feature_engine, '_load_custom_transformer', return_value=None):
            # Act
            result = feature_engine._generate_table_features(
                dataset_name="test_dataset",
                backend=mock_backend,
                table_name="train_table",
                table_type="train",
                target_column="target",
                id_columns=["id"]
            )
        
        # Assert
        assert result['feature_count'] == 0