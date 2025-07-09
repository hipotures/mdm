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
        
        # Mock the signal detector to return the features as-is
        feature_engine.signal_detector.filter_features.return_value = {
            'new_feature1': pd.Series(range(100)),
            'new_feature2': pd.Series(['X'] * 100)
        }
        
        with patch.object(feature_engine, '_generate_generic_features', return_value={}):
            with patch.object(feature_engine, '_generate_custom_features', return_value={}):
                with patch.object(mock_backend, 'write_table') as mock_write:
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
        assert 'feature_shape' in result
        assert 'original_shape' in result
        assert result['filtered_features'] == 2
        # write_table is called once for the final feature table
        assert mock_write.call_count >= 1

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
        
        # Mock signal detector
        feature_engine.signal_detector.filter_features.return_value = {
            'new_feature': pd.Series(range(2500))
        }
        
        with patch.object(feature_engine, '_generate_generic_features', return_value={}):
            with patch.object(feature_engine, '_generate_custom_features', return_value={}):
                with patch.object(mock_backend, 'write_table'):
                    # Act
                    result = feature_engine._generate_table_features(
                        dataset_name="test_dataset",
                        backend=mock_backend,
                        table_name="train_table",
                        table_type="train"
                    )
        
        # Assert
        # Just check that it completed successfully
        assert 'feature_table' in result
        assert result is not None


    def test_generate_table_features_no_features(self, feature_engine, mock_backend):
        """Test when no features are generated."""
        # Arrange
        df = pd.DataFrame({'id': [1, 2, 3], 'target': [0, 1, 0]})
        mock_backend.read_table.return_value = df
        
        # Mock signal detector to return empty features
        feature_engine.signal_detector.filter_features.return_value = {}
        
        with patch.object(feature_engine, '_generate_generic_features', return_value={}):
            with patch.object(feature_engine, '_generate_custom_features', return_value={}):
                with patch.object(mock_backend, 'write_table'):
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
        assert result['filtered_features'] == 0