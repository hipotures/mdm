"""Unit tests for FeatureGenerator."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import pandas as pd

from mdm.features.generator import FeatureGenerator
from mdm.models.enums import ColumnType


class TestFeatureGenerator:
    """Test cases for FeatureGenerator."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.features.enable_at_registration = True
        config.features.min_column_variance = 0.01
        return config

    @pytest.fixture
    def mock_registry(self):
        """Mock feature registry."""
        registry = Mock()
        return registry

    @pytest.fixture
    def feature_generator(self, mock_config, mock_registry):
        """Create FeatureGenerator instance."""
        with patch('mdm.config.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = Path("/test")
            mock_get_config.return_value = mock_manager
            
            with patch('mdm.features.generator.feature_registry', mock_registry):
                generator = FeatureGenerator()
                generator.registry = mock_registry
                return generator

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'numeric_col': [10.5, 20.3, 30.1, 40.7, 50.2],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'text_col': ['hello world', 'test data', 'feature gen', 'sample text', 'mdm test'],
            'target': [0, 1, 0, 1, 0]
        })

    @pytest.fixture
    def column_types(self):
        """Sample column type mapping."""
        return {
            'id': ColumnType.NUMERIC,
            'numeric_col': ColumnType.NUMERIC,
            'categorical_col': ColumnType.CATEGORICAL,
            'text_col': ColumnType.TEXT,
            'target': ColumnType.NUMERIC
        }

    def test_generate_features_basic(self, feature_generator, sample_dataframe, column_types, mock_registry):
        """Test basic feature generation."""
        # Arrange
        mock_transformer = Mock()
        mock_transformer.generate_features.return_value = {
            'numeric_col_mean': [25.36] * 5,
            'numeric_col_std': [15.8] * 5
        }
        mock_registry.get_transformers.return_value = [mock_transformer]
        
        # Act
        result = feature_generator.generate_features(
            df=sample_dataframe,
            dataset_name="test_dataset",
            column_types=column_types,
            target_column="target",
            id_columns=["id"]
        )
        
        # Assert
        assert 'numeric_col_mean' in result.columns
        assert 'numeric_col_std' in result.columns
        assert len(result) == len(sample_dataframe)
        # ID and target should be excluded from feature generation
        assert mock_registry.get_transformers.call_count == 3  # numeric, categorical, text

    def test_generate_features_multiple_transformers(self, feature_generator, sample_dataframe, column_types, mock_registry):
        """Test feature generation with multiple transformers per type."""
        # Arrange
        transformer1 = Mock()
        transformer1.generate_features.return_value = {'feature1': [1] * 5}
        transformer1.__class__.__name__ = "Transformer1"
        
        transformer2 = Mock()
        transformer2.generate_features.return_value = {'feature2': [2] * 5}
        transformer2.__class__.__name__ = "Transformer2"
        
        mock_registry.get_transformers.return_value = [transformer1, transformer2]
        
        # Act
        result = feature_generator.generate_features(
            df=sample_dataframe,
            dataset_name="test_dataset",
            column_types=column_types
        )
        
        # Assert
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns

    def test_generate_features_no_duplicate_columns(self, feature_generator, sample_dataframe, column_types, mock_registry):
        """Test that duplicate feature names are not added."""
        # Arrange
        mock_transformer = Mock()
        mock_transformer.generate_features.return_value = {
            'numeric_col': [99] * 5,  # Same name as original column
            'new_feature': [88] * 5
        }
        mock_registry.get_transformers.return_value = [mock_transformer]
        
        # Act
        result = feature_generator.generate_features(
            df=sample_dataframe,
            dataset_name="test_dataset",
            column_types=column_types
        )
        
        # Assert
        assert result['numeric_col'].equals(sample_dataframe['numeric_col'])  # Original preserved
        assert 'new_feature' in result.columns

    @patch.object(FeatureGenerator, '_load_custom_features')
    def test_generate_features_with_custom(self, mock_load_custom, feature_generator, 
                                         sample_dataframe, column_types, mock_registry):
        """Test feature generation with custom features."""
        # Arrange
        mock_registry.get_transformers.return_value = []
        
        mock_custom = Mock()
        mock_custom.generate_all_features.return_value = {
            'custom_feature1': [100] * 5,
            'custom_feature2': [200] * 5
        }
        mock_load_custom.return_value = mock_custom
        
        # Act
        result = feature_generator.generate_features(
            df=sample_dataframe,
            dataset_name="test_dataset",
            column_types=column_types
        )
        
        # Assert
        assert 'custom_feature1' in result.columns
        assert 'custom_feature2' in result.columns
        mock_custom.generate_all_features.assert_called_once_with(sample_dataframe)

    def test_generate_features_empty_dataframe(self, feature_generator, column_types, mock_registry):
        """Test feature generation with empty DataFrame."""
        # Arrange
        empty_df = pd.DataFrame(columns=['id', 'numeric_col', 'target'])
        mock_registry.get_transformers.return_value = []
        
        # Act
        result = feature_generator.generate_features(
            df=empty_df,
            dataset_name="test_dataset",
            column_types=column_types
        )
        
        # Assert
        assert len(result) == 0
        assert list(result.columns) == list(empty_df.columns)

    def test_generate_features_transformer_error(self, feature_generator, sample_dataframe, 
                                               column_types, mock_registry):
        """Test handling of transformer errors."""
        # Arrange
        mock_transformer = Mock()
        mock_transformer.generate_features.side_effect = Exception("Transform failed")
        mock_transformer.__class__.__name__ = "FailingTransformer"
        mock_registry.get_transformers.return_value = [mock_transformer]
        
        # Act - should not raise, just skip the failing transformer
        result = feature_generator.generate_features(
            df=sample_dataframe,
            dataset_name="test_dataset",
            column_types=column_types
        )
        
        # Assert - original columns preserved
        assert set(result.columns) == set(sample_dataframe.columns)

    def test_load_custom_features_exists(self, feature_generator):
        """Test loading existing custom features."""
        # Arrange
        custom_path = feature_generator.base_path / "config" / "custom_features" / "test_dataset.py"
        
        mock_spec = Mock()
        mock_module = Mock()
        mock_features_class = Mock()
        mock_module.CustomFeatures = mock_features_class
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('importlib.util.spec_from_file_location', return_value=mock_spec):
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    # Act
                    result = feature_generator._load_custom_features("test_dataset")
        
        # Assert
        assert result == mock_features_class.return_value

    def test_load_custom_features_not_exists(self, feature_generator):
        """Test loading custom features when file doesn't exist."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=False):
            # Act
            result = feature_generator._load_custom_features("test_dataset")
        
        # Assert
        assert result is None

    def test_load_custom_features_import_error(self, feature_generator):
        """Test handling import errors for custom features."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=True):
            with patch('importlib.util.spec_from_file_location', side_effect=ImportError("Import failed")):
                with patch('mdm.features.generator.logger') as mock_logger:
                    # Act
                    result = feature_generator._load_custom_features("test_dataset")
        
        # Assert
        assert result is None
        mock_logger.error.assert_called_once()

    def test_generate_features_performance_logging(self, feature_generator, sample_dataframe, 
                                                 column_types, mock_registry):
        """Test performance logging during feature generation."""
        # Arrange
        mock_registry.get_transformers.return_value = []
        
        with patch('mdm.features.generator.logger') as mock_logger:
            # Act
            result = feature_generator.generate_features(
                df=sample_dataframe,
                dataset_name="test_dataset",
                column_types=column_types
            )
        
        # Assert
        # Check that performance info was logged
        info_calls = [call for call in mock_logger.info.call_args_list]
        assert any("Generating features" in str(call) for call in info_calls)
        assert any("Feature generation complete" in str(call) for call in info_calls)