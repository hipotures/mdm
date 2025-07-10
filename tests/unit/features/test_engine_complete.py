"""Comprehensive unit tests for feature engineering engine."""

import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

import pandas as pd
import pytest

from mdm.features.engine import FeatureEngine


class TestFeatureEngineComplete:
    """Comprehensive test coverage for FeatureEngine."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.get.return_value = {"save_intermediate": False}
        return config

    @pytest.fixture
    def mock_backend(self):
        """Create mock storage backend."""
        backend = Mock()
        backend.read_table = Mock()
        backend.write_table = Mock()
        return backend

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'numeric_col': [10.5, 20.3, 15.7, 30.2, 25.1],
            'category_col': ['A', 'B', 'A', 'C', 'B'],
            'text_col': ['hello world', 'test data', 'feature engineering', 'machine learning', 'data science'],
            'date_col': pd.date_range('2024-01-01', periods=5)
        })

    @pytest.fixture
    def engine(self, mock_config):
        """Create FeatureEngine instance."""
        with patch('mdm.features.engine.get_config', return_value=mock_config):
            return FeatureEngine()

    def test_init(self, mock_config):
        """Test FeatureEngine initialization."""
        with patch('mdm.features.engine.get_config', return_value=mock_config):
            engine = FeatureEngine()
            
            assert engine.config == mock_config
            assert hasattr(engine, 'signal_detector')
            assert hasattr(engine, 'generic_transformers')
            assert 'temporal' in engine.generic_transformers
            assert 'categorical' in engine.generic_transformers
            assert 'statistical' in engine.generic_transformers
            assert 'text' in engine.generic_transformers

    def test_generate_features_single_table(self, engine, mock_backend, sample_df):
        """Test feature generation for a single table."""
        tables = {'train': 'dataset_train'}
        mock_backend.read_table.return_value = sample_df
        
        # Mock the individual feature generation methods
        with patch.object(engine, '_generate_table_features') as mock_gen:
            mock_gen.return_value = {
                'original_shape': (5, 5),
                'feature_shape': (5, 10),
                'generic_features': 20,
                'custom_features': 5,
                'filtered_features': 15,
                'discarded_features': 10,
                'processing_time': 0.5,
                'feature_table': 'dataset_train_features'
            }
            
            result = engine.generate_features(
                dataset_name='test_dataset',
                backend=mock_backend,
                tables=tables,
                target_column='target',
                id_columns=['id']
            )
            
            assert 'train' in result
            assert result['train']['generic_features'] == 20
            assert result['train']['custom_features'] == 5
            mock_gen.assert_called_once()

    def test_generate_features_multiple_tables(self, engine, mock_backend, sample_df):
        """Test feature generation for multiple tables."""
        tables = {
            'train': 'dataset_train',
            'test': 'dataset_test',
            'submission': 'dataset_submission'
        }
        mock_backend.read_table.return_value = sample_df
        
        with patch.object(engine, '_generate_table_features') as mock_gen:
            mock_gen.return_value = {
                'original_shape': (5, 5),
                'feature_shape': (5, 10),
                'generic_features': 20,
                'processing_time': 0.5
            }
            
            result = engine.generate_features(
                dataset_name='test_dataset',
                backend=mock_backend,
                tables=tables
            )
            
            # Should process train and test, but skip submission
            assert 'train' in result
            assert 'test' in result
            assert 'submission' not in result
            assert mock_gen.call_count == 2

    def test_generate_features_error_handling(self, engine, mock_backend):
        """Test error handling during feature generation."""
        tables = {'train': 'dataset_train'}
        
        with patch.object(engine, '_generate_table_features') as mock_gen:
            mock_gen.side_effect = Exception("Feature generation failed")
            
            result = engine.generate_features(
                dataset_name='test_dataset',
                backend=mock_backend,
                tables=tables
            )
            
            # Should return empty dict on error
            assert result == {}

    def test_generate_table_features_success(self, engine, mock_backend, sample_df):
        """Test successful table feature generation."""
        mock_backend.read_table.return_value = sample_df
        
        # Mock the feature generation methods
        generic_features = {
            'numeric_col_mean': pd.Series([15.0] * 5),
            'category_col_encoded': pd.Series([0, 1, 0, 2, 1])
        }
        custom_features = {
            'custom_feature_1': pd.Series([1, 2, 3, 4, 5])
        }
        filtered_features = {**generic_features}  # Assume one custom feature was filtered
        
        with patch.object(engine, '_generate_generic_features', return_value=generic_features):
            with patch.object(engine, '_generate_custom_features', return_value=custom_features):
                with patch.object(engine.signal_detector, 'filter_features', return_value=filtered_features):
                    with patch.object(engine, '_create_feature_dataframe') as mock_create:
                        mock_create.return_value = pd.DataFrame({
                            'id': [1, 2, 3, 4, 5],
                            **filtered_features
                        })
                        
                        result = engine._generate_table_features(
                            dataset_name='test_dataset',
                            backend=mock_backend,
                            table_name='dataset_train',
                            table_type='train',
                            target_column='target',
                            id_columns=['id']
                        )
                        
                        assert result['original_shape'] == (5, 5)
                        assert result['generic_features'] == 2
                        assert result['custom_features'] == 1
                        assert result['filtered_features'] == 2
                        assert result['discarded_features'] == 1
                        assert 'processing_time' in result
                        assert result['feature_table'] == 'dataset_train_features'
                        
                        # Verify table was written
                        mock_backend.write_table.assert_called_once()

    def test_generate_table_features_with_intermediate_save(self, engine, mock_backend, sample_df):
        """Test table feature generation with intermediate saving."""
        mock_backend.read_table.return_value = sample_df
        
        # Enable intermediate saving
        engine.config.get.return_value = {"save_intermediate": True}
        
        generic_features = {'feat1': pd.Series([1, 2, 3, 4, 5])}
        custom_features = {'feat2': pd.Series([5, 4, 3, 2, 1])}
        
        with patch.object(engine, '_generate_generic_features', return_value=generic_features):
            with patch.object(engine, '_generate_custom_features', return_value=custom_features):
                with patch.object(engine.signal_detector, 'filter_features', return_value={**generic_features, **custom_features}):
                    with patch.object(engine, '_create_feature_dataframe', return_value=sample_df):
                        
                        engine._generate_table_features(
                            dataset_name='test_dataset',
                            backend=mock_backend,
                            table_name='dataset_train',
                            table_type='train'
                        )
                        
                        # Should save 3 tables: generic, custom, and final
                        assert mock_backend.write_table.call_count == 3
                        call_args = [call[0] for call in mock_backend.write_table.call_args_list]
                        assert 'dataset_train_generic' in call_args[0]
                        assert 'dataset_train_custom' in call_args[1]
                        assert 'dataset_train_features' in call_args[2]

    def test_generate_generic_features(self, engine, sample_df):
        """Test generic feature generation."""
        # Mock transformers
        for transformer_name, transformer in engine.generic_transformers.items():
            if transformer_name == 'temporal':
                transformer.get_applicable_columns = Mock(return_value=['date_col'])
                transformer.generate_features = Mock(return_value={
                    'date_col_year': pd.Series([2024] * 5),
                    'date_col_month': pd.Series([1] * 5)
                })
            elif transformer_name == 'categorical':
                transformer.get_applicable_columns = Mock(return_value=['category_col'])
                transformer.generate_features = Mock(return_value={
                    'category_col_encoded': pd.Series([0, 1, 0, 2, 1])
                })
            elif transformer_name == 'statistical':
                transformer.get_applicable_columns = Mock(return_value=['numeric_col'])
                transformer.generate_features = Mock(return_value={
                    'numeric_col_zscore': pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
                })
            else:  # text
                transformer.get_applicable_columns = Mock(return_value=['text_col'])
                transformer.generate_features = Mock(return_value={
                    'text_col_length': pd.Series([11, 9, 19, 16, 12])
                })
        
        features = engine._generate_generic_features(sample_df)
        
        assert len(features) == 5  # 2 temporal + 1 categorical + 1 statistical + 1 text
        assert 'date_col_year' in features
        assert 'category_col_encoded' in features
        assert 'numeric_col_zscore' in features
        assert 'text_col_length' in features

    def test_generate_generic_features_no_applicable_columns(self, engine):
        """Test generic feature generation with no applicable columns."""
        df = pd.DataFrame({'id': [1, 2, 3]})
        
        # Mock all transformers to return no applicable columns
        for transformer in engine.generic_transformers.values():
            transformer.get_applicable_columns = Mock(return_value=[])
        
        features = engine._generate_generic_features(df)
        
        assert features == {}

    def test_generate_generic_features_error_handling(self, engine, sample_df):
        """Test error handling in generic feature generation."""
        # Make one transformer fail
        engine.generic_transformers['temporal'].get_applicable_columns = Mock(
            side_effect=Exception("Transformer error")
        )
        
        # Others work normally
        engine.generic_transformers['categorical'].get_applicable_columns = Mock(return_value=['category_col'])
        engine.generic_transformers['categorical'].generate_features = Mock(return_value={
            'category_col_encoded': pd.Series([0, 1, 0, 2, 1])
        })
        
        features = engine._generate_generic_features(sample_df)
        
        # Should still get features from working transformers
        assert 'category_col_encoded' in features

    def test_generate_custom_features_no_module(self, engine, sample_df):
        """Test custom feature generation when no module exists."""
        with patch('pathlib.Path.exists', return_value=False):
            features = engine._generate_custom_features('test_dataset', sample_df)
            assert features == {}

    def test_generate_custom_features_success(self, engine, sample_df, tmp_path):
        """Test successful custom feature generation."""
        # Create mock custom features module
        custom_module_content = '''
class CustomFeatureOperations:
    def generate_all_features(self, df):
        return {
            'custom_feature_1': df['numeric_col'] * 2,
            'custom_feature_2': df['numeric_col'] ** 2
        }
'''
        custom_features_dir = tmp_path / ".mdm" / "custom_features"
        custom_features_dir.mkdir(parents=True)
        custom_module_path = custom_features_dir / "test_dataset.py"
        custom_module_path.write_text(custom_module_content)
        
        with patch('pathlib.Path.home', return_value=tmp_path):
            features = engine._generate_custom_features('test_dataset', sample_df)
            
            assert len(features) == 2
            assert 'custom_feature_1' in features
            assert 'custom_feature_2' in features

    def test_generate_custom_features_no_class(self, engine, sample_df, tmp_path):
        """Test custom features module without CustomFeatureOperations class."""
        # Create module without the required class
        custom_module_content = '''
def some_function():
    pass
'''
        custom_features_dir = tmp_path / ".mdm" / "custom_features"
        custom_features_dir.mkdir(parents=True)
        custom_module_path = custom_features_dir / "test_dataset.py"
        custom_module_path.write_text(custom_module_content)
        
        with patch('pathlib.Path.home', return_value=tmp_path):
            features = engine._generate_custom_features('test_dataset', sample_df)
            assert features == {}

    def test_generate_custom_features_error(self, engine, sample_df, tmp_path):
        """Test error handling in custom feature generation."""
        # Create module with error
        custom_module_content = '''
class CustomFeatureOperations:
    def generate_all_features(self, df):
        raise Exception("Custom feature error")
'''
        custom_features_dir = tmp_path / ".mdm" / "custom_features"
        custom_features_dir.mkdir(parents=True)
        custom_module_path = custom_features_dir / "test_dataset.py"
        custom_module_path.write_text(custom_module_content)
        
        with patch('pathlib.Path.home', return_value=tmp_path):
            features = engine._generate_custom_features('test_dataset', sample_df)
            assert features == {}

    def test_create_feature_dataframe_all_components(self, engine, sample_df):
        """Test feature DataFrame creation with all components."""
        features = {
            'new_feature_1': pd.Series([1, 2, 3, 4, 5]),
            'new_feature_2': pd.Series([5, 4, 3, 2, 1])
        }
        preserve_columns = {'id', 'target'}
        
        # Add a fake target column
        sample_df['target'] = [0, 1, 0, 1, 0]
        
        result = engine._create_feature_dataframe(sample_df, features, preserve_columns)
        
        # Should have preserved columns (lowercase) + original columns (lowercase) + new features
        assert 'id' in result.columns
        assert 'target' in result.columns
        assert 'numeric_col' in result.columns
        assert 'new_feature_1' in result.columns
        assert 'new_feature_2' in result.columns
        
        # Check no duplicates
        assert len(result.columns) == len(set(result.columns))

    def test_create_feature_dataframe_empty_features(self, engine, sample_df):
        """Test feature DataFrame creation with no new features."""
        features = {}
        preserve_columns = {'id'}
        
        result = engine._create_feature_dataframe(sample_df, features, preserve_columns)
        
        # Should still have original columns
        assert 'id' in result.columns
        assert 'numeric_col' in result.columns
        assert len(result) == len(sample_df)

    def test_create_feature_dataframe_no_preserve(self, engine, sample_df):
        """Test feature DataFrame creation without preserve columns."""
        features = {
            'new_feature': pd.Series([1, 2, 3, 4, 5])
        }
        preserve_columns = set()
        
        result = engine._create_feature_dataframe(sample_df, features, preserve_columns)
        
        # Should have all original columns + new feature
        assert 'numeric_col' in result.columns
        assert 'new_feature' in result.columns

    def test_create_feature_dataframe_column_case_handling(self, engine):
        """Test that columns are properly lowercased."""
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'NumericCol': [10, 20, 30],
            'CategoryCol': ['A', 'B', 'C']
        })
        
        features = {
            'new_feature': pd.Series([1, 2, 3])
        }
        preserve_columns = {'ID'}
        
        result = engine._create_feature_dataframe(df, features, preserve_columns)
        
        # All columns should be lowercase
        assert 'id' in result.columns
        assert 'numericcol' in result.columns
        assert 'categorycol' in result.columns
        assert 'new_feature' in result.columns
        assert 'ID' not in result.columns

    def test_performance_timing(self, engine, mock_backend, sample_df):
        """Test that timing is properly recorded."""
        mock_backend.read_table.return_value = sample_df
        
        with patch('time.time', side_effect=[1000, 1001]):  # 1 second processing
            with patch.object(engine, '_generate_generic_features', return_value={}):
                with patch.object(engine, '_generate_custom_features', return_value={}):
                    with patch.object(engine.signal_detector, 'filter_features', return_value={}):
                        with patch.object(engine, '_create_feature_dataframe', return_value=sample_df):
                            
                            result = engine._generate_table_features(
                                dataset_name='test_dataset',
                                backend=mock_backend,
                                table_name='dataset_train',
                                table_type='train'
                            )
                            
                            assert result['processing_time'] == 1.0