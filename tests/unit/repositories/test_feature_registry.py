"""Unit tests for FeatureRegistry."""

import pytest
from unittest.mock import Mock, patch

from mdm.features.registry import FeatureRegistry, feature_registry
from mdm.features.base import GenericFeatureOperation
from mdm.features.generic import (
    CategoricalFeatures,
    StatisticalFeatures,
    TemporalFeatures,
    TextFeatures,
)
from mdm.models.enums import ColumnType


class MockFeatureOperation(GenericFeatureOperation):
    """Mock feature operation for testing."""
    
    def get_applicable_columns(self, df):
        """Mock implementation."""
        return []
    
    def _generate_column_features(self, df, column, **kwargs):
        """Mock implementation."""
        return {}


class TestFeatureRegistry:
    """Test cases for FeatureRegistry."""

    def test_init_default_transformers(self):
        """Test registry initialization with default transformers."""
        registry = FeatureRegistry()
        
        # Check default transformers are registered
        assert ColumnType.DATETIME in registry._transformers
        assert TemporalFeatures in registry._transformers[ColumnType.DATETIME]
        
        assert ColumnType.CATEGORICAL in registry._transformers
        assert CategoricalFeatures in registry._transformers[ColumnType.CATEGORICAL]
        
        assert ColumnType.NUMERIC in registry._transformers
        assert StatisticalFeatures in registry._transformers[ColumnType.NUMERIC]
        
        assert ColumnType.TEXT in registry._transformers
        assert TextFeatures in registry._transformers[ColumnType.TEXT]

    def test_get_transformers_existing_type(self):
        """Test getting transformers for existing column type."""
        registry = FeatureRegistry()
        
        # Get transformers for datetime
        transformers = registry.get_transformers(ColumnType.DATETIME)
        
        assert len(transformers) == 1
        assert isinstance(transformers[0], TemporalFeatures)
        
        # Get again - should return same instance
        transformers2 = registry.get_transformers(ColumnType.DATETIME)
        assert transformers[0] is transformers2[0]

    def test_get_transformers_nonexistent_type(self):
        """Test getting transformers for non-existent column type."""
        registry = FeatureRegistry()
        
        # Create a mock column type not in defaults
        mock_type = Mock(spec=ColumnType)
        
        transformers = registry.get_transformers(mock_type)
        assert transformers == []

    def test_register_transformer_new_type(self):
        """Test registering transformer for new column type."""
        registry = FeatureRegistry()
        
        # Create a mock column type
        mock_type = Mock(spec=ColumnType)
        mock_type.value = "custom"
        
        # Register transformer
        with patch('mdm.features.registry.logger') as mock_logger:
            registry.register_transformer(mock_type, MockFeatureOperation)
        
        # Check it was registered
        assert mock_type in registry._transformers
        assert MockFeatureOperation in registry._transformers[mock_type]
        mock_logger.info.assert_called_once()

    def test_register_transformer_existing_type(self):
        """Test registering additional transformer for existing type."""
        registry = FeatureRegistry()
        
        # Register additional transformer for NUMERIC
        registry.register_transformer(ColumnType.NUMERIC, MockFeatureOperation)
        
        # Check both transformers are registered
        assert StatisticalFeatures in registry._transformers[ColumnType.NUMERIC]
        assert MockFeatureOperation in registry._transformers[ColumnType.NUMERIC]
        assert len(registry._transformers[ColumnType.NUMERIC]) == 2

    def test_register_transformer_duplicate(self):
        """Test registering duplicate transformer (should not add twice)."""
        registry = FeatureRegistry()
        
        # Register transformer
        registry.register_transformer(ColumnType.NUMERIC, MockFeatureOperation)
        initial_count = len(registry._transformers[ColumnType.NUMERIC])
        
        # Register same transformer again
        registry.register_transformer(ColumnType.NUMERIC, MockFeatureOperation)
        
        # Should not be added twice
        assert len(registry._transformers[ColumnType.NUMERIC]) == initial_count

    def test_get_all_transformers(self):
        """Test getting all registered transformers."""
        registry = FeatureRegistry()
        
        # Get all transformers
        all_transformers = registry.get_all_transformers()
        
        # Should have at least 4 unique transformers (one for each default type)
        # Plus any global transformers
        assert len(all_transformers) >= 4
        
        # Check that default transformers are present
        transformer_types = {type(t) for t in all_transformers}
        expected_types = {
            TemporalFeatures,
            CategoricalFeatures,
            StatisticalFeatures,
            TextFeatures
        }
        # Check that all expected types are present (may have additional global adapters)
        assert expected_types.issubset(transformer_types)

    def test_get_all_transformers_with_custom(self):
        """Test getting all transformers including custom ones."""
        registry = FeatureRegistry()
        
        # Get initial count
        initial_transformers = registry.get_all_transformers()
        initial_count = len(initial_transformers)
        
        # Register custom transformer
        registry.register_transformer(ColumnType.NUMERIC, MockFeatureOperation)
        
        # Get all transformers
        all_transformers = registry.get_all_transformers()
        
        # Should have one more transformer than before
        assert len(all_transformers) == initial_count + 1
        
        # Check mock transformer is included
        transformer_types = {type(t) for t in all_transformers}
        assert MockFeatureOperation in transformer_types

    def test_transformer_instance_reuse(self):
        """Test that transformer instances are reused."""
        registry = FeatureRegistry()
        
        # Get transformers multiple times
        trans1 = registry.get_transformers(ColumnType.NUMERIC)
        trans2 = registry.get_transformers(ColumnType.NUMERIC)
        all_trans = registry.get_all_transformers()
        
        # Should be same instance
        assert trans1[0] is trans2[0]
        
        # Same instance should be in all_transformers
        assert trans1[0] in all_trans

    def test_global_registry_instance(self):
        """Test that global registry instance works correctly."""
        # Check it's a FeatureRegistry instance
        assert isinstance(feature_registry, FeatureRegistry)
        
        # Check it has default transformers
        transformers = feature_registry.get_transformers(ColumnType.TEXT)
        assert len(transformers) > 0
        assert isinstance(transformers[0], TextFeatures)

    def test_register_multiple_transformers_at_once(self):
        """Test registering multiple transformers for a type."""
        registry = FeatureRegistry()
        
        # Create mock types
        class MockTransformer1(MockFeatureOperation):
            pass
        
        class MockTransformer2(MockFeatureOperation):
            pass
        
        # Register both
        registry.register_transformer(ColumnType.NUMERIC, MockTransformer1)
        registry.register_transformer(ColumnType.NUMERIC, MockTransformer2)
        
        # Get transformers
        transformers = registry.get_transformers(ColumnType.NUMERIC)
        transformer_types = {type(t) for t in transformers}
        
        # Check all are present
        assert StatisticalFeatures in transformer_types
        assert MockTransformer1 in transformer_types
        assert MockTransformer2 in transformer_types

    def test_empty_registry(self):
        """Test behavior with empty registry."""
        registry = FeatureRegistry()
        registry._transformers = {}  # Clear all transformers
        registry._global_transformer_instances = []  # Clear global transformers too
        
        # Get transformers for any type
        transformers = registry.get_transformers(ColumnType.NUMERIC)
        assert transformers == []
        
        # Get all transformers
        all_transformers = registry.get_all_transformers()
        assert all_transformers == []