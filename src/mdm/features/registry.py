"""Feature transformer registry."""


from loguru import logger

from mdm.features.base import GenericFeatureOperation
from mdm.features.generic import (
    CategoricalFeatures,
    StatisticalFeatures,
    TemporalFeatures,
    TextFeatures,
)
from mdm.features.generic.adapters import create_global_adapters
from mdm.models.enums import ColumnType


class FeatureRegistry:
    """Registry for feature transformers."""

    def __init__(self):
        """Initialize feature registry."""
        self._transformers: dict[ColumnType, list[type[GenericFeatureOperation]]] = {
            ColumnType.DATETIME: [TemporalFeatures],
            ColumnType.CATEGORICAL: [CategoricalFeatures],
            ColumnType.NUMERIC: [StatisticalFeatures],
            ColumnType.TEXT: [TextFeatures],
        }
        
        # Global transformers that apply to entire dataset
        # These are instances, not classes, created via adapters
        self._global_transformer_instances = create_global_adapters()

        # Default transformer instances
        self._default_instances: dict[type[GenericFeatureOperation], GenericFeatureOperation] = {}

    def get_transformers(
        self, column_type: ColumnType
    ) -> list[GenericFeatureOperation]:
        """Get transformer instances for a column type.

        Args:
            column_type: Type of column

        Returns:
            List of transformer instances
        """
        transformer_classes = self._transformers.get(column_type, [])
        instances = []

        for transformer_class in transformer_classes:
            # Create instance if not already created
            if transformer_class not in self._default_instances:
                self._default_instances[transformer_class] = transformer_class()

            instances.append(self._default_instances[transformer_class])

        return instances

    def register_transformer(
        self,
        column_type: ColumnType,
        transformer_class: type[GenericFeatureOperation],
    ) -> None:
        """Register a new transformer for a column type.

        Args:
            column_type: Type of column
            transformer_class: Transformer class to register
        """
        if column_type not in self._transformers:
            self._transformers[column_type] = []

        if transformer_class not in self._transformers[column_type]:
            self._transformers[column_type].append(transformer_class)
            logger.info(
                f"Registered {transformer_class.__name__} for {column_type.value}"
            )

    def get_all_transformers(self) -> list[GenericFeatureOperation]:
        """Get all registered transformer instances.

        Returns:
            List of all transformer instances
        """
        all_instances = []

        # Add column-specific transformers
        for transformer_list in self._transformers.values():
            for transformer_class in transformer_list:
                if transformer_class not in self._default_instances:
                    self._default_instances[transformer_class] = transformer_class()

                instance = self._default_instances[transformer_class]
                if instance not in all_instances:
                    all_instances.append(instance)

        # Add global transformers
        for instance in self._global_transformer_instances:
            if instance not in all_instances:
                all_instances.append(instance)

        return all_instances
    
    def get_global_transformers(self) -> list[GenericFeatureOperation]:
        """Get global transformer instances that apply to entire dataset.
        
        Returns:
            List of global transformer instances
        """
        return self._global_transformer_instances.copy()


# Global registry instance
feature_registry = FeatureRegistry()
