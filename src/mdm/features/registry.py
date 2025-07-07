"""Feature transformer registry."""

from typing import Dict, List, Type

from loguru import logger

from mdm.features.base import GenericFeatureOperation
from mdm.features.generic import (
    CategoricalFeatures,
    StatisticalFeatures,
    TemporalFeatures,
    TextFeatures,
)
from mdm.models.enums import ColumnType


class FeatureRegistry:
    """Registry for feature transformers."""

    def __init__(self):
        """Initialize feature registry."""
        self._transformers: Dict[ColumnType, List[Type[GenericFeatureOperation]]] = {
            ColumnType.DATETIME: [TemporalFeatures],
            ColumnType.CATEGORICAL: [CategoricalFeatures],
            ColumnType.NUMERIC: [StatisticalFeatures],
            ColumnType.TEXT: [TextFeatures],
        }

        # Default transformer instances
        self._default_instances: Dict[Type[GenericFeatureOperation], GenericFeatureOperation] = {}

    def get_transformers(
        self, column_type: ColumnType
    ) -> List[GenericFeatureOperation]:
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
        transformer_class: Type[GenericFeatureOperation],
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

    def get_all_transformers(self) -> List[GenericFeatureOperation]:
        """Get all registered transformer instances.

        Returns:
            List of all transformer instances
        """
        all_instances = []

        for transformer_list in self._transformers.values():
            for transformer_class in transformer_list:
                if transformer_class not in self._default_instances:
                    self._default_instances[transformer_class] = transformer_class()

                instance = self._default_instances[transformer_class]
                if instance not in all_instances:
                    all_instances.append(instance)

        return all_instances


# Global registry instance
feature_registry = FeatureRegistry()
