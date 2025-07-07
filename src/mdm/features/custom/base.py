"""Base class for custom domain-specific features."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import pandas as pd
from loguru import logger

from mdm.features.base import CustomFeatureOperation


class BaseDomainFeatures(CustomFeatureOperation, ABC):
    """Base class for domain-specific custom features.
    
    Users should create a subclass of this in their custom features file.
    """

    def __init__(self, dataset_name: str, min_signal_ratio: float = 0.01):
        """Initialize domain features.

        Args:
            dataset_name: Name of the dataset
            min_signal_ratio: Minimum unique value ratio for signal detection
        """
        super().__init__(dataset_name, min_signal_ratio)
        self._operation_registry: Dict[str, Callable] = {}
        self._register_operations()

    @abstractmethod
    def _register_operations(self) -> None:
        """Register all custom operations.
        
        Should populate self._operation_registry with operation name to method mapping.
        """
        pass

    def generate_all_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate all registered custom features.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary of feature name to Series
        """
        all_features = {}

        for operation_name, operation_func in self._operation_registry.items():
            logger.info(f"Generating {operation_name} for {self.dataset_name}")
            try:
                features = operation_func(df)

                # Apply signal detection
                for feature_name, feature_values in features.items():
                    if self._check_feature_signal(feature_values):
                        all_features[feature_name] = feature_values
                    else:
                        logger.info(
                            f"Feature '{feature_name}' has no signal, discarded"
                        )

            except Exception as e:
                logger.error(
                    f"Error in {operation_name} for {self.dataset_name}: {e}"
                )

        return all_features

    def _generate_column_features(
        self, df: pd.DataFrame, column: str, **kwargs: Any
    ) -> Dict[str, pd.Series]:
        """Not used for custom features - we use generate_all_features instead."""
        return {}
