"""Base classes for feature operations."""

import time
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from loguru import logger


class BaseFeatureOperation(ABC):
    """Base class for all feature operations with signal detection and timing."""

    def __init__(self, min_signal_ratio: float = 0.01):
        """Initialize feature operation.

        Args:
            min_signal_ratio: Minimum unique value ratio for signal detection
        """
        self.min_signal_ratio = min_signal_ratio

    def generate_features(
        self, df: pd.DataFrame, columns: list[str], **kwargs: Any
    ) -> dict[str, pd.Series]:
        """Generate features with automatic signal checking and timing.

        Args:
            df: Input DataFrame
            columns: Columns to process
            **kwargs: Additional arguments for feature generation

        Returns:
            Dictionary of feature name to Series
        """
        features = {}

        for column in columns:
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in DataFrame")
                continue

            start_time = time.time()

            try:
                # Generate features for this column
                column_features = self._generate_column_features(
                    df, column, **kwargs
                )

                # Check each generated feature for signal
                for feature_name, feature_values in column_features.items():
                    if self._check_feature_signal(feature_values):
                        features[feature_name] = feature_values
                        elapsed = time.time() - start_time
                        logger.debug(
                            f"Generated feature '{feature_name}' in {elapsed:.3f}s"
                        )
                    else:
                        logger.info(
                            f"Feature '{feature_name}' has no signal, discarded"
                        )

            except Exception as e:
                logger.error(f"Error generating features for column '{column}': {e}")

        return features

    @abstractmethod
    def _generate_column_features(
        self, df: pd.DataFrame, column: str, **kwargs: Any
    ) -> dict[str, pd.Series]:
        """Generate features for a single column.

        Args:
            df: Input DataFrame
            column: Column name to process
            **kwargs: Additional arguments

        Returns:
            Dictionary of feature name to Series
        """
        pass

    def _check_feature_signal(
        self, series: pd.Series, sample_size: int = 2000
    ) -> bool:
        """Check if feature has meaningful signal (variance).

        Args:
            series: Feature values
            sample_size: Sample size for large datasets

        Returns:
            True if feature has signal
        """
        # Handle empty series
        if len(series) == 0:
            return False

        # For large datasets, use sampling
        if len(series) > sample_size:
            sample = series.sample(n=sample_size, random_state=42)
        else:
            sample = series

        # Check for null values
        non_null_count = sample.count()
        if non_null_count == 0:
            return False

        # Check unique ratio
        unique_ratio = sample.nunique() / len(sample)
        if unique_ratio < self.min_signal_ratio:
            return False

        # For numeric features, check variance
        return not (pd.api.types.is_numeric_dtype(sample) and sample.std() == 0)


class GenericFeatureOperation(BaseFeatureOperation):
    """Base class for generic feature operations that apply to all datasets."""

    @abstractmethod
    def get_applicable_columns(self, df: pd.DataFrame) -> list[str]:
        """Get columns this transformer can process.

        Args:
            df: Input DataFrame

        Returns:
            List of applicable column names
        """
        pass


class CustomFeatureOperation(BaseFeatureOperation):
    """Base class for custom dataset-specific feature operations."""

    def __init__(self, dataset_name: str, min_signal_ratio: float = 0.01):
        """Initialize custom feature operation.

        Args:
            dataset_name: Name of the dataset
            min_signal_ratio: Minimum unique value ratio for signal detection
        """
        super().__init__(min_signal_ratio)
        self.dataset_name = dataset_name
