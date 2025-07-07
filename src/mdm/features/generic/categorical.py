"""Categorical feature transformations for low cardinality columns."""

from typing import Any

import pandas as pd
from loguru import logger

from mdm.features.base import GenericFeatureOperation


class CategoricalFeatures(GenericFeatureOperation):
    """Generate features from categorical columns."""

    def __init__(
        self,
        max_cardinality: int = 50,
        min_frequency: float = 0.01,
        min_signal_ratio: float = 0.01,
    ):
        """Initialize categorical features transformer.

        Args:
            max_cardinality: Maximum unique values for one-hot encoding
            min_frequency: Minimum frequency for rare category grouping
            min_signal_ratio: Minimum unique value ratio for signal detection
        """
        super().__init__(min_signal_ratio)
        self.max_cardinality = max_cardinality
        self.min_frequency = min_frequency

    def get_applicable_columns(self, df: pd.DataFrame) -> list[str]:
        """Get categorical columns from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            List of categorical column names
        """
        categorical_columns = []

        for col in df.columns:
            # Skip if already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue

            # Check cardinality ratio
            unique_ratio = df[col].nunique() / len(df)

            # Consider categorical if:
            # 1. Explicitly categorical dtype
            # 2. Object dtype with low cardinality
            # 3. Numeric with very low cardinality (like binary)
            if (
                pd.api.types.is_categorical_dtype(df[col])
                or (df[col].dtype == 'object' and unique_ratio < 0.5)
                or (pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < 20)
            ):
                categorical_columns.append(col)

        return categorical_columns

    def _generate_column_features(
        self, df: pd.DataFrame, column: str, **kwargs: Any
    ) -> dict[str, pd.Series]:
        """Generate categorical features for a column.

        Args:
            df: Input DataFrame
            column: Categorical column name
            **kwargs: Additional arguments

        Returns:
            Dictionary of feature name to Series
        """
        features = {}

        # Frequency encoding
        value_counts = df[column].value_counts()
        features[f"{column}_frequency"] = df[column].map(value_counts)
        features[f"{column}_frequency_ratio"] = features[f"{column}_frequency"] / len(df)

        # One-hot encoding if cardinality is reasonable
        n_unique = df[column].nunique()
        if n_unique <= self.max_cardinality:
            # Get dummies
            dummies = pd.get_dummies(df[column], prefix=column, dummy_na=False)
            for col_name in dummies.columns:
                features[col_name] = dummies[col_name]
        else:
            logger.info(
                f"Skipping one-hot encoding for '{column}' "
                f"(cardinality {n_unique} > {self.max_cardinality})"
            )

        # Rare category indicator
        freq_threshold = len(df) * self.min_frequency
        rare_mask = df[column].map(value_counts) < freq_threshold
        features[f"{column}_is_rare"] = rare_mask.astype(int)

        # Number of unique values (useful for text-like categoricals)
        if df[column].dtype == 'object':
            features[f"{column}_nunique_chars"] = df[column].fillna('').str.len()

        return features
