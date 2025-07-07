"""Temporal feature transformations for datetime columns."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from mdm.features.base import GenericFeatureOperation


class TemporalFeatures(GenericFeatureOperation):
    """Generate temporal features from datetime columns."""

    def get_applicable_columns(self, df: pd.DataFrame) -> list[str]:
        """Get datetime columns from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            List of datetime column names
        """
        datetime_columns = []

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_columns.append(col)
            # Try to parse as datetime if it's object type
            elif df[col].dtype == 'object':
                try:
                    # Test on small sample
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0:
                        pd.to_datetime(sample, errors='coerce')
                        # If more than 50% parsed successfully, consider it datetime
                        parsed = pd.to_datetime(sample, errors='coerce')
                        if parsed.notna().sum() / len(sample) > 0.5:
                            datetime_columns.append(col)
                except Exception:
                    pass

        return datetime_columns

    def _generate_column_features(
        self, df: pd.DataFrame, column: str, **kwargs: Any
    ) -> Dict[str, pd.Series]:
        """Generate temporal features for a datetime column.

        Args:
            df: Input DataFrame
            column: Datetime column name
            **kwargs: Additional arguments

        Returns:
            Dictionary of feature name to Series
        """
        features = {}

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            dt_series = pd.to_datetime(df[column], errors='coerce')
        else:
            dt_series = df[column]

        # Basic components
        features[f"{column}_year"] = dt_series.dt.year
        features[f"{column}_month"] = dt_series.dt.month
        features[f"{column}_day"] = dt_series.dt.day
        features[f"{column}_dayofweek"] = dt_series.dt.dayofweek
        features[f"{column}_hour"] = dt_series.dt.hour

        # Derived features
        features[f"{column}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
        features[f"{column}_is_month_start"] = dt_series.dt.is_month_start.astype(int)
        features[f"{column}_is_month_end"] = dt_series.dt.is_month_end.astype(int)

        # Cyclical encoding for month and hour
        features[f"{column}_month_sin"] = np.sin(2 * np.pi * dt_series.dt.month / 12)
        features[f"{column}_month_cos"] = np.cos(2 * np.pi * dt_series.dt.month / 12)
        features[f"{column}_hour_sin"] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
        features[f"{column}_hour_cos"] = np.cos(2 * np.pi * dt_series.dt.hour / 24)

        # Days since start
        if not dt_series.isna().all():
            min_date = dt_series.min()
            features[f"{column}_days_since_start"] = (dt_series - min_date).dt.days

        return features
