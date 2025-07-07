"""Statistical feature transformations for numeric columns."""

import contextlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from mdm.features.base import GenericFeatureOperation


class StatisticalFeatures(GenericFeatureOperation):
    """Generate statistical features from numeric columns."""

    def __init__(
        self,
        enable_binning: bool = True,
        n_bins: int = 10,
        min_signal_ratio: float = 0.01,
    ):
        """Initialize statistical features transformer.

        Args:
            enable_binning: Whether to create binned features
            n_bins: Number of bins for binning
            min_signal_ratio: Minimum unique value ratio for signal detection
        """
        super().__init__(min_signal_ratio)
        self.enable_binning = enable_binning
        self.n_bins = n_bins

    def get_applicable_columns(self, df: pd.DataFrame) -> list[str]:
        """Get numeric columns from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            List of numeric column names
        """
        numeric_columns = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Exclude likely ID columns
                if not (col.lower().endswith('_id') or col.lower().startswith('id')):
                    # Exclude binary columns (handled by categorical)
                    if df[col].nunique() > 2:
                        numeric_columns.append(col)

        return numeric_columns

    def _generate_column_features(
        self, df: pd.DataFrame, column: str, **kwargs: Any
    ) -> dict[str, pd.Series]:
        """Generate statistical features for a numeric column.

        Args:
            df: Input DataFrame
            column: Numeric column name
            **kwargs: Additional arguments

        Returns:
            Dictionary of feature name to Series
        """
        features = {}
        series = df[column]

        # Handle missing values for calculations
        non_null_series = series.dropna()

        if len(non_null_series) == 0:
            return features

        # Log transformation (for positive values)
        if (non_null_series > 0).all():
            features[f"{column}_log"] = np.log1p(series)

        # Z-score normalization
        if non_null_series.std() > 0:
            features[f"{column}_zscore"] = (
                (series - non_null_series.mean()) / non_null_series.std()
            )

        # Percentile rank
        features[f"{column}_percentile"] = series.rank(pct=True)

        # Outlier indicators
        q1 = non_null_series.quantile(0.25)
        q3 = non_null_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        features[f"{column}_is_outlier"] = (
            (series < lower_bound) | (series > upper_bound)
        ).astype(int)

        # Binning
        if self.enable_binning and len(non_null_series.unique()) > self.n_bins:
            # Equal-width bins
            with contextlib.suppress(Exception):
                features[f"{column}_bin_equal"] = pd.cut(
                    series, bins=self.n_bins, labels=False
                )

            # Quantile bins
            with contextlib.suppress(Exception):
                features[f"{column}_bin_quantile"] = pd.qcut(
                    series, q=self.n_bins, labels=False, duplicates='drop'
                )

        # Power transformations
        if (non_null_series > 0).all():
            # Square root
            features[f"{column}_sqrt"] = np.sqrt(series)

            # Box-Cox transformation (requires strictly positive values)
            if non_null_series.min() > 0:
                try:
                    transformed, _ = stats.boxcox(non_null_series)
                    features[f"{column}_boxcox"] = pd.Series(
                        transformed, index=non_null_series.index
                    ).reindex(series.index)
                except Exception:
                    pass

        # Interaction with mean
        mean_val = non_null_series.mean()
        if mean_val != 0:
            features[f"{column}_div_mean"] = series / mean_val

        return features
