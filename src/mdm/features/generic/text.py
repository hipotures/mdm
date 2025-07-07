"""Text feature transformations for text columns."""

from typing import Any, Dict

import pandas as pd

from mdm.features.base import GenericFeatureOperation


class TextFeatures(GenericFeatureOperation):
    """Generate features from text columns."""

    def __init__(
        self,
        min_avg_length: int = 50,
        min_signal_ratio: float = 0.01,
    ):
        """Initialize text features transformer.

        Args:
            min_avg_length: Minimum average length to consider as text
            min_signal_ratio: Minimum unique value ratio for signal detection
        """
        super().__init__(min_signal_ratio)
        self.min_avg_length = min_avg_length

    def get_applicable_columns(self, df: pd.DataFrame) -> list[str]:
        """Get text columns from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            List of text column names
        """
        text_columns = []

        for col in df.columns:
            if df[col].dtype == 'object':
                # Sample to check if it's long text
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    avg_length = sample.str.len().mean()
                    # High cardinality and long strings indicate text
                    unique_ratio = df[col].nunique() / len(df)
                    if avg_length >= self.min_avg_length or unique_ratio > 0.8:
                        text_columns.append(col)

        return text_columns

    def _generate_column_features(
        self, df: pd.DataFrame, column: str, **kwargs: Any
    ) -> Dict[str, pd.Series]:
        """Generate text features for a column.

        Args:
            df: Input DataFrame
            column: Text column name
            **kwargs: Additional arguments

        Returns:
            Dictionary of feature name to Series
        """
        features = {}

        # Fill NaN with empty string for text processing
        text_series = df[column].fillna('')

        # Length features
        features[f"{column}_length"] = text_series.str.len()
        features[f"{column}_word_count"] = text_series.str.split().str.len()

        # Average word length
        def avg_word_length(text):
            words = str(text).split()
            if not words:
                return 0
            return sum(len(word) for word in words) / len(words)

        features[f"{column}_avg_word_length"] = text_series.apply(avg_word_length)

        # Character type features
        features[f"{column}_n_uppercase"] = text_series.str.count(r'[A-Z]')
        features[f"{column}_n_lowercase"] = text_series.str.count(r'[a-z]')
        features[f"{column}_n_digits"] = text_series.str.count(r'\d')
        features[f"{column}_n_spaces"] = text_series.str.count(r'\s')
        features[f"{column}_n_special"] = text_series.str.count(r'[^A-Za-z0-9\s]')

        # Binary features
        features[f"{column}_has_digits"] = (
            text_series.str.contains(r'\d', na=False)
        ).astype(int)
        features[f"{column}_has_special"] = (
            text_series.str.contains(r'[^A-Za-z0-9\s]', na=False)
        ).astype(int)
        features[f"{column}_is_upper"] = (
            text_series.str.isupper()
        ).astype(int)
        features[f"{column}_is_lower"] = (
            text_series.str.islower()
        ).astype(int)

        # Complexity features
        features[f"{column}_unique_word_count"] = text_series.apply(
            lambda x: len(set(str(x).lower().split()))
        )
        features[f"{column}_unique_word_ratio"] = (
            features[f"{column}_unique_word_count"] /
            features[f"{column}_word_count"].replace(0, 1)
        )

        # Pattern features
        features[f"{column}_n_urls"] = text_series.str.count(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        features[f"{column}_n_emails"] = text_series.str.count(
            r'[\w\.-]+@[\w\.-]+\.\w+'
        )
        features[f"{column}_n_hashtags"] = text_series.str.count(r'#\w+')
        features[f"{column}_n_mentions"] = text_series.str.count(r'@\w+')

        return features
