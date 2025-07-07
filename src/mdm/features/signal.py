"""Signal detection for feature engineering."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class SignalDetector:
    """Detects and removes features without signal."""

    def __init__(self, unique_threshold: float = 0.01, imbalance_threshold: float = 0.99):
        """Initialize signal detector.

        Args:
            unique_threshold: Minimum unique ratio for continuous features
            imbalance_threshold: Maximum class ratio for binary features
        """
        self.unique_threshold = unique_threshold
        self.imbalance_threshold = imbalance_threshold

    def has_signal(self, feature: pd.Series, feature_name: str) -> bool:
        """Check if feature has sufficient signal.

        Returns False if:
        - All values are null
        - All values identical (no variance)
        - Unique ratio < threshold (default 1%)
        - Binary feature with extreme imbalance (>99% one class)

        Args:
            feature: Feature series to check
            feature_name: Name of the feature for logging

        Returns:
            True if feature has signal, False otherwise
        """
        # Null check
        if feature.isna().all():
            logger.debug(f"Feature '{feature_name}' rejected: all values null")
            return False

        # Get non-null values for further checks
        non_null = feature.dropna()
        if len(non_null) == 0:
            logger.debug(f"Feature '{feature_name}' rejected: all values null after dropna")
            return False

        # Variance check
        nunique = non_null.nunique()
        if nunique <= 1:
            logger.debug(f"Feature '{feature_name}' rejected: no variance (nunique={nunique})")
            return False

        # Unique ratio check for continuous features
        unique_ratio = nunique / len(non_null)
        if unique_ratio < self.unique_threshold and nunique > 2:
            logger.debug(
                f"Feature '{feature_name}' rejected: low unique ratio "
                f"({unique_ratio:.4f} < {self.unique_threshold})"
            )
            return False

        # Binary imbalance check
        if nunique == 2:
            value_counts = non_null.value_counts(normalize=True)
            max_ratio = value_counts.max()
            if max_ratio > self.imbalance_threshold:
                logger.debug(
                    f"Feature '{feature_name}' rejected: extreme binary imbalance "
                    f"({max_ratio:.4f} > {self.imbalance_threshold})"
                )
                return False

        return True

    def filter_features(self, features_dict: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Filter features dictionary to keep only those with signal.

        Args:
            features_dict: Dictionary of feature_name -> feature_series

        Returns:
            Filtered dictionary with only features that have signal
        """
        filtered = {}
        discarded = 0

        for name, feature in features_dict.items():
            if self.has_signal(feature, name):
                filtered[name] = feature
            else:
                discarded += 1

        if discarded > 0:
            logger.info(f"Discarded {discarded} features without signal")

        return filtered
