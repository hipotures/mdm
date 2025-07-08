"""Main feature engineering engine."""

import importlib.util
from loguru import logger
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from mdm.config import get_config
from mdm.features.generic import (
    CategoricalFeatures,
    StatisticalFeatures,
    TemporalFeatures,
    TextFeatures,
)
from mdm.features.signal import SignalDetector
from mdm.storage.base import StorageBackend

console = Console()


class FeatureEngine:
    """Main feature engineering engine that coordinates all transformers."""

    def __init__(self):
        """Initialize feature engine."""
        self.config = get_config()
        self.signal_detector = SignalDetector()

        # Initialize generic transformers
        self.generic_transformers = {
            "temporal": TemporalFeatures(),
            "categorical": CategoricalFeatures(),
            "statistical": StatisticalFeatures(),
            "text": TextFeatures(),
        }

    def generate_features(
        self,
        dataset_name: str,
        backend: StorageBackend,
        tables: dict[str, str],
        target_column: Optional[str] = None,
        id_columns: Optional[list[str]] = None,
    ) -> dict[str, dict[str, Any]]:
        """Generate features for all tables in a dataset.

        Args:
            dataset_name: Name of the dataset
            backend: Storage backend instance
            tables: Dictionary of table_type -> table_name
            target_column: Name of target column
            id_columns: List of ID column names

        Returns:
            Dictionary of table_type -> feature_info
        """
        feature_info = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            for table_type, table_name in tables.items():
                if table_type == "submission":
                    # Skip submission table
                    continue

                task = progress.add_task(
                    f"Generating features for {table_type} table...",
                    total=None
                )

                try:
                    # Generate features for this table
                    table_features = self._generate_table_features(
                        dataset_name=dataset_name,
                        backend=backend,
                        table_name=table_name,
                        table_type=table_type,
                        target_column=target_column,
                        id_columns=id_columns,
                    )

                    feature_info[table_type] = table_features
                    progress.update(task, completed=100)

                except Exception as e:
                    logger.error(f"Failed to generate features for {table_type}: {e}")
                    progress.update(task, completed=100)

        return feature_info

    def _generate_table_features(
        self,
        dataset_name: str,
        backend: StorageBackend,
        table_name: str,
        table_type: str,
        target_column: Optional[str] = None,
        id_columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Generate features for a single table.

        Args:
            dataset_name: Name of the dataset
            backend: Storage backend
            table_name: Name of the table
            table_type: Type of table (train, test, etc.)
            target_column: Name of target column
            id_columns: List of ID column names

        Returns:
            Feature information dictionary
        """
        start_time = time.time()

        # Load the table
        df = backend.read_table(table_name)
        original_shape = df.shape

        logger.info(
            f"Generating features for {table_type} table "
            f"({original_shape[0]} rows, {original_shape[1]} columns)"
        )

        # Columns to preserve (IDs and target)
        preserve_columns = set()
        if id_columns:
            preserve_columns.update(id_columns)
        if target_column and target_column in df.columns:
            preserve_columns.add(target_column)

        # Generate generic features
        generic_features = self._generate_generic_features(df)

        # Generate custom features if available
        custom_features = self._generate_custom_features(dataset_name, df)

        # Combine all features
        all_features = {**generic_features, **custom_features}

        # Filter features with signal detector
        filtered_features = self.signal_detector.filter_features(all_features)

        # Create feature dataframe
        feature_df = self._create_feature_dataframe(
            df, filtered_features, preserve_columns
        )

        # Save intermediate tables if configured
        if self.config.get("feature_engineering", {}).get("save_intermediate", False):
            # Save generic features
            if generic_features:
                generic_df = pd.DataFrame(generic_features)
                backend.write_table(f"{table_name}_generic", generic_df)

            # Save custom features
            if custom_features:
                custom_df = pd.DataFrame(custom_features)
                backend.write_table(f"{table_name}_custom", custom_df)

        # Save final feature table
        feature_table_name = f"{table_name}_features"
        backend.write_table(feature_table_name, feature_df)

        elapsed = time.time() - start_time

        # Prepare feature info
        feature_info = {
            "original_shape": original_shape,
            "feature_shape": feature_df.shape,
            "generic_features": len(generic_features),
            "custom_features": len(custom_features),
            "filtered_features": len(filtered_features),
            "discarded_features": len(all_features) - len(filtered_features),
            "processing_time": elapsed,
            "feature_table": feature_table_name,
        }

        logger.info(
            f"Generated {len(filtered_features)} features for {table_type} "
            f"in {elapsed:.2f}s ({len(all_features) - len(filtered_features)} discarded)"
        )

        return feature_info

    def _generate_generic_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate all generic features.

        Args:
            df: Input dataframe

        Returns:
            Dictionary of feature_name -> feature_series
        """
        all_features = {}

        for transformer_name, transformer in self.generic_transformers.items():
            try:
                # Get applicable columns
                columns = transformer.get_applicable_columns(df)

                if not columns:
                    logger.debug(f"No applicable columns for {transformer_name}")
                    continue

                logger.info(
                    f"Applying {transformer_name} to {len(columns)} columns: {columns[:5]}"
                    + ("..." if len(columns) > 5 else "")
                )

                # Generate features
                features = transformer.generate_features(df, columns)

                logger.info(
                    f"{transformer_name} generated {len(features)} features"
                )

                all_features.update(features)

            except Exception as e:
                logger.error(f"Error in {transformer_name}: {e}")

        return all_features

    def _generate_custom_features(
        self, dataset_name: str, df: pd.DataFrame
    ) -> dict[str, pd.Series]:
        """Generate custom features if available.

        Args:
            dataset_name: Name of the dataset
            df: Input dataframe

        Returns:
            Dictionary of feature_name -> feature_series
        """
        # Check for custom features module
        custom_features_dir = Path.home() / ".mdm" / "custom_features"
        custom_module_path = custom_features_dir / f"{dataset_name}.py"

        if not custom_module_path.exists():
            logger.debug(f"No custom features found for {dataset_name}")
            return {}

        try:
            # Load custom module
            spec = importlib.util.spec_from_file_location(
                f"custom_{dataset_name}", custom_module_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for CustomFeatureOperations class
                if hasattr(module, "CustomFeatureOperations"):
                    logger.info(f"Found custom features for {dataset_name}")
                    custom_transformer = module.CustomFeatureOperations()
                    return custom_transformer.generate_all_features(df)
                logger.warning(
                    f"Custom module {dataset_name}.py does not contain "
                    "CustomFeatureOperations class"
                )

        except Exception as e:
            logger.error(f"Error loading custom features for {dataset_name}: {e}")

        return {}

    def _create_feature_dataframe(
        self,
        original_df: pd.DataFrame,
        features: dict[str, pd.Series],
        preserve_columns: set,
    ) -> pd.DataFrame:
        """Create final feature dataframe.

        Args:
            original_df: Original dataframe
            features: Dictionary of generated features
            preserve_columns: Set of column names to preserve from original

        Returns:
            Combined feature dataframe
        """
        # Start with preserved columns
        dfs_to_concat = []

        # Add preserved columns (lowercase)
        preserved_df = pd.DataFrame()
        for col in preserve_columns:
            if col in original_df.columns:
                preserved_df[col.lower()] = original_df[col]

        if not preserved_df.empty:
            dfs_to_concat.append(preserved_df)

        # Add original columns (lowercase)
        original_cols = [
            col for col in original_df.columns
            if col not in preserve_columns
        ]
        if original_cols:
            original_lowercase = original_df[original_cols].copy()
            original_lowercase.columns = [col.lower() for col in original_lowercase.columns]
            dfs_to_concat.append(original_lowercase)

        # Add generated features (already lowercase)
        if features:
            feature_df = pd.DataFrame(features)
            dfs_to_concat.append(feature_df)

        # Concatenate all dataframes
        result = pd.concat(dfs_to_concat, axis=1)

        # Ensure no duplicate columns
        return result.loc[:, ~result.columns.duplicated()]

