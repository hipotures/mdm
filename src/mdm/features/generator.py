"""Feature generation for datasets."""

import importlib.util
import time
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import Engine

from mdm.core.config import MDMConfig
from mdm.features.custom.base import BaseDomainFeatures
from mdm.features.registry import feature_registry
from mdm.models.enums import ColumnType


class FeatureGenerator:
    """Generate features for datasets during registration."""

    def __init__(self, config: Optional[MDMConfig] = None):
        """Initialize feature generator.

        Args:
            config: MDM configuration
        """
        self.config = config or MDMConfig()
        self.registry = feature_registry

    def generate_features(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        column_types: Dict[str, ColumnType],
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate features for a DataFrame.

        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset
            column_types: Mapping of column names to types
            target_column: Target column name (if any)
            id_columns: ID column names

        Returns:
            DataFrame with original and generated features
        """
        logger.info(f"Generating features for {dataset_name} ({len(df)} rows)")
        start_time = time.time()

        # Start with original columns
        feature_df = df.copy()
        feature_count = 0
        discarded_count = 0

        # Exclude ID and target columns from feature generation
        exclude_columns = set()
        if id_columns:
            exclude_columns.update(id_columns)
        if target_column:
            exclude_columns.add(target_column)

        # Apply generic transformers based on column types
        for column, col_type in column_types.items():
            if column in exclude_columns:
                continue

            # Get transformers for this column type
            transformers = self.registry.get_transformers(col_type)

            for transformer in transformers:
                logger.debug(
                    f"[{transformer.__class__.__name__}] Processing column '{column}'"
                )

                # Generate features
                features = transformer.generate_features(df, [column])

                # Add to feature DataFrame
                for feature_name, feature_values in features.items():
                    if feature_name not in feature_df.columns:
                        feature_df[feature_name] = feature_values
                        feature_count += 1

        # Apply custom transformers if available
        custom_features = self._load_custom_features(dataset_name)
        if custom_features:
            logger.info(f"[CustomFeatures] Applying {dataset_name}-specific features")
            custom_feature_dict = custom_features.generate_all_features(df)

            for feature_name, feature_values in custom_feature_dict.items():
                if feature_name not in feature_df.columns:
                    feature_df[feature_name] = feature_values
                    feature_count += 1

        elapsed = time.time() - start_time
        logger.info(
            f"Feature generation complete: {feature_count} features created "
            f"({discarded_count} discarded) in {elapsed:.2f}s"
        )

        return feature_df

    def generate_feature_tables(
        self,
        engine: Engine,
        dataset_name: str,
        source_tables: Dict[str, str],
        column_types: Dict[str, ColumnType],
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Generate feature tables for all source tables.

        Args:
            engine: SQLAlchemy engine
            dataset_name: Name of the dataset
            source_tables: Mapping of table type to table name
            column_types: Mapping of column names to types
            target_column: Target column name
            id_columns: ID column names

        Returns:
            Mapping of feature table type to table name
        """
        feature_tables = {}

        for table_type, table_name in source_tables.items():
            logger.info(f"Generating features for {table_type} table")

            # Read source table
            df = pd.read_sql_table(table_name, engine)

            # Generate features
            feature_df = self.generate_features(
                df, dataset_name, column_types, target_column, id_columns
            )

            # Save feature table
            feature_table_name = f"{table_type}_features"
            feature_df.to_sql(
                feature_table_name, engine, if_exists="replace", index=False
            )

            feature_tables[f"{table_type}_features"] = feature_table_name
            logger.info(
                f"Created {feature_table_name} with {len(feature_df.columns)} columns"
            )

        return feature_tables

    def _load_custom_features(
        self, dataset_name: str
    ) -> Optional[BaseDomainFeatures]:
        """Load custom features for a dataset if available.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Custom feature instance or None
        """
        custom_features_path = (
            self.config.config_dir / "custom_features" / f"{dataset_name}.py"
        )

        if not custom_features_path.exists():
            return None

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                f"custom_features_{dataset_name}", custom_features_path
            )
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the CustomFeatureOperations class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseDomainFeatures)
                    and attr is not BaseDomainFeatures
                ):
                    # Create instance
                    return attr(dataset_name)

        except Exception as e:
            logger.error(f"Failed to load custom features for {dataset_name}: {e}")

        return None
