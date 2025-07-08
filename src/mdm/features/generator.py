"""Feature generation for datasets."""

import importlib.util
import time
from typing import Optional, Any

import pandas as pd
from loguru import logger
from sqlalchemy import Engine

from mdm.config import get_config_manager
from mdm.features.custom.base import BaseDomainFeatures
from mdm.features.registry import feature_registry
from mdm.models.enums import ColumnType


class FeatureGenerator:
    """Generate features for datasets during registration."""

    def __init__(self):
        """Initialize feature generator."""
        config_manager = get_config_manager()
        self.config = config_manager.config
        self.base_path = config_manager.base_path
        self.registry = feature_registry

    def generate_features(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        column_types: dict[str, ColumnType],
        target_column: Optional[str] = None,
        id_columns: Optional[list[str]] = None,
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
        source_tables: dict[str, str],
        column_types: dict[str, ColumnType],
        target_column: Optional[str] = None,
        id_columns: Optional[list[str]] = None,
        progress: Optional[Any] = None,
    ) -> dict[str, str]:
        """Generate feature tables for all source tables.
        
        This now processes data in chunks to avoid loading entire dataset into memory.

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
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        import sqlalchemy as sa
        
        feature_tables = {}
        
        # Get batch size from config
        try:
            from mdm.config import get_config_manager
            config = get_config_manager().config
            batch_size = config.performance.batch_size
        except:
            batch_size = 10000

        for table_type, table_name in source_tables.items():
            logger.info(f"Generating features for {table_type} table")
            
            feature_table_name = f"{table_type}_features"
            
            # Get total row count
            with engine.connect() as conn:
                row_count_query = sa.text(f"SELECT COUNT(*) FROM {table_name}")
                result = conn.execute(row_count_query)
                total_rows = result.scalar()
            
            # Use passed progress or create new one
            if progress is None:
                from rich.progress import Progress as RichProgress
                progress_ctx = RichProgress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=None,
                    transient=True,
                )
                progress = progress_ctx.__enter__()
                own_progress = True
            else:
                own_progress = False
                
            try:
                task = progress.add_task(
                    f"Generating and saving features for {feature_table_name}",
                    total=total_rows
                )
                
                first_batch = True
                
                # Process data in chunks
                for offset in range(0, total_rows, batch_size):
                    # Read chunk from source table
                    query = f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
                    chunk_df = pd.read_sql_query(query, engine)
                    
                    if len(chunk_df) == 0:
                        break
                    
                    # Generate features for this chunk
                    chunk_features = self.generate_features(
                        chunk_df, dataset_name, column_types, target_column, id_columns
                    )
                    
                    # Save chunk to feature table
                    if first_batch:
                        chunk_features.to_sql(
                            feature_table_name, engine, if_exists="replace", index=False
                        )
                        first_batch = False
                    else:
                        chunk_features.to_sql(
                            feature_table_name, engine, if_exists="append", index=False
                        )
                    
                    progress.update(task, advance=len(chunk_df))
                    
                    # Free memory
                    del chunk_df
                    del chunk_features
                    
            finally:
                if own_progress:
                    progress_ctx.__exit__(None, None, None)

            feature_tables[f"{table_type}_features"] = feature_table_name
            logger.info(
                f"Created {feature_table_name} with features generated in chunks"
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
            self.base_path / self.config.paths.custom_features_path / f"{dataset_name}.py"
        )

        if not custom_features_path.exists():
            logger.debug(f"No custom features file found at {custom_features_path}")
            return None
        
        logger.info(f"Loading custom features from {custom_features_path}")

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
                    logger.debug(f"Found custom feature class: {attr_name}")
                    instance = attr(dataset_name)
                    # Log registered operations
                    if hasattr(instance, '_operation_registry'):
                        operations = list(instance._operation_registry.keys())
                        logger.debug(f"Custom operations for {dataset_name}: {operations}")
                    return instance

        except Exception as e:
            logger.error(f"Failed to load custom features for {dataset_name}: {e}")

        return None
