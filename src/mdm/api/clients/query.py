"""Dataset query and loading client."""

from typing import Optional, Dict, Any, List
import pandas as pd
from loguru import logger

from mdm.core.exceptions import DatasetError
from mdm.models.dataset import DatasetInfo
from mdm.storage.factory import BackendFactory

from .base import BaseClient


class QueryClient(BaseClient):
    """Client for dataset querying and loading operations."""
    
    def get_dataset(self, name: str) -> Optional[DatasetInfo]:
        """Get dataset information.

        Args:
            name: Dataset name (case-insensitive)

        Returns:
            DatasetInfo object or None if not found
        """
        try:
            return self.manager.get_dataset(name)
        except DatasetError:
            return None
    
    def list_datasets(
        self,
        limit: Optional[int] = None,
        tags: Optional[List[str]] = None,
        sort_by: str = "registration_date",
        ascending: bool = False
    ) -> List[DatasetInfo]:
        """List all registered datasets.

        Args:
            limit: Maximum number of datasets to return
            tags: Filter datasets by tags (any match)
            sort_by: Sort by field (name, registration_date, size)
            ascending: Sort order

        Returns:
            List of DatasetInfo objects
        """
        datasets = self.manager.list_datasets()
        
        # Filter by tags if provided
        if tags:
            datasets = [
                d for d in datasets
                if any(tag in d.tags for tag in tags)
            ]
        
        # Sort datasets
        if sort_by == "name":
            datasets.sort(key=lambda d: d.name, reverse=not ascending)
        elif sort_by == "registration_date":
            datasets.sort(key=lambda d: d.registration_date, reverse=not ascending)
        elif sort_by == "size":
            # Get sizes if available
            datasets.sort(
                key=lambda d: d.metadata.get("statistics", {}).get("memory_size_bytes", 0),
                reverse=not ascending
            )
        
        # Apply limit
        if limit:
            datasets = datasets[:limit]
        
        return datasets
    
    def dataset_exists(self, name: str) -> bool:
        """Check if dataset exists.

        Args:
            name: Dataset name (case-insensitive)

        Returns:
            True if dataset exists
        """
        return self.manager.dataset_exists(name)
    
    def load_dataset_files(
        self, 
        name: str, 
        include_features: bool = True,
        limit: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load all files from a dataset.

        Args:
            name: Dataset name
            include_features: Whether to include feature tables
            limit: Optional row limit for each table

        Returns:
            Dictionary mapping table names to DataFrames

        Raises:
            DatasetError: If dataset not found or loading fails
        """
        dataset = self.get_dataset(name)
        if not dataset:
            raise DatasetError(f"Dataset '{name}' not found")

        backend = BackendFactory.create(
            dataset.database["backend"],
            dataset.database
        )

        # Get database path/connection
        if "path" in dataset.database:
            db_path = dataset.database["path"]
        else:
            # Server-based backend
            db_info = dataset.database
            db_path = f"{db_info['backend']}://{db_info['user']}:{db_info['password']}@{db_info['host']}:{db_info['port']}/{db_info['database']}"

        # Load tables
        result = {}
        tables_to_load = dict(dataset.tables)
        
        if include_features and dataset.feature_tables:
            tables_to_load.update(dataset.feature_tables)

        for table_key, table_name in tables_to_load.items():
            logger.info(f"Loading table {table_name}")
            if limit:
                df = backend.read_table_to_dataframe(table_name, db_path, limit=limit)
            else:
                df = backend.read_table_to_dataframe(table_name, db_path)
            result[table_key] = df

        return result
    
    def load_table(self, name: str, table_name: str) -> pd.DataFrame:
        """Load a specific table from dataset.

        Args:
            name: Dataset name
            table_name: Table name to load

        Returns:
            DataFrame with table data

        Raises:
            DatasetError: If dataset or table not found
        """
        dataset = self.get_dataset(name)
        if not dataset:
            raise DatasetError(f"Dataset '{name}' not found")

        # Check if table exists
        all_tables = dict(dataset.tables)
        if dataset.feature_tables:
            all_tables.update(dataset.feature_tables)
        
        if table_name not in all_tables.values():
            available = list(all_tables.keys())
            raise DatasetError(
                f"Table '{table_name}' not found in dataset '{name}'. "
                f"Available tables: {available}"
            )

        backend = BackendFactory.create(
            dataset.database["backend"],
            dataset.database
        )

        # Get database path/connection
        if "path" in dataset.database:
            db_path = dataset.database["path"]
        else:
            # Server-based backend
            db_info = dataset.database
            db_path = f"{db_info['backend']}://{db_info['user']}:{db_info['password']}@{db_info['host']}:{db_info['port']}/{db_info['database']}"

        return backend.read_table_to_dataframe(table_name, db_path)
    
    def query_dataset(self, name: str, query: str) -> pd.DataFrame:
        """Execute SQL query on dataset.

        Args:
            name: Dataset name
            query: SQL query to execute

        Returns:
            Query results as DataFrame

        Raises:
            DatasetError: If dataset not found or query fails
        """
        dataset = self.get_dataset(name)
        if not dataset:
            raise DatasetError(f"Dataset '{name}' not found")

        backend = BackendFactory.create(
            dataset.database["backend"],
            dataset.database
        )
        
        return backend.query(query)
    
    def load_dataset(
        self,
        name: str,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        id_columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.DataFrame]]:
        """Load dataset in ML-ready format.

        Args:
            name: Dataset name
            target_column: Override target column
            feature_columns: Specific features to load (None = all)
            id_columns: ID columns to include
            limit: Row limit for sampling

        Returns:
            Tuple of (features, target, ids) where target and ids may be None

        Raises:
            DatasetError: If dataset not found
        """
        dataset = self.get_dataset(name)
        if not dataset:
            raise DatasetError(f"Dataset '{name}' not found")

        # Load main data table
        main_table = dataset.tables.get("train", dataset.tables.get("data"))
        if not main_table:
            raise DatasetError(f"No main data table found in dataset '{name}'")

        df = self.load_table(name, main_table)
        
        # Apply row limit if specified
        if limit:
            df = df.head(limit)

        # Use dataset's target if not overridden
        if target_column is None:
            target_column = dataset.target_column

        # Split features and target
        target = None
        if target_column and target_column in df.columns:
            target = df[target_column]
            df = df.drop(columns=[target_column])

        # Handle ID columns
        ids = None
        id_cols = id_columns or dataset.id_columns
        if id_cols:
            existing_id_cols = [col for col in id_cols if col in df.columns]
            if existing_id_cols:
                ids = df[existing_id_cols]
                df = df.drop(columns=existing_id_cols)

        # Filter features if specified
        if feature_columns:
            missing = [col for col in feature_columns if col not in df.columns]
            if missing:
                logger.warning(f"Missing feature columns: {missing}")
            df = df[[col for col in feature_columns if col in df.columns]]

        return df, target, ids
    
    def get_column_info(
        self,
        name: str,
        table_name: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get column information for dataset tables.

        Args:
            name: Dataset name
            table_name: Specific table (None = all tables)

        Returns:
            Dictionary mapping table names to column info

        Raises:
            DatasetError: If dataset not found
        """
        dataset = self.get_dataset(name)
        if not dataset:
            raise DatasetError(f"Dataset '{name}' not found")

        backend = BackendFactory.create(
            dataset.database["backend"],
            dataset.database
        )

        # Get database path/connection
        if "path" in dataset.database:
            db_path = dataset.database["path"]
        else:
            # Server-based backend
            db_info = dataset.database
            db_path = f"{db_info['backend']}://{db_info['user']}:{db_info['password']}@{db_info['host']}:{db_info['port']}/{db_info['database']}"

        result = {}
        
        # Get tables to analyze
        if table_name:
            tables_to_analyze = {table_name: table_name}
        else:
            tables_to_analyze = dict(dataset.tables)
            if dataset.feature_tables:
                tables_to_analyze.update(dataset.feature_tables)

        # Get column info for each table
        for key, table in tables_to_analyze.items():
            df_sample = backend.read_table_to_dataframe(table, db_path, limit=1000)
            
            col_info = {}
            for col in df_sample.columns:
                col_info[col] = {
                    "dtype": str(df_sample[col].dtype),
                    "null_count": df_sample[col].isnull().sum(),
                    "unique_count": df_sample[col].nunique(),
                    "sample_values": df_sample[col].dropna().head(5).tolist()
                }
            
            result[key] = col_info

        return result