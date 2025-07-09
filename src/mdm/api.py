"""MDM Programmatic API.

This module provides the main public API for MDM.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Union, List, Dict

import pandas as pd

from mdm.config import get_config
from mdm.core.exceptions import DatasetError
from mdm.dataset.manager import DatasetManager
from mdm.dataset.registrar import DatasetRegistrar
from mdm.models.dataset import DatasetInfo
from mdm.storage.factory import BackendFactory
from mdm.utils.integration import MLFrameworkAdapter, SubmissionCreator
from mdm.utils.performance import ChunkProcessor, PerformanceMonitor
from mdm.utils.time_series import TimeSeriesSplitter


class MDMClient:
    """High-level client for MDM operations."""

    def __init__(self, config=None):
        """Initialize MDM client.

        Args:
            config: Optional configuration object. If not provided,
                   loads from default location.
        """
        self.config = config or get_config()
        self.manager = DatasetManager()

    def register_dataset(
        self,
        name: str,
        dataset_path: str,
        target_column: Optional[str] = None,
        id_columns: Optional[list[str]] = None,
        problem_type: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        auto_analyze: bool = True,
        force: bool = False,
        **kwargs
    ) -> DatasetInfo:
        """Register a new dataset.

        Args:
            name: Dataset name (case-insensitive)
            dataset_path: Path to dataset directory
            target_column: Name of target column
            id_columns: List of ID column names
            problem_type: Type of ML problem (classification, regression, etc.)
            description: Dataset description
            tags: List of tags
            auto_analyze: Whether to auto-detect structure
            force: Whether to overwrite existing dataset
            **kwargs: Additional options

        Returns:
            DatasetInfo object

        Raises:
            DatasetError: If registration fails
        """
        registrar = DatasetRegistrar(self.manager)

        # Convert path to Path object
        path = Path(dataset_path)
        
        # Register dataset (registrar will check if path exists)
        return registrar.register(
            name=name,
            path=path,
            auto_detect=auto_analyze,
            target_column=target_column,
            id_columns=id_columns,
            problem_type=problem_type,
            description=description,
            tags=tags,
            force=force,
            **kwargs
        )

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
        filter_func: Optional[Callable[[DatasetInfo], bool]] = None,
        limit: Optional[int] = None,
        sort_by: Optional[str] = None,
        filter_backend: Optional[str] = None
    ) -> list[DatasetInfo]:
        """List all datasets.

        Args:
            filter_func: Optional filter function
            limit: Maximum number of datasets to return
            sort_by: Field to sort by (e.g., 'name', 'registered_at')
            filter_backend: Filter by backend type

        Returns:
            List of DatasetInfo objects
        """
        # Get all datasets from manager
        datasets = self.manager.list_datasets()

        # Filter by backend if requested
        if filter_backend:
            datasets = [d for d in datasets if d.database.get('backend') == filter_backend]

        # Apply additional filter if provided
        if filter_func:
            datasets = [d for d in datasets if filter_func(d)]

        # Sort if requested
        if sort_by:
            reverse = False
            if sort_by.startswith('-'):
                reverse = True
                sort_by = sort_by[1:]
            try:
                datasets = sorted(datasets, key=lambda d: getattr(d, sort_by, ''), reverse=reverse)
            except AttributeError:
                pass  # Ignore if attribute doesn't exist

        # Limit if requested
        if limit is not None:
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
        sample_size: Optional[int] = None
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load dataset files as DataFrames.

        Args:
            name: Dataset name
            sample_size: Optional sample size

        Returns:
            Tuple of (train_df, test_df). test_df may be None.

        Raises:
            DatasetError: If dataset not found
        """
        dataset_info = self.get_dataset(name)
        if not dataset_info:
            raise ValueError(f"Dataset '{name}' not found")

        backend = self.manager.get_backend(name)

        # Load train table (or data table for single-file datasets)
        train_table = dataset_info.tables.get("train")
        if not train_table:
            # Check if this is a single-file dataset with 'data' table
            train_table = dataset_info.tables.get("data")
            if not train_table:
                raise ValueError(f"Dataset '{name}' has no train or data table")

        # Get engine for the dataset
        db_path = dataset_info.database.get('path')
        if not db_path:
            raise ValueError(f"No database path found for dataset '{name}'")
        
        engine = backend.get_engine(db_path)
        train_df = backend.read_table_to_dataframe(train_table, engine)

        # Sample if requested
        if sample_size and len(train_df) > sample_size:
            train_df = train_df.sample(n=sample_size, random_state=42)

        # Load test table if exists
        test_df = None
        test_table = dataset_info.tables.get("test")
        if test_table:
            test_df = backend.read_table_to_dataframe(test_table, engine)
            if sample_size and test_df is not None and len(test_df) > sample_size:
                test_df = test_df.sample(n=sample_size, random_state=42)

        return train_df, test_df

    def load_table(self, name: str, table_name: str) -> pd.DataFrame:
        """Load a specific table from dataset.

        Args:
            name: Dataset name
            table_name: Table name (e.g., 'train', 'test', 'validation')

        Returns:
            DataFrame

        Raises:
            DatasetError: If dataset or table not found
        """
        dataset_info = self.get_dataset(name)
        if not dataset_info:
            raise ValueError(f"Dataset '{name}' not found")

        table_full_name = dataset_info.tables.get(table_name)
        if not table_full_name:
            available = list(dataset_info.tables.keys())
            raise ValueError(
                f"Table '{table_name}' not found in dataset '{name}'. "
                f"Available tables: {available}"
            )

        backend = self.manager.get_backend(name)
        
        # Get engine for the dataset
        db_path = dataset_info.database.get('path')
        if not db_path:
            raise ValueError(f"No database path found for dataset '{name}'")
        
        engine = backend.get_engine(db_path)
        return backend.read_table_to_dataframe(table_full_name, engine)

    def query_dataset(self, name: str, query: str) -> pd.DataFrame:
        """Execute SQL query on dataset.

        Args:
            name: Dataset name
            query: SQL query string

        Returns:
            Query result as DataFrame

        Raises:
            DatasetError: If dataset not found
        """
        backend = self.manager.get_backend(name)
        return backend.execute_query(query)

    def get_dataset_connection(self, name: str):
        """Get direct database connection.

        Args:
            name: Dataset name

        Returns:
            Database connection object (DuckDB, SQLite, etc.)

        Raises:
            DatasetError: If dataset not found
        """
        backend = self.manager.get_backend(name)

        # Get the underlying connection
        if hasattr(backend, 'get_connection'):
            return backend.get_connection()
        if hasattr(backend, 'connection'):
            return backend.connection
        raise NotImplementedError(
            f"Backend {type(backend).__name__} does not support direct connections"
        )

    def update_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
        id_columns: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> DatasetInfo:
        """Update dataset metadata.

        Args:
            name: Dataset name
            description: New description
            target_column: New target column
            problem_type: New problem type
            id_columns: New ID columns
            tags: New tags

        Returns:
            Updated DatasetInfo

        Raises:
            DatasetError: If dataset not found or update fails
        """
        updates = {}
        if description is not None:
            updates['description'] = description
        if target_column is not None:
            updates['target_column'] = target_column
        if problem_type is not None:
            updates['problem_type'] = problem_type
        if id_columns is not None:
            updates['id_columns'] = id_columns
        if tags is not None:
            updates['tags'] = tags

        return self.manager.update_dataset(name, updates)

    def remove_dataset(self, name: str, force: bool = False) -> None:
        """Remove a dataset.

        Args:
            name: Dataset name
            force: Skip confirmation

        Raises:
            DatasetError: If dataset not found or removal fails
        """
        self.manager.remove_dataset(name, force=force)

    def export_dataset(
        self,
        name: str,
        output_dir: str,
        format: str = "csv",
        tables: Optional[list[str]] = None,
        compression: Optional[str] = None,
    ) -> list[str]:
        """Export dataset to files.

        Args:
            name: Dataset name
            output_dir: Output directory path
            format: Export format (csv, parquet, json)
            tables: Specific tables to export (default: all)
            compression: Compression type (zip, gzip, etc.)

        Returns:
            List of exported file paths

        Raises:
            DatasetError: If dataset not found or export fails
        """
        from pathlib import Path

        from mdm.dataset.operations import ExportOperation

        export_op = ExportOperation()

        # Convert tables list to specific table name if only one
        table_name = None
        if tables and len(tables) == 1:
            table_name = tables[0]

        exported_files = export_op.execute(
            name=name,
            format=format,
            output_dir=Path(output_dir),
            table=table_name,
            compression=compression,
        )

        return [str(f) for f in exported_files]

    def get_statistics(self, name: str, full: bool = False) -> dict[str, Any]:
        """Get dataset statistics.

        Args:
            name: Dataset name
            full: Whether to compute full statistics

        Returns:
            Statistics dictionary

        Raises:
            DatasetError: If dataset not found
        """
        from mdm.dataset.operations import StatsOperation

        stats_op = StatsOperation()
        return stats_op.execute(name, full=full)

    def split_time_series(
        self,
        name: str,
        test_size: Union[float, int],
        validation_size: Optional[Union[float, int]] = None,
        time_column: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Split time series dataset.

        Args:
            name: Dataset name
            test_size: Size of test set (fraction or days)
            validation_size: Size of validation set
            time_column: Time column (uses dataset's time_column if not specified)

        Returns:
            Dictionary with train, validation (optional), and test DataFrames

        Raises:
            DatasetError: If dataset not found
        """
        dataset_info = self.get_dataset(name)
        if not dataset_info:
            raise ValueError(f"Dataset '{name}' not found")

        # Use dataset's time column if not specified
        time_col = time_column or dataset_info.time_column
        if not time_col:
            raise ValueError(f"No time column specified for dataset '{name}'")

        # Load data
        train_df, _ = self.load_dataset_files(name)

        # Split data
        splitter = TimeSeriesSplitter(time_col, dataset_info.group_column)
        return splitter.split_by_time(train_df, test_size, validation_size)

    def prepare_for_ml(
        self,
        name: str,
        framework: str = 'auto',
        sample_size: Optional[int] = None,
    ) -> dict[str, Any]:
        """Prepare dataset for ML framework.

        Args:
            name: Dataset name
            framework: ML framework ('sklearn', 'pytorch', 'tensorflow', 'auto')
            sample_size: Optional sample size

        Returns:
            Dictionary with prepared data for the framework

        Raises:
            DatasetError: If dataset not found
        """
        dataset_info = self.get_dataset(name)
        if not dataset_info:
            raise ValueError(f"Dataset '{name}' not found")

        # Load data
        train_df, test_df = self.load_dataset_files(name, sample_size)

        # Prepare for framework
        adapter = MLFrameworkAdapter(framework)
        return adapter.prepare_data(
            train_df,
            test_df,
            dataset_info.target_column,
            dataset_info.id_columns,
        )

    def create_submission(
        self,
        name: str,
        predictions: Union[pd.Series, pd.DataFrame, list],
        output_path: Optional[str] = None,
    ) -> str:
        """Create submission file.

        Args:
            name: Dataset name
            predictions: Model predictions
            output_path: Output file path

        Returns:
            Path to created submission file

        Raises:
            DatasetError: If dataset not found
        """
        creator = SubmissionCreator(self.manager)
        return creator.create_submission(name, predictions, output_path)

    def process_in_chunks(
        self,
        name: str,
        process_func: Callable,
        table_name: str = 'train',
        chunk_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> list[Any]:
        """Process dataset in chunks.

        Args:
            name: Dataset name
            process_func: Function to apply to each chunk
            table_name: Table to process
            chunk_size: Chunk size (default from config)
            show_progress: Whether to show progress

        Returns:
            List of results from each chunk

        Raises:
            DatasetError: If dataset not found
        """
        # Load data
        df = self.load_table(name, table_name)

        # Process in chunks
        processor = ChunkProcessor(chunk_size)
        return processor.process_dataframe(df, process_func, show_progress)

    @property
    def performance_monitor(self) -> PerformanceMonitor:
        """Get performance monitor instance."""
        if not hasattr(self, '_perf_monitor'):
            self._perf_monitor = PerformanceMonitor()
        return self._perf_monitor


# Convenience functions
def load_dataset(name: str) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load dataset files.

    Args:
        name: Dataset name

    Returns:
        Tuple of (train_df, test_df)
    """
    client = MDMClient()
    return client.load_dataset_files(name)


def list_datasets() -> list[str]:
    """List all dataset names.

    Returns:
        List of dataset names
    """
    client = MDMClient()
    datasets = client.list_datasets()
    return [d.name for d in datasets]


def get_dataset_info(name: str) -> Optional[DatasetInfo]:
    """Get dataset information.

    Args:
        name: Dataset name

    Returns:
        DatasetInfo object or None
    """
    client = MDMClient()
    return client.get_dataset(name)
