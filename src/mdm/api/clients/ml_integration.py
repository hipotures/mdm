"""Machine Learning integration client."""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
import pandas as pd
from loguru import logger

from mdm.core.exceptions import DatasetError
from mdm.utils.integration import MLFrameworkAdapter, SubmissionCreator
from mdm.utils.time_series import TimeSeriesSplitter
from mdm.utils.performance import ChunkProcessor, PerformanceMonitor

from .base import BaseClient


class MLIntegrationClient(BaseClient):
    """Client for ML framework integration and utilities."""
    
    def split_time_series(
        self,
        name: str,
        n_splits: int = 5,
        test_size: Union[int, float] = 0.2,
        gap: int = 0,
        strategy: str = "expanding"
    ) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
        """Create time series cross-validation splits.

        Args:
            name: Dataset name
            n_splits: Number of splits
            test_size: Test set size (rows if int, fraction if float)
            gap: Gap between train and test
            strategy: Split strategy ('expanding' or 'sliding')

        Returns:
            List of (train, test) DataFrame tuples

        Raises:
            DatasetError: If dataset not found or has no time column
        """
        from .query import QueryClient
        query_client = QueryClient(self.config, self.manager)
        
        dataset = query_client.get_dataset(name)
        if not dataset:
            raise DatasetError(f"Dataset '{name}' not found")

        if not dataset.time_column:
            raise DatasetError(f"Dataset '{name}' has no time column configured")

        # Load main data
        main_table = dataset.tables.get("train", dataset.tables.get("data"))
        if not main_table:
            raise DatasetError(f"No main data table found")

        df = query_client.load_table(name, main_table)

        # Create splitter
        splitter = TimeSeriesSplitter(
            time_column=dataset.time_column,
            group_column=dataset.group_column
        )

        return splitter.split(
            df,
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            strategy=strategy
        )
    
    def prepare_for_ml(
        self,
        name: str,
        framework: str = "auto",
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        return_sparse: bool = False
    ):
        """Prepare dataset for specific ML framework.

        Args:
            name: Dataset name
            framework: ML framework ('sklearn', 'pytorch', 'tensorflow', 'auto')
            target_column: Override target column
            feature_columns: Specific features to use
            return_sparse: Return sparse matrices if possible

        Returns:
            Framework-specific data format

        Raises:
            DatasetError: If dataset not found
        """
        from .query import QueryClient
        query_client = QueryClient(self.config, self.manager)
        
        # Load dataset
        X, y, ids = query_client.load_dataset(
            name, target_column, feature_columns
        )

        # Get adapter
        adapter = MLFrameworkAdapter.create(framework)
        return adapter.prepare_data(X, y, ids, return_sparse)
    
    def create_submission(
        self,
        name: str,
        predictions: Union[pd.Series, pd.DataFrame],
        submission_file: str,
        id_column: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> Path:
        """Create a submission file for competitions.

        Args:
            name: Dataset name
            predictions: Model predictions
            submission_file: Output file path
            id_column: ID column name (auto-detected if None)
            target_column: Target column name (auto-detected if None)

        Returns:
            Path to submission file

        Raises:
            DatasetError: If dataset not found
        """
        from .query import QueryClient
        query_client = QueryClient(self.config, self.manager)
        
        dataset = query_client.get_dataset(name)
        if not dataset:
            raise DatasetError(f"Dataset '{name}' not found")

        # Get ID column
        if not id_column:
            id_column = dataset.id_columns[0] if dataset.id_columns else None
            if not id_column:
                raise DatasetError("No ID column found or specified")

        # Get target column
        if not target_column:
            target_column = dataset.target_column
            if not target_column:
                raise DatasetError("No target column found or specified")

        creator = SubmissionCreator()
        return creator.create(
            predictions,
            submission_file,
            id_column,
            target_column
        )
    
    def get_framework_adapter(
        self,
        framework: str = "auto"
    ) -> MLFrameworkAdapter:
        """Get ML framework adapter.

        Args:
            framework: Framework name or 'auto'

        Returns:
            MLFrameworkAdapter instance
        """
        return MLFrameworkAdapter.create(framework)
    
    def process_in_chunks(
        self,
        name: str,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        chunk_size: int = 10000,
        output_table: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Process dataset in chunks to save memory.

        Args:
            name: Dataset name
            func: Function to apply to each chunk
            chunk_size: Rows per chunk
            output_table: Save results to this table (None = return DataFrame)

        Returns:
            Processed DataFrame if output_table is None

        Raises:
            DatasetError: If dataset not found
        """
        processor = ChunkProcessor(chunk_size)
        return processor.process_dataset(name, func, output_table)
    
    def create_time_series_splits(
        self,
        data: pd.DataFrame,
        time_column: str,
        n_splits: int = 5,
        test_size: Union[int, float] = 0.2,
        gap: int = 0,
        strategy: str = "expanding",
        group_column: Optional[str] = None
    ) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
        """Create time series cross-validation splits from DataFrame.

        Args:
            data: Input DataFrame
            time_column: Name of time column
            n_splits: Number of splits
            test_size: Test set size
            gap: Gap between train and test
            strategy: Split strategy ('expanding' or 'sliding')
            group_column: Optional group column for grouped splits

        Returns:
            List of (train, test) DataFrame tuples
        """
        splitter = TimeSeriesSplitter(
            time_column=time_column,
            group_column=group_column
        )
        
        return splitter.split(
            data,
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            strategy=strategy
        )