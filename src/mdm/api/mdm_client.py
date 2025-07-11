"""Refactored MDM Client - facade for specialized clients."""

from typing import Optional, Dict
import time
import pandas as pd

from mdm.config import get_config
from mdm.dataset.manager import DatasetManager
from mdm.utils.performance import PerformanceMonitor

from .clients import (
    RegistrationClient,
    QueryClient,
    MLIntegrationClient,
    ExportClient,
    ManagementClient,
)


class MDMClient:
    """High-level client for MDM operations.
    
    This is a facade that provides access to specialized clients for different
    types of operations. For a single-user local application, this provides
    a clean API while keeping the implementation modular.
    """

    def __init__(
        self,
        registration: Optional[RegistrationClient] = None,
        query: Optional[QueryClient] = None,
        ml: Optional[MLIntegrationClient] = None,
        export: Optional[ExportClient] = None,
        management: Optional[ManagementClient] = None,
        config: Optional[dict] = None
    ):
        """Initialize MDM client with dependency injection.

        Args:
            registration: Registration client instance
            query: Query client instance  
            ml: ML integration client instance
            export: Export client instance
            management: Management client instance
            config: Optional configuration dict
        """
        # If clients are not provided, get them from DI container
        from mdm.core import get_service
        
        self.config = config or get_config()
        self.registration = registration or get_service(RegistrationClient)
        self.query = query or get_service(QueryClient)
        self.ml = ml or get_service(MLIntegrationClient)
        self.export = export or get_service(ExportClient)
        self.management = management or get_service(ManagementClient)
        
        # Performance monitoring
        self._performance_monitor = None

    # Convenience methods that delegate to specialized clients
    # These provide backward compatibility and a simpler API
    
    def register_dataset(self, name: str, dataset_path: str, **kwargs):
        """Register a new dataset. See RegistrationClient.register_dataset."""
        return self.registration.register_dataset(name, dataset_path, **kwargs)
    
    def get_dataset(self, name: str):
        """Get dataset information. See QueryClient.get_dataset."""
        return self.query.get_dataset(name)
    
    def list_datasets(self, **kwargs):
        """List all datasets. See QueryClient.list_datasets."""
        return self.query.list_datasets(**kwargs)
    
    def load_dataset(self, name: str, **kwargs):
        """Load dataset in ML-ready format. See QueryClient.load_dataset."""
        return self.query.load_dataset(name, **kwargs)
    
    def query_dataset(self, name: str, query: str):
        """Execute SQL query on dataset. See QueryClient.query_dataset."""
        return self.query.query_dataset(name, query)
    
    def update_dataset(self, name: str, **kwargs):
        """Update dataset metadata. See ManagementClient.update_dataset."""
        return self.management.update_dataset(name, **kwargs)
    
    def remove_dataset(self, name: str, force: bool = False):
        """Remove a dataset. See ManagementClient.remove_dataset."""
        return self.management.remove_dataset(name, force)
    
    def export_dataset(self, name: str, output_dir: str, **kwargs):
        """Export dataset to files. See ExportClient.export_dataset."""
        return self.export.export_dataset(name, output_dir, **kwargs)
    
    def prepare_for_ml(self, name: str, framework: str = "auto", **kwargs):
        """Prepare dataset for ML framework. See MLIntegrationClient.prepare_for_ml."""
        return self.ml.prepare_for_ml(name, framework, **kwargs)
    
    def create_submission(self, name: str, predictions, submission_file: str, **kwargs):
        """Create submission file. See MLIntegrationClient.create_submission."""
        return self.ml.create_submission(name, predictions, submission_file, **kwargs)
    
    def compute_statistics(self, name: str, full: bool = False):
        """Compute dataset statistics. See ManagementClient.compute_statistics."""
        return self.management.compute_statistics(name, full)
    
    def get_statistics(self, name: str, full: bool = False):
        """Get pre-computed statistics for a dataset. See ManagementClient.get_statistics."""
        return self.management.get_statistics(name, full)
    
    def search_datasets(self, pattern: str, deep: bool = False, case_sensitive: bool = False):
        """Search datasets by pattern. See ManagementClient.search_datasets."""
        # Management client doesn't support deep and case_sensitive params currently
        return self.management.search_datasets(pattern)
    
    def search_datasets_by_tag(self, tag: str):
        """Search datasets by tag. See ManagementClient.search_datasets_by_tag."""
        return self.management.search_datasets_by_tag(tag)
    
    def load_dataset_files(self, name: str, include_features: bool = True, limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load dataset files. See QueryClient.load_dataset_files."""
        return self.query.load_dataset_files(name, include_features, limit)
    
    def split_time_series(self, name: str, n_splits: int = 5, test_size: float = 0.2, gap: int = 0, strategy: str = "expanding"):
        """Split time series for cross-validation. See MLIntegrationClient.split_time_series."""
        return self.ml.split_time_series(name, n_splits, test_size, gap, strategy)
    
    def get_column_info(self, name: str, table: str = "train"):
        """Get column information for a dataset. See QueryClient.get_column_info."""
        return self.query.get_column_info(name, table)
    
    def get_framework_adapter(self, framework: str):
        """Get ML framework adapter. See MLIntegrationClient.get_framework_adapter."""
        return self.ml.get_framework_adapter(framework)
    
    def process_in_chunks(self, data, process_func, chunk_size: Optional[int] = None):
        """Process data in chunks for memory efficiency.
        
        Args:
            data: DataFrame to process
            process_func: Function to apply to each chunk
            chunk_size: Optional chunk size (default from config)
            
        Returns:
            List of results from each chunk
        """
        from mdm.utils.performance import ChunkProcessor
        processor = ChunkProcessor(chunk_size=chunk_size)
        return processor.process_dataframe(data, process_func, show_progress=False)
    
    def monitor_performance(self):
        """Context manager for performance monitoring.
        
        Returns:
            Context manager that yields PerformanceMonitor
        """
        if self._performance_monitor is None:
            self._performance_monitor = PerformanceMonitor()
        
        from contextlib import contextmanager
        
        @contextmanager
        def _monitor():
            """Context manager wrapper for PerformanceMonitor."""
            self._performance_monitor.start_time = time.time()
            yield self._performance_monitor
            # Could add cleanup or reporting here if needed
            
        return _monitor()
    
    def create_time_series_splits(self, data, time_column: str, n_splits: int = 5, gap_days: int = 0):
        """Create time series splits for cross-validation.
        
        Args:
            data: DataFrame with time series data
            time_column: Name of the time column
            n_splits: Number of splits
            gap_days: Gap between train and test sets in days
            
        Returns:
            List of (train, test) DataFrame tuples
        """
        from mdm.utils.time_series import TimeSeriesSplitter
        splitter = TimeSeriesSplitter(time_column=time_column)
        folds = splitter.split_by_folds(data, n_folds=n_splits, gap_days=gap_days)
        
        # Convert folds dict format to tuples
        return [(fold['train'], fold['test']) for fold in folds]
    
    @property
    def performance_monitor(self) -> PerformanceMonitor:
        """Get performance monitor instance."""
        if self._performance_monitor is None:
            self._performance_monitor = PerformanceMonitor()
        return self._performance_monitor