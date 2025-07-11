"""Refactored MDM Client - facade for specialized clients."""

from typing import Optional, Dict
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
    
    def load_dataset_files(self, name: str, include_features: bool = True, limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load dataset files. See QueryClient.load_dataset_files."""
        return self.query.load_dataset_files(name, include_features, limit)
    
    def split_time_series(self, name: str, n_splits: int = 5, test_size: float = 0.2, gap: int = 0, strategy: str = "expanding"):
        """Split time series for cross-validation. See MLIntegrationClient.split_time_series."""
        return self.ml.split_time_series(name, n_splits, test_size, gap, strategy)
    
    @property
    def performance_monitor(self) -> PerformanceMonitor:
        """Get performance monitor instance."""
        if self._performance_monitor is None:
            self._performance_monitor = PerformanceMonitor()
        return self._performance_monitor