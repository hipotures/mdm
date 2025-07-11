"""Refactored MDM Client - facade for specialized clients."""

from typing import Optional

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

    def __init__(self, config=None):
        """Initialize MDM client.

        Args:
            config: Optional configuration object. If not provided,
                   loads from default location.
        """
        self.config = config or get_config()
        self._manager = DatasetManager()
        
        # Initialize specialized clients
        self.registration = RegistrationClient(self.config, self._manager)
        self.query = QueryClient(self.config, self._manager)
        self.ml = MLIntegrationClient(self.config, self._manager)
        self.export = ExportClient(self.config, self._manager)
        self.management = ManagementClient(self.config, self._manager)
        
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
    
    @property
    def performance_monitor(self) -> PerformanceMonitor:
        """Get performance monitor instance."""
        if self._performance_monitor is None:
            self._performance_monitor = PerformanceMonitor()
        return self._performance_monitor