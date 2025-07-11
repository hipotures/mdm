"""Dataset management client."""

from typing import Optional, List, Dict, Any

from mdm.core.exceptions import DatasetError
from mdm.dataset.operations import UpdateOperation, RemoveOperation
from mdm.models.dataset import DatasetInfo

from .base import BaseClient


class ManagementClient(BaseClient):
    """Client for dataset management operations."""
    
    def update_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
        problem_type: Optional[str] = None,
        **kwargs
    ) -> DatasetInfo:
        """Update dataset metadata.

        Args:
            name: Dataset name
            description: New description
            tags: New tags (replaces existing)
            target_column: New target column
            id_columns: New ID columns
            problem_type: New problem type
            **kwargs: Additional metadata

        Returns:
            Updated DatasetInfo

        Raises:
            DatasetError: If dataset not found or update fails
        """
        update_op = UpdateOperation()
        # UpdateOperation uses 'target' not 'target_column'
        result = update_op.execute(
            name=name,
            description=description,
            target=target_column,
            id_columns=id_columns,
            problem_type=problem_type,
            tags=tags
        )
        
        # Convert result dict to DatasetInfo
        return self.manager.get_dataset(name)
    
    def remove_dataset(self, name: str, force: bool = False) -> None:
        """Remove a dataset.

        Args:
            name: Dataset name
            force: Skip confirmation

        Raises:
            DatasetError: If dataset not found or removal fails
        """
        remove_op = RemoveOperation()
        remove_op.execute(name, force=force, dry_run=False)
    
    def search_datasets(
        self,
        pattern: str,
        search_fields: Optional[List[str]] = None
    ) -> List[DatasetInfo]:
        """Search datasets by pattern.

        Args:
            pattern: Search pattern (case-insensitive)
            search_fields: Fields to search in (default: name, description, tags)

        Returns:
            List of matching datasets
        """
        if search_fields is None:
            search_fields = ["name", "description", "tags"]
        
        return self.manager.search_datasets(pattern, search_fields)
    
    def search_datasets_by_tag(self, tag: str) -> List[DatasetInfo]:
        """Search datasets by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of datasets with the tag
        """
        datasets = self.manager.list_datasets()
        return [d for d in datasets if tag in d.tags]
    
    def get_statistics(self, name: str, full: bool = False) -> Optional[Dict[str, Any]]:
        """Get dataset statistics.

        Args:
            name: Dataset name
            full: Whether to compute full statistics (slow for large datasets)

        Returns:
            Statistics dictionary or None if not found

        Raises:
            DatasetError: If dataset not found
        """
        dataset = self.manager.get_dataset(name)
        if not dataset:
            raise DatasetError(f"Dataset '{name}' not found")

        # Return cached statistics if available and not requesting full
        # But only if they have the full structure (not just initial stats)
        if not full and dataset.metadata.get("statistics"):
            cached_stats = dataset.metadata["statistics"]
            # Check if this is a full statistics structure
            if 'tables' in cached_stats and 'dataset_name' in cached_stats:
                return cached_stats
            # Otherwise, these are just initial stats from registration, need to compute

        # Otherwise compute statistics
        from mdm.dataset.statistics import compute_dataset_statistics

        # Call the statistics function with correct signature
        stats = compute_dataset_statistics(
            dataset_name=name,
            full=full,
            save=full  # Save if computing full stats
        )

        return stats
    
    def get_dataset_connection(self, name: str):
        """Get database connection for dataset.

        Args:
            name: Dataset name

        Returns:
            SQLAlchemy engine object

        Raises:
            DatasetError: If dataset not found
        """
        dataset = self.manager.get_dataset(name)
        if not dataset:
            raise DatasetError(f"Dataset '{name}' not found")

        from mdm.storage.factory import BackendFactory
        
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
        
        return backend.get_engine(db_path)