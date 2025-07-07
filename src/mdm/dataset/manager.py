"""Dataset manager for MDM."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import yaml

from mdm.config import get_config
from mdm.core.exceptions import DatasetError, StorageError
from mdm.models.dataset import DatasetInfo, DatasetStatistics
from mdm.storage.factory import BackendFactory
from mdm.utils.serialization import serialize_for_yaml

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset operations including registration, loading, and metadata."""

    def __init__(self, datasets_path: Optional[Path] = None):
        """Initialize dataset manager.
        
        Args:
            datasets_path: Optional path to datasets directory.
                          If not provided, uses config default.
        """
        from mdm.config import get_config_manager
        config_manager = get_config_manager()
        config = config_manager.config
        self.config = config
        self.base_path = config_manager.base_path
        
        # Get datasets path
        if datasets_path:
            self.datasets_path = datasets_path
        else:
            self.datasets_path = self.base_path / config.paths.datasets_path
        self.datasets_path.mkdir(parents=True, exist_ok=True)

        # Also ensure dataset registry directory exists
        self.dataset_registry_dir = self.base_path / config.paths.configs_path
        self.dataset_registry_dir.mkdir(parents=True, exist_ok=True)

    def register_dataset(self, dataset_info: DatasetInfo) -> None:
        """Register a new dataset.
        
        Args:
            dataset_info: Dataset information
            
        Raises:
            DatasetError: If dataset already exists or registration fails
        """
        # Normalize name for storage
        dataset_name = dataset_info.name.lower()
        dataset_path = self.datasets_path / dataset_name

        # Check if YAML registry exists (more reliable than directory check)
        yaml_path = self.dataset_registry_dir / f"{dataset_name}.yaml"
        if yaml_path.exists():
            raise DatasetError(f"Dataset '{dataset_name}' already exists")

        try:
            # Create dataset directory
            dataset_path.mkdir(parents=True, exist_ok=True)

            # Save dataset info to JSON (for backward compatibility)
            info_path = dataset_path / "dataset_info.json"
            with open(info_path, 'w') as f:
                json.dump(dataset_info.model_dump(), f, indent=2, default=str)

            # Also save to YAML in registry directory
            yaml_path = self.dataset_registry_dir / f"{dataset_name}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(serialize_for_yaml(dataset_info.model_dump()), f, default_flow_style=False, sort_keys=False)

            # Initialize metadata directory
            metadata_path = dataset_path / "metadata"
            metadata_path.mkdir(exist_ok=True)

            logger.info(f"Dataset '{dataset_name}' registered successfully")

        except Exception as e:
            # Clean up on failure
            if dataset_path.exists():
                import shutil
                shutil.rmtree(dataset_path)
            raise DatasetError(f"Failed to register dataset: {e}") from e

    def get_dataset(self, name: str) -> Optional[DatasetInfo]:
        """Get dataset information.
        
        Args:
            name: Dataset name (case-insensitive)
            
        Returns:
            DatasetInfo or None if not found
        """
        dataset_name = name.lower()

        # Try YAML first (new format)
        yaml_path = self.dataset_registry_dir / f"{dataset_name}.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)
                return DatasetInfo(**data)
            except Exception as e:
                logger.error(f"Failed to load dataset '{dataset_name}' from YAML: {e}")

        # Fall back to JSON (backward compatibility)
        info_path = self.datasets_path / dataset_name / "dataset_info.json"
        if info_path.exists():
            try:
                with open(info_path) as f:
                    data = json.load(f)
                return DatasetInfo(**data)
            except Exception as e:
                logger.error(f"Failed to load dataset '{dataset_name}' from JSON: {e}")

        return None

    def update_dataset(self, name: str, updates: dict[str, Any]) -> DatasetInfo:
        """Update dataset information.
        
        Args:
            name: Dataset name
            updates: Fields to update
            
        Returns:
            Updated DatasetInfo
            
        Raises:
            DatasetError: If dataset not found or update fails
        """
        dataset_info = self.get_dataset(name)
        if not dataset_info:
            raise DatasetError(f"Dataset '{name}' not found")

        try:
            # Update fields
            for key, value in updates.items():
                if hasattr(dataset_info, key):
                    setattr(dataset_info, key, value)

            # Update timestamp
            dataset_info.last_updated_at = datetime.now(timezone.utc)

            # Save updated info to both locations
            dataset_name = name.lower()

            # Update JSON (backward compatibility)
            info_path = self.datasets_path / dataset_name / "dataset_info.json"
            if info_path.exists():
                with open(info_path, 'w') as f:
                    json.dump(dataset_info.model_dump(), f, indent=2, default=str)

            # Update YAML (primary format)
            yaml_path = self.dataset_registry_dir / f"{dataset_name}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(serialize_for_yaml(dataset_info.model_dump()), f, default_flow_style=False, sort_keys=False)

            logger.info(f"Dataset '{name}' updated successfully")
            return dataset_info

        except Exception as e:
            raise DatasetError(f"Failed to update dataset: {e}") from e

    def list_datasets(self) -> list[DatasetInfo]:
        """List all registered datasets.
        
        Returns:
            List of DatasetInfo objects
        """
        datasets = []
        seen_names = set()

        # First, scan YAML files in registry (primary source)
        for yaml_file in self.dataset_registry_dir.glob("*.yaml"):
            try:
                dataset_info = self.get_dataset(yaml_file.stem)
                if dataset_info and dataset_info.name not in seen_names:
                    datasets.append(dataset_info)
                    seen_names.add(dataset_info.name)
            except Exception as e:
                logger.error(f"Failed to load dataset from {yaml_file}: {e}")

        # Then scan dataset directories for any missing (backward compatibility)
        for dataset_dir in self.datasets_path.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name not in seen_names:
                dataset_info = self.get_dataset(dataset_dir.name)
                if dataset_info:
                    datasets.append(dataset_info)
                    seen_names.add(dataset_info.name)

        return sorted(datasets, key=lambda d: d.name)

    def dataset_exists(self, name: str) -> bool:
        """Check if dataset exists.
        
        Args:
            name: Dataset name (case-insensitive)
            
        Returns:
            True if dataset exists
        """
        dataset_name = name.lower()
        # Check both YAML registry and dataset directory
        yaml_exists = (self.dataset_registry_dir / f"{dataset_name}.yaml").exists()
        dir_exists = (self.datasets_path / dataset_name).exists()
        return yaml_exists or dir_exists

    def delete_dataset(self, name: str, force: bool = False) -> None:
        """Delete a dataset.
        
        Args:
            name: Dataset name
            force: Force deletion without confirmation
            
        Raises:
            DatasetError: If dataset not found or deletion fails
        """
        dataset_name = name.lower()
        dataset_path = self.datasets_path / dataset_name

        if not dataset_path.exists():
            raise DatasetError(f"Dataset '{name}' not found")

        if not force:
            # In a real CLI, this would prompt for confirmation
            logger.warning(f"Deleting dataset '{name}' without confirmation (force=False not implemented)")

        try:
            # Remove YAML config first (atomic operation)
            yaml_path = self.dataset_registry_dir / f"{dataset_name}.yaml"
            if yaml_path.exists():
                yaml_path.unlink()
                logger.info(f"Removed YAML config: {yaml_path}")

            # Then remove dataset directory
            import shutil
            shutil.rmtree(dataset_path)
            logger.info(f"Dataset '{name}' deleted successfully")
        except Exception as e:
            raise DatasetError(f"Failed to delete dataset: {e}") from e

    def remove_dataset(self, name: str) -> None:
        """Remove a registered dataset (alias for delete_dataset).
        
        Args:
            name: Dataset name
            
        Raises:
            DatasetError: If dataset not found or deletion fails
        """
        self.delete_dataset(name, force=True)

    def save_statistics(self, name: str, statistics: DatasetStatistics) -> None:
        """Save dataset statistics.
        
        Args:
            name: Dataset name
            statistics: Dataset statistics
            
        Raises:
            DatasetError: If save fails
        """
        dataset_name = name.lower()
        metadata_path = self.datasets_path / dataset_name / "metadata"

        if not metadata_path.exists():
            raise DatasetError(f"Dataset '{name}' not found")

        try:
            stats_path = metadata_path / "statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(statistics.model_dump(), f, indent=2, default=str)

            logger.info(f"Statistics saved for dataset '{name}'")

        except Exception as e:
            raise DatasetError(f"Failed to save statistics: {e}") from e

    def get_statistics(self, name: str) -> Optional[DatasetStatistics]:
        """Get dataset statistics.
        
        Args:
            name: Dataset name
            
        Returns:
            DatasetStatistics or None if not found
        """
        dataset_name = name.lower()
        stats_path = self.datasets_path / dataset_name / "metadata" / "statistics.json"

        if not stats_path.exists():
            return None

        try:
            with open(stats_path) as f:
                data = json.load(f)
            return DatasetStatistics(**data)
        except Exception as e:
            logger.error(f"Failed to load statistics for '{name}': {e}")
            return None

    def get_backend(self, dataset_name: str) -> Any:
        """Get storage backend for a dataset.
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Storage backend instance
            
        Raises:
            DatasetError: If dataset not found
            StorageError: If backend creation fails
        """
        dataset_info = self.get_dataset(dataset_name)
        if not dataset_info:
            raise DatasetError(f"Dataset '{dataset_name}' not found")

        # Get backend type and config
        config = get_config()
        backend_type = dataset_info.database.get('backend', config.database.default_backend)

        # Merge dataset-specific config with global config
        backend_config = getattr(config.database, backend_type).model_dump()
        backend_config.update(dataset_info.database)

        try:
            return BackendFactory.create(backend_type, backend_config)
        except Exception as e:
            raise StorageError(f"Failed to create backend for dataset '{dataset_name}': {e}") from e

    def validate_dataset_name(self, name: str) -> str:
        """Validate and normalize dataset name.
        
        Args:
            name: Dataset name to validate
            
        Returns:
            Normalized dataset name
            
        Raises:
            DatasetError: If name is invalid
        """
        if not name:
            raise DatasetError("Dataset name cannot be empty")

        # Normalize to lowercase
        normalized = name.lower()

        # Check for valid characters
        if not all(c.isalnum() or c in "_-" for c in normalized):
            raise DatasetError(
                "Dataset name can only contain alphanumeric characters, underscores, and dashes"
            )

        # Check length
        if len(normalized) > 100:
            raise DatasetError("Dataset name cannot exceed 100 characters")

        return normalized

    def export_metadata(self, name: str, output_path: Path) -> None:
        """Export dataset metadata to a file.
        
        Args:
            name: Dataset name
            output_path: Path to export metadata to
            
        Raises:
            DatasetError: If export fails
        """
        dataset_info = self.get_dataset(name)
        if not dataset_info:
            raise DatasetError(f"Dataset '{name}' not found")

        try:
            metadata = {
                'dataset_info': dataset_info.model_dump(),
                'statistics': None
            }

            # Add statistics if available
            stats = self.get_statistics(name)
            if stats:
                metadata['statistics'] = stats.model_dump()

            # Export based on file extension
            if output_path.suffix.lower() == '.yaml':
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(serialize_for_yaml(metadata), f, default_flow_style=False)
            else:
                # Default to JSON
                with open(output_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Metadata exported to {output_path}")

        except Exception as e:
            raise DatasetError(f"Failed to export metadata: {e}") from e

    def search_datasets(
        self,
        query: str,
        deep: bool = False,
        case_sensitive: bool = False
    ) -> List[DatasetInfo]:
        """Search for datasets matching query.
        
        Args:
            query: Search query string
            deep: Whether to search in database metadata (slower)
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            List of matching DatasetInfo objects
        """
        matches = []

        # Process query
        if not case_sensitive:
            query = query.lower()

        # Quick search in YAML files
        for yaml_file in self.dataset_registry_dir.glob("*.yaml"):
            try:
                # Check filename first
                filename = yaml_file.stem
                if not case_sensitive:
                    filename = filename.lower()

                if query in filename:
                    dataset_info = self.get_dataset(yaml_file.stem)
                    if dataset_info:
                        matches.append(dataset_info)
                        continue

                # Check file contents
                with open(yaml_file) as f:
                    content = f.read()
                    if not case_sensitive:
                        content = content.lower()

                    if query in content:
                        dataset_info = self.get_dataset(yaml_file.stem)
                        if dataset_info:
                            matches.append(dataset_info)

                # Deep search would require opening databases
                if deep:
                    # TODO: Implement deep search in database metadata
                    pass

            except Exception as e:
                logger.error(f"Error searching {yaml_file}: {e}")

        return matches
