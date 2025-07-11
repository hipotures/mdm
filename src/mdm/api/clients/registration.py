"""Dataset registration client."""

from pathlib import Path
from typing import Optional, List

from mdm.core.exceptions import DatasetError
from mdm.dataset.registrar import DatasetRegistrar
from mdm.models.dataset import DatasetInfo

from .base import BaseClient


class RegistrationClient(BaseClient):
    """Client for dataset registration operations."""
    
    def register_dataset(
        self,
        name: str,
        dataset_path: str,
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
        problem_type: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
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
        
        # Register dataset
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