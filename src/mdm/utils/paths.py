"""Path utilities for MDM."""

from pathlib import Path
from typing import Optional

from mdm.config import get_config


class PathManager:
    """Manages MDM paths."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize path manager.

        Args:
            base_path: Base path for MDM (defaults to ~/.mdm)
        """
        self.base_path = base_path or Path.home() / ".mdm"
        self._config = get_config()

    @property
    def datasets_path(self) -> Path:
        """Get datasets path."""
        return self._config.get_full_path("datasets_path", self.base_path)

    @property
    def configs_path(self) -> Path:
        """Get configs path."""
        return self._config.get_full_path("configs_path", self.base_path)

    @property
    def logs_path(self) -> Path:
        """Get logs path."""
        return self._config.get_full_path("logs_path", self.base_path)

    @property
    def custom_features_path(self) -> Path:
        """Get custom features path."""
        return self._config.get_full_path("custom_features_path", self.base_path)

    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get path for specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Path to dataset directory
        """
        return self.datasets_path / dataset_name.lower()

    def get_dataset_db_path(self, dataset_name: str, backend: str) -> Path:
        """Get database file path for dataset.

        Args:
            dataset_name: Name of the dataset
            backend: Database backend type

        Returns:
            Path to database file
        """
        dataset_path = self.get_dataset_path(dataset_name)

        if backend == "sqlite":
            return dataset_path / "dataset.sqlite"
        if backend == "duckdb":
            return dataset_path / "dataset.duckdb"
        # PostgreSQL doesn't use file path
        return dataset_path / "dataset.info"

    def get_dataset_config_path(self, dataset_name: str) -> Path:
        """Get configuration file path for dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Path to dataset config file
        """
        return self.configs_path / f"{dataset_name.lower()}.yaml"

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for path in [
            self.datasets_path,
            self.configs_path,
            self.logs_path,
            self.custom_features_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global path manager instance
_path_manager: Optional[PathManager] = None


def get_path_manager() -> PathManager:
    """Get global path manager instance."""
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager()
    return _path_manager

