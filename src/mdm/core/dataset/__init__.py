"""Core dataset management implementation.

This package provides the new dataset registration and management implementation
that follows clean architecture principles and supports gradual migration.
"""

from .registrar import NewDatasetRegistrar
from .manager import NewDatasetManager
from .validators import (
    DatasetNameValidator,
    DatasetPathValidator,
    DatasetStructureDetector,
)
from .loaders import (
    DataLoader,
    CSVLoader,
    ParquetLoader,
    ExcelLoader,
    JSONLoader,
)

__all__ = [
    'NewDatasetRegistrar',
    'NewDatasetManager',
    'DatasetNameValidator',
    'DatasetPathValidator',
    'DatasetStructureDetector',
    'DataLoader',
    'CSVLoader',
    'ParquetLoader',
    'ExcelLoader',
    'JSONLoader',
]
