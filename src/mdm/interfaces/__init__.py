"""
Interfaces for MDM components.

These Protocol classes define the contracts that implementations must follow.
"""

from .storage import IStorageBackend
from .features import IFeatureGenerator, IFeatureTransformer
from .dataset import IDatasetRegistrar, IDatasetManager

__all__ = [
    'IStorageBackend',
    'IFeatureGenerator',
    'IFeatureTransformer',
    'IDatasetRegistrar',
    'IDatasetManager',
]