"""Feature engineering domain layer.

This layer contains pure business logic for feature generation,
without any infrastructure or I/O concerns.
"""

from .feature_service import FeatureService
from .feature_repository import IFeatureRepository
from .value_objects import (
    FeatureSet,
    FeatureMetadata,
    FeatureGenerationConfig,
    BatchProcessingConfig
)
from .interfaces import (
    IFeatureLoader,
    ICustomFeatureLoader,
    IFeatureWriter
)

__all__ = [
    # Services
    "FeatureService",
    
    # Repository
    "IFeatureRepository",
    
    # Value Objects
    "FeatureSet",
    "FeatureMetadata",
    "FeatureGenerationConfig",
    "BatchProcessingConfig",
    
    # Interfaces
    "IFeatureLoader",
    "ICustomFeatureLoader",
    "IFeatureWriter",
]