"""
MDM - ML Data Manager

A standalone, enterprise-grade dataset management system for machine learning.
"""

import os

# Suppress ydata-profiling promotional banner
os.environ["YDATA_SUPPRESS_BANNER"] = "1"

__version__ = "0.1.0"
__author__ = "MDM Development Team"

# Public API
from mdm.api import (
    MDMClient,
    get_dataset_info,
    list_datasets,
    load_dataset,
)
from mdm.config import get_config
from mdm.core.exceptions import MDMError
from mdm.dataset.manager import DatasetManager
from mdm.models.dataset import DatasetInfo

__all__ = [
    # Main client
    "MDMClient",
    
    # Convenience functions
    "load_dataset",
    "list_datasets", 
    "get_dataset_info",
    
    # Core classes
    "DatasetManager",
    "DatasetInfo",
    
    # Configuration
    "get_config",
    
    # Exceptions
    "MDMError",
    
    # Version
    "__version__",
]

