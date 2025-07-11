"""
MDM - ML Data Manager

A standalone, enterprise-grade dataset management system for machine learning.
"""

import os

# Suppress ydata-profiling promotional banner
os.environ["YDATA_SUPPRESS_BANNER"] = "1"

# Suppress all tqdm progress bars globally
os.environ["TQDM_DISABLE"] = "1"

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("mdm")
except PackageNotFoundError:
    # Package is not installed, use development version
    __version__ = "0.0.0+dev"

__author__ = "MDM Development Team"

# Lazy imports - only import when actually used
# This significantly speeds up CLI startup time

def __getattr__(name):
    """Lazy import mechanism for heavy modules."""
    # Main client
    if name == "MDMClient":
        from mdm.api import MDMClient
        return MDMClient
    
    # Convenience functions
    elif name == "load_dataset":
        from mdm.api import load_dataset
        return load_dataset
    elif name == "list_datasets":
        from mdm.api import list_datasets
        return list_datasets
    elif name == "get_dataset_info":
        from mdm.api import get_dataset_info
        return get_dataset_info
    
    # Core classes
    elif name == "DatasetManager":
        from mdm.dataset.manager import DatasetManager
        return DatasetManager
    elif name == "DatasetInfo":
        from mdm.models.dataset import DatasetInfo
        return DatasetInfo
    
    # Configuration
    elif name == "get_config":
        from mdm.config import get_config
        return get_config
    
    # Exceptions - this one is lightweight, we can import it directly
    elif name == "MDMError":
        from mdm.core.exceptions import MDMError
        return MDMError
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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