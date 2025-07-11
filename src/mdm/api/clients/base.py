"""Base client class for MDM API clients."""

from mdm.config import get_config
from mdm.dataset.manager import DatasetManager


class BaseClient:
    """Base class for MDM client classes."""
    
    def __init__(self, config=None, manager=None):
        """Initialize base client.
        
        Args:
            config: Optional configuration object
            manager: Optional DatasetManager instance
        """
        self.config = config or get_config()
        self.manager = manager or DatasetManager()