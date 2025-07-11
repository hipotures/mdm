"""Base client class for MDM API clients."""

from typing import Optional
from mdm.config.config import MDMConfig
from mdm.dataset.manager import DatasetManager


class BaseClient:
    """Base class for MDM client classes."""
    
    def __init__(
        self,
        manager: DatasetManager,
        config: Optional[MDMConfig] = None
    ):
        """Initialize base client with dependency injection.
        
        Args:
            manager: DatasetManager instance (required)
            config: Optional configuration object
        """
        self.manager = manager
        self.config = config
        
        # If config not provided, get from manager's config if available
        if self.config is None and hasattr(manager, 'config'):
            self.config = manager.config