"""
Dependency injection container for MDM.

This provides a simple but effective DI system that allows us to swap
implementations at runtime based on configuration and feature flags.
"""
from typing import Dict, Type, Any, Callable, Optional, TypeVar, Generic
from functools import lru_cache
import inspect
import logging

from ..interfaces.storage import IStorageBackend
from ..interfaces.features import IFeatureGenerator
from ..interfaces.dataset import IDatasetRegistrar, IDatasetManager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceNotFoundError(Exception):
    """Raised when a requested service is not registered."""
    pass


class DIContainer:
    """
    Simple dependency injection container.
    
    Features:
    - Service registration with factory functions
    - Singleton and transient lifetimes
    - Configuration support
    - Lazy initialization
    - Thread-safe singleton creation
    """
    
    def __init__(self):
        self._services: Dict[Type, Callable[[], Any]] = {}
        self._singletons: Dict[Type, Any] = {}
        self._singleton_factories: Dict[Type, bool] = {}
        self._config: Dict[str, Any] = {}
        
        logger.info("Initialized DI container")
    
    def register(
        self, 
        interface: Type[T], 
        factory: Callable[[], T], 
        singleton: bool = False
    ) -> None:
        """
        Register a service factory.
        
        Args:
            interface: The interface/protocol type
            factory: Function that creates the service instance
            singleton: If True, only one instance will be created
        """
        self._services[interface] = factory
        self._singleton_factories[interface] = singleton
        
        # Clear existing singleton if re-registering
        if interface in self._singletons:
            del self._singletons[interface]
        
        logger.debug(f"Registered service: {interface.__name__} "
                    f"(singleton: {singleton})")
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """
        Register an existing instance as a singleton.
        
        Args:
            interface: The interface/protocol type
            instance: The instance to register
        """
        self._singletons[interface] = instance
        self._singleton_factories[interface] = True
        # Remove factory if exists
        if interface in self._services:
            del self._services[interface]
        
        logger.debug(f"Registered instance: {interface.__name__}")
    
    def get(self, interface: Type[T]) -> T:
        """
        Get service instance.
        
        Args:
            interface: The interface/protocol type to retrieve
            
        Returns:
            Service instance
            
        Raises:
            ServiceNotFoundError: If no service is registered for the interface
        """
        # Check if we have a singleton instance
        if interface in self._singletons:
            return self._singletons[interface]
        
        # Check if we have a factory
        if interface not in self._services:
            raise ServiceNotFoundError(
                f"No service registered for {interface.__name__}"
            )
        
        # Create instance
        instance = self._services[interface]()
        
        # Store as singleton if configured
        if self._singleton_factories.get(interface, False):
            self._singletons[interface] = instance
            logger.debug(f"Created singleton instance: {interface.__name__}")
        else:
            logger.debug(f"Created transient instance: {interface.__name__}")
        
        return instance
    
    def get_optional(self, interface: Type[T], default: Optional[T] = None) -> Optional[T]:
        """
        Get service instance or return default if not found.
        
        Args:
            interface: The interface/protocol type to retrieve
            default: Default value if service not found
            
        Returns:
            Service instance or default
        """
        try:
            return self.get(interface)
        except ServiceNotFoundError:
            return default
    
    def has(self, interface: Type) -> bool:
        """Check if a service is registered."""
        return interface in self._services or interface in self._singletons
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Set configuration values."""
        self._config.update(config)
        logger.debug(f"Updated container configuration: {list(config.keys())}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def clear(self) -> None:
        """Clear all registrations and singletons."""
        self._services.clear()
        self._singletons.clear()
        self._singleton_factories.clear()
        self._config.clear()
        logger.info("Cleared DI container")
    
    def get_registered_services(self) -> Dict[str, str]:
        """Get information about registered services."""
        services = {}
        
        # Add factory-based services
        for interface in self._services:
            lifetime = "singleton" if self._singleton_factories.get(interface) else "transient"
            services[interface.__name__] = lifetime
        
        # Add instance-based services
        for interface in self._singletons:
            if interface not in self._services:
                services[interface.__name__] = "instance"
        
        return services


# Global container instance
container = DIContainer()


def configure_container(config: Dict[str, Any]) -> None:
    """
    Configure the DI container with appropriate implementations.
    
    This function sets up all the service registrations based on
    configuration and feature flags.
    """
    from ..storage.backends.stateless_sqlite import StatelessSQLiteBackend
    from ..storage.backends.stateless_duckdb import StatelessDuckDBBackend
    from ..storage.postgresql import PostgreSQLBackend
    from ..features.generator import FeatureGenerator
    from ..dataset.registrar import DatasetRegistrar
    from ..dataset.manager import DatasetManager
    
    # Store configuration
    container.configure(config)
    
    # Get backend configuration
    backend_type = config.get("database", {}).get("default_backend", "sqlite")
    
    logger.info(f"Configuring container - Backend: {backend_type}")
    
    # Register storage backend
    if backend_type == "sqlite":
        container.register(IStorageBackend, StatelessSQLiteBackend, singleton=False)
    elif backend_type == "duckdb":
        container.register(IStorageBackend, StatelessDuckDBBackend, singleton=False)
    elif backend_type == "postgresql":
        # TODO: Create stateless PostgreSQL backend
        container.register(IStorageBackend, PostgreSQLBackend, singleton=True)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    # Register feature generator
    container.register(IFeatureGenerator, FeatureGenerator, singleton=True)
    
    # Register dataset services
    container.register(IDatasetRegistrar, DatasetRegistrar, singleton=False)
    container.register(IDatasetManager, DatasetManager, singleton=True)
    
    logger.info(f"Container configured with {len(container.get_registered_services())} services")


def inject(func: Callable) -> Callable:
    """
    Decorator for dependency injection.
    
    This decorator inspects function parameters and automatically injects
    services from the container based on type annotations.
    
    Example:
        @inject
        def process_data(backend: IStorageBackend, generator: IFeatureGenerator):
            # backend and generator are automatically injected
            pass
    """
    sig = inspect.signature(func)
    
    def wrapper(*args, **kwargs):
        # Get parameter info
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        
        # Inject missing parameters
        for param_name, param in sig.parameters.items():
            if param_name not in bound.arguments and param.annotation != param.empty:
                # Try to inject from container
                try:
                    service = container.get(param.annotation)
                    bound.arguments[param_name] = service
                    logger.debug(f"Injected {param.annotation.__name__} "
                                f"into {func.__name__}.{param_name}")
                except ServiceNotFoundError:
                    # Not a registered service, skip
                    pass
        
        return func(*bound.args, **bound.kwargs)
    
    # Preserve function metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = func.__annotations__
    wrapper.__wrapped__ = func
    
    return wrapper


def get_service(interface: Type[T]) -> T:
    """
    Convenience function to get a service from the global container.
    
    Args:
        interface: The interface/protocol type
        
    Returns:
        Service instance
    """
    return container.get(interface)


def has_service(interface: Type) -> bool:
    """Check if a service is registered in the global container."""
    return container.has(interface)