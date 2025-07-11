"""
Modern dependency injection system for MDM.

This provides a simplified DI container focused on:
- Type-based registration and resolution
- Constructor injection
- Scoped lifetime management
- Configuration injection
"""
from typing import Dict, Type, Any, Callable, Optional, TypeVar, get_type_hints, get_args, get_origin
from dataclasses import dataclass
from contextlib import contextmanager
from functools import wraps
import inspect
import logging
from abc import ABC

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifetime:
    """Service lifetime options."""
    TRANSIENT = "transient"  # New instance each time
    SINGLETON = "singleton"  # Single instance for app lifetime
    SCOPED = "scoped"       # Single instance per scope


@dataclass
class ServiceDescriptor:
    """Describes how to create a service."""
    service_type: Type
    implementation: Type | Callable | Any
    lifetime: str = ServiceLifetime.TRANSIENT
    factory: Optional[Callable] = None


class ServiceNotRegisteredError(Exception):
    """Raised when a requested service is not registered."""
    pass


class Container:
    """
    Modern dependency injection container for MDM.
    
    Features:
    - Constructor-based injection
    - Multiple lifetimes (transient, singleton, scoped)
    - Type-safe resolution
    - Configuration injection
    - Simple API
    """
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._config: Dict[str, Any] = {}
        
    def add_transient(self, service_type: Type[T], implementation: Type[T] | Callable[[], T]) -> 'Container':
        """Register a transient service (new instance each time)."""
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT
        )
        return self
        
    def add_singleton(self, service_type: Type[T], implementation: Type[T] | Callable[[], T] | T) -> 'Container':
        """Register a singleton service (one instance for app lifetime)."""
        # If it's already an instance, store it directly
        if not (inspect.isclass(implementation) or callable(implementation)):
            self._singletons[service_type] = implementation
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                implementation=implementation,
                lifetime=ServiceLifetime.SINGLETON
            )
        else:
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                implementation=implementation,
                lifetime=ServiceLifetime.SINGLETON
            )
        return self
        
    def add_scoped(self, service_type: Type[T], implementation: Type[T] | Callable[[], T]) -> 'Container':
        """Register a scoped service (one instance per scope)."""
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            lifetime=ServiceLifetime.SCOPED
        )
        return self
    
    def add_config(self, config: Dict[str, Any]) -> 'Container':
        """Add configuration values."""
        self._config.update(config)
        return self
        
    def get(self, service_type: Type[T]) -> T:
        """Resolve a service."""
        # Check if it's a configuration request
        if service_type == dict and hasattr(service_type, '__args__'):
            return self._config
            
        if service_type not in self._services:
            raise ServiceNotRegisteredError(f"Service {service_type} is not registered")
            
        descriptor = self._services[service_type]
        
        # Handle different lifetimes
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type not in self._singletons:
                self._singletons[service_type] = self._create_instance(descriptor)
            return self._singletons[service_type]
            
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if service_type not in self._scoped_instances:
                self._scoped_instances[service_type] = self._create_instance(descriptor)
            return self._scoped_instances[service_type]
            
        else:  # TRANSIENT
            return self._create_instance(descriptor)
            
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create an instance of a service."""
        implementation = descriptor.implementation
        
        # If it's already an instance, return it
        if not (inspect.isclass(implementation) or callable(implementation)):
            return implementation
            
        # If it's a factory function
        if callable(implementation) and not inspect.isclass(implementation):
            # Check if factory needs injection
            sig = inspect.signature(implementation)
            if sig.parameters:
                # Inject dependencies into factory
                kwargs = self._resolve_dependencies(implementation)
                return implementation(**kwargs)
            else:
                return implementation()
                
        # It's a class - use constructor injection
        return self._instantiate_class(implementation)
        
    def _instantiate_class(self, cls: Type) -> Any:
        """Instantiate a class with constructor injection."""
        # Get constructor parameters
        kwargs = self._resolve_dependencies(cls)
        
        # Create instance
        return cls(**kwargs)
        
    def _resolve_dependencies(self, target: Type | Callable) -> Dict[str, Any]:
        """Resolve dependencies for a class or function."""
        # Get the signature
        if inspect.isclass(target):
            sig = inspect.signature(target.__init__)
        else:
            sig = inspect.signature(target)
            
        kwargs = {}
        
        # Get type hints for better resolution
        hints = get_type_hints(target) if inspect.isclass(target) else get_type_hints(target)
        
        for param_name, param in sig.parameters.items():
            # Skip self
            if param_name == 'self':
                continue
                
            # Try to resolve by type annotation
            if param.annotation != param.empty:
                param_type = param.annotation
                
                # Handle Optional types
                if get_origin(param_type) is type(Optional):
                    args = get_args(param_type)
                    if args:
                        param_type = args[0]
                        
                try:
                    # Special handling for config dict
                    if param_type == dict and param_name == 'config':
                        kwargs[param_name] = self._config
                    else:
                        kwargs[param_name] = self.get(param_type)
                except ServiceNotRegisteredError:
                    # If optional and not found, use default
                    if param.default != param.empty:
                        kwargs[param_name] = param.default
                    # Otherwise let it fail with a clear error
                    
        return kwargs
        
    @contextmanager
    def create_scope(self):
        """Create a new scope for scoped services."""
        # Save current scoped instances
        old_scoped = self._scoped_instances.copy()
        self._scoped_instances.clear()
        
        try:
            yield self
        finally:
            # Restore old scoped instances
            self._scoped_instances = old_scoped
            
    def clear(self):
        """Clear all registrations."""
        self._services.clear()
        self._singletons.clear()
        self._scoped_instances.clear()
        self._config.clear()


# Global container instance
_container = Container()


def get_container() -> Container:
    """Get the global container instance."""
    return _container


def configure_services(config: Dict[str, Any]) -> Container:
    """
    Configure the DI container with MDM services.
    
    This is called once at startup to register all services.
    """
    container = get_container()
    container.clear()
    
    # Add configuration
    container.add_config(config)
    
    # Register storage backends
    from ..storage.backends.stateless_sqlite import StatelessSQLiteBackend
    from ..storage.backends.stateless_duckdb import StatelessDuckDBBackend
    from ..storage.factory import BackendFactory
    
    backend_type = config.get("database", {}).get("default_backend", "sqlite")
    
    # Register the appropriate backend as singleton
    if backend_type == "sqlite":
        container.add_singleton(StatelessSQLiteBackend, StatelessSQLiteBackend)
    elif backend_type == "duckdb":
        container.add_singleton(StatelessDuckDBBackend, StatelessDuckDBBackend)
        
    # Register backend factory
    container.add_singleton(BackendFactory, BackendFactory)
    
    # Register feature services
    from ..features.generator import FeatureGenerator
    container.add_singleton(FeatureGenerator, FeatureGenerator)
    
    # Register dataset services
    from ..dataset.registrar import DatasetRegistrar
    from ..dataset.manager import DatasetManager
    
    container.add_transient(DatasetRegistrar, DatasetRegistrar)
    container.add_singleton(DatasetManager, DatasetManager)
    
    # Register API clients
    from ..api.clients import (
        RegistrationClient, QueryClient, ManagementClient,
        ExportClient, MLIntegrationClient
    )
    from ..api.mdm_client import MDMClient
    
    container.add_scoped(RegistrationClient, RegistrationClient)
    container.add_scoped(QueryClient, QueryClient)
    container.add_scoped(ManagementClient, ManagementClient)
    container.add_scoped(ExportClient, ExportClient)
    container.add_scoped(MLIntegrationClient, MLIntegrationClient)
    container.add_scoped(MDMClient, MDMClient)
    
    logger.info(f"Configured {len(container._services)} services")
    return container


def inject(func: Callable) -> Callable:
    """
    Decorator for method injection.
    
    This is simpler than the old version - it only injects into
    functions, not constructors (which are handled automatically).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        container = get_container()
        
        # Get function signature
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        
        # Inject missing parameters
        for param_name, param in sig.parameters.items():
            if param_name not in bound.arguments and param.annotation != param.empty:
                try:
                    bound.arguments[param_name] = container.get(param.annotation)
                except ServiceNotRegisteredError:
                    pass  # Let it use default or fail naturally
                    
        return func(**bound.arguments)
        
    return wrapper


# Convenience functions
def get_service(service_type: Type[T]) -> T:
    """Get a service from the container."""
    return get_container().get(service_type)


def create_scope():
    """Create a new DI scope."""
    return get_container().create_scope()