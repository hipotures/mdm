#!/usr/bin/env python3
"""
Example of using the new dependency injection system in MDM.

This shows how the simplified DI container provides:
- Constructor injection
- Multiple lifetimes
- Configuration injection
- Scoped services
"""

from mdm.core import configure_services, get_service, create_scope
from mdm.config import get_config
from mdm.api import MDMClient
from mdm.dataset.manager import DatasetManager
from mdm.features.generator import FeatureGenerator


def main():
    """Demonstrate dependency injection usage."""
    
    # 1. Configure services at application startup
    config = get_config()
    configure_services(config.model_dump())
    
    print("=== Dependency Injection Example ===\n")
    
    # 2. Get services from container
    # Singleton - same instance every time
    manager1 = get_service(DatasetManager)
    manager2 = get_service(DatasetManager)
    print(f"DatasetManager is singleton: {manager1 is manager2}")  # True
    
    # Transient - new instance every time
    from mdm.dataset.registrar import DatasetRegistrar
    registrar1 = get_service(DatasetRegistrar)
    registrar2 = get_service(DatasetRegistrar)
    print(f"DatasetRegistrar is transient: {registrar1 is registrar2}")  # False
    
    # 3. Use scoped services
    print("\n--- Scoped Services ---")
    with create_scope():
        # Within scope, scoped services are singletons
        client1 = get_service(MDMClient)
        client2 = get_service(MDMClient)
        print(f"Within scope, MDMClient instances are same: {client1 is client2}")  # True
        
    # Outside scope, new instances
    with create_scope():
        client3 = get_service(MDMClient)
        print(f"Different scope, different instance: {client1 is client3}")  # False
    
    # 4. Constructor injection happens automatically
    print("\n--- Constructor Injection ---")
    # When we get MDMClient, its dependencies are automatically injected
    client = get_service(MDMClient)
    print(f"MDMClient has registration client: {client.registration is not None}")
    print(f"MDMClient has query client: {client.query is not None}")
    
    # 5. Using the @inject decorator for method injection
    from mdm.core import inject
    
    @inject
    def process_data(manager: DatasetManager, generator: FeatureGenerator):
        """This function gets its dependencies injected."""
        print(f"\nInjected manager: {type(manager).__name__}")
        print(f"Injected generator: {type(generator).__name__}")
        return manager, generator
    
    # Call without parameters - they're injected!
    m, g = process_data()
    
    # 6. Manual usage without DI (still works)
    print("\n--- Manual Construction ---")
    # You can still create objects manually if needed
    manual_manager = DatasetManager()
    print(f"Manual DatasetManager created: {type(manual_manager).__name__}")
    
    # 7. Configuration injection
    print("\n--- Configuration Access ---")
    # Services can access config through the container
    print(f"Default backend: {config.database.default_backend}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()