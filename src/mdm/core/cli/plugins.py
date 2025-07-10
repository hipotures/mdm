"""CLI plugin system for MDM.

This module provides a plugin architecture for extending CLI functionality
with custom commands and features.
"""
import logging
import importlib
import importlib.util
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
import inspect

from ...interfaces.cli import ICLIPluginManager

logger = logging.getLogger(__name__)


class CLIPlugin:
    """Base class for CLI plugins."""
    
    def __init__(self):
        """Initialize plugin."""
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.description = "CLI plugin"
        self.commands = []
    
    def register_commands(self) -> List[Dict[str, Any]]:
        """Register plugin commands.
        
        Returns:
            List of command definitions
        """
        return self.commands
    
    def on_load(self) -> None:
        """Called when plugin is loaded."""
        pass
    
    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        pass


class CLIPluginManager(ICLIPluginManager):
    """Manages CLI plugins."""
    
    def __init__(self):
        """Initialize plugin manager."""
        self._plugins: Dict[str, CLIPlugin] = {}
        self._plugin_paths: List[Path] = [
            Path.home() / '.mdm' / 'plugins',
            Path(__file__).parent / 'plugins'
        ]
        logger.info("Initialized CLIPluginManager")
    
    def register_plugin(
        self,
        name: str,
        plugin: Any,
        commands: List[str]
    ) -> None:
        """Register a CLI plugin.
        
        Args:
            name: Plugin name
            plugin: Plugin instance
            commands: List of command names provided by plugin
        """
        if name in self._plugins:
            logger.warning(f"Plugin {name} already registered, replacing")
        
        self._plugins[name] = plugin
        
        # Call plugin's on_load method
        if hasattr(plugin, 'on_load'):
            try:
                plugin.on_load()
            except Exception as e:
                logger.error(f"Error in plugin {name} on_load: {e}")
        
        logger.info(f"Registered plugin: {name} with {len(commands)} commands")
    
    def get_plugin(self, name: str) -> Optional[Any]:
        """Get a registered plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins.
        
        Returns:
            List of plugin information
        """
        plugins = []
        
        for name, plugin in self._plugins.items():
            info = {
                'name': name,
                'version': getattr(plugin, 'version', 'unknown'),
                'description': getattr(plugin, 'description', ''),
                'commands': []
            }
            
            # Get registered commands
            if hasattr(plugin, 'register_commands'):
                try:
                    commands = plugin.register_commands()
                    info['commands'] = [cmd.get('name', '') for cmd in commands]
                except Exception as e:
                    logger.error(f"Error getting commands from plugin {name}: {e}")
            
            plugins.append(info)
        
        return plugins
    
    def load_plugins(self, directory: Path) -> int:
        """Load plugins from directory.
        
        Args:
            directory: Directory containing plugins
            
        Returns:
            Number of plugins loaded
        """
        if not directory.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return 0
        
        loaded = 0
        
        # Find Python files
        for plugin_file in directory.glob("*.py"):
            if plugin_file.name.startswith('_'):
                continue
            
            try:
                # Load plugin module
                plugin_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(
                    f"mdm_plugin_{plugin_name}",
                    plugin_file
                )
                
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find plugin classes
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, CLIPlugin) and 
                            obj != CLIPlugin):
                            
                            # Instantiate plugin
                            plugin = obj()
                            commands = plugin.register_commands()
                            
                            self.register_plugin(
                                plugin_name,
                                plugin,
                                [cmd.get('name', '') for cmd in commands]
                            )
                            loaded += 1
                            break
                
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")
        
        return loaded
    
    def load_all_plugins(self) -> int:
        """Load plugins from all configured directories.
        
        Returns:
            Total number of plugins loaded
        """
        total = 0
        
        for path in self._plugin_paths:
            if path.exists():
                loaded = self.load_plugins(path)
                total += loaded
                logger.info(f"Loaded {loaded} plugins from {path}")
        
        return total
    
    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Whether plugin was unloaded
        """
        if name not in self._plugins:
            return False
        
        plugin = self._plugins[name]
        
        # Call plugin's on_unload method
        if hasattr(plugin, 'on_unload'):
            try:
                plugin.on_unload()
            except Exception as e:
                logger.error(f"Error in plugin {name} on_unload: {e}")
        
        del self._plugins[name]
        logger.info(f"Unloaded plugin: {name}")
        return True
    
    def get_plugin_commands(self, plugin_name: str) -> List[Dict[str, Any]]:
        """Get commands provided by a plugin.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            List of command definitions
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return []
        
        if hasattr(plugin, 'register_commands'):
            try:
                return plugin.register_commands()
            except Exception as e:
                logger.error(f"Error getting commands from plugin {plugin_name}: {e}")
        
        return []
    
    def execute_plugin_command(
        self,
        plugin_name: str,
        command_name: str,
        **kwargs
    ) -> Any:
        """Execute a command from a plugin.
        
        Args:
            plugin_name: Plugin name
            command_name: Command name
            **kwargs: Command arguments
            
        Returns:
            Command result
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        # Find command method
        method_name = f"cmd_{command_name}"
        if not hasattr(plugin, method_name):
            raise ValueError(f"Command not found: {command_name}")
        
        method = getattr(plugin, method_name)
        
        # Execute command
        try:
            return method(**kwargs)
        except Exception as e:
            logger.error(f"Error executing plugin command {plugin_name}.{command_name}: {e}")
            raise


# Example plugin template
PLUGIN_TEMPLATE = '''"""Example CLI plugin for MDM.

This plugin adds custom commands to the MDM CLI.
"""
from mdm.core.cli.plugins import CLIPlugin


class ExamplePlugin(CLIPlugin):
    """Example plugin implementation."""
    
    def __init__(self):
        """Initialize plugin."""
        super().__init__()
        self.name = "example"
        self.version = "1.0.0"
        self.description = "Example plugin for MDM CLI"
        
        # Define commands
        self.commands = [
            {
                'name': 'hello',
                'description': 'Say hello',
                'handler': self.cmd_hello
            },
            {
                'name': 'info',
                'description': 'Show plugin info',
                'handler': self.cmd_info
            }
        ]
    
    def cmd_hello(self, name: str = "World") -> str:
        """Say hello command.
        
        Args:
            name: Name to greet
            
        Returns:
            Greeting message
        """
        return f"Hello, {name}!"
    
    def cmd_info(self) -> dict:
        """Show plugin information.
        
        Returns:
            Plugin information
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'commands': [cmd['name'] for cmd in self.commands]
        }
'''