"""New CLI configuration implementation.

This module provides enhanced configuration management for the CLI
with support for command-specific settings and user preferences.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import yaml

from ...interfaces.cli import ICLIConfig
from ...config import get_config_manager

logger = logging.getLogger(__name__)


class NewCLIConfig(ICLIConfig):
    """New CLI configuration with enhanced features."""
    
    def __init__(self):
        """Initialize CLI configuration."""
        self._config = get_config_manager()
        self._cli_config_file = Path.home() / '.mdm' / 'cli_config.yaml'
        self._user_prefs = self._load_user_preferences()
        self._command_configs = self._load_command_configs()
        logger.info("Initialized NewCLIConfig")
    
    @property
    def output_format(self) -> str:
        """Default output format."""
        # Check in order: user prefs, config file, default
        return (
            self._user_prefs.get('output_format') or
            self._config.get('cli.output_format', 'table')
        )
    
    @property
    def color_enabled(self) -> bool:
        """Whether color output is enabled."""
        # Check environment variable first
        import os
        if os.environ.get('NO_COLOR'):
            return False
        
        return (
            self._user_prefs.get('color', True) and
            self._config.get('cli.color', True)
        )
    
    @property
    def verbose(self) -> bool:
        """Whether verbose output is enabled."""
        return (
            self._user_prefs.get('verbose', False) or
            self._config.get('cli.verbose', False)
        )
    
    @property
    def quiet(self) -> bool:
        """Whether quiet mode is enabled."""
        return (
            self._user_prefs.get('quiet', False) or
            self._config.get('cli.quiet', False)
        )
    
    def get_command_config(self, command: str) -> Dict[str, Any]:
        """Get configuration for specific command.
        
        Args:
            command: Command name (e.g., 'dataset.register')
            
        Returns:
            Command-specific configuration
        """
        # Check multiple sources
        config = {}
        
        # Base config from main config file
        base_config = self._config.get(f'cli.commands.{command}', {})
        if isinstance(base_config, dict):
            config.update(base_config)
        
        # Command-specific config file
        if command in self._command_configs:
            config.update(self._command_configs[command])
        
        # User preferences override
        user_command_config = self._user_prefs.get('commands', {}).get(command, {})
        if isinstance(user_command_config, dict):
            config.update(user_command_config)
        
        return config
    
    def set_user_preference(self, key: str, value: Any) -> None:
        """Set a user preference.
        
        Args:
            key: Preference key
            value: Preference value
        """
        self._user_prefs[key] = value
        self._save_user_preferences()
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference.
        
        Args:
            key: Preference key
            default: Default value if not found
            
        Returns:
            Preference value
        """
        return self._user_prefs.get(key, default)
    
    def reset_preferences(self) -> None:
        """Reset all user preferences to defaults."""
        self._user_prefs = {}
        self._save_user_preferences()
        logger.info("Reset all user preferences")
    
    def get_aliases(self) -> Dict[str, str]:
        """Get command aliases.
        
        Returns:
            Dictionary of alias -> command mappings
        """
        aliases = {}
        
        # System aliases
        system_aliases = self._config.get('cli.aliases', {})
        if isinstance(system_aliases, dict):
            aliases.update(system_aliases)
        
        # User aliases
        user_aliases = self._user_prefs.get('aliases', {})
        if isinstance(user_aliases, dict):
            aliases.update(user_aliases)
        
        return aliases
    
    def get_theme(self) -> Dict[str, Any]:
        """Get CLI theme configuration.
        
        Returns:
            Theme configuration
        """
        theme = {
            'table_style': 'rounded',
            'syntax_theme': 'monokai',
            'progress_style': 'default',
            'panel_style': 'blue',
            'error_style': 'red',
            'warning_style': 'yellow',
            'success_style': 'green',
            'info_style': 'blue'
        }
        
        # Override with config
        config_theme = self._config.get('cli.theme', {})
        if isinstance(config_theme, dict):
            theme.update(config_theme)
        
        # Override with user preferences
        user_theme = self._user_prefs.get('theme', {})
        if isinstance(user_theme, dict):
            theme.update(user_theme)
        
        return theme
    
    def get_defaults(self, command: str) -> Dict[str, Any]:
        """Get default arguments for a command.
        
        Args:
            command: Command name
            
        Returns:
            Default arguments
        """
        defaults = {}
        
        # Get command config
        cmd_config = self.get_command_config(command)
        
        # Extract defaults
        if 'defaults' in cmd_config and isinstance(cmd_config['defaults'], dict):
            defaults.update(cmd_config['defaults'])
        
        return defaults
    
    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences from file."""
        if self._cli_config_file.exists():
            try:
                with open(self._cli_config_file, 'r') as f:
                    prefs = yaml.safe_load(f) or {}
                    logger.debug(f"Loaded user preferences from {self._cli_config_file}")
                    return prefs
            except Exception as e:
                logger.warning(f"Failed to load user preferences: {e}")
        
        return {}
    
    def _save_user_preferences(self) -> None:
        """Save user preferences to file."""
        try:
            self._cli_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cli_config_file, 'w') as f:
                yaml.dump(self._user_prefs, f, default_flow_style=False)
            logger.debug(f"Saved user preferences to {self._cli_config_file}")
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
    
    def _load_command_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load command-specific configuration files."""
        configs = {}
        
        # Look for command config directory
        cmd_config_dir = Path.home() / '.mdm' / 'cli' / 'commands'
        if cmd_config_dir.exists():
            for config_file in cmd_config_dir.glob('*.yaml'):
                try:
                    with open(config_file, 'r') as f:
                        command_name = config_file.stem
                        configs[command_name] = yaml.safe_load(f) or {}
                        logger.debug(f"Loaded config for command: {command_name}")
                except Exception as e:
                    logger.warning(f"Failed to load command config {config_file}: {e}")
        
        return configs