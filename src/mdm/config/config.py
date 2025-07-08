"""Configuration management for MDM."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from mdm.core.exceptions import ConfigError
from mdm.models.config import MDMConfig


class ConfigManager:
    """Manages MDM configuration with YAML and environment variable support."""

    CONFIG_FILE_NAME = "mdm.yaml"
    ENV_PREFIX = "MDM_"

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager.

        Args:
            config_path: Optional path to configuration file.
                        If not provided, uses ~/.mdm/mdm.yaml
        """
        # Respect MDM_HOME_DIR environment variable
        self.base_path = Path(os.environ.get("MDM_HOME_DIR", str(Path.home() / ".mdm")))
        self.config_path = config_path or self.base_path / self.CONFIG_FILE_NAME
        self._config: Optional[MDMConfig] = None

    def load(self) -> MDMConfig:
        """Load configuration from YAML file and environment variables.

        Configuration precedence:
        1. Default values (from Pydantic models)
        2. YAML file values
        3. Environment variables (highest priority)

        Returns:
            MDMConfig: Loaded configuration

        Raises:
            ConfigError: If configuration is invalid
        """
        if self._config is not None:
            return self._config

        # Start with default configuration
        config_dict: dict[str, Any] = {}

        # Load from YAML if exists
        if self.config_path.exists():
            try:
                config_dict = self._load_yaml()
            except Exception as e:
                raise ConfigError(f"Failed to load configuration from {self.config_path}: {e}") from e

        # Apply environment variables (override YAML)
        config_dict = self._apply_environment_variables(config_dict)

        # Validate and create configuration
        try:
            self._config = MDMConfig(**config_dict)
        except ValidationError as e:
            raise ConfigError(f"Invalid configuration: {e}") from e

        return self._config

    def save(self, config: MDMConfig, path: Optional[Path] = None) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration to save
            path: Optional path to save to (defaults to config_path)

        Raises:
            ConfigError: If save fails
        """
        save_path = path or self.config_path

        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict and save
            config_dict = config.model_dump(exclude_defaults=True)

            with save_path.open("w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        except Exception as e:
            raise ConfigError(f"Failed to save configuration to {save_path}: {e}") from e

    def initialize_defaults(self) -> None:
        """Initialize default configuration if it doesn't exist."""
        if not self.config_path.exists():
            # Create base directory
            self.base_path.mkdir(parents=True, exist_ok=True)

            # Create default config
            default_config = MDMConfig()
            self.save(default_config)

            # Create subdirectories
            for path_attr in ["datasets_path", "configs_path", "logs_path", "custom_features_path"]:
                full_path = default_config.get_full_path(path_attr, self.base_path)
                full_path.mkdir(parents=True, exist_ok=True)

    def _load_yaml(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        with self.config_path.open() as f:
            return yaml.safe_load(f) or {}

    def _apply_environment_variables(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variables to configuration.

        Environment variables follow the pattern:
        MDM_<SECTION>_<KEY> or MDM_<SECTION>_<SUBSECTION>_<KEY>

        Examples:
        - MDM_DATABASE_DEFAULT_BACKEND=postgresql
        - MDM_DATABASE_POSTGRESQL_HOST=myhost.com
        - MDM_PERFORMANCE_BATCH_SIZE=50000
        """
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(self.ENV_PREFIX):
                continue

            # Parse environment variable name
            parts = env_key[len(self.ENV_PREFIX):].lower().split("_")
            if not parts:
                continue

            # Handle special cases where underscore is part of the key name
            # e.g., MDM_DATABASE_DEFAULT_BACKEND -> database.default_backend
            if len(parts) >= 2 and parts[0] == "feature" and parts[1] == "engineering":
                # Combine feature_engineering
                parts = ["feature_engineering"] + parts[2:]
            if len(parts) >= 3 and parts[1] == "default" and parts[2] == "backend":
                # Combine default_backend
                parts = [parts[0], "default_backend"] + parts[3:]
            elif len(parts) >= 3 and parts[1] == "batch" and parts[2] == "size":
                # Combine batch_size
                parts = [parts[0], "batch_size"] + parts[3:]
            elif len(parts) >= 3 and parts[1] == "connection" and parts[2] == "timeout":
                # Combine connection_timeout
                parts = [parts[0], "connection_timeout"] + parts[3:]
            elif len(parts) >= 3 and parts[1] == "show" and parts[2] == "progress":
                # Combine show_progress
                parts = [parts[0], "show_progress"] + parts[3:]
            elif len(parts) >= 3 and parts[-2] == "n" and parts[-1] == "bins":
                # Combine n_bins at the end
                parts = parts[:-2] + ["n_bins"]

            # Navigate config dict and set value
            current = config_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    # If the value exists and is not a dict, we can't navigate further
                    break
                current = current[part]

            # Set the value with type conversion
            key = parts[-1]
            current[key] = self._convert_env_value(env_value)

        return config_dict

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # List (comma-separated) - check this before numbers
        if "," in value:
            items = [v.strip() for v in value.split(",")]
            # Try to convert list items to numbers
            converted_items: list[Any] = []
            for item in items:
                try:
                    converted_items.append(int(item))
                except ValueError:
                    try:
                        converted_items.append(float(item))
                    except ValueError:
                        converted_items.append(item)
            return converted_items

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # String
        return value

    @property
    def config(self) -> MDMConfig:
        """Get current configuration (load if necessary)."""
        if self._config is None:
            self._config = self.load()
        return self._config


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reset_config_manager() -> None:
    """Reset configuration manager (useful for testing)."""
    global _config_manager
    _config_manager = None


def get_config() -> MDMConfig:
    """Get current configuration."""
    return get_config_manager().config

