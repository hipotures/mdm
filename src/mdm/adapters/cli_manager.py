"""CLI Manager for MDM refactoring.

This module provides a unified interface for CLI commands that can
switch between old and new implementations based on feature flags.
"""
import logging
from typing import Dict, Any, Optional, List, Type
from datetime import datetime

from ..core import feature_flags
from ..interfaces.cli import (
    IDatasetCommands,
    IBatchCommands,
    ITimeSeriesCommands,
    IStatsCommands,
    ICLIFormatter,
    ICLIConfig,
)
from ..core.exceptions import MDMError

logger = logging.getLogger(__name__)


class CLIManager:
    """Manages CLI command implementations with feature flag support."""
    
    def __init__(self):
        """Initialize CLI manager."""
        self._dataset_commands = None
        self._batch_commands = None
        self._timeseries_commands = None
        self._stats_commands = None
        self._formatter = None
        self._config = None
        self._cache = {}
        self._metrics = {
            'commands_executed': 0,
            'cache_hits': 0,
            'errors': 0,
            'feature_flag_checks': 0
        }
        logger.info("Initialized CLIManager")
    
    def get_dataset_commands(self, force_new: Optional[bool] = None) -> IDatasetCommands:
        """Get dataset commands implementation.
        
        Args:
            force_new: Force use of new implementation (overrides feature flag)
            
        Returns:
            Dataset commands implementation
        """
        use_new = self._check_feature_flag("use_new_cli", force_new)
        
        cache_key = f"dataset_commands_{use_new}"
        if cache_key in self._cache:
            self._metrics['cache_hits'] += 1
            return self._cache[cache_key]
        
        if use_new:
            logger.info("Using new dataset commands implementation")
            from ..core.cli.dataset_commands import NewDatasetCommands
            commands = NewDatasetCommands()
        else:
            logger.info("Using legacy dataset commands implementation")
            from ..cli.dataset import LegacyDatasetCommands
            commands = LegacyDatasetCommands()
        
        self._cache[cache_key] = commands
        return commands
    
    def get_batch_commands(self, force_new: Optional[bool] = None) -> IBatchCommands:
        """Get batch commands implementation.
        
        Args:
            force_new: Force use of new implementation (overrides feature flag)
            
        Returns:
            Batch commands implementation
        """
        use_new = self._check_feature_flag("use_new_cli", force_new)
        
        cache_key = f"batch_commands_{use_new}"
        if cache_key in self._cache:
            self._metrics['cache_hits'] += 1
            return self._cache[cache_key]
        
        if use_new:
            logger.info("Using new batch commands implementation")
            from ..core.cli.batch_commands import NewBatchCommands
            commands = NewBatchCommands()
        else:
            logger.info("Using legacy batch commands implementation")
            from ..cli.batch import LegacyBatchCommands
            commands = LegacyBatchCommands()
        
        self._cache[cache_key] = commands
        return commands
    
    def get_timeseries_commands(self, force_new: Optional[bool] = None) -> ITimeSeriesCommands:
        """Get time series commands implementation.
        
        Args:
            force_new: Force use of new implementation (overrides feature flag)
            
        Returns:
            Time series commands implementation
        """
        use_new = self._check_feature_flag("use_new_cli", force_new)
        
        cache_key = f"timeseries_commands_{use_new}"
        if cache_key in self._cache:
            self._metrics['cache_hits'] += 1
            return self._cache[cache_key]
        
        if use_new:
            logger.info("Using new time series commands implementation")
            from ..core.cli.timeseries_commands import NewTimeSeriesCommands
            commands = NewTimeSeriesCommands()
        else:
            logger.info("Using legacy time series commands implementation")
            from ..cli.timeseries import LegacyTimeSeriesCommands
            commands = LegacyTimeSeriesCommands()
        
        self._cache[cache_key] = commands
        return commands
    
    def get_stats_commands(self, force_new: Optional[bool] = None) -> IStatsCommands:
        """Get stats commands implementation.
        
        Args:
            force_new: Force use of new implementation (overrides feature flag)
            
        Returns:
            Stats commands implementation
        """
        use_new = self._check_feature_flag("use_new_cli", force_new)
        
        cache_key = f"stats_commands_{use_new}"
        if cache_key in self._cache:
            self._metrics['cache_hits'] += 1
            return self._cache[cache_key]
        
        if use_new:
            logger.info("Using new stats commands implementation")
            from ..core.cli.stats_commands import NewStatsCommands
            commands = NewStatsCommands()
        else:
            logger.info("Using legacy stats commands implementation")
            from ..cli.stats import LegacyStatsCommands
            commands = LegacyStatsCommands()
        
        self._cache[cache_key] = commands
        return commands
    
    def get_formatter(self, force_new: Optional[bool] = None) -> ICLIFormatter:
        """Get CLI formatter implementation.
        
        Args:
            force_new: Force use of new implementation (overrides feature flag)
            
        Returns:
            CLI formatter implementation
        """
        use_new = self._check_feature_flag("use_new_cli", force_new)
        
        cache_key = f"formatter_{use_new}"
        if cache_key in self._cache:
            self._metrics['cache_hits'] += 1
            return self._cache[cache_key]
        
        if use_new:
            logger.info("Using new CLI formatter implementation")
            from ..core.cli.formatter import NewCLIFormatter
            formatter = NewCLIFormatter()
        else:
            logger.info("Using legacy CLI formatter implementation")
            from ..cli.utils import LegacyCLIFormatter
            formatter = LegacyCLIFormatter()
        
        self._cache[cache_key] = formatter
        return formatter
    
    def get_config(self, force_new: Optional[bool] = None) -> ICLIConfig:
        """Get CLI configuration implementation.
        
        Args:
            force_new: Force use of new implementation (overrides feature flag)
            
        Returns:
            CLI configuration implementation
        """
        use_new = self._check_feature_flag("use_new_cli", force_new)
        
        cache_key = f"config_{use_new}"
        if cache_key in self._cache:
            self._metrics['cache_hits'] += 1
            return self._cache[cache_key]
        
        if use_new:
            logger.info("Using new CLI config implementation")
            from ..core.cli.config import NewCLIConfig
            config = NewCLIConfig()
        else:
            logger.info("Using legacy CLI config implementation")
            from ..cli.config import LegacyCLIConfig
            config = LegacyCLIConfig()
        
        self._cache[cache_key] = config
        return config
    
    def execute_command(
        self,
        command_group: str,
        command: str,
        force_new: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a CLI command through the appropriate implementation.
        
        Args:
            command_group: Command group (dataset, batch, timeseries, stats)
            command: Command name
            force_new: Force use of new implementation
            **kwargs: Command arguments
            
        Returns:
            Command result
        """
        try:
            self._metrics['commands_executed'] += 1
            
            # Get the appropriate command handler
            if command_group == "dataset":
                handler = self.get_dataset_commands(force_new)
            elif command_group == "batch":
                handler = self.get_batch_commands(force_new)
            elif command_group == "timeseries":
                handler = self.get_timeseries_commands(force_new)
            elif command_group == "stats":
                handler = self.get_stats_commands(force_new)
            else:
                raise MDMError(f"Unknown command group: {command_group}")
            
            # Execute the command
            method = getattr(handler, command, None)
            if method is None:
                raise MDMError(f"Unknown command: {command_group}.{command}")
            
            result = method(**kwargs)
            
            # Track success
            logger.info(f"Successfully executed {command_group}.{command}")
            return result
            
        except Exception as e:
            self._metrics['errors'] += 1
            logger.error(f"Error executing {command_group}.{command}: {e}")
            raise
    
    def clear_cache(self) -> None:
        """Clear the command cache."""
        self._cache.clear()
        logger.info("Cleared CLI command cache")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get CLI manager metrics.
        
        Returns:
            Metrics dictionary
        """
        return {
            **self._metrics,
            'cache_size': len(self._cache),
            'cache_keys': list(self._cache.keys())
        }
    
    def _check_feature_flag(self, flag: str, force: Optional[bool] = None) -> bool:
        """Check feature flag with optional override.
        
        Args:
            flag: Feature flag name
            force: Force value (overrides flag)
            
        Returns:
            Whether to use new implementation
        """
        self._metrics['feature_flag_checks'] += 1
        
        if force is not None:
            logger.debug(f"Forcing {flag} to {force}")
            return force
        
        return feature_flags.get(flag, False)


# Global instance
_manager = CLIManager()


# Public API functions
def get_dataset_commands(force_new: Optional[bool] = None) -> IDatasetCommands:
    """Get dataset commands implementation with feature flag support."""
    return _manager.get_dataset_commands(force_new)


def get_batch_commands(force_new: Optional[bool] = None) -> IBatchCommands:
    """Get batch commands implementation with feature flag support."""
    return _manager.get_batch_commands(force_new)


def get_timeseries_commands(force_new: Optional[bool] = None) -> ITimeSeriesCommands:
    """Get time series commands implementation with feature flag support."""
    return _manager.get_timeseries_commands(force_new)


def get_stats_commands(force_new: Optional[bool] = None) -> IStatsCommands:
    """Get stats commands implementation with feature flag support."""
    return _manager.get_stats_commands(force_new)


def get_cli_formatter(force_new: Optional[bool] = None) -> ICLIFormatter:
    """Get CLI formatter implementation with feature flag support."""
    return _manager.get_formatter(force_new)


def get_cli_config(force_new: Optional[bool] = None) -> ICLIConfig:
    """Get CLI configuration implementation with feature flag support."""
    return _manager.get_config(force_new)


def execute_command(
    command_group: str,
    command: str,
    force_new: Optional[bool] = None,
    **kwargs
) -> Dict[str, Any]:
    """Execute a CLI command through the appropriate implementation."""
    return _manager.execute_command(command_group, command, force_new, **kwargs)


def clear_cli_cache() -> None:
    """Clear the CLI command cache."""
    _manager.clear_cache()


def get_cli_metrics() -> Dict[str, Any]:
    """Get CLI manager metrics."""
    return _manager.get_metrics()