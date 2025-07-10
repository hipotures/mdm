"""New modular CLI implementation for MDM.

This package provides a clean, modular implementation of CLI commands
with better separation of concerns and enhanced features.
"""

from .dataset_commands import NewDatasetCommands
from .batch_commands import NewBatchCommands
from .timeseries_commands import NewTimeSeriesCommands
from .stats_commands import NewStatsCommands
from .formatter import NewCLIFormatter
from .config import NewCLIConfig
from .plugins import CLIPluginManager

__all__ = [
    'NewDatasetCommands',
    'NewBatchCommands',
    'NewTimeSeriesCommands',
    'NewStatsCommands',
    'NewCLIFormatter',
    'NewCLIConfig',
    'CLIPluginManager',
]