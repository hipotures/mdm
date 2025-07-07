"""MDM utilities package."""

from mdm.utils.integration import DatasetIterator, MLFrameworkAdapter, SubmissionCreator
from mdm.utils.performance import (
    ChunkProcessor,
    PerformanceMonitor,
    QueryOptimizer,
    estimate_memory_usage,
    optimize_dtypes,
)
from mdm.utils.serialization import (
    MDMJSONEncoder,
    deserialize_datetime,
    is_serializable,
    json_dumps,
    json_loads,
    serialize_for_yaml,
)
from mdm.utils.time_series import TimeSeriesAnalyzer, TimeSeriesSplitter

__all__ = [
    # Integration
    "MLFrameworkAdapter",
    "DatasetIterator",
    "SubmissionCreator",
    # Performance
    "PerformanceMonitor",
    "ChunkProcessor",
    "QueryOptimizer",
    "estimate_memory_usage",
    "optimize_dtypes",
    # Serialization
    "MDMJSONEncoder",
    "json_dumps",
    "json_loads",
    "serialize_for_yaml",
    "deserialize_datetime",
    "is_serializable",
    # Time Series
    "TimeSeriesSplitter",
    "TimeSeriesAnalyzer",
]