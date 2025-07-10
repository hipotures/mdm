"""
Performance optimization module for MDM.

This module provides:
- Query optimization for storage backends
- Caching layer for frequently accessed data
- Connection pooling for database operations
- Batch processing optimizations
- Performance monitoring and metrics
"""

from .query_optimizer import QueryOptimizer, QueryPlan
from .cache_manager import CacheManager, CachePolicy
from .connection_pool import ConnectionPool, PoolConfig
from .batch_optimizer import BatchOptimizer, BatchConfig
from .performance_monitor import PerformanceMonitor, MetricCollector

__all__ = [
    'QueryOptimizer',
    'QueryPlan',
    'CacheManager',
    'CachePolicy',
    'ConnectionPool',
    'PoolConfig',
    'BatchOptimizer',
    'BatchConfig',
    'PerformanceMonitor',
    'MetricCollector',
]