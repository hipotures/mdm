"""Performance monitoring and metrics collection."""
import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from contextlib import contextmanager
import json
from pathlib import Path
from collections import deque, defaultdict

from ..core.logging import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Single performance metric."""
    name: str
    type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.type.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags
        }


@dataclass
class TimerContext:
    """Context for timing operations."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def stop(self) -> float:
        """Stop timer and return duration."""
        self.end_time = time.time()
        return self.duration


class MetricCollector:
    """Collects performance metrics."""
    
    def __init__(self, buffer_size: int = 10000):
        """Initialize metric collector.
        
        Args:
            buffer_size: Maximum metrics to buffer
        """
        self.buffer_size = buffer_size
        self._metrics: deque = deque(maxlen=buffer_size)
        self._lock = threading.RLock()
        
        # Aggregated metrics
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
    
    def record_counter(self, name: str, value: float = 1.0, 
                      tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric.
        
        Args:
            name: Metric name
            value: Value to increment by
            tags: Optional tags
        """
        with self._lock:
            metric = Metric(name, MetricType.COUNTER, value, tags=tags or {})
            self._metrics.append(metric)
            self._counters[name] += value
    
    def record_gauge(self, name: str, value: float,
                    tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric.
        
        Args:
            name: Metric name
            value: Current value
            tags: Optional tags
        """
        with self._lock:
            metric = Metric(name, MetricType.GAUGE, value, tags=tags or {})
            self._metrics.append(metric)
            self._gauges[name] = value
    
    def record_histogram(self, name: str, value: float,
                        tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric.
        
        Args:
            name: Metric name
            value: Value to record
            tags: Optional tags
        """
        with self._lock:
            metric = Metric(name, MetricType.HISTOGRAM, value, tags=tags or {})
            self._metrics.append(metric)
            self._histograms[name].append(value)
            
            # Keep only recent values
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]
    
    def record_timer(self, name: str, duration: float,
                    tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric.
        
        Args:
            name: Metric name
            duration: Duration in seconds
            tags: Optional tags
        """
        with self._lock:
            metric = Metric(name, MetricType.TIMER, duration, tags=tags or {})
            self._metrics.append(metric)
            self._timers[name].append(duration)
            
            # Keep only recent values
            if len(self._timers[name]) > 1000:
                self._timers[name] = self._timers[name][-1000:]
    
    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations.
        
        Args:
            name: Timer name
            tags: Optional tags
            
        Yields:
            TimerContext
        """
        context = TimerContext()
        try:
            yield context
        finally:
            self.record_timer(name, context.duration, tags)
    
    def get_metrics(self, since: Optional[float] = None) -> List[Metric]:
        """Get recorded metrics.
        
        Args:
            since: Optional timestamp to get metrics since
            
        Returns:
            List of metrics
        """
        with self._lock:
            if since is None:
                return list(self._metrics)
            
            return [m for m in self._metrics if m.timestamp >= since]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            summary = {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {},
                'timers': {}
            }
            
            # Calculate histogram statistics
            for name, values in self._histograms.items():
                if values:
                    summary['histograms'][name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'p50': self._percentile(values, 0.5),
                        'p95': self._percentile(values, 0.95),
                        'p99': self._percentile(values, 0.99)
                    }
            
            # Calculate timer statistics
            for name, durations in self._timers.items():
                if durations:
                    summary['timers'][name] = {
                        'count': len(durations),
                        'min': min(durations),
                        'max': max(durations),
                        'avg': sum(durations) / len(durations),
                        'p50': self._percentile(durations, 0.5),
                        'p95': self._percentile(durations, 0.95),
                        'p99': self._percentile(durations, 0.99)
                    }
            
            return summary
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()


class PerformanceMonitor:
    """Monitors system and application performance."""
    
    def __init__(self,
                 enable_system_metrics: bool = True,
                 sample_interval: float = 60.0,
                 metrics_file: Optional[Path] = None):
        """Initialize performance monitor.
        
        Args:
            enable_system_metrics: Whether to collect system metrics
            sample_interval: Interval for system metrics sampling
            metrics_file: Optional file to persist metrics
        """
        self.enable_system_metrics = enable_system_metrics
        self.sample_interval = sample_interval
        self.metrics_file = metrics_file
        
        self.collector = MetricCollector()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Operation tracking
        self._active_operations: Dict[str, TimerContext] = {}
        self._operation_lock = threading.RLock()
    
    def start(self) -> None:
        """Start performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        if self.enable_system_metrics:
            self._monitor_thread = threading.Thread(target=self._monitor_system)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop(self) -> None:
        """Stop performance monitoring."""
        self._monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join()
            self._monitor_thread = None
        
        # Save final metrics
        if self.metrics_file:
            self.save_metrics()
        
        logger.info("Performance monitoring stopped")
    
    def _monitor_system(self) -> None:
        """Monitor system metrics."""
        process = psutil.Process()
        
        while self._monitoring:
            try:
                # CPU metrics
                cpu_percent = process.cpu_percent(interval=1)
                self.collector.record_gauge("system.cpu.percent", cpu_percent)
                
                # Memory metrics
                memory_info = process.memory_info()
                self.collector.record_gauge("system.memory.rss_mb", 
                                          memory_info.rss / 1024 / 1024)
                self.collector.record_gauge("system.memory.vms_mb", 
                                          memory_info.vms / 1024 / 1024)
                
                # Disk I/O (if available)
                try:
                    io_counters = process.io_counters()
                    self.collector.record_counter("system.io.read_bytes", 
                                                io_counters.read_bytes)
                    self.collector.record_counter("system.io.write_bytes", 
                                                io_counters.write_bytes)
                except:
                    pass
                
                # Thread count
                self.collector.record_gauge("system.threads", 
                                          process.num_threads())
                
                # Wait for next sample
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                time.sleep(self.sample_interval)
    
    @contextmanager
    def track_operation(self, operation: str, **tags):
        """Track a specific operation.
        
        Args:
            operation: Operation name
            **tags: Additional tags
            
        Yields:
            TimerContext
        """
        op_id = f"{operation}:{threading.get_ident()}:{time.time()}"
        
        with self._operation_lock:
            context = TimerContext()
            self._active_operations[op_id] = context
        
        try:
            # Record start
            self.collector.record_counter(f"operation.{operation}.started")
            
            yield context
            
            # Record success
            self.collector.record_counter(f"operation.{operation}.completed")
            self.collector.record_timer(f"operation.{operation}.duration", 
                                      context.duration, tags)
            
        except Exception as e:
            # Record failure
            self.collector.record_counter(f"operation.{operation}.failed")
            raise
            
        finally:
            with self._operation_lock:
                del self._active_operations[op_id]
    
    def track_query(self, query_type: str, duration: float, 
                   rows: Optional[int] = None) -> None:
        """Track database query performance.
        
        Args:
            query_type: Type of query
            duration: Query duration in seconds
            rows: Number of rows affected/returned
        """
        self.collector.record_timer(f"query.{query_type}.duration", duration)
        
        if rows is not None:
            self.collector.record_histogram(f"query.{query_type}.rows", rows)
    
    def track_cache(self, operation: str, hit: bool) -> None:
        """Track cache operations.
        
        Args:
            operation: Cache operation name
            hit: Whether it was a cache hit
        """
        if hit:
            self.collector.record_counter(f"cache.{operation}.hits")
        else:
            self.collector.record_counter(f"cache.{operation}.misses")
    
    def track_batch(self, operation: str, batch_size: int, 
                   duration: float) -> None:
        """Track batch processing.
        
        Args:
            operation: Batch operation name
            batch_size: Size of batch
            duration: Processing duration
        """
        self.collector.record_histogram(f"batch.{operation}.size", batch_size)
        self.collector.record_timer(f"batch.{operation}.duration", duration)
        
        # Calculate throughput
        if duration > 0:
            throughput = batch_size / duration
            self.collector.record_gauge(f"batch.{operation}.throughput", throughput)
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get currently active operations."""
        with self._operation_lock:
            return [
                {
                    'id': op_id,
                    'duration': context.duration,
                    'start_time': context.start_time
                }
                for op_id, context in self._active_operations.items()
            ]
    
    def get_report(self) -> Dict[str, Any]:
        """Get performance report."""
        summary = self.collector.get_summary()
        
        # Add computed metrics
        report = {
            'summary': summary,
            'active_operations': len(self._active_operations),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate operation success rates
        if 'counters' in summary:
            counters = summary['counters']
            for op in set(k.split('.')[1] for k in counters.keys() 
                         if k.startswith('operation.')):
                started = counters.get(f'operation.{op}.started', 0)
                completed = counters.get(f'operation.{op}.completed', 0)
                failed = counters.get(f'operation.{op}.failed', 0)
                
                if started > 0:
                    success_rate = completed / started
                    report[f'operation_{op}_success_rate'] = success_rate
        
        return report
    
    def save_metrics(self, file_path: Optional[Path] = None) -> None:
        """Save metrics to file.
        
        Args:
            file_path: Optional file path (uses default if not provided)
        """
        file_path = file_path or self.metrics_file
        if not file_path:
            return
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            metrics = [m.to_dict() for m in self.collector.get_metrics()]
            
            with open(file_path, 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'summary': self.collector.get_summary(),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
                
            logger.info(f"Saved metrics to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


# Global performance monitor instance
_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
        _monitor.start()
    return _monitor