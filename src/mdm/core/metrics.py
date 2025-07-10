"""
Metrics collection system for MDM.

This module provides unified metrics collection for both old and new
implementations, enabling performance comparison and monitoring.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from pathlib import Path
import threading
from collections import defaultdict
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


class MetricsCollector:
    """Collect and aggregate metrics from both implementations."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path.home() / ".mdm" / "metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._metrics: List[Metric] = []
        self._counters: Dict[str, int] = defaultdict(int)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        
        self._lock = threading.Lock()
        self._start_time = time.time()
        
        logger.info(f"Initialized MetricsCollector with output dir: {self.output_dir}")
    
    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        with self._lock:
            self._counters[name] += value
            self._metrics.append(Metric(name, value, tags=tags or {}))
            logger.debug(f"Incremented counter {name} by {value}")
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value
            self._metrics.append(Metric(name, value, tags=tags or {}))
            logger.debug(f"Set gauge {name} to {value}")
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        class Timer:
            def __init__(self, collector, metric_name, metric_tags):
                self.collector = collector
                self.name = metric_name
                self.tags = metric_tags or {}
                self.start = None
            
            def __enter__(self):
                self.start = time.perf_counter()
                return self
            
            def __exit__(self, *args):
                duration = time.perf_counter() - self.start
                self.collector.record_time(self.name, duration, self.tags)
        
        return Timer(self, name, tags)
    
    def record_time(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing measurement."""
        with self._lock:
            self._timers[name].append(duration)
            self._metrics.append(Metric(f"{name}.duration", duration, tags=tags or {}))
            logger.debug(f"Recorded time for {name}: {duration:.3f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            summary = {
                "uptime_seconds": time.time() - self._start_time,
                "total_metrics": len(self._metrics),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timers": {}
            }
            
            # Calculate timer statistics
            for name, values in self._timers.items():
                if values:
                    summary["timers"][name] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "stddev": statistics.stdev(values) if len(values) > 1 else 0,
                        "p95": self._percentile(values, 0.95),
                        "p99": self._percentile(values, 0.99)
                    }
            
            return summary
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def export(self, filename: Optional[str] = None) -> Path:
        """Export metrics to file."""
        if filename is None:
            filename = f"metrics_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        filepath = self.output_dir / filename
        
        with self._lock:
            data = {
                "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
                "export_time": datetime.now().isoformat(),
                "summary": self.get_summary(),
                "metrics": [m.to_dict() for m in self._metrics[-10000:]]  # Last 10k metrics
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported metrics to: {filepath}")
        return filepath
    
    def compare_implementations(self, metric_prefix: str = "implementation") -> Dict[str, Any]:
        """Compare metrics between old and new implementations."""
        with self._lock:
            old_metrics = {k: v for k, v in self._timers.items() 
                          if k.startswith(f"{metric_prefix}.old")}
            new_metrics = {k: v for k, v in self._timers.items() 
                          if k.startswith(f"{metric_prefix}.new")}
        
        comparison = {}
        
        for old_key, old_values in old_metrics.items():
            # Find corresponding new metric
            base_name = old_key.replace(f"{metric_prefix}.old.", "")
            new_key = f"{metric_prefix}.new.{base_name}"
            
            if new_key in new_metrics:
                new_values = new_metrics[new_key]
                
                old_mean = statistics.mean(old_values) if old_values else 0
                new_mean = statistics.mean(new_values) if new_values else 0
                
                comparison[base_name] = {
                    "old_mean": old_mean,
                    "new_mean": new_mean,
                    "delta": new_mean - old_mean,
                    "delta_percent": ((new_mean - old_mean) / old_mean * 100) if old_mean else 0,
                    "old_p95": self._percentile(old_values, 0.95),
                    "new_p95": self._percentile(new_values, 0.95),
                    "samples": {"old": len(old_values), "new": len(new_values)}
                }
        
        return comparison
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._timers.clear()
            self._gauges.clear()
            self._start_time = time.time()
            logger.info("Reset all metrics")
    
    def get_recent_metrics(self, name: str, limit: int = 100) -> List[Metric]:
        """Get recent metrics by name."""
        with self._lock:
            matching = [m for m in self._metrics if m.name == name]
            return matching[-limit:]


# Global metrics collector
metrics_collector = MetricsCollector()


# Decorator for automatic metrics collection
def track_metrics(name: str, implementation: str = "unknown"):
    """
    Decorator to automatically track function metrics.
    
    Args:
        name: Base name for metrics
        implementation: Implementation type ('old' or 'new')
        
    Example:
        @track_metrics("dataset.register", implementation="new")
        def register_dataset():
            pass
    """
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            tags = {"implementation": implementation, "function": func.__name__}
            
            # Track call count
            metrics_collector.increment(f"{name}.calls", tags=tags)
            
            # Track execution time
            with metrics_collector.timer(f"{name}.{implementation}", tags=tags):
                try:
                    result = func(*args, **kwargs)
                    metrics_collector.increment(f"{name}.success", tags=tags)
                    return result
                except Exception as e:
                    metrics_collector.increment(f"{name}.errors", tags=tags)
                    metrics_collector.increment(f"{name}.errors.{type(e).__name__}", tags=tags)
                    raise
        
        return wrapper
    return decorator


# Convenience functions for common metrics
def track_operation(operation: str):
    """Track a named operation."""
    return metrics_collector.timer(f"operation.{operation}")


def track_query(query_type: str, backend: str):
    """Track database query."""
    return metrics_collector.timer(f"query.{backend}.{query_type}")


def track_feature_generation(feature_type: str):
    """Track feature generation."""
    return metrics_collector.timer(f"features.{feature_type}")


def record_dataset_size(dataset_name: str, rows: int, columns: int, size_mb: float):
    """Record dataset size metrics."""
    tags = {"dataset": dataset_name}
    metrics_collector.gauge(f"dataset.rows", rows, tags)
    metrics_collector.gauge(f"dataset.columns", columns, tags)
    metrics_collector.gauge(f"dataset.size_mb", size_mb, tags)