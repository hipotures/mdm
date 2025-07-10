"""Simple monitoring implementation for single-user MDM."""

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from mdm.utils.paths import PathManager


class MetricType(str, Enum):
    """Types of metrics we track."""
    
    DATASET_REGISTER = "dataset_register"
    DATASET_READ = "dataset_read"
    DATASET_EXPORT = "dataset_export"
    FEATURE_GENERATION = "feature_generation"
    QUERY_EXECUTION = "query_execution"
    ERROR = "error"
    STORAGE_OPERATION = "storage_operation"


class Metric(BaseModel):
    """Single metric entry."""
    
    timestamp: datetime
    metric_type: MetricType
    operation: str
    duration_ms: Optional[float] = None
    success: bool = True
    dataset_name: Optional[str] = None
    row_count: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class SimpleMonitor:
    """Lightweight monitoring for MDM - no external dependencies."""
    
    def __init__(self):
        self.path_manager = PathManager()
        self.metrics_db_path = self.path_manager.base_path / "metrics.db"
        self._init_database()
        self._configure_logging()
    
    def _init_database(self):
        """Initialize metrics database."""
        with sqlite3.connect(str(self.metrics_db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metric_type TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    duration_ms REAL,
                    success INTEGER DEFAULT 1,
                    dataset_name TEXT,
                    row_count INTEGER,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            
            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON metrics(timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_type 
                ON metrics(metric_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_dataset 
                ON metrics(dataset_name)
            """)
            
            conn.commit()
    
    def _configure_logging(self):
        """Configure loguru for file logging with rotation."""
        log_path = self.path_manager.base_path / "logs" / "mdm.log"
        log_path.parent.mkdir(exist_ok=True)
        
        # Remove default handler
        logger.remove()
        
        # Console logging - only warnings and above
        logger.add(
            lambda msg: print(msg, end=""),
            format="<level>{message}</level>",
            level="WARNING",
            colorize=True
        )
        
        # File logging - all messages
        logger.add(
            str(log_path),
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="7 days",  # Keep logs for 7 days
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True
        )
        
        logger.info("MDM monitoring initialized")
    
    @contextmanager
    def track_operation(self, metric_type: MetricType, operation: str, 
                       dataset_name: Optional[str] = None, **metadata):
        """Context manager to track operation metrics."""
        start_time = time.time()
        success = True
        error_message = None
        row_count = metadata.pop('row_count', None)
        
        try:
            logger.debug(f"Starting {operation}" + (f" for {dataset_name}" if dataset_name else ""))
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Error in {operation}: {error_message}")
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            self.record_metric(
                metric_type=metric_type,
                operation=operation,
                duration_ms=duration_ms,
                success=success,
                dataset_name=dataset_name,
                row_count=row_count,
                error_message=error_message,
                metadata=metadata
            )
            
            if success:
                logger.info(f"Completed {operation} in {duration_ms:.1f}ms")
    
    def record_metric(self, metric_type: MetricType, operation: str, 
                     duration_ms: Optional[float] = None, success: bool = True,
                     dataset_name: Optional[str] = None, row_count: Optional[int] = None,
                     error_message: Optional[str] = None, metadata: Optional[Dict] = None):
        """Record a single metric."""
        metric = Metric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            dataset_name=dataset_name,
            row_count=row_count,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        with sqlite3.connect(str(self.metrics_db_path)) as conn:
            conn.execute("""
                INSERT INTO metrics 
                (timestamp, metric_type, operation, duration_ms, success, 
                 dataset_name, row_count, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.timestamp,
                metric.metric_type,
                metric.operation,
                metric.duration_ms,
                1 if metric.success else 0,
                metric.dataset_name,
                metric.row_count,
                metric.error_message,
                str(metric.metadata) if metric.metadata else None
            ))
            conn.commit()
    
    def get_recent_metrics(self, limit: int = 100, 
                          metric_type: Optional[MetricType] = None,
                          dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent metrics with optional filtering."""
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if metric_type:
            query += " AND metric_type = ?"
            params.append(metric_type)
        
        if dataset_name:
            query += " AND dataset_name = ?"
            params.append(dataset_name)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(str(self.metrics_db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with sqlite3.connect(str(self.metrics_db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            # Overall stats
            overall = conn.execute("""
                SELECT 
                    COUNT(*) as total_operations,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_operations,
                    AVG(CASE WHEN success = 1 THEN duration_ms ELSE NULL END) as avg_duration_ms,
                    MAX(timestamp) as last_operation
                FROM metrics
            """).fetchone()
            
            # Stats by type
            by_type = conn.execute("""
                SELECT 
                    metric_type,
                    COUNT(*) as count,
                    AVG(CASE WHEN success = 1 THEN duration_ms ELSE NULL END) as avg_duration_ms,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as error_count
                FROM metrics
                GROUP BY metric_type
            """).fetchall()
            
            # Dataset stats
            dataset_stats = conn.execute("""
                SELECT 
                    COUNT(DISTINCT dataset_name) as total_datasets,
                    SUM(CASE WHEN metric_type = 'dataset_register' THEN 1 ELSE 0 END) as datasets_registered
                FROM metrics
                WHERE dataset_name IS NOT NULL
            """).fetchone()
            
            # Recent errors
            recent_errors = conn.execute("""
                SELECT timestamp, operation, error_message
                FROM metrics
                WHERE success = 0
                ORDER BY timestamp DESC
                LIMIT 5
            """).fetchall()
            
            return {
                'overall': dict(overall) if overall else {},
                'by_type': [dict(row) for row in by_type],
                'dataset_stats': dict(dataset_stats) if dataset_stats else {},
                'recent_errors': [dict(row) for row in recent_errors]
            }
    
    def get_dataset_metrics(self, dataset_name: str) -> Dict[str, Any]:
        """Get metrics for a specific dataset."""
        with sqlite3.connect(str(self.metrics_db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            metrics = conn.execute("""
                SELECT 
                    metric_type,
                    COUNT(*) as count,
                    AVG(duration_ms) as avg_duration_ms,
                    SUM(row_count) as total_rows_processed,
                    MAX(timestamp) as last_operation
                FROM metrics
                WHERE dataset_name = ?
                GROUP BY metric_type
            """, (dataset_name,)).fetchall()
            
            return [dict(row) for row in metrics]
    
    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """Clean up metrics older than specified days."""
        with sqlite3.connect(str(self.metrics_db_path)) as conn:
            deleted = conn.execute("""
                DELETE FROM metrics
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep)).rowcount
            conn.commit()
            
        logger.info(f"Cleaned up {deleted} old metrics")
        return deleted


# Global instance for easy access
_monitor = None

def get_monitor() -> SimpleMonitor:
    """Get global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = SimpleMonitor()
    return _monitor