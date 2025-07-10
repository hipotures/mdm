"""Rollout monitoring and alerting system.

This module provides monitoring capabilities during and after the rollout.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
import threading
import time
from collections import deque
import statistics

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn
from rich import box

from mdm.performance import get_monitor as get_perf_monitor
from mdm.core.exceptions import MonitoringError


class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data."""
    name: str
    type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'unit': self.unit
        }


@dataclass
class Alert:
    """Alert information."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    metric_name: Optional[str] = None
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'metric_name': self.metric_name,
            'threshold': self.threshold,
            'current_value': self.current_value,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class MetricHistory:
    """History of metric values."""
    max_size: int = 1000
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add(self, value: float, timestamp: datetime) -> None:
        """Add value to history."""
        self.values.append((timestamp, value))
    
    def get_recent(self, minutes: int = 5) -> List[tuple]:
        """Get recent values."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [(ts, val) for ts, val in self.values if ts > cutoff]
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics for recent values."""
        recent = self.get_recent()
        if not recent:
            return {}
        
        values = [val for _, val in recent]
        return {
            'min': min(values),
            'max': max(values),
            'avg': statistics.mean(values),
            'median': statistics.median(values),
            'count': len(values)
        }


class RolloutMonitor:
    """Monitors rollout progress and system health."""
    
    def __init__(self):
        """Initialize monitor."""
        self.console = Console()
        self.metrics: Dict[str, Metric] = {}
        self.metric_history: Dict[str, MetricHistory] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Monitoring configuration
        self.thresholds = self._default_thresholds()
        self.monitoring_interval = 5  # seconds
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Performance monitor integration
        self._perf_monitor = get_perf_monitor()
    
    def _default_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Get default alert thresholds."""
        return {
            'error_rate': {
                'warning': 0.01,  # 1%
                'critical': 0.05  # 5%
            },
            'response_time_p95': {
                'warning': 1.0,   # 1 second
                'critical': 5.0   # 5 seconds
            },
            'memory_usage_percent': {
                'warning': 80,
                'critical': 95
            },
            'disk_usage_percent': {
                'warning': 80,
                'critical': 90
            },
            'migration_failure_rate': {
                'warning': 0.01,
                'critical': 0.05
            }
        }
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect application metrics
                self._collect_app_metrics()
                
                # Check thresholds and generate alerts
                self._check_thresholds()
                
                # Clean up old alerts
                self._cleanup_alerts()
                
            except Exception as e:
                self.console.print(f"[red]Monitoring error: {e}[/red]")
            
            # Wait for next interval
            self._stop_monitoring.wait(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric(
            "system.cpu.usage",
            cpu_percent,
            MetricType.GAUGE,
            unit="percent"
        )
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.record_metric(
            "system.memory.usage",
            memory.percent,
            MetricType.GAUGE,
            unit="percent"
        )
        self.record_metric(
            "system.memory.available",
            memory.available / (1024**3),
            MetricType.GAUGE,
            unit="GB"
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.record_metric(
            "system.disk.usage",
            disk.percent,
            MetricType.GAUGE,
            unit="percent"
        )
        self.record_metric(
            "system.disk.free",
            disk.free / (1024**3),
            MetricType.GAUGE,
            unit="GB"
        )
        
        # Network I/O
        net_io = psutil.net_io_counters()
        self.record_metric(
            "system.network.bytes_sent",
            net_io.bytes_sent,
            MetricType.COUNTER
        )
        self.record_metric(
            "system.network.bytes_recv",
            net_io.bytes_recv,
            MetricType.COUNTER
        )
    
    def _collect_app_metrics(self) -> None:
        """Collect application-level metrics."""
        # Get performance metrics
        perf_report = self._perf_monitor.get_report()
        
        if 'summary' in perf_report:
            summary = perf_report['summary']
            
            # Operation counts
            if 'counters' in summary:
                for metric, count in summary['counters'].items():
                    self.record_metric(
                        f"app.{metric}",
                        count,
                        MetricType.COUNTER
                    )
            
            # Timers
            if 'timers' in summary:
                for operation, stats in summary['timers'].items():
                    self.record_metric(
                        f"app.{operation}.duration.avg",
                        stats.get('avg', 0),
                        MetricType.GAUGE,
                        unit="seconds"
                    )
                    self.record_metric(
                        f"app.{operation}.duration.p95",
                        stats.get('p95', 0),
                        MetricType.GAUGE,
                        unit="seconds"
                    )
        
        # Active operations
        self.record_metric(
            "app.active_operations",
            perf_report.get('active_operations', 0),
            MetricType.GAUGE
        )
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ) -> None:
        """Record a metric value."""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            tags=tags or {},
            unit=unit
        )
        
        self.metrics[name] = metric
        
        # Add to history
        if name not in self.metric_history:
            self.metric_history[name] = MetricHistory()
        self.metric_history[name].add(value, metric.timestamp)
    
    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        metric_name: Optional[str] = None,
        threshold: Optional[float] = None,
        current_value: Optional[float] = None
    ) -> Alert:
        """Create and register an alert."""
        alert_id = f"{severity.value}_{metric_name or 'manual'}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value
        )
        
        self.alerts[alert_id] = alert
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.console.print(f"[red]Alert handler error: {e}[/red]")
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> None:
        """Mark alert as resolved."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    def _check_thresholds(self) -> None:
        """Check metrics against thresholds."""
        for metric_name, thresholds in self.thresholds.items():
            if metric_name not in self.metrics:
                continue
            
            metric = self.metrics[metric_name]
            value = metric.value
            
            # Check critical threshold
            if 'critical' in thresholds and value >= thresholds['critical']:
                self.create_alert(
                    AlertSeverity.CRITICAL,
                    f"Critical: {metric_name}",
                    f"{metric_name} is {value:.2f} (threshold: {thresholds['critical']})",
                    metric_name=metric_name,
                    threshold=thresholds['critical'],
                    current_value=value
                )
            # Check warning threshold
            elif 'warning' in thresholds and value >= thresholds['warning']:
                # Only create warning if no critical alert exists
                existing_critical = any(
                    a.severity == AlertSeverity.CRITICAL and
                    a.metric_name == metric_name and
                    not a.resolved
                    for a in self.alerts.values()
                )
                
                if not existing_critical:
                    self.create_alert(
                        AlertSeverity.WARNING,
                        f"Warning: {metric_name}",
                        f"{metric_name} is {value:.2f} (threshold: {thresholds['warning']})",
                        metric_name=metric_name,
                        threshold=thresholds['warning'],
                        current_value=value
                    )
    
    def _cleanup_alerts(self) -> None:
        """Clean up old resolved alerts."""
        cutoff = datetime.utcnow() - timedelta(hours=24)
        
        to_remove = []
        for alert_id, alert in self.alerts.items():
            if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff:
                to_remove.append(alert_id)
        
        for alert_id in to_remove:
            del self.alerts[alert_id]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display."""
        # Get active alerts
        active_alerts = [
            alert for alert in self.alerts.values()
            if not alert.resolved
        ]
        
        # Group by severity
        alerts_by_severity = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 0,
            AlertSeverity.WARNING: 0,
            AlertSeverity.INFO: 0
        }
        
        for alert in active_alerts:
            alerts_by_severity[alert.severity] += 1
        
        # Get key metrics
        key_metrics = {}
        for name in ['system.cpu.usage', 'system.memory.usage', 'app.active_operations']:
            if name in self.metrics:
                metric = self.metrics[name]
                stats = self.metric_history[name].get_stats() if name in self.metric_history else {}
                key_metrics[name] = {
                    'current': metric.value,
                    'unit': metric.unit,
                    **stats
                }
        
        return {
            'alerts': {
                'total': len(active_alerts),
                'by_severity': alerts_by_severity,
                'recent': sorted(
                    active_alerts,
                    key=lambda a: a.timestamp,
                    reverse=True
                )[:5]
            },
            'metrics': key_metrics,
            'last_updated': datetime.utcnow()
        }
    
    def display_dashboard(self) -> None:
        """Display monitoring dashboard."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="metrics", ratio=2),
            Layout(name="alerts", ratio=1)
        )
        
        def generate_dashboard():
            """Generate dashboard content."""
            data = self.get_dashboard_data()
            
            # Header
            layout["header"].update(
                Panel(
                    f"[bold cyan]MDM Rollout Monitor[/bold cyan] - {data['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}",
                    box=box.ROUNDED
                )
            )
            
            # Metrics panel
            metrics_table = Table(
                title="System Metrics",
                box=box.SIMPLE,
                show_header=True
            )
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Current", justify="right")
            metrics_table.add_column("Avg (5m)", justify="right")
            metrics_table.add_column("Min/Max", justify="right")
            
            for name, values in data['metrics'].items():
                current = f"{values['current']:.1f}{values.get('unit', '')}"
                avg = f"{values.get('avg', 0):.1f}" if 'avg' in values else "-"
                min_max = f"{values.get('min', 0):.1f}/{values.get('max', 0):.1f}" if 'min' in values else "-"
                
                metrics_table.add_row(
                    name.replace('system.', '').replace('app.', ''),
                    current,
                    avg,
                    min_max
                )
            
            layout["metrics"].update(Panel(metrics_table, title="Metrics", border_style="blue"))
            
            # Alerts panel
            alerts_table = Table(
                title="Active Alerts",
                box=box.SIMPLE,
                show_header=True
            )
            alerts_table.add_column("Severity", style="bold")
            alerts_table.add_column("Alert")
            alerts_table.add_column("Time")
            
            for alert in data['alerts']['recent']:
                severity_style = {
                    AlertSeverity.CRITICAL: "red",
                    AlertSeverity.ERROR: "red",
                    AlertSeverity.WARNING: "yellow",
                    AlertSeverity.INFO: "blue"
                }.get(alert.severity, "white")
                
                time_ago = (datetime.utcnow() - alert.timestamp).total_seconds()
                if time_ago < 60:
                    time_str = f"{int(time_ago)}s ago"
                elif time_ago < 3600:
                    time_str = f"{int(time_ago/60)}m ago"
                else:
                    time_str = f"{int(time_ago/3600)}h ago"
                
                alerts_table.add_row(
                    f"[{severity_style}]{alert.severity.value.upper()}[/{severity_style}]",
                    alert.title,
                    time_str
                )
            
            if not data['alerts']['recent']:
                alerts_table.add_row("", "[green]No active alerts[/green]", "")
            
            layout["alerts"].update(Panel(alerts_table, title="Alerts", border_style="yellow"))
            
            # Footer
            alert_summary = data['alerts']['by_severity']
            footer_text = (
                f"Critical: {alert_summary[AlertSeverity.CRITICAL]} | "
                f"Error: {alert_summary[AlertSeverity.ERROR]} | "
                f"Warning: {alert_summary[AlertSeverity.WARNING]} | "
                f"Info: {alert_summary[AlertSeverity.INFO]}"
            )
            layout["footer"].update(
                Panel(footer_text, box=box.ROUNDED, style="dim")
            )
            
            return layout
        
        # Display with live updates
        with Live(generate_dashboard(), refresh_per_second=1, console=self.console) as live:
            try:
                while True:
                    time.sleep(1)
                    live.update(generate_dashboard())
            except KeyboardInterrupt:
                pass
    
    def export_metrics(self, path: Path, format: str = "json") -> None:
        """Export metrics to file."""
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': [
                metric.to_dict() for metric in self.metrics.values()
            ],
            'alerts': [
                alert.to_dict() for alert in self.alerts.values()
            ]
        }
        
        if format == "json":
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "prometheus":
            # Export in Prometheus format
            with open(path, 'w') as f:
                for metric in self.metrics.values():
                    # Format: metric_name{tags} value timestamp
                    tags_str = ','.join(f'{k}="{v}"' for k, v in metric.tags.items())
                    tags_part = f"{{{tags_str}}}" if tags_str else ""
                    
                    f.write(
                        f"{metric.name}{tags_part} {metric.value} "
                        f"{int(metric.timestamp.timestamp() * 1000)}\n"
                    )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        active_alerts = [a for a in self.alerts.values() if not a.resolved]
        
        # Determine health based on alerts
        has_critical = any(a.severity == AlertSeverity.CRITICAL for a in active_alerts)
        has_error = any(a.severity == AlertSeverity.ERROR for a in active_alerts)
        has_warning = any(a.severity == AlertSeverity.WARNING for a in active_alerts)
        
        if has_critical:
            status = "critical"
            status_color = "red"
        elif has_error:
            status = "error"
            status_color = "red"
        elif has_warning:
            status = "warning"
            status_color = "yellow"
        else:
            status = "healthy"
            status_color = "green"
        
        return {
            'status': status,
            'status_color': status_color,
            'active_alerts': len(active_alerts),
            'metrics_count': len(self.metrics),
            'uptime': self._get_uptime()
        }
    
    def _get_uptime(self) -> float:
        """Get monitor uptime in seconds."""
        if self.metrics:
            first_metric = min(self.metrics.values(), key=lambda m: m.timestamp)
            return (datetime.utcnow() - first_metric.timestamp).total_seconds()
        return 0.0