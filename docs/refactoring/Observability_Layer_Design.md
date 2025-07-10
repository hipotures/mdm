# Observability Layer Design (DEPRECATED)

> **Note**: This document describes an enterprise-grade observability stack that was initially planned but later replaced with a simpler solution more appropriate for MDM's single-user design. 
> 
> **Please see [Simple_Monitoring_Design.md](./Simple_Monitoring_Design.md) for the actual monitoring implementation.**

## Why This Was Changed

The original design included:
- Multiple external services (Prometheus, Grafana, Jaeger, Elasticsearch, Kibana)
- Docker Compose setup with 5+ containers
- Network ports and complex configuration
- Significant resource overhead (GBs of RAM)
- Maintenance burden of monitoring the monitoring system

This was replaced with a simple, built-in monitoring system that:
- Uses only SQLite and log files (no external services)
- Requires zero configuration
- Has minimal resource usage (<10MB RAM)
- Provides all essential monitoring for a single user
- Auto-maintains itself (log rotation, old metrics cleanup)

## What to Use Instead

1. **For Monitoring**: See [Simple_Monitoring_Design.md](./Simple_Monitoring_Design.md)
2. **For User Guide**: See [Monitoring_Guide.md](/docs/Monitoring_Guide.md)
3. **For Implementation**: The simple monitoring is already integrated into MDM

---

## Original Architecture (For Historical Reference Only)

```
┌─────────────────────────────────────────────────────────────┐
│                      Observability Layer                     │
├─────────────────┬─────────────────┬────────────────┬────────┤
│   Structured    │   Distributed   │    Metrics     │ Health │
│    Logging      │     Tracing     │  Collection    │ Checks │
├─────────────────┼─────────────────┼────────────────┼────────┤
│     Loguru      │  OpenTelemetry  │  Prometheus    │ Custom │
└─────────────────┴─────────────────┴────────────────┴────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Exporters       │
                    ├───────────────────┤
                    │ • Console         │
                    │ • File            │
                    │ • Elasticsearch   │
                    │ • Jaeger          │
                    │ • Grafana         │
                    └───────────────────┘
```

## Core Components

### 1. Structured Logging

```python
# src/mdm/observability/logging.py
import structlog
from loguru import logger
from typing import Dict, Any, Optional
import contextvars
from datetime import datetime

# Context variable for correlation ID
correlation_id_var = contextvars.ContextVar('correlation_id', default=None)

class StructuredLogger:
    """Structured logging with correlation ID support"""
    
    def __init__(self):
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                self._add_correlation_id,
                self._add_context,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure loguru for backward compatibility
        logger.add(
            self._structured_sink,
            format="{message}",
            serialize=True
        )
        
    def _add_correlation_id(self, logger, method_name, event_dict):
        """Add correlation ID to all log entries"""
        correlation_id = correlation_id_var.get()
        if correlation_id:
            event_dict['correlation_id'] = correlation_id
        return event_dict
        
    def _add_context(self, logger, method_name, event_dict):
        """Add contextual information"""
        event_dict['timestamp'] = datetime.utcnow().isoformat()
        event_dict['service'] = 'mdm'
        event_dict['version'] = get_mdm_version()
        event_dict['environment'] = os.getenv('MDM_ENVIRONMENT', 'development')
        return event_dict
        
    def _structured_sink(self, message):
        """Custom sink for structured output"""
        record = message.record
        
        # Extract structured data
        structured_data = {
            'level': record['level'].name,
            'message': record['message'],
            'timestamp': record['time'].isoformat(),
            'module': record['module'],
            'function': record['function'],
            'line': record['line'],
        }
        
        # Add extra fields
        if record.get('extra'):
            structured_data.update(record['extra'])
            
        # Send to appropriate backend
        self._send_to_backend(structured_data)
        
    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a logger instance"""
        return structlog.get_logger(name)

# Usage example
logger = StructuredLogger().get_logger(__name__)

@contextmanager
def log_operation(operation: str, **kwargs):
    """Context manager for logging operations with timing"""
    correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    
    start_time = time.time()
    logger.info(f"{operation}_started", correlation_id=correlation_id, **kwargs)
    
    try:
        yield correlation_id
        duration = time.time() - start_time
        logger.info(
            f"{operation}_completed",
            correlation_id=correlation_id,
            duration_ms=duration * 1000,
            **kwargs
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"{operation}_failed",
            correlation_id=correlation_id,
            duration_ms=duration * 1000,
            error=str(e),
            error_type=type(e).__name__,
            **kwargs
        )
        raise
    finally:
        correlation_id_var.set(None)
```

### 2. Distributed Tracing

```python
# src/mdm/observability/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from typing import Optional, Dict, Any
import functools

class TracingManager:
    """Manages distributed tracing with OpenTelemetry"""
    
    def __init__(self, service_name: str = "mdm"):
        self.service_name = service_name
        self.tracer_provider = TracerProvider()
        trace.set_tracer_provider(self.tracer_provider)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=os.getenv('JAEGER_HOST', 'localhost'),
            agent_port=int(os.getenv('JAEGER_PORT', '6831')),
            service_name=service_name,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        self.tracer_provider.add_span_processor(span_processor)
        
        # Auto-instrument libraries
        SQLAlchemyInstrumentor().instrument()
        RequestsInstrumentor().instrument()
        
        self.tracer = trace.get_tracer(__name__)
        
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Decorator for tracing operations"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(operation_name) as span:
                    # Add attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, str(value))
                            
                    # Add function context
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(
                            trace.Status(trace.StatusCode.ERROR, str(e))
                        )
                        span.record_exception(e)
                        raise
                        
            return wrapper
        return decorator
        
    def trace_dataset_operation(self, dataset_name: str, operation: str):
        """Specialized tracing for dataset operations"""
        return self.trace_operation(
            f"dataset.{operation}",
            attributes={
                "dataset.name": dataset_name,
                "dataset.operation": operation
            }
        )

# Usage
tracing = TracingManager()

@tracing.trace_operation("register_dataset")
def register_dataset(name: str, path: str):
    # Implementation
    pass

# Manual spans for complex operations
def complex_operation():
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("complex_operation") as span:
        span.set_attribute("operation.type", "complex")
        
        # Step 1
        with tracer.start_as_current_span("step1"):
            # Do step 1
            pass
            
        # Step 2
        with tracer.start_as_current_span("step2"):
            # Do step 2
            pass
```

### 3. Metrics Collection

```python
# src/mdm/observability/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary
from prometheus_client import start_http_server, generate_latest
from typing import Dict, Any, Callable
import time
import functools

class MetricsCollector:
    """Centralized metrics collection with Prometheus"""
    
    def __init__(self):
        # Operation metrics
        self.operation_counter = Counter(
            'mdm_operations_total',
            'Total number of operations',
            ['operation_type', 'status']
        )
        
        self.operation_duration = Histogram(
            'mdm_operation_duration_seconds',
            'Operation duration in seconds',
            ['operation_type'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Dataset metrics
        self.dataset_count = Gauge(
            'mdm_datasets_total',
            'Total number of datasets'
        )
        
        self.dataset_size_bytes = Gauge(
            'mdm_dataset_size_bytes',
            'Dataset size in bytes',
            ['dataset_name']
        )
        
        self.dataset_rows = Gauge(
            'mdm_dataset_rows_total',
            'Number of rows in dataset',
            ['dataset_name']
        )
        
        # Storage backend metrics
        self.connection_pool_size = Gauge(
            'mdm_connection_pool_size',
            'Connection pool size',
            ['backend', 'pool_name']
        )
        
        self.connection_pool_active = Gauge(
            'mdm_connection_pool_active',
            'Active connections in pool',
            ['backend', 'pool_name']
        )
        
        self.query_duration = Histogram(
            'mdm_query_duration_seconds',
            'Query execution time',
            ['backend', 'query_type'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
        )
        
        # Feature engineering metrics
        self.feature_generation_duration = Summary(
            'mdm_feature_generation_duration_seconds',
            'Feature generation duration',
            ['feature_type']
        )
        
        self.feature_count = Gauge(
            'mdm_features_total',
            'Total number of features',
            ['dataset_name']
        )
        
        # Error metrics
        self.error_counter = Counter(
            'mdm_errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
    def track_operation(self, operation_type: str):
        """Decorator to track operation metrics"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                status = 'success'
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = 'error'
                    self.error_counter.labels(
                        error_type=type(e).__name__,
                        component=func.__module__
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.operation_counter.labels(
                        operation_type=operation_type,
                        status=status
                    ).inc()
                    self.operation_duration.labels(
                        operation_type=operation_type
                    ).observe(duration)
                    
            return wrapper
        return decorator
        
    def update_dataset_metrics(self, dataset_name: str, metadata: Dict[str, Any]):
        """Update dataset-specific metrics"""
        self.dataset_size_bytes.labels(dataset_name=dataset_name).set(
            metadata.get('size_bytes', 0)
        )
        self.dataset_rows.labels(dataset_name=dataset_name).set(
            metadata.get('row_count', 0)
        )
        
    def update_pool_metrics(self, backend: str, pool_name: str, pool_info: Dict[str, Any]):
        """Update connection pool metrics"""
        self.connection_pool_size.labels(
            backend=backend,
            pool_name=pool_name
        ).set(pool_info.get('size', 0))
        
        self.connection_pool_active.labels(
            backend=backend,
            pool_name=pool_name
        ).set(pool_info.get('active', 0))
        
    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics server"""
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")

# Global metrics instance
metrics = MetricsCollector()

# Usage
@metrics.track_operation("dataset_registration")
def register_dataset(name: str, path: str):
    # Implementation
    pass
```

### 4. Health Checks

```python
# src/mdm/observability/health.py
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import asyncio

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        
    def register_check(self, name: str, check_func: Callable, critical: bool = True):
        """Register a health check"""
        self.checks[name] = {
            'func': check_func,
            'critical': critical
        }
        
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        tasks = []
        for name, check_info in self.checks.items():
            task = asyncio.create_task(self._run_single_check(name, check_info))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update results
        for name, result in zip(self.checks.keys(), results):
            if isinstance(result, Exception):
                self.results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(result)}"
                )
            else:
                self.results[name] = result
                
        return self._aggregate_results()
        
    async def _run_single_check(self, name: str, check_info: Dict) -> HealthCheckResult:
        """Run a single health check"""
        try:
            check_func = check_info['func']
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
                
            return HealthCheckResult(
                name=name,
                status=result.get('status', HealthStatus.HEALTHY),
                message=result.get('message', 'OK'),
                details=result.get('details')
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )
            
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate health check results"""
        overall_status = HealthStatus.HEALTHY
        critical_checks_failed = False
        
        for name, check_info in self.checks.items():
            result = self.results.get(name)
            if result and result.status == HealthStatus.UNHEALTHY:
                if check_info['critical']:
                    critical_checks_failed = True
                    overall_status = HealthStatus.UNHEALTHY
                elif overall_status != HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
        return {
            'status': overall_status.value,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'details': result.details,
                    'timestamp': result.timestamp.isoformat()
                }
                for name, result in self.results.items()
            }
        }

# Standard health checks
def check_database_connectivity() -> Dict[str, Any]:
    """Check database connectivity"""
    try:
        from mdm.storage import test_all_backends
        results = test_all_backends()
        
        if all(r['connected'] for r in results):
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'All backends connected',
                'details': results
            }
        else:
            return {
                'status': HealthStatus.DEGRADED,
                'message': 'Some backends not connected',
                'details': results
            }
    except Exception as e:
        return {
            'status': HealthStatus.UNHEALTHY,
            'message': f'Database check failed: {str(e)}'
        }

def check_disk_space() -> Dict[str, Any]:
    """Check available disk space"""
    import shutil
    
    mdm_path = Path.home() / '.mdm'
    stat = shutil.disk_usage(str(mdm_path))
    
    free_percent = (stat.free / stat.total) * 100
    
    if free_percent > 20:
        status = HealthStatus.HEALTHY
        message = f"Sufficient disk space: {free_percent:.1f}% free"
    elif free_percent > 10:
        status = HealthStatus.DEGRADED
        message = f"Low disk space: {free_percent:.1f}% free"
    else:
        status = HealthStatus.UNHEALTHY
        message = f"Critical disk space: {free_percent:.1f}% free"
        
    return {
        'status': status,
        'message': message,
        'details': {
            'total_bytes': stat.total,
            'free_bytes': stat.free,
            'used_bytes': stat.used,
            'free_percent': free_percent
        }
    }

def check_memory_usage() -> Dict[str, Any]:
    """Check memory usage"""
    import psutil
    
    memory = psutil.virtual_memory()
    
    if memory.percent < 80:
        status = HealthStatus.HEALTHY
        message = f"Memory usage normal: {memory.percent:.1f}%"
    elif memory.percent < 90:
        status = HealthStatus.DEGRADED
        message = f"High memory usage: {memory.percent:.1f}%"
    else:
        status = HealthStatus.UNHEALTHY
        message = f"Critical memory usage: {memory.percent:.1f}%"
        
    return {
        'status': status,
        'message': message,
        'details': {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used
        }
    }

# Initialize health checker
health_checker = HealthChecker()
health_checker.register_check('database', check_database_connectivity)
health_checker.register_check('disk_space', check_disk_space)
health_checker.register_check('memory', check_memory_usage)
```

## Integration Points

### 1. CLI Integration

```python
# src/mdm/cli/observability.py
import typer
from rich.console import Console
from rich.table import Table
import asyncio

app = typer.Typer(help="Observability commands")
console = Console()

@app.command()
def metrics(
    format: str = typer.Option("table", help="Output format: table, json, prometheus"),
    filter: str = typer.Option(None, help="Filter metrics by name")
):
    """Display current metrics"""
    from mdm.observability.metrics import metrics
    
    if format == "prometheus":
        from prometheus_client import generate_latest
        print(generate_latest().decode('utf-8'))
    else:
        # Format as table or JSON
        pass

@app.command()
def health(
    format: str = typer.Option("table", help="Output format: table, json"),
    check: str = typer.Option(None, help="Run specific health check")
):
    """Check system health"""
    from mdm.observability.health import health_checker
    
    results = asyncio.run(health_checker.run_checks())
    
    if format == "json":
        console.print_json(results)
    else:
        table = Table(title="Health Check Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Message")
        
        for check_name, check_result in results['checks'].items():
            status_style = {
                'healthy': 'green',
                'degraded': 'yellow',
                'unhealthy': 'red'
            }.get(check_result['status'], 'white')
            
            table.add_row(
                check_name,
                f"[{status_style}]{check_result['status']}[/{status_style}]",
                check_result['message']
            )
            
        console.print(table)

@app.command()
def trace(
    operation_id: str = typer.Argument(None, help="Operation ID to trace"),
    follow: bool = typer.Option(False, help="Follow trace in real-time")
):
    """View operation traces"""
    # Implementation for viewing traces
    pass
```

### 2. API Integration

```python
# src/mdm/api/observability.py
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
import json

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from mdm.observability.health import health_checker
    
    results = await health_checker.run_checks()
    status_code = 200 if results['status'] == 'healthy' else 503
    
    return Response(
        content=json.dumps(results),
        status_code=status_code,
        media_type="application/json"
    )

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest
    return generate_latest()

@app.get("/debug/pprof")
async def debug_profile():
    """Python profiling endpoint"""
    import cProfile
    import pstats
    import io
    
    pr = cProfile.Profile()
    pr.enable()
    
    # Run some operations
    await asyncio.sleep(0.1)
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    return PlainTextResponse(s.getvalue())
```

### 3. Storage Backend Integration

```python
# src/mdm/storage/observable_backend.py
from mdm.observability import logger, metrics, tracing

class ObservableStorageBackend(StorageBackend):
    """Storage backend with observability"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger.get_logger(f"{__name__}.{self.backend_type}")
        
    @tracing.trace_operation("storage.execute_query")
    @metrics.track_operation("query_execution")
    def execute_query(self, query: str, params: Optional[Dict] = None):
        """Execute query with observability"""
        with log_operation("query_execution", 
                          backend=self.backend_type,
                          query_type=self._get_query_type(query)) as correlation_id:
            
            self.logger.debug("Executing query", 
                            query=query, 
                            params=params,
                            correlation_id=correlation_id)
            
            start_time = time.time()
            try:
                result = super().execute_query(query, params)
                duration = time.time() - start_time
                
                metrics.query_duration.labels(
                    backend=self.backend_type,
                    query_type=self._get_query_type(query)
                ).observe(duration)
                
                self.logger.debug("Query completed",
                                duration_ms=duration * 1000,
                                row_count=len(result) if hasattr(result, '__len__') else None)
                
                return result
                
            except Exception as e:
                self.logger.error("Query failed",
                                error=str(e),
                                error_type=type(e).__name__)
                raise
```

## Deployment Configuration

### Docker Compose Setup

```yaml
# docker-compose.observability.yml
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "6831:6831/udp"  # Jaeger agent
      - "16686:16686"    # Jaeger UI
    environment:
      - COLLECTOR_ZIPKIN_HTTP_PORT=9411

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200

volumes:
  prometheus_data:
  grafana_data:
  elasticsearch_data:
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mdm'
    static_configs:
      - targets: ['localhost:8000']  # MDM metrics endpoint
    metrics_path: '/metrics'
    
  - job_name: 'mdm_health'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/health'
    scrape_interval: 30s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "MDM Observability Dashboard",
    "panels": [
      {
        "title": "Operation Rate",
        "targets": [{
          "expr": "rate(mdm_operations_total[5m])"
        }]
      },
      {
        "title": "Operation Duration",
        "targets": [{
          "expr": "histogram_quantile(0.95, mdm_operation_duration_seconds)"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(mdm_errors_total[5m])"
        }]
      },
      {
        "title": "Connection Pool Usage",
        "targets": [{
          "expr": "mdm_connection_pool_active / mdm_connection_pool_size"
        }]
      }
    ]
  }
}
```

## Alert Rules

```yaml
# alerts.yml
groups:
  - name: mdm_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(mdm_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"
          
      - alert: ConnectionPoolExhausted
        expr: mdm_connection_pool_active / mdm_connection_pool_size > 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Connection pool nearly exhausted"
          description: "Pool {{ $labels.pool_name }} is {{ $value }}% full"
          
      - alert: DiskSpaceLow
        expr: mdm_disk_free_percent < 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Only {{ $value }}% disk space remaining"
```

## Best Practices

1. **Correlation IDs**: Always use correlation IDs to trace operations across services
2. **Structured Logging**: Use structured logging for easy parsing and searching
3. **Metric Naming**: Follow Prometheus naming conventions
4. **Trace Sampling**: Use intelligent sampling to reduce overhead
5. **Health Check Granularity**: Balance between comprehensive checks and performance
6. **Dashboard Organization**: Group related metrics for easy troubleshooting
7. **Alert Fatigue**: Set appropriate thresholds to avoid alert noise

## Performance Considerations

- **Async Health Checks**: Run health checks asynchronously to avoid blocking
- **Metric Cardinality**: Limit label combinations to prevent explosion
- **Log Sampling**: Sample verbose logs in production
- **Trace Sampling**: Use head-based sampling for high-volume operations
- **Buffer Sizes**: Configure appropriate buffer sizes for exporters

## Security Considerations

- **Sensitive Data**: Never log passwords, tokens, or PII
- **Access Control**: Secure observability endpoints with authentication
- **Data Retention**: Set appropriate retention policies
- **Encryption**: Use TLS for all observability data in transit
- **Audit Logging**: Log all access to observability systems