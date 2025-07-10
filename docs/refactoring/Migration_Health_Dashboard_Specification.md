# Migration Health Dashboard Specification (SIMPLIFIED)

> **Note**: This document originally specified a complex real-time dashboard with streaming data, ML predictions, and multiple data sources. For MDM's single-user design, migration monitoring is handled through the simple monitoring system.
> 
> **For actual migration monitoring, use:**
> - `mdm stats show` - View recent operations
> - `mdm stats summary` - Overall statistics  
> - `mdm stats dashboard` - Generate simple HTML report
> - See [Simple_Monitoring_Design.md](./Simple_Monitoring_Design.md) for details

## Overview (Original Design - For Reference)
The Migration Health Dashboard provides real-time visibility into the MDM refactoring migration process, enabling teams to monitor progress, identify issues, and make data-driven decisions throughout the 21-week migration timeline.

## Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Migration Health Dashboard                    │
├─────────────────────┬───────────────────┬──────────────────────┤
│   Data Collection   │   Processing      │   Visualization      │
├─────────────────────┼───────────────────┼──────────────────────┤
│ • Migration Events  │ • Aggregation     │ • React Dashboard    │
│ • System Metrics    │ • Trend Analysis  │ • Real-time Updates  │
│ • Test Results      │ • Alert Engine    │ • Mobile Responsive  │
│ • Error Logs        │ • ML Predictions  │ • Export Reports     │
└─────────────────────┴───────────────────┴──────────────────────┘
```

## Core Components

### 1. Executive Summary View

```typescript
interface ExecutiveSummary {
  overall_health: 'healthy' | 'at_risk' | 'critical';
  migration_progress: number; // 0-100%
  estimated_completion: Date;
  blockers: Blocker[];
  key_metrics: {
    datasets_migrated: number;
    datasets_total: number;
    success_rate: number;
    rollback_count: number;
    performance_delta: number; // % change
  };
  recommendations: string[];
}
```

**Visual Design:**
```
┌─────────────────────────────────────────────────────────┐
│ MDM Migration Health                    Week 8 of 21    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Overall Status: ● AT RISK                              │
│                                                         │
│  Progress: ████████████░░░░░░░░░░ 38%                  │
│                                                         │
│  ┌─────────────┬──────────────┬─────────────────────┐  │
│  │ Migrated    │ Success Rate │ Est. Completion     │  │
│  │ 156/412     │ 94.2%        │ May 15, 2025 ⚠️     │  │
│  └─────────────┴──────────────┴─────────────────────┘  │
│                                                         │
│  ⚠️ 2 Critical Blockers Require Attention               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2. Migration Timeline View

```typescript
interface MigrationPhase {
  id: string;
  name: string;
  start_date: Date;
  end_date: Date;
  status: 'not_started' | 'in_progress' | 'completed' | 'blocked';
  progress: number;
  milestones: Milestone[];
  dependencies: string[]; // phase IDs
  risk_level: 'low' | 'medium' | 'high';
}

interface Milestone {
  name: string;
  date: Date;
  status: 'pending' | 'completed' | 'missed';
  blocker?: string;
}
```

**Visual Timeline:**
```
Week:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests  ████████ ✓
Config     ██████████ ✓
Abstr.        ████████████ ● (in progress)
Storage          ░░░░░░░░░░░░░░░
Feature                ░░░░░░░░░░░░░
Registr                      ░░░░░░░░░
Valid.                            ░░░░░░
Cleanup                                ░░░

Legend: █ Completed  ● In Progress  ░ Planned  ⚠️ At Risk  ❌ Blocked
```

### 3. Component Health Matrix

```typescript
interface ComponentHealth {
  component: string;
  old_version: VersionInfo;
  new_version: VersionInfo;
  migration_status: MigrationStatus;
  test_coverage: number;
  performance_metrics: PerformanceMetrics;
  issues: Issue[];
}

interface VersionInfo {
  version: string;
  health: 'healthy' | 'degraded' | 'failing';
  metrics: {
    error_rate: number;
    response_time_p95: number;
    throughput: number;
  };
}
```

**Component Matrix Display:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Component Health Matrix                                         │
├──────────────┬────────┬────────┬──────────┬─────────┬─────────┤
│ Component    │ Old    │ New    │ Migration│ Tests   │ Perf Δ  │
├──────────────┼────────┼────────┼──────────┼─────────┼─────────┤
│ Storage      │ ✓ 100% │ ✓ 98%  │ ████ 45% │ 156/200 │ +12%    │
│ Config       │ ✓ 100% │ ✓ 100% │ ████ 100%│ 45/45   │ +5%     │
│ Features     │ ✓ 99%  │ ⚠️ 92%  │ ██░░ 23% │ 78/150  │ -3%     │
│ Registration │ ✓ 98%  │ ● 95%  │ ░░░░ 0%  │ 0/89    │ N/A     │
└──────────────┴────────┴────────┴──────────┴─────────┴─────────┘

✓ Healthy  ⚠️ Warning  ● Degraded  ❌ Critical
```

### 4. Real-time Metrics Dashboard

```typescript
interface MetricStream {
  timestamp: Date;
  metric_type: 'operation' | 'error' | 'performance' | 'resource';
  component: string;
  value: number;
  unit: string;
  threshold?: {
    warning: number;
    critical: number;
  };
}
```

**Live Metrics Display:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Live Migration Metrics                          ⟳ Auto-refresh  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Operations/sec     Errors/min        Avg Response      Memory   │
│ ▁▃▅▇▅▃▁▃▅▇        ▁▁▁▃▁▁▁▁▁        ▁▂▃▄▅▆▇▆▅▄      ▄▄▅▅▆▆▇▇ │
│ 1,234 ops/s       3 errors          145ms            4.2GB     │
│ ↑ 12%             ↓ 70%             ↑ 8%             ↑ 15%     │
│                                                                 │
│ Active Migrations: 3    Queue Depth: 12    Workers: 8/10       │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Issue Tracker & Alerts

```typescript
interface Issue {
  id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  component: string;
  title: string;
  description: string;
  detected_at: Date;
  auto_resolved: boolean;
  resolution?: {
    resolved_at: Date;
    resolved_by: string;
    resolution_type: 'manual' | 'automatic' | 'rollback';
  };
  impact: {
    affected_datasets: string[];
    affected_users: number;
    estimated_downtime: number; // minutes
  };
}
```

**Issue Display:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Active Issues (5)                              🔔 Notifications │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 🔴 CRITICAL: Storage backend connection pool exhausted          │
│    Component: PostgreSQL | Detected: 2 min ago                  │
│    Impact: 15 datasets offline | Action Required               │
│                                                                 │
│ 🟡 WARNING: Feature generation slower than baseline             │
│    Component: Features | Detected: 45 min ago                   │
│    Impact: +35% processing time | Monitoring...                 │
│                                                                 │
│ 🔵 INFO: Configuration cache cleared successfully               │
│    Component: Config | Resolved: 1 hour ago                     │
│    Impact: None | Auto-resolved ✓                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6. Test Results & Validation

```typescript
interface TestSuite {
  name: string;
  phase: string;
  last_run: Date;
  results: {
    total: number;
    passed: number;
    failed: number;
    skipped: number;
    duration: number; // seconds
  };
  failing_tests: FailingTest[];
  trend: number[]; // pass rate history
}

interface FailingTest {
  name: string;
  error: string;
  first_failed: Date;
  attempts: number;
  related_issue?: string;
}
```

**Test Results Grid:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Test Suite Results                                              │
├────────────────┬────────┬────────┬────────┬──────────┬────────┤
│ Suite          │ Pass % │ Failed │ Time   │ Trend    │ Status │
├────────────────┼────────┼────────┼────────┼──────────┼────────┤
│ Unit Tests     │ 98.5%  │ 12     │ 2m 34s │ ▅▆▇▇▇    │ ✓      │
│ Integration    │ 94.2%  │ 8      │ 5m 12s │ ▃▅▆▅▇    │ ⚠️      │
│ E2E Tests      │ 87.3%  │ 23     │ 15m 8s │ ▇▅▃▂▁    │ ❌      │
│ Performance    │ 100%   │ 0      │ 8m 45s │ ▇▇▇▇▇    │ ✓      │
│ Regression     │ 91.0%  │ 5      │ 3m 22s │ ▆▅▆▇▆    │ ⚠️      │
└────────────────┴────────┴────────┴────────┴──────────┴────────┘

Recent Failures: [View Details] [Run Failed Tests] [Create Issue]
```

### 7. Performance Comparison

```typescript
interface PerformanceComparison {
  operation: string;
  old_system: PerformanceMetric;
  new_system: PerformanceMetric;
  delta_percent: number;
  regression: boolean;
  sample_size: number;
}

interface PerformanceMetric {
  p50: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  mean: number;
  std_dev: number;
}
```

**Performance Dashboard:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Performance Comparison: Old vs New System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Dataset Registration Time (seconds)                             │
│                                                                 │
│        P50     P95     P99     Mean    Std Dev                 │
│ Old:   2.3     5.1     8.2     2.8     1.2     ───────         │
│ New:   1.9     4.2     6.9     2.1     0.9     ━━━━━━━         │
│ Delta: -17%    -18%    -16%    -25%    -25%    ✓ Improved     │
│                                                                 │
│ Query Performance (ms)                                          │
│ ┌─────────────────────────────────────────┐                   │
│ │  Old █████████████ 145ms                │                   │
│ │  New ████████ 89ms (-39%)               │                   │
│ └─────────────────────────────────────────┘                   │
│                                                                 │
│ ⚠️ 2 operations showing regression > 10% [View Details]         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8. Resource Utilization

```typescript
interface ResourceMetrics {
  timestamp: Date;
  cpu: {
    usage_percent: number;
    cores_used: number;
  };
  memory: {
    used_gb: number;
    available_gb: number;
    percent: number;
  };
  disk: {
    read_mb_s: number;
    write_mb_s: number;
    used_gb: number;
    free_gb: number;
  };
  network: {
    in_mb_s: number;
    out_mb_s: number;
  };
  database: {
    connections_active: number;
    connections_idle: number;
    queries_per_sec: number;
  };
}
```

**Resource Monitor:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Resource Utilization                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ CPU Usage          Memory Usage        Disk I/O                │
│ ████████░░ 78%     ███████░░░ 68%     R: ▃▅▇▅▃ 125 MB/s      │
│ 12.5/16 cores      10.9/16 GB          W: ▁▂▃▂▁ 45 MB/s       │
│                                                                 │
│ Database Connections                   Network                  │
│ Active: ████ 45/200                   In:  12.3 MB/s          │
│ Idle:   ██░░ 23/200                   Out: 8.7 MB/s           │
│ Pool:   34% utilized                  Total: 21.0 MB/s        │
│                                                                 │
│ Top Resource Consumers:                                         │
│ 1. Feature Generation Worker #3 - CPU: 85%, Mem: 2.1GB         │
│ 2. PostgreSQL Backend - Connections: 35, Queries: 1.2k/s       │
│ 3. Data Validation Process - Disk I/O: 89 MB/s                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### Technology Stack

```yaml
frontend:
  framework: React 18
  state_management: Redux Toolkit
  ui_components: Ant Design
  charts: Recharts + D3.js
  real_time: WebSockets (Socket.io)
  
backend:
  api: FastAPI
  database: PostgreSQL (metrics storage)
  cache: Redis (real-time data)
  queue: RabbitMQ (event processing)
  
monitoring:
  metrics: Prometheus
  tracing: Jaeger
  logs: Elasticsearch
  
deployment:
  container: Docker
  orchestration: Kubernetes
  ci_cd: GitHub Actions
```

### Data Collection Architecture

```python
# src/mdm/dashboard/collector.py
from typing import Dict, Any, List
import asyncio
from datetime import datetime

class MigrationDataCollector:
    """Collects migration health data from various sources"""
    
    def __init__(self):
        self.sources = {
            'prometheus': PrometheusCollector(),
            'test_results': TestResultCollector(),
            'git': GitCollector(),
            'database': DatabaseCollector(),
            'logs': LogCollector()
        }
        
    async def collect_all(self) -> Dict[str, Any]:
        """Collect data from all sources"""
        tasks = []
        for name, collector in self.sources.items():
            task = asyncio.create_task(
                self._collect_with_timeout(name, collector)
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'data': dict(zip(self.sources.keys(), results))
        }
        
    async def _collect_with_timeout(self, name: str, collector, timeout: int = 30):
        """Collect with timeout to prevent hanging"""
        try:
            return await asyncio.wait_for(
                collector.collect(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Collector {name} timed out")
            return {'error': 'timeout', 'collector': name}

class PrometheusCollector:
    """Collects metrics from Prometheus"""
    
    async def collect(self) -> Dict[str, Any]:
        queries = {
            'operation_rate': 'rate(mdm_operations_total[5m])',
            'error_rate': 'rate(mdm_errors_total[5m])',
            'p95_latency': 'histogram_quantile(0.95, mdm_operation_duration_seconds)',
            'connection_pool_usage': 'mdm_connection_pool_active / mdm_connection_pool_size',
            'dataset_count': 'mdm_datasets_total'
        }
        
        results = {}
        for name, query in queries.items():
            results[name] = await self._execute_query(query)
            
        return results
```

### Real-time Updates

```typescript
// src/dashboard/websocket.ts
import { io, Socket } from 'socket.io-client';
import { store } from './store';
import { updateMetrics, addAlert, updateProgress } from './actions';

class DashboardWebSocket {
  private socket: Socket;
  
  constructor() {
    this.socket = io('ws://localhost:8001', {
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });
    
    this.setupEventHandlers();
  }
  
  private setupEventHandlers() {
    this.socket.on('metrics:update', (data) => {
      store.dispatch(updateMetrics(data));
    });
    
    this.socket.on('alert:new', (alert) => {
      store.dispatch(addAlert(alert));
      this.showNotification(alert);
    });
    
    this.socket.on('progress:update', (progress) => {
      store.dispatch(updateProgress(progress));
    });
    
    this.socket.on('test:results', (results) => {
      store.dispatch(updateTestResults(results));
    });
  }
  
  private showNotification(alert: Alert) {
    if (alert.severity === 'critical') {
      // Show browser notification
      if (Notification.permission === 'granted') {
        new Notification('Critical Migration Alert', {
          body: alert.message,
          icon: '/alert-icon.png',
          requireInteraction: true
        });
      }
    }
  }
  
  public subscribeToComponent(component: string) {
    this.socket.emit('subscribe', { component });
  }
  
  public unsubscribeFromComponent(component: string) {
    this.socket.emit('unsubscribe', { component });
  }
}
```

### Alert Engine

```python
# src/mdm/dashboard/alerts.py
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertRule:
    """Base class for alert rules"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def evaluate(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate rule and return alert if triggered"""
        raise NotImplementedError

class ThresholdRule(AlertRule):
    """Alert when metric exceeds threshold"""
    
    def __init__(self, name: str, metric: str, threshold: float, 
                 severity: AlertSeverity, operator: str = ">"):
        super().__init__(name, f"Alert when {metric} {operator} {threshold}")
        self.metric = metric
        self.threshold = threshold
        self.severity = severity
        self.operator = operator
        
    def evaluate(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        value = metrics.get(self.metric)
        if value is None:
            return None
            
        triggered = False
        if self.operator == ">" and value > self.threshold:
            triggered = True
        elif self.operator == "<" and value < self.threshold:
            triggered = True
        elif self.operator == ">=" and value >= self.threshold:
            triggered = True
        elif self.operator == "<=" and value <= self.threshold:
            triggered = True
            
        if triggered:
            return Alert(
                rule_name=self.name,
                severity=self.severity,
                message=f"{self.metric} is {value} (threshold: {self.operator} {self.threshold})",
                metric_value=value,
                threshold=self.threshold
            )
            
        return None

class AlertEngine:
    """Evaluates alert rules and manages alerts"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
    def _initialize_rules(self) -> List[AlertRule]:
        """Initialize standard alert rules"""
        return [
            # Performance alerts
            ThresholdRule("high_error_rate", "error_rate", 0.05, 
                         AlertSeverity.WARNING),
            ThresholdRule("critical_error_rate", "error_rate", 0.1, 
                         AlertSeverity.CRITICAL),
            ThresholdRule("slow_response", "p95_latency", 1.0, 
                         AlertSeverity.WARNING),
            
            # Resource alerts
            ThresholdRule("high_memory", "memory_percent", 80, 
                         AlertSeverity.WARNING),
            ThresholdRule("critical_memory", "memory_percent", 90, 
                         AlertSeverity.CRITICAL),
            ThresholdRule("pool_exhaustion", "connection_pool_usage", 0.9, 
                         AlertSeverity.CRITICAL),
            
            # Migration alerts
            ThresholdRule("low_success_rate", "migration_success_rate", 0.9, 
                         AlertSeverity.WARNING, operator="<"),
            ThresholdRule("behind_schedule", "schedule_delay_days", 7, 
                         AlertSeverity.WARNING),
            
            # Custom rules
            RollbackDetectionRule(),
            DataIntegrityRule(),
            PerformanceRegressionRule()
        ]
        
    async def evaluate_all(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate all rules and return new alerts"""
        new_alerts = []
        
        for rule in self.rules:
            alert = rule.evaluate(metrics)
            if alert:
                # Check if this is a new alert or existing one
                alert_key = f"{rule.name}:{alert.component}"
                
                if alert_key not in self.active_alerts:
                    # New alert
                    alert.triggered_at = datetime.utcnow()
                    self.active_alerts[alert_key] = alert
                    new_alerts.append(alert)
                    self.alert_history.append(alert)
                else:
                    # Update existing alert
                    self.active_alerts[alert_key].last_seen = datetime.utcnow()
                    
        # Check for resolved alerts
        self._check_resolved_alerts(metrics)
        
        return new_alerts
```

### Dashboard API

```python
# src/mdm/dashboard/api.py
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import asyncio

app = FastAPI(title="MDM Migration Dashboard API")

@app.get("/api/v1/health/summary")
async def get_health_summary() -> Dict[str, Any]:
    """Get executive summary of migration health"""
    collector = MigrationDataCollector()
    data = await collector.collect_all()
    
    analyzer = HealthAnalyzer()
    summary = analyzer.generate_summary(data)
    
    return summary

@app.get("/api/v1/migration/timeline")
async def get_migration_timeline() -> List[Dict[str, Any]]:
    """Get migration timeline with progress"""
    phases = MigrationPhaseTracker.get_all_phases()
    
    return [
        {
            'id': phase.id,
            'name': phase.name,
            'status': phase.status,
            'progress': phase.calculate_progress(),
            'start_date': phase.start_date.isoformat(),
            'end_date': phase.end_date.isoformat(),
            'milestones': phase.get_milestones(),
            'blockers': phase.get_blockers()
        }
        for phase in phases
    ]

@app.get("/api/v1/components/health")
async def get_component_health() -> Dict[str, Any]:
    """Get health status of all components"""
    components = [
        'storage', 'configuration', 'features', 
        'registration', 'api', 'cli'
    ]
    
    health_data = {}
    for component in components:
        health_data[component] = await check_component_health(component)
        
    return health_data

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    
    try:
        # Add client to subscribers
        await dashboard_manager.add_subscriber(websocket)
        
        # Send initial data
        await websocket.send_json({
            'type': 'initial_data',
            'data': await get_dashboard_state()
        })
        
        # Keep connection alive and handle messages
        while True:
            message = await websocket.receive_json()
            await handle_websocket_message(websocket, message)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await dashboard_manager.remove_subscriber(websocket)

@app.post("/api/v1/alerts/acknowledge/{alert_id}")
async def acknowledge_alert(alert_id: str, user: str):
    """Acknowledge an alert"""
    alert_manager = AlertManager()
    
    if not alert_manager.acknowledge(alert_id, user):
        raise HTTPException(status_code=404, detail="Alert not found")
        
    return {"status": "acknowledged", "alert_id": alert_id}

@app.get("/api/v1/reports/generate")
async def generate_report(
    start_date: str,
    end_date: str,
    format: str = "pdf"
) -> Dict[str, Any]:
    """Generate migration report"""
    reporter = MigrationReporter()
    
    report_data = await reporter.collect_report_data(start_date, end_date)
    
    if format == "pdf":
        report_url = await reporter.generate_pdf(report_data)
    elif format == "html":
        report_url = await reporter.generate_html(report_data)
    else:
        raise HTTPException(status_code=400, detail="Invalid format")
        
    return {
        "report_url": report_url,
        "generated_at": datetime.utcnow().isoformat()
    }
```

## Security & Access Control

### Authentication & Authorization

```python
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication

# Role-based access control
ROLES = {
    'viewer': ['read'],
    'operator': ['read', 'acknowledge_alerts'],
    'admin': ['read', 'write', 'rollback', 'configure']
}

@app.get("/api/v1/secure/rollback", dependencies=[Depends(require_role("admin"))])
async def initiate_rollback(component: str, target_version: str):
    """Initiate component rollback (admin only)"""
    # Implementation
    pass
```

### Audit Logging

```python
@dataclass
class AuditEntry:
    timestamp: datetime
    user: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

class AuditLogger:
    """Log all dashboard actions for compliance"""
    
    async def log_action(self, request: Request, action: str, **kwargs):
        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            user=request.user.username,
            action=action,
            resource=request.url.path,
            details=kwargs,
            ip_address=request.client.host
        )
        
        # Store in database and forward to SIEM
        await self.store_entry(entry)
        await self.forward_to_siem(entry)
```

## Deployment Configuration

### Docker Compose

```yaml
# docker-compose.dashboard.yml
version: '3.8'

services:
  dashboard-api:
    build: ./dashboard/api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/dashboard
      - REDIS_URL=redis://redis:6379
      - PROMETHEUS_URL=http://prometheus:9090
    depends_on:
      - db
      - redis
      
  dashboard-ui:
    build: ./dashboard/ui
    ports:
      - "3000:80"
    environment:
      - API_URL=http://dashboard-api:8000
      - WS_URL=ws://dashboard-api:8001
      
  dashboard-collector:
    build: ./dashboard/collector
    environment:
      - COLLECTION_INTERVAL=30
      - RETENTION_DAYS=90
    depends_on:
      - db
      - redis
```

### Kubernetes Deployment

```yaml
# k8s/dashboard-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mdm-dashboard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mdm-dashboard
  template:
    metadata:
      labels:
        app: mdm-dashboard
    spec:
      containers:
      - name: dashboard-api
        image: mdm/dashboard-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: dashboard-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          periodSeconds: 10
```

## Performance Requirements

- **Page Load Time**: < 2 seconds for initial dashboard load
- **Real-time Updates**: < 100ms latency for metric updates
- **API Response Time**: < 200ms for 95th percentile
- **Concurrent Users**: Support 100+ concurrent dashboard users
- **Data Retention**: 90 days of historical data
- **Chart Rendering**: < 500ms for complex visualizations

## Mobile Responsiveness

The dashboard must be fully responsive and functional on:
- Desktop (1920x1080 and above)
- Tablet (768x1024)
- Mobile (375x667)

Key mobile features:
- Collapsible navigation
- Touch-friendly controls
- Simplified views for small screens
- Offline capability with service workers

## Accessibility Requirements

- WCAG 2.1 AA compliance
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode
- Configurable font sizes
- Color-blind friendly palettes

## Integration Points

1. **CI/CD Pipeline**: Automatic updates on deployment
2. **Alerting Systems**: PagerDuty, Slack, Email
3. **Documentation**: Links to runbooks and procedures
4. **Ticketing System**: Create Jira tickets from alerts
5. **Chat Ops**: Slack commands for dashboard queries

## Success Metrics

- **Adoption Rate**: 90% of team using dashboard daily
- **Alert Response Time**: < 5 minutes for critical alerts
- **Issue Detection**: 95% of issues detected before user reports
- **Dashboard Availability**: 99.9% uptime
- **User Satisfaction**: > 4.5/5 rating