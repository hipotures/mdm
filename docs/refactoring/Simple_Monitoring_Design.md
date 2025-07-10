# Simple Monitoring Design for Single-User MDM

## Overview

This document describes the lightweight monitoring system implemented for MDM's single-user deployment. Unlike enterprise systems that require complex infrastructure (Prometheus, Grafana, Jaeger, etc.), this solution provides essential monitoring capabilities with zero external dependencies.

## Design Principles

1. **Zero Infrastructure**: No Docker, no external services, no network ports
2. **Minimal Dependencies**: Uses only libraries already in MDM (loguru, SQLite)
3. **Low Overhead**: <1ms per operation, small disk footprint
4. **User-Friendly**: Simple CLI commands, optional HTML dashboard
5. **Automatic**: Works out of the box, no configuration required

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Simple Monitoring Layer                    │
├─────────────────────┬──────────────────┬───────────────────┤
│   File Logging      │  Metrics Storage │   Visualization   │
├─────────────────────┼──────────────────┼───────────────────┤
│  Loguru (rotating)  │  SQLite (local)  │  HTML Dashboard   │
└─────────────────────┴──────────────────┴───────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Data Location   │
                    ├───────────────────┤
                    │ ~/.mdm/metrics.db │
                    │ ~/.mdm/logs/      │
                    │ ~/.mdm/dashboard  │
                    └───────────────────┘
```

## Core Components

### 1. Simple Monitoring Module (`src/mdm/monitoring/simple.py`)

```python
class SimpleMonitor:
    """Lightweight monitoring for MDM - no external dependencies."""
    
    def __init__(self):
        self.metrics_db_path = Path.home() / ".mdm" / "metrics.db"
        self._init_database()     # Create SQLite tables
        self._configure_logging()  # Setup loguru
```

Key features:
- SQLite database for metrics storage
- Context manager for operation tracking
- Automatic metric recording
- Built-in cleanup for old data

### 2. Metrics Storage Schema

```sql
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_type TEXT NOT NULL,      -- e.g., 'dataset_register'
    operation TEXT NOT NULL,        -- Specific operation name
    duration_ms REAL,              -- Operation duration
    success INTEGER DEFAULT 1,      -- Success/failure flag
    dataset_name TEXT,             -- Related dataset
    row_count INTEGER,             -- Rows processed
    error_message TEXT,            -- Error details if failed
    metadata TEXT                  -- JSON metadata
);
```

### 3. Logging Configuration

- **Console**: Only warnings and errors (clean output)
- **File**: All messages with DEBUG level
- **Rotation**: 10MB max file size
- **Retention**: 7 days of logs
- **Location**: `~/.mdm/logs/mdm.log`

### 4. CLI Commands (`src/mdm/cli/stats.py`)

```bash
mdm stats show       # Recent operations
mdm stats summary    # Overall statistics
mdm stats dataset    # Dataset-specific metrics
mdm stats logs       # View log entries
mdm stats cleanup    # Remove old metrics
mdm stats dashboard  # Generate HTML report
```

### 5. HTML Dashboard (`src/mdm/monitoring/dashboard.py`)

Simple, self-contained HTML file with:
- Chart.js from CDN (no local dependencies)
- Operations summary
- Performance charts
- Recent activity table
- No server required - just open in browser

## Integration Points

### Dataset Registration

```python
# Automatic tracking in registrar.py
duration_ms = (end_time - start_time).total_seconds() * 1000
self.monitor.record_metric(
    MetricType.DATASET_REGISTER,
    f"register_{dataset_name}",
    duration_ms=duration_ms,
    success=True,
    dataset_name=dataset_name,
    row_count=total_rows
)
```

### Error Handling

All exceptions are automatically logged with context:
- Operation name
- Dataset involved
- Error message
- Stack trace (in log file)

## Usage Examples

### Check Migration Progress

```bash
# View recent operations
mdm stats show --limit 50

# Check for errors
mdm stats logs --level ERROR

# Generate progress report
mdm stats dashboard
```

### Monitor Specific Dataset

```bash
# See all operations for a dataset
mdm stats show --dataset my_dataset

# Get dataset-specific metrics
mdm stats dataset my_dataset
```

### Troubleshooting

```bash
# Check error logs
mdm stats logs --grep "connection error"

# View system summary
mdm stats summary

# Clean up old data
mdm stats cleanup --days 7
```

## Performance Impact

- **Metric Recording**: ~0.5ms per operation
- **Log Writing**: Buffered, negligible impact
- **Database Size**: ~1KB per operation
- **Dashboard Generation**: <100ms for 10,000 metrics

## File Structure

```
~/.mdm/
├── metrics.db          # SQLite metrics database
├── logs/
│   └── mdm.log        # Rotating log file
└── dashboard.html     # Generated dashboard (optional)
```

## Migration Monitoring

During the refactoring migration, monitoring helps track:
- Test execution times
- Error rates during parallel testing
- Performance comparisons (old vs new)
- Migration checkpoint success

Example migration tracking script:
```bash
#!/bin/bash
# Track migration progress
mdm stats summary
mdm stats logs --level ERROR --tail 10
mdm stats dashboard --output migration-status.html
```

## Future Enhancements (Keeping it Simple)

Potential improvements while maintaining simplicity:
1. CSV export for metrics
2. Email alerts for critical errors (optional)
3. Configurable retention policies
4. Basic anomaly detection (sudden performance drops)

## Comparison with Enterprise Observability

| Feature | Enterprise Stack | MDM Simple Monitoring |
|---------|-----------------|---------------------|
| Infrastructure | Prometheus, Grafana, Jaeger | None (SQLite file) |
| Setup Time | Hours/Days | Zero (automatic) |
| Resource Usage | GBs RAM, Multiple services | <10MB RAM, 1 file |
| Network Ports | Multiple | None |
| Configuration | Complex YAML files | None required |
| Maintenance | Regular updates, monitoring the monitors | Self-maintaining |
| Cost | High (infrastructure + time) | Zero |

## Conclusion

This simple monitoring system provides everything needed for a single-user MDM deployment:
- Track what's happening
- Find and debug errors  
- Monitor performance
- Generate reports

All without the complexity of enterprise monitoring stacks. Perfect for the refactoring migration where you need visibility without overhead.