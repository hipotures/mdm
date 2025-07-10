# MDM Simple Monitoring Guide

## Overview

MDM now includes a lightweight, built-in monitoring system designed for single-user deployments. No external services or complex infrastructure required - everything runs locally and stores data in SQLite.

## Features

- **Automatic Logging**: All operations logged to rotating file with loguru
- **Metrics Collection**: Operation timings, success rates, and dataset statistics stored in SQLite
- **CLI Statistics**: View recent operations and summaries from command line
- **HTML Dashboard**: Optional browser-based dashboard with charts

## Getting Started

The monitoring system starts automatically when you use MDM. No configuration needed!

### View Recent Operations

```bash
# Show last 20 operations
mdm stats show

# Show last 50 operations
mdm stats show --limit 50

# Filter by dataset
mdm stats show --dataset my_dataset

# Filter by operation type
mdm stats show --type dataset_register
```

### View Summary Statistics

```bash
# Show overall statistics
mdm stats summary
```

Output includes:
- Total operations count
- Success rate
- Average duration
- Operations by type
- Recent errors

### View Dataset-Specific Metrics

```bash
# Show metrics for a specific dataset
mdm stats dataset my_dataset
```

### View Logs

```bash
# Show last 50 log entries
mdm stats logs

# Filter by log level
mdm stats logs --level ERROR

# Search logs
mdm stats logs --grep "connection"
```

### Generate HTML Dashboard

```bash
# Generate and open dashboard in browser
mdm stats dashboard

# Generate without opening
mdm stats dashboard --no-open

# Save to specific location
mdm stats dashboard --output /tmp/mdm-dashboard.html
```

## Monitoring Integration

The monitoring system automatically tracks:

### Dataset Operations
- Registration time and success
- Number of rows processed
- Files discovered
- Features generated

### Storage Operations
- Query execution times
- Connection pool usage
- Backend-specific metrics

### Errors and Issues
- Failed operations with error messages
- Slow operations (configurable threshold)
- Resource usage warnings

## File Locations

All monitoring data is stored in your MDM home directory:

```
~/.mdm/
├── metrics.db          # SQLite database with metrics
├── logs/
│   └── mdm.log        # Rotating log file (max 10MB, 7 days retention)
└── dashboard.html     # Generated dashboard (optional)
```

## Configuration

The monitoring system uses sensible defaults, but you can customize through environment variables:

```bash
# Set log level (DEBUG, INFO, WARNING, ERROR)
export MDM_LOGGING_LEVEL=DEBUG

# Set custom log file location
export MDM_LOGGING_FILE=/var/log/mdm.log
```

## Performance Impact

The monitoring system is designed to have minimal impact:
- Metrics are written asynchronously to SQLite
- Logs are buffered and written in batches
- No network calls or external dependencies
- Typical overhead: <1ms per operation

## Maintenance

### Clean Up Old Metrics

```bash
# Keep only last 30 days of metrics (default)
mdm stats cleanup

# Keep only last 7 days
mdm stats cleanup --days 7
```

### Log Rotation

Logs are automatically rotated:
- When file reaches 10MB
- Files older than 7 days are deleted
- No manual intervention needed

## Example Workflow

```bash
# 1. Register a dataset
mdm dataset register sales_data /path/to/data.csv

# 2. Check if it succeeded
mdm stats show --limit 5

# 3. View detailed metrics
mdm stats dataset sales_data

# 4. Generate dashboard to see trends
mdm stats dashboard

# 5. If there were errors, check logs
mdm stats logs --level ERROR
```

## Troubleshooting

### No metrics appearing
- Check if `~/.mdm/metrics.db` exists
- Ensure write permissions on `~/.mdm/` directory
- Look for errors in `~/.mdm/logs/mdm.log`

### Dashboard not opening
- Check if default browser is configured
- Try generating without auto-open: `mdm stats dashboard --no-open`
- Open the file manually: `~/.mdm/dashboard.html`

### High disk usage
- Run cleanup: `mdm stats cleanup --days 7`
- Check log file size: `ls -lh ~/.mdm/logs/`
- Metrics database is typically <10MB even with thousands of operations

## Integration with Your Code

If you're using MDM programmatically, monitoring is automatic:

```python
from mdm import MDMClient

# Monitoring happens automatically
client = MDMClient()
client.register_dataset("my_data", "/path/to/data")

# You can also access metrics directly
from mdm.monitoring import SimpleMonitor

monitor = SimpleMonitor()
stats = monitor.get_summary_stats()
print(f"Total operations: {stats['overall']['total_operations']}")
```

## Privacy and Security

- All data stored locally - nothing sent to external services
- No telemetry or usage tracking
- Logs may contain file paths and dataset names
- No sensitive data (passwords, keys) is ever logged

## Future Enhancements

Planned improvements (keeping it simple):
- CSV export for metrics
- Email alerts for errors (optional)
- Configurable retention policies
- Basic anomaly detection