# Performance Tuning Guidelines for Connection Pools

## Overview
This guide provides comprehensive performance tuning recommendations for connection pools in the refactored MDM architecture. Proper tuning is critical for achieving optimal performance while avoiding resource exhaustion.

## Connection Pool Architecture

### Pool Components
```python
ConnectionPool
├── Active Connections (in use)
├── Idle Connections (available)
├── Connection Factory
├── Health Checker
├── Metrics Collector
└── Eviction Policy
```

## Backend-Specific Tuning

### SQLite Connection Pools

SQLite has unique characteristics that require special consideration:

```python
# Recommended SQLite pool configuration
SQLITE_POOL_CONFIG = {
    'pool_size': 5,          # Maximum connections (SQLite handles concurrency differently)
    'max_overflow': 0,       # Don't allow overflow for SQLite
    'pool_pre_ping': True,   # Validate connections before use
    'pool_recycle': 3600,    # Recycle connections every hour
    'timeout': 30.0,         # Connection timeout in seconds
    'connect_args': {
        'check_same_thread': False,  # Allow cross-thread usage
        'timeout': 30.0,
        'isolation_level': None,     # Autocommit mode
    }
}

# Performance optimizations
SQLITE_PRAGMAS = {
    'journal_mode': 'WAL',           # Write-Ahead Logging for concurrency
    'synchronous': 'NORMAL',         # Balance safety/performance
    'cache_size': -64000,            # 64MB cache (negative = KB)
    'temp_store': 'MEMORY',          # Use memory for temp tables
    'mmap_size': 268435456,          # 256MB memory-mapped I/O
    'page_size': 4096,               # Optimal page size
    'optimize': True,                # Run OPTIMIZE on connection
}
```

### DuckDB Connection Pools

DuckDB is designed for analytics workloads:

```python
# Recommended DuckDB pool configuration
DUCKDB_POOL_CONFIG = {
    'pool_size': 10,         # DuckDB handles concurrency well
    'max_overflow': 5,       # Allow some overflow
    'pool_pre_ping': True,   # Validate connections
    'pool_recycle': 7200,    # Recycle every 2 hours
    'timeout': 60.0,         # Longer timeout for complex queries
}

# Performance settings
DUCKDB_SETTINGS = {
    'threads': 4,                    # Number of threads per connection
    'memory_limit': '4GB',           # Memory limit per connection
    'max_memory': '8GB',             # Maximum memory for all connections
    'temp_directory': '/tmp/duckdb', # SSD-backed temp storage
    'enable_profiling': False,       # Disable in production
    'enable_progress_bar': False,    # Disable for performance
    'checkpoint_threshold': '1GB',   # Auto checkpoint size
}
```

### PostgreSQL Connection Pools

PostgreSQL for production deployments:

```python
# Recommended PostgreSQL pool configuration
POSTGRES_POOL_CONFIG = {
    'pool_size': 20,         # Base pool size
    'max_overflow': 10,      # Allow overflow to 30 total
    'pool_pre_ping': True,   # Essential for network databases
    'pool_recycle': 3600,    # Recycle hourly
    'timeout': 30.0,         # Connection timeout
    'echo_pool': False,      # Disable in production
}

# Connection parameters
POSTGRES_CONNECT_ARGS = {
    'connect_timeout': 10,
    'application_name': 'mdm',
    'options': '-c statement_timeout=300000',  # 5 min statement timeout
    'keepalives': 1,
    'keepalives_idle': 30,
    'keepalives_interval': 10,
    'keepalives_count': 5,
}

# Server-side settings (postgresql.conf)
POSTGRES_SERVER_TUNING = {
    'max_connections': 200,
    'shared_buffers': '4GB',
    'effective_cache_size': '12GB',
    'maintenance_work_mem': '1GB',
    'work_mem': '32MB',
    'wal_buffers': '16MB',
    'checkpoint_completion_target': 0.9,
    'random_page_cost': 1.1,  # For SSD storage
}
```

## Pool Sizing Formulas

### Basic Formula
```
pool_size = (number_of_workers × connections_per_worker) + overhead

Where:
- number_of_workers = CPU cores or thread count
- connections_per_worker = average concurrent operations
- overhead = buffer for management operations (typically 2-5)
```

### Advanced Sizing Calculation
```python
def calculate_optimal_pool_size(
    cpu_cores: int,
    expected_concurrent_users: int,
    avg_query_time_ms: float,
    target_response_time_ms: float,
    connection_overhead_ms: float = 50
) -> int:
    """Calculate optimal pool size based on Little's Law"""
    
    # Effective processing time including overhead
    effective_time = avg_query_time_ms + connection_overhead_ms
    
    # Maximum concurrent requests to meet target response time
    max_concurrent = (target_response_time_ms / effective_time) * cpu_cores
    
    # Pool size with safety factor
    pool_size = min(
        int(max_concurrent * 1.2),  # 20% safety factor
        expected_concurrent_users * 2,  # Don't over-provision
        cpu_cores * 4  # Reasonable upper limit
    )
    
    return max(pool_size, cpu_cores)  # At least one per core
```

## Monitoring and Metrics

### Key Metrics to Track

```python
class PoolMetrics:
    # Health indicators
    active_connections: int      # Currently in use
    idle_connections: int        # Available for use
    pending_requests: int        # Waiting for connection
    
    # Performance metrics
    avg_wait_time: float        # Time to acquire connection
    avg_usage_time: float       # Time connection is held
    connection_errors: int      # Failed connection attempts
    timeout_count: int          # Requests that timed out
    
    # Resource utilization
    pool_utilization: float     # active / pool_size
    overflow_utilization: float # overflow_active / max_overflow
    connection_churn: float     # connections created/destroyed per minute
```

### Monitoring Implementation

```python
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ConnectionPoolMonitor:
    """Monitor connection pool performance"""
    
    def __init__(self, pool_name: str):
        self.pool_name = pool_name
        self.metrics = PoolMetrics()
        
    @contextmanager
    def track_connection_acquisition(self):
        """Track time to acquire connection"""
        start = time.time()
        try:
            yield
        finally:
            self.metrics.avg_wait_time = time.time() - start
            
    def check_pool_health(self) -> Dict[str, Any]:
        """Health check with recommendations"""
        health = {
            'status': 'healthy',
            'warnings': [],
            'recommendations': []
        }
        
        # Check utilization
        if self.metrics.pool_utilization > 0.8:
            health['warnings'].append('High pool utilization')
            health['recommendations'].append('Increase pool_size')
            
        # Check wait times
        if self.metrics.avg_wait_time > 1.0:  # 1 second
            health['warnings'].append('High connection wait time')
            health['recommendations'].append('Increase pool_size or optimize queries')
            
        # Check errors
        if self.metrics.connection_errors > 10:
            health['status'] = 'unhealthy'
            health['warnings'].append('Excessive connection errors')
            
        return health
```

## Performance Optimization Strategies

### 1. Connection Pooling Patterns

```python
# Pattern 1: Per-Dataset Pools (Recommended)
class PerDatasetPoolManager:
    """Separate pool for each dataset for isolation"""
    def __init__(self, base_config: Dict[str, Any]):
        self.pools: Dict[str, Pool] = {}
        self.base_config = base_config
        
    def get_pool(self, dataset_name: str) -> Pool:
        if dataset_name not in self.pools:
            config = self._adjust_config_for_dataset(dataset_name)
            self.pools[dataset_name] = create_pool(**config)
        return self.pools[dataset_name]
        
    def _adjust_config_for_dataset(self, dataset_name: str) -> Dict:
        """Adjust pool size based on dataset characteristics"""
        config = self.base_config.copy()
        dataset_size = get_dataset_size(dataset_name)
        
        if dataset_size > 1_000_000_000:  # 1B rows
            config['pool_size'] *= 2
        elif dataset_size < 1_000_000:  # 1M rows
            config['pool_size'] = max(2, config['pool_size'] // 2)
            
        return config

# Pattern 2: Read/Write Pool Separation
class ReadWritePoolManager:
    """Separate pools for read and write operations"""
    def __init__(self, read_config: Dict, write_config: Dict):
        self.read_pool = create_pool(**read_config)
        self.write_pool = create_pool(**write_config)
        
    def get_connection(self, operation_type: str):
        pool = self.read_pool if operation_type == 'read' else self.write_pool
        return pool.connect()
```

### 2. Query Optimization Tips

```python
# Use prepared statements
PREPARED_STATEMENTS = {
    'get_by_id': 'SELECT * FROM {table} WHERE id = %s',
    'batch_insert': 'INSERT INTO {table} VALUES %s',
    'update_stats': 'UPDATE {table} SET stats = %s WHERE id = %s'
}

# Batch operations
def batch_insert_optimized(pool: Pool, data: List[Dict], batch_size: int = 1000):
    """Optimized batch insert with connection reuse"""
    with pool.connect() as conn:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            conn.execute_batch(PREPARED_STATEMENTS['batch_insert'], batch)
            
# Connection pinning for transactions
@contextmanager
def pinned_transaction(pool: Pool):
    """Pin connection for entire transaction"""
    conn = pool.connect()
    trans = conn.begin()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
        conn.close()
```

### 3. Resource Limits and Throttling

```python
import asyncio
from typing import Optional

class ResourceLimiter:
    """Prevent resource exhaustion"""
    
    def __init__(
        self,
        max_memory_mb: int = 4096,
        max_cpu_percent: int = 80,
        max_connections_per_second: int = 100
    ):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.max_connections_per_second = max_connections_per_second
        self.semaphore = asyncio.Semaphore(max_connections_per_second)
        
    async def acquire_if_available(self) -> bool:
        """Check resource availability before acquiring connection"""
        if not self._check_memory():
            return False
        if not self._check_cpu():
            return False
            
        async with self.semaphore:
            return True
            
    def _check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        import psutil
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < (self.max_memory_mb / psutil.virtual_memory().total * 100)
        
    def _check_cpu(self) -> bool:
        """Check if CPU usage is within limits"""
        import psutil
        return psutil.cpu_percent(interval=0.1) < self.max_cpu_percent
```

## Troubleshooting Common Issues

### Issue 1: Connection Pool Exhaustion
**Symptoms**: Timeouts, "too many connections" errors
**Solutions**:
```python
# 1. Increase pool size
config['pool_size'] = 30
config['max_overflow'] = 20

# 2. Reduce connection hold time
config['pool_recycle'] = 1800  # 30 minutes

# 3. Add connection timeout
config['pool_timeout'] = 30  # Fail fast

# 4. Implement circuit breaker
from circuit_breaker import CircuitBreaker
pool = CircuitBreaker(pool, failure_threshold=5)
```

### Issue 2: Memory Leaks
**Symptoms**: Growing memory usage, OOM errors
**Solutions**:
```python
# 1. Enable connection recycling
config['pool_recycle'] = 3600  # Recycle every hour

# 2. Limit result set size
conn.execute("SET max_result_size = 1000000")

# 3. Use streaming for large results
def stream_large_results(pool: Pool, query: str):
    with pool.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(query)
        for row in result:
            yield row  # Process one row at a time
```

### Issue 3: Slow Connection Acquisition
**Symptoms**: High wait times, poor response times
**Solutions**:
```python
# 1. Pre-warm the pool
def prewarm_pool(pool: Pool):
    """Create all connections upfront"""
    connections = []
    for _ in range(pool.size()):
        connections.append(pool.connect())
    for conn in connections:
        conn.close()

# 2. Use connection pooling middleware
class PoolingMiddleware:
    def __init__(self, pool: Pool):
        self.pool = pool
        self.local = threading.local()
        
    def get_connection(self):
        if not hasattr(self.local, 'conn'):
            self.local.conn = self.pool.connect()
        return self.local.conn
```

## Best Practices Summary

1. **Size pools based on workload**: Use formulas and monitoring to find optimal size
2. **Monitor continuously**: Track metrics and adjust based on patterns
3. **Use backend-specific optimizations**: Each database has unique characteristics
4. **Implement graceful degradation**: Handle pool exhaustion without crashing
5. **Test under load**: Validate settings with realistic workloads
6. **Document settings**: Keep track of why each value was chosen
7. **Plan for growth**: Leave headroom for traffic increases

## Configuration Templates

### Development Environment
```yaml
# config/pool_config_dev.yaml
sqlite:
  pool_size: 5
  max_overflow: 0
  timeout: 30

duckdb:
  pool_size: 5
  max_overflow: 2
  memory_limit: "1GB"

postgresql:
  pool_size: 10
  max_overflow: 5
  echo_pool: true  # Enable for debugging
```

### Production Environment
```yaml
# config/pool_config_prod.yaml
sqlite:
  pool_size: 10
  max_overflow: 0
  timeout: 30
  pragmas:
    journal_mode: "WAL"
    synchronous: "NORMAL"

duckdb:
  pool_size: 20
  max_overflow: 10
  memory_limit: "8GB"
  threads: 8

postgresql:
  pool_size: 50
  max_overflow: 25
  pool_pre_ping: true
  pool_recycle: 3600
  echo_pool: false
```

## Performance Testing Script

```bash
#!/bin/bash
# test_pool_performance.sh

# Test different pool sizes
for pool_size in 5 10 20 50; do
    echo "Testing pool_size=$pool_size"
    export MDM_POOL_SIZE=$pool_size
    
    # Run load test
    mdm benchmark \
        --concurrent-users 100 \
        --duration 300 \
        --operation mixed \
        --report pool_test_${pool_size}.json
done

# Analyze results
mdm analyze-benchmarks pool_test_*.json --output pool_optimization_report.html
```