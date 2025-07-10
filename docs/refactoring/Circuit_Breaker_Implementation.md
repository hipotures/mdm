# Circuit Breaker Implementation for Storage Backends

## Overview

This document describes the circuit breaker pattern implementation for MDM storage backends, providing automatic failure detection, fallback mechanisms, and self-healing capabilities to ensure system resilience during backend failures.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Circuit Breaker System                    │
├──────────────┬─────────────────┬────────────────┬──────────┤
│    State     │   Monitoring    │   Fallback     │  Metrics │
│   Machine    │    & Health     │  Strategies    │ & Alerts │
├──────────────┼─────────────────┼────────────────┼──────────┤
│ • Closed     │ • Error Rate    │ • Cache        │ • State  │
│ • Open       │ • Latency       │ • Read-only    │ • Errors │
│ • Half-Open  │ • Timeouts      │ • Alternative  │ • Recovery│
└──────────────┴─────────────────┴────────────────┴──────────┘
```

## Core Implementation

### 1. Circuit Breaker Base Class

```python
# src/mdm/resilience/circuit_breaker.py
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes before closing
    timeout: float = 60.0               # Seconds before trying half-open
    window_size: int = 10               # Rolling window for metrics
    error_types: List[type] = field(default_factory=lambda: [Exception])
    exclude_types: List[type] = field(default_factory=list)
    
    # Advanced settings
    failure_rate_threshold: float = 0.5  # 50% failure rate
    slow_call_duration: float = 1.0      # Slow call threshold
    slow_call_rate_threshold: float = 0.5  # 50% slow calls
    
    # Recovery settings
    recovery_timeout: float = 120.0      # Max time in half-open
    backoff_factor: float = 2.0         # Exponential backoff

@dataclass
class CallMetrics:
    """Metrics for a single call"""
    timestamp: datetime
    duration: float
    success: bool
    error: Optional[Exception] = None

class CircuitBreaker:
    """Circuit breaker implementation with advanced features"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change: datetime = datetime.now()
        self._metrics: deque = deque(maxlen=config.window_size)
        self._lock = threading.RLock()
        self._half_open_calls = 0
        self._listeners: List[Callable] = []
        
    @property
    def state(self) -> CircuitState:
        """Get current state with automatic transition check"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        return self._execute_with_breaker(func, args, kwargs)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator usage"""
        def wrapper(*args, **kwargs):
            return self._execute_with_breaker(func, args, kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    def _execute_with_breaker(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with circuit breaker logic"""
        # Check if circuit allows the call
        if not self._can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Failed {self._failure_count} times. "
                f"Will retry after {self._get_retry_after()}"
            )
        
        # Execute and track metrics
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self._on_success(duration)
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            if self._should_count_as_failure(e):
                self._on_failure(duration, e)
            raise
    
    def _can_execute(self) -> bool:
        """Check if execution is allowed"""
        state = self.state  # This checks for automatic transitions
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            return False
        else:  # HALF_OPEN
            with self._lock:
                # Limit concurrent calls in half-open state
                if self._half_open_calls >= self.config.success_threshold:
                    return False
                self._half_open_calls += 1
                return True
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset from OPEN state"""
        if self._last_failure_time is None:
            return True
            
        timeout = timedelta(seconds=self.config.timeout)
        return datetime.now() - self._last_failure_time >= timeout
    
    def _should_count_as_failure(self, error: Exception) -> bool:
        """Determine if error should count as failure"""
        # Check excluded types
        for exc_type in self.config.exclude_types:
            if isinstance(error, exc_type):
                return False
                
        # Check included types
        for exc_type in self.config.error_types:
            if isinstance(error, exc_type):
                return True
                
        return False
    
    def _on_success(self, duration: float):
        """Handle successful call"""
        with self._lock:
            # Record metrics
            self._metrics.append(CallMetrics(
                timestamp=datetime.now(),
                duration=duration,
                success=True
            ))
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                self._half_open_calls -= 1
                
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0
    
    def _on_failure(self, duration: float, error: Exception):
        """Handle failed call"""
        with self._lock:
            # Record metrics
            self._metrics.append(CallMetrics(
                timestamp=datetime.now(),
                duration=duration,
                success=False,
                error=error
            ))
            
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls -= 1
                self._transition_to(CircuitState.OPEN)
                
            elif self._state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self._should_open_circuit():
                    self._transition_to(CircuitState.OPEN)
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on metrics"""
        # Simple threshold check
        if self._failure_count >= self.config.failure_threshold:
            return True
            
        # Advanced: Check failure rate
        recent_metrics = list(self._metrics)
        if len(recent_metrics) >= self.config.window_size:
            failures = sum(1 for m in recent_metrics if not m.success)
            failure_rate = failures / len(recent_metrics)
            
            if failure_rate >= self.config.failure_rate_threshold:
                return True
                
        # Advanced: Check slow call rate
        slow_calls = sum(
            1 for m in recent_metrics 
            if m.duration >= self.config.slow_call_duration
        )
        if len(recent_metrics) > 0:
            slow_rate = slow_calls / len(recent_metrics)
            if slow_rate >= self.config.slow_call_rate_threshold:
                return True
                
        return False
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state"""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.now()
        
        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.OPEN:
            self._success_count = 0
            
        # Log transition
        logger.info(
            f"Circuit breaker '{self.name}' transitioned: "
            f"{old_state.value} → {new_state.value}"
        )
        
        # Notify listeners
        self._notify_listeners(old_state, new_state)
    
    def _get_retry_after(self) -> str:
        """Get human-readable retry time"""
        if self._last_failure_time is None:
            return "unknown"
            
        retry_time = self._last_failure_time + timedelta(seconds=self.config.timeout)
        seconds_until = (retry_time - datetime.now()).total_seconds()
        
        if seconds_until <= 0:
            return "now"
        elif seconds_until < 60:
            return f"{int(seconds_until)} seconds"
        else:
            return f"{int(seconds_until / 60)} minutes"
    
    def _notify_listeners(self, old_state: CircuitState, new_state: CircuitState):
        """Notify state change listeners"""
        for listener in self._listeners:
            try:
                listener(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"Error notifying listener: {e}")
    
    def add_listener(self, listener: Callable):
        """Add state change listener"""
        self._listeners.append(listener)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self._lock:
            recent_metrics = list(self._metrics)
            
            if not recent_metrics:
                return {
                    'state': self._state.value,
                    'total_calls': 0,
                    'failure_rate': 0.0,
                    'avg_duration': 0.0
                }
            
            total_calls = len(recent_metrics)
            failures = sum(1 for m in recent_metrics if not m.success)
            total_duration = sum(m.duration for m in recent_metrics)
            
            return {
                'state': self._state.value,
                'total_calls': total_calls,
                'failures': failures,
                'failure_rate': failures / total_calls if total_calls > 0 else 0.0,
                'avg_duration': total_duration / total_calls if total_calls > 0 else 0.0,
                'last_failure': self._last_failure_time.isoformat() if self._last_failure_time else None,
                'last_state_change': self._last_state_change.isoformat()
            }
    
    def reset(self):
        """Manually reset the circuit breaker"""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._metrics.clear()

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass
```

### 2. Storage Backend Integration

```python
# src/mdm/storage/resilient_backend.py
from mdm.storage.base import StorageBackend
from mdm.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from typing import Optional, Any, Dict, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ResilientStorageBackend(StorageBackend):
    """Storage backend with circuit breaker protection"""
    
    def __init__(self, backend: StorageBackend, circuit_config: Optional[CircuitBreakerConfig] = None):
        self.backend = backend
        self.circuit_config = circuit_config or CircuitBreakerConfig(
            failure_threshold=3,
            timeout=30.0,
            error_types=[ConnectionError, TimeoutError],
            slow_call_duration=5.0
        )
        
        # Create circuit breakers for different operations
        self.read_circuit = CircuitBreaker(f"{backend.name}_read", self.circuit_config)
        self.write_circuit = CircuitBreaker(f"{backend.name}_write", self.circuit_config)
        self.query_circuit = CircuitBreaker(f"{backend.name}_query", self.circuit_config)
        
        # Fallback strategies
        self.cache = {}  # Simple in-memory cache for read fallback
        self.read_only_mode = False
        
        # Add listeners for monitoring
        self.read_circuit.add_listener(self._on_circuit_state_change)
        self.write_circuit.add_listener(self._on_circuit_state_change)
        self.query_circuit.add_listener(self._on_circuit_state_change)
    
    def read_data(self, dataset_name: str, **kwargs) -> pd.DataFrame:
        """Read data with circuit breaker protection"""
        cache_key = f"{dataset_name}:{str(kwargs)}"
        
        try:
            # Try to read through circuit breaker
            @self.read_circuit
            def _read():
                return self.backend.read_data(dataset_name, **kwargs)
                
            data = _read()
            
            # Update cache on success
            self.cache[cache_key] = data.copy()
            return data
            
        except CircuitBreakerOpenError:
            # Circuit is open, try fallback
            logger.warning(f"Read circuit open for {dataset_name}, attempting fallback")
            return self._read_fallback(cache_key, dataset_name)
    
    def write_data(self, dataset_name: str, data: pd.DataFrame, **kwargs):
        """Write data with circuit breaker protection"""
        if self.read_only_mode:
            raise ReadOnlyModeError("System is in read-only mode due to backend issues")
        
        try:
            @self.write_circuit
            def _write():
                return self.backend.write_data(dataset_name, data, **kwargs)
                
            return _write()
            
        except CircuitBreakerOpenError:
            # Circuit is open, queue for later or use alternative
            logger.error(f"Write circuit open for {dataset_name}")
            self._queue_write_operation(dataset_name, data, kwargs)
            raise
    
    def execute_query(self, query: str, **kwargs) -> Any:
        """Execute query with circuit breaker protection"""
        try:
            @self.query_circuit
            def _query():
                return self.backend.execute_query(query, **kwargs)
                
            return _query()
            
        except CircuitBreakerOpenError:
            # Circuit is open
            logger.error(f"Query circuit open, cannot execute: {query[:50]}...")
            
            # Try to return cached results if available
            query_hash = hash(query + str(kwargs))
            if query_hash in self.cache:
                logger.info("Returning cached query results")
                return self.cache[query_hash]
            raise
    
    def _read_fallback(self, cache_key: str, dataset_name: str) -> pd.DataFrame:
        """Fallback strategy for read operations"""
        # Strategy 1: Return from cache
        if cache_key in self.cache:
            logger.info(f"Returning cached data for {dataset_name}")
            return self.cache[cache_key].copy()
        
        # Strategy 2: Try alternative backend
        if hasattr(self, 'fallback_backend'):
            try:
                logger.info(f"Attempting read from fallback backend")
                return self.fallback_backend.read_data(dataset_name)
            except Exception as e:
                logger.error(f"Fallback backend also failed: {e}")
        
        # Strategy 3: Return empty dataset with warning
        logger.warning(f"No fallback available for {dataset_name}, returning empty dataset")
        return pd.DataFrame()
    
    def _queue_write_operation(self, dataset_name: str, data: pd.DataFrame, kwargs: dict):
        """Queue write operation for later retry"""
        # In a real implementation, this would use a persistent queue
        # For now, we'll just log it
        logger.warning(
            f"Queueing write operation for {dataset_name} "
            f"({len(data)} rows) for later retry"
        )
    
    def _on_circuit_state_change(self, circuit_name: str, old_state: CircuitState, new_state: CircuitState):
        """Handle circuit state changes"""
        logger.info(f"Circuit {circuit_name} changed: {old_state.value} → {new_state.value}")
        
        # If any write circuit is open, enable read-only mode
        if 'write' in circuit_name and new_state == CircuitState.OPEN:
            self.read_only_mode = True
            logger.warning("Enabling read-only mode due to write circuit failure")
            
        # If write circuit recovers, disable read-only mode
        elif 'write' in circuit_name and new_state == CircuitState.CLOSED:
            self.read_only_mode = False
            logger.info("Disabling read-only mode, write circuit recovered")
            
        # Send alerts for critical state changes
        if new_state == CircuitState.OPEN:
            self._send_alert(f"Circuit {circuit_name} is now OPEN - backend failing")
        elif old_state == CircuitState.OPEN and new_state == CircuitState.CLOSED:
            self._send_alert(f"Circuit {circuit_name} recovered - backend operational")
    
    def _send_alert(self, message: str):
        """Send alert to monitoring system"""
        # Integration with alerting system
        logger.critical(f"ALERT: {message}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all circuits"""
        return {
            'backend': self.backend.name,
            'read_only_mode': self.read_only_mode,
            'circuits': {
                'read': self.read_circuit.get_metrics(),
                'write': self.write_circuit.get_metrics(),
                'query': self.query_circuit.get_metrics()
            },
            'cache_size': len(self.cache),
            'cache_memory_mb': sum(
                df.memory_usage(deep=True).sum() / 1024 / 1024 
                for df in self.cache.values() 
                if isinstance(df, pd.DataFrame)
            )
        }

class ReadOnlyModeError(Exception):
    """Raised when system is in read-only mode"""
    pass
```

### 3. Advanced Circuit Breaker Features

```python
# src/mdm/resilience/advanced_circuit_breaker.py
from mdm.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
import asyncio
from typing import Optional, Callable, Any
import aiohttp
from dataclasses import dataclass

@dataclass
class BulkheadConfig:
    """Configuration for bulkhead pattern"""
    max_concurrent_calls: int = 10
    max_wait_duration: float = 1.0  # seconds

class BulkheadCircuitBreaker(CircuitBreaker):
    """Circuit breaker with bulkhead isolation"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig, bulkhead_config: BulkheadConfig):
        super().__init__(name, config)
        self.bulkhead_config = bulkhead_config
        self._semaphore = asyncio.Semaphore(bulkhead_config.max_concurrent_calls)
        self._active_calls = 0
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Async execution with bulkhead protection"""
        # Check circuit state
        if not self._can_execute():
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        # Try to acquire semaphore
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.bulkhead_config.max_wait_duration
            )
        except asyncio.TimeoutError:
            raise BulkheadRejectedError(
                f"Bulkhead full: {self._active_calls}/{self.bulkhead_config.max_concurrent_calls} calls active"
            )
        
        try:
            self._active_calls += 1
            # Execute function
            result = await self._execute_async(func, args, kwargs)
            return result
        finally:
            self._active_calls -= 1
            self._semaphore.release()
    
    async def _execute_async(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute async function with metrics"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await func(*args, **kwargs)
            duration = asyncio.get_event_loop().time() - start_time
            self._on_success(duration)
            return result
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            if self._should_count_as_failure(e):
                self._on_failure(duration, e)
            raise

class BulkheadRejectedError(Exception):
    """Raised when bulkhead rejects call due to capacity"""
    pass

class AdaptiveCircuitBreaker(CircuitBreaker):
    """Circuit breaker that adapts thresholds based on system load"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        super().__init__(name, config)
        self._baseline_latency: Optional[float] = None
        self._load_factor = 1.0
    
    def _on_success(self, duration: float):
        """Update baseline and adapt thresholds"""
        super()._on_success(duration)
        
        # Update baseline latency
        if self._baseline_latency is None:
            self._baseline_latency = duration
        else:
            # Exponential moving average
            self._baseline_latency = 0.9 * self._baseline_latency + 0.1 * duration
        
        # Adapt slow call threshold based on baseline
        self.config.slow_call_duration = max(
            self._baseline_latency * 3,  # 3x baseline
            1.0  # Minimum 1 second
        )
    
    def _should_open_circuit(self) -> bool:
        """Adaptive circuit opening based on load"""
        # Get system load metrics
        load_metrics = self._get_system_load()
        
        # Adjust thresholds based on load
        if load_metrics['cpu_percent'] > 80:
            # Under high load, be more aggressive
            adjusted_threshold = self.config.failure_threshold * 0.7
        else:
            adjusted_threshold = self.config.failure_threshold
        
        # Check with adjusted threshold
        if self._failure_count >= adjusted_threshold:
            return True
            
        return super()._should_open_circuit()
    
    def _get_system_load(self) -> Dict[str, float]:
        """Get current system load metrics"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent
            }
        except ImportError:
            return {'cpu_percent': 0, 'memory_percent': 0}
```

### 4. Fallback Strategies

```python
# src/mdm/resilience/fallback_strategies.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import pickle
import redis

class FallbackStrategy(ABC):
    """Base class for fallback strategies"""
    
    @abstractmethod
    def can_handle(self, operation: str, **kwargs) -> bool:
        """Check if this strategy can handle the operation"""
        pass
    
    @abstractmethod
    def execute(self, operation: str, **kwargs) -> Any:
        """Execute the fallback strategy"""
        pass

class CacheFallbackStrategy(FallbackStrategy):
    """Fallback to cached data"""
    
    def __init__(self, cache_ttl: int = 3600):
        self.cache: Dict[str, Any] = {}
        self.cache_times: Dict[str, datetime] = {}
        self.cache_ttl = cache_ttl
    
    def can_handle(self, operation: str, **kwargs) -> bool:
        return operation in ['read', 'query']
    
    def execute(self, operation: str, **kwargs) -> Any:
        cache_key = self._generate_cache_key(operation, kwargs)
        
        # Check if cached data exists and is valid
        if cache_key in self.cache:
            if self._is_cache_valid(cache_key):
                logger.info(f"Returning cached data for {cache_key}")
                return self.cache[cache_key]
        
        raise FallbackNotAvailableError("No valid cached data available")
    
    def update_cache(self, operation: str, kwargs: dict, result: Any):
        """Update cache with fresh data"""
        cache_key = self._generate_cache_key(operation, kwargs)
        self.cache[cache_key] = result
        self.cache_times[cache_key] = datetime.now()
    
    def _generate_cache_key(self, operation: str, kwargs: dict) -> str:
        """Generate cache key from operation and parameters"""
        return f"{operation}:{hash(str(sorted(kwargs.items())))}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_times:
            return False
        
        age = datetime.now() - self.cache_times[cache_key]
        return age.total_seconds() < self.cache_ttl

class RedisFallbackStrategy(FallbackStrategy):
    """Fallback to Redis cache"""
    
    def __init__(self, redis_client: redis.Redis, ttl: int = 3600):
        self.redis_client = redis_client
        self.ttl = ttl
    
    def can_handle(self, operation: str, **kwargs) -> bool:
        return operation in ['read', 'query']
    
    def execute(self, operation: str, **kwargs) -> Any:
        cache_key = f"mdm:fallback:{operation}:{hash(str(sorted(kwargs.items())))}"
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.error(f"Redis fallback failed: {e}")
        
        raise FallbackNotAvailableError("No cached data in Redis")
    
    def update_cache(self, operation: str, kwargs: dict, result: Any):
        """Update Redis cache"""
        cache_key = f"mdm:fallback:{operation}:{hash(str(sorted(kwargs.items())))}"
        
        try:
            self.redis_client.setex(
                cache_key,
                self.ttl,
                pickle.dumps(result)
            )
        except Exception as e:
            logger.error(f"Failed to update Redis cache: {e}")

class DegradedModeFallback(FallbackStrategy):
    """Operate in degraded mode with limited functionality"""
    
    def __init__(self):
        self.degraded_operations = {
            'read': self._degraded_read,
            'write': self._degraded_write,
            'query': self._degraded_query
        }
    
    def can_handle(self, operation: str, **kwargs) -> bool:
        return operation in self.degraded_operations
    
    def execute(self, operation: str, **kwargs) -> Any:
        if operation in self.degraded_operations:
            return self.degraded_operations[operation](**kwargs)
        
        raise FallbackNotAvailableError(f"No degraded mode for {operation}")
    
    def _degraded_read(self, dataset_name: str, **kwargs) -> pd.DataFrame:
        """Return sample data or schema only"""
        logger.warning(f"Degraded read for {dataset_name} - returning empty DataFrame with schema")
        
        # Try to return at least the schema
        schema = self._get_cached_schema(dataset_name)
        if schema:
            return pd.DataFrame(columns=schema['columns'])
        
        return pd.DataFrame()
    
    def _degraded_write(self, dataset_name: str, data: pd.DataFrame, **kwargs):
        """Queue writes for later processing"""
        logger.warning(f"Degraded write for {dataset_name} - queueing {len(data)} rows")
        
        # In production, this would write to a persistent queue
        # For now, just acknowledge the write
        return {'status': 'queued', 'rows': len(data)}
    
    def _degraded_query(self, query: str, **kwargs) -> Any:
        """Return limited query results"""
        logger.warning("Degraded query mode - limited functionality")
        
        # Could return cached aggregates or pre-computed results
        return {'status': 'degraded', 'message': 'Query service temporarily unavailable'}
    
    def _get_cached_schema(self, dataset_name: str) -> Optional[Dict]:
        """Get cached schema information"""
        # In production, this would read from a schema cache
        return None

class FallbackNotAvailableError(Exception):
    """Raised when no fallback is available"""
    pass

class CompositeFallbackStrategy(FallbackStrategy):
    """Combine multiple fallback strategies"""
    
    def __init__(self, strategies: List[FallbackStrategy]):
        self.strategies = strategies
    
    def can_handle(self, operation: str, **kwargs) -> bool:
        return any(s.can_handle(operation, **kwargs) for s in self.strategies)
    
    def execute(self, operation: str, **kwargs) -> Any:
        """Try each strategy in order until one succeeds"""
        errors = []
        
        for strategy in self.strategies:
            if strategy.can_handle(operation, **kwargs):
                try:
                    return strategy.execute(operation, **kwargs)
                except Exception as e:
                    errors.append((strategy.__class__.__name__, str(e)))
                    continue
        
        raise FallbackNotAvailableError(
            f"All fallback strategies failed: {errors}"
        )
```

### 5. Monitoring and Metrics

```python
# src/mdm/resilience/circuit_breaker_metrics.py
from mdm.monitoring import SimpleMonitor, MetricType
from typing import Dict, List
import time
import json

class CircuitBreakerMetrics:
    """Simple monitoring integration for circuit breakers"""
    
    def __init__(self):
        self.monitor = SimpleMonitor()
        self.state_history: Dict[str, List[tuple]] = {}
        
    def record_call(self, circuit_name: str, duration: float, success: bool):
        """Record a call metric"""
        self.monitor.record_metric(
            MetricType.STORAGE_OPERATION,
            f"circuit_call_{circuit_name}",
            duration_ms=duration * 1000,
            success=success,
            metadata={
                "circuit_name": circuit_name,
                "operation_type": "circuit_call"
            }
        )
    
    def record_rejection(self, circuit_name: str):
        """Record a rejected call"""
        self.monitor.record_metric(
            MetricType.STORAGE_OPERATION,
            f"circuit_rejection_{circuit_name}",
            success=False,
            error_message="Call rejected by circuit breaker",
            metadata={
                "circuit_name": circuit_name,
                "operation_type": "circuit_rejection"
            }
        )
    
    def update_state(self, circuit_name: str, state: CircuitState):
        """Record state change"""
        # Track state in history
        if circuit_name not in self.state_history:
            self.state_history[circuit_name] = []
        
        self.state_history[circuit_name].append((time.time(), state.value))
        
        # Record state change as metric
        self.monitor.record_metric(
            MetricType.STORAGE_OPERATION,
            f"circuit_state_change_{circuit_name}",
            success=True,
            metadata={
                "circuit_name": circuit_name,
                "new_state": state.value,
                "operation_type": "state_change"
            }
    
    def record_transition(self, circuit_name: str, from_state: CircuitState, to_state: CircuitState):
        """Record state transition"""
        self.monitor.record_metric(
            MetricType.STORAGE_OPERATION,
            f"circuit_transition_{circuit_name}",
            success=True,
            metadata={
                "circuit_name": circuit_name,
                "from_state": from_state.value,
                "to_state": to_state.value,
                "operation_type": "state_transition"
            }
        )
    
    def record_fallback(self, circuit_name: str, strategy: str, success: bool):
        """Record fallback execution"""
        self.monitor.record_metric(
            MetricType.STORAGE_OPERATION,
            f"circuit_fallback_{circuit_name}",
            success=success,
            metadata={
                "circuit_name": circuit_name,
                "strategy": strategy,
                "operation_type": "fallback_execution"
            }
        )
    
    def register_circuit(self, circuit_name: str, config: CircuitBreakerConfig):
        """Register circuit breaker configuration"""
        # Log circuit configuration for reference
        logger.info(
            f"Circuit breaker registered: {circuit_name}",
            extra={
                "failure_threshold": config.failure_threshold,
                "timeout": config.timeout,
                "window_size": config.window_size
            }
        )

# Global metrics instance
circuit_breaker_metrics = CircuitBreakerMetrics()
```

### 6. CLI Integration

```python
# src/mdm/cli/resilience.py
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from typing import Optional
import time

app = typer.Typer(help="Circuit breaker management commands")
console = Console()

@app.command()
def status(
    backend: Optional[str] = typer.Option(None, help="Specific backend to check"),
    watch: bool = typer.Option(False, help="Watch status in real-time")
):
    """Check circuit breaker status"""
    from mdm.storage import get_all_backends
    
    if watch:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                table = _create_status_table(backend)
                live.update(table)
                time.sleep(1)
    else:
        table = _create_status_table(backend)
        console.print(table)

def _create_status_table(backend_filter: Optional[str]) -> Table:
    """Create status table"""
    table = Table(title="Circuit Breaker Status")
    table.add_column("Backend", style="cyan")
    table.add_column("Circuit", style="magenta")
    table.add_column("State", style="bold")
    table.add_column("Failures", style="red")
    table.add_column("Success Rate", style="green")
    table.add_column("Avg Duration", style="yellow")
    
    backends = get_all_backends()
    if backend_filter:
        backends = [b for b in backends if b.name == backend_filter]
    
    for backend in backends:
        if hasattr(backend, 'get_health_status'):
            health = backend.get_health_status()
            
            for circuit_name, metrics in health['circuits'].items():
                state_color = {
                    'closed': 'green',
                    'open': 'red',
                    'half_open': 'yellow'
                }.get(metrics['state'], 'white')
                
                success_rate = 1 - metrics['failure_rate']
                
                table.add_row(
                    backend.name,
                    circuit_name,
                    f"[{state_color}]{metrics['state']}[/{state_color}]",
                    str(metrics.get('failures', 0)),
                    f"{success_rate:.1%}",
                    f"{metrics['avg_duration']:.3f}s"
                )
    
    return table

@app.command()
def reset(
    backend: str = typer.Argument(..., help="Backend name"),
    circuit: str = typer.Argument(..., help="Circuit type (read/write/query)"),
    force: bool = typer.Option(False, help="Force reset without confirmation")
):
    """Reset a circuit breaker"""
    if not force:
        confirm = typer.confirm(
            f"Reset {circuit} circuit for {backend}? This may cause errors if backend is still failing."
        )
        if not confirm:
            raise typer.Abort()
    
    from mdm.storage import get_backend
    backend_instance = get_backend(backend)
    
    if hasattr(backend_instance, f'{circuit}_circuit'):
        circuit_breaker = getattr(backend_instance, f'{circuit}_circuit')
        circuit_breaker.reset()
        console.print(f"[green]Reset {circuit} circuit for {backend}[/green]")
    else:
        console.print(f"[red]Circuit {circuit} not found for {backend}[/red]")

@app.command()
def test(
    backend: str = typer.Argument(..., help="Backend to test"),
    operation: str = typer.Option("read", help="Operation to test (read/write/query)"),
    simulate_failure: bool = typer.Option(False, help="Simulate failures")
):
    """Test circuit breaker behavior"""
    from mdm.storage import get_backend
    import random
    
    backend_instance = get_backend(backend)
    
    console.print(f"Testing {operation} circuit for {backend}...")
    
    for i in range(20):
        try:
            if simulate_failure and random.random() < 0.3:
                # Simulate 30% failure rate
                raise ConnectionError("Simulated failure")
            
            # Attempt operation
            if operation == "read":
                backend_instance.read_data("test_dataset")
            elif operation == "write":
                backend_instance.write_data("test_dataset", pd.DataFrame({'test': [1]}))
            elif operation == "query":
                backend_instance.execute_query("SELECT 1")
                
            console.print(f"[green]✓[/green] Call {i+1}: Success")
            
        except CircuitBreakerOpenError as e:
            console.print(f"[red]✗[/red] Call {i+1}: Circuit OPEN - {e}")
        except Exception as e:
            console.print(f"[yellow]![/yellow] Call {i+1}: Failed - {type(e).__name__}")
        
        time.sleep(0.5)
    
    # Show final status
    if hasattr(backend_instance, 'get_health_status'):
        health = backend_instance.get_health_status()
        console.print("\nFinal Circuit Status:")
        console.print(health['circuits'][operation])
```

## Configuration

### MDM Configuration Integration

```yaml
# ~/.mdm/mdm.yaml
resilience:
  circuit_breaker:
    enabled: true
    default_config:
      failure_threshold: 5
      success_threshold: 3
      timeout: 60.0
      window_size: 10
      failure_rate_threshold: 0.5
      slow_call_duration: 1.0
    
    backend_configs:
      postgresql:
        failure_threshold: 3
        timeout: 30.0
        error_types:
          - "psycopg2.OperationalError"
          - "psycopg2.InterfaceError"
      
      duckdb:
        failure_threshold: 5
        slow_call_duration: 5.0
      
      sqlite:
        failure_threshold: 10
        timeout: 10.0
  
  fallback:
    strategies:
      - type: "cache"
        ttl: 3600
      - type: "redis"
        ttl: 7200
        host: "localhost"
        port: 6379
      - type: "degraded"
    
  monitoring:
    export_metrics: true
    alert_on_open: true
    alert_channels:
      - "slack"
      - "email"
```

## Testing

### Unit Tests

```python
# tests/test_circuit_breaker.py
import pytest
from mdm.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError
import time

class TestCircuitBreaker:
    def test_circuit_opens_after_failures(self):
        """Test that circuit opens after threshold failures"""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        # Simulate failures
        for i in range(3):
            with pytest.raises(Exception):
                cb.call(lambda: 1/0)  # Division by zero
        
        # Circuit should be open
        assert cb.state == CircuitState.OPEN
        
        # Next call should be rejected
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: "success")
    
    def test_circuit_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout"""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        cb = CircuitBreaker("test", config)
        
        # Open the circuit
        with pytest.raises(Exception):
            cb.call(lambda: 1/0)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Check state (accessing state property triggers transition)
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_circuit_closes_after_success(self):
        """Test circuit closes after successful calls in half-open"""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout=0.1
        )
        cb = CircuitBreaker("test", config)
        
        # Open the circuit
        with pytest.raises(Exception):
            cb.call(lambda: 1/0)
        
        # Wait for half-open
        time.sleep(0.2)
        
        # Successful calls
        for i in range(2):
            result = cb.call(lambda: "success")
            assert result == "success"
        
        assert cb.state == CircuitState.CLOSED
```

## Best Practices

1. **Configure Appropriately**: Set thresholds based on your SLA and backend characteristics
2. **Monitor Actively**: Use metrics and alerts to detect issues early
3. **Test Failure Scenarios**: Regularly test circuit breaker behavior
4. **Implement Fallbacks**: Always have a fallback strategy for critical operations
5. **Document Dependencies**: Make circuit breaker dependencies clear in documentation
6. **Gradual Rollout**: Enable circuit breakers gradually, starting with non-critical operations

## Troubleshooting

### Circuit Stuck Open
- Check if backend has recovered: `mdm storage test <backend>`
- Manually reset if needed: `mdm resilience reset <backend> <circuit>`
- Review timeout configuration

### High False Positive Rate
- Increase failure threshold
- Adjust error types to exclude transient errors
- Increase window size for more stable metrics

### Performance Impact
- Check circuit breaker overhead with metrics
- Consider using async circuit breakers
- Optimize fallback strategies

## Future Enhancements

1. **Predictive Circuit Breaking**: Use ML to predict failures before they happen
2. **Adaptive Timeouts**: Automatically adjust timeouts based on recovery patterns  
3. **Distributed Circuit Breakers**: Coordinate circuit state across multiple instances
4. **Smart Fallbacks**: Choose fallback strategy based on failure type
5. **Integration with Service Mesh**: Integrate with Istio/Linkerd for microservices