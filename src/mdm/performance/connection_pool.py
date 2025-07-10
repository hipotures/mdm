"""Connection pooling for database operations."""
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from queue import Queue, Empty, Full
import weakref

from sqlalchemy import create_engine, event, pool
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool, NullPool, StaticPool

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PoolConfig:
    """Configuration for connection pooling."""
    pool_size: int = 5
    max_overflow: int = 10
    timeout: float = 30.0
    recycle: int = 3600  # Recycle connections after 1 hour
    pre_ping: bool = True  # Test connections before use
    echo_pool: bool = False
    poolclass: Optional[type] = None  # Custom pool class
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.max_overflow < 0:
            raise ValueError("max_overflow must be non-negative")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass
class ConnectionStats:
    """Statistics for a connection."""
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    total_time: float = 0.0
    errors: int = 0
    
    def record_use(self, duration: float) -> None:
        """Record connection use."""
        self.last_used = time.time()
        self.use_count += 1
        self.total_time += duration
    
    def record_error(self) -> None:
        """Record connection error."""
        self.errors += 1
    
    @property
    def average_time(self) -> float:
        """Get average use time."""
        return self.total_time / max(1, self.use_count)


class ConnectionPool:
    """Manages database connection pooling."""
    
    def __init__(self, 
                 connection_string: str,
                 config: Optional[PoolConfig] = None):
        """Initialize connection pool.
        
        Args:
            connection_string: Database connection string
            config: Pool configuration
        """
        self.connection_string = connection_string
        self.config = config or PoolConfig()
        self.config.validate()
        
        self._engine: Optional[Engine] = None
        self._lock = threading.RLock()
        self._stats: Dict[int, ConnectionStats] = {}
        self._pool_stats = {
            'connections_created': 0,
            'connections_recycled': 0,
            'connection_errors': 0,
            'total_checkouts': 0,
            'total_checkins': 0,
            'overflow_created': 0
        }
        
        # Track active connections
        self._active_connections = weakref.WeakSet()
        
        self._initialize_engine()
    
    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine with pooling."""
        # Determine pool class
        poolclass = self.config.poolclass
        if poolclass is None:
            # Auto-select based on database
            if 'sqlite' in self.connection_string:
                if ':memory:' in self.connection_string:
                    poolclass = StaticPool
                else:
                    poolclass = NullPool  # SQLite doesn't benefit from pooling
            else:
                poolclass = QueuePool
        
        # Create engine with pool configuration
        engine_kwargs = {
            'poolclass': poolclass,
            'echo_pool': self.config.echo_pool
        }
        
        # Add pool-specific configuration
        if poolclass == QueuePool:
            engine_kwargs.update({
                'pool_size': self.config.pool_size,
                'max_overflow': self.config.max_overflow,
                'pool_timeout': self.config.timeout,
                'pool_recycle': self.config.recycle,
                'pool_pre_ping': self.config.pre_ping
            })
        elif poolclass == StaticPool:
            engine_kwargs['connect_args'] = {'check_same_thread': False}
        
        self._engine = create_engine(self.connection_string, **engine_kwargs)
        
        # Set up event listeners
        self._setup_event_listeners()
        
        logger.info(f"Initialized connection pool with {poolclass.__name__}")
    
    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners."""
        @event.listens_for(self._engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            """Handle new connection."""
            with self._lock:
                conn_id = id(dbapi_conn)
                self._stats[conn_id] = ConnectionStats()
                self._pool_stats['connections_created'] += 1
                logger.debug(f"New connection created: {conn_id}")
        
        @event.listens_for(self._engine, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            """Handle connection checkout."""
            with self._lock:
                self._pool_stats['total_checkouts'] += 1
                self._active_connections.add(dbapi_conn)
        
        @event.listens_for(self._engine, "checkin")
        def on_checkin(dbapi_conn, connection_record):
            """Handle connection checkin."""
            with self._lock:
                self._pool_stats['total_checkins'] += 1
                self._active_connections.discard(dbapi_conn)
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool.
        
        Yields:
            Database connection
        """
        start_time = time.time()
        conn = None
        conn_id = None
        
        try:
            conn = self._engine.connect()
            conn_id = id(conn.connection.dbapi_connection)
            
            yield conn
            
            # Record successful use
            duration = time.time() - start_time
            with self._lock:
                if conn_id in self._stats:
                    self._stats[conn_id].record_use(duration)
                    
        except Exception as e:
            # Record error
            with self._lock:
                if conn_id and conn_id in self._stats:
                    self._stats[conn_id].record_error()
                self._pool_stats['connection_errors'] += 1
            
            logger.error(f"Connection error: {e}")
            raise
            
        finally:
            if conn:
                conn.close()
    
    def execute(self, query: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query using a pooled connection.
        
        Args:
            query: SQL query to execute
            params: Optional query parameters
            
        Returns:
            Query result
        """
        with self.get_connection() as conn:
            if params:
                result = conn.execute(query, params)
            else:
                result = conn.execute(query)
            return result
    
    def execute_many(self, queries: List[Tuple[Any, Optional[Dict[str, Any]]]]) -> List[Any]:
        """Execute multiple queries in a single connection.
        
        Args:
            queries: List of (query, params) tuples
            
        Returns:
            List of results
        """
        results = []
        
        with self.get_connection() as conn:
            for query, params in queries:
                if params:
                    result = conn.execute(query, params)
                else:
                    result = conn.execute(query)
                results.append(result)
        
        return results
    
    def transaction(self, func: Callable[[Any], Any]) -> Any:
        """Execute function within a transaction.
        
        Args:
            func: Function to execute with connection
            
        Returns:
            Function result
        """
        with self.get_connection() as conn:
            trans = conn.begin()
            try:
                result = func(conn)
                trans.commit()
                return result
            except Exception:
                trans.rollback()
                raise
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status."""
        status = {
            'size': 0,
            'checked_out': 0,
            'overflow': 0,
            'total': 0
        }
        
        if hasattr(self._engine.pool, 'size'):
            status['size'] = self._engine.pool.size()
        if hasattr(self._engine.pool, 'checked_out'):
            status['checked_out'] = self._engine.pool.checked_out()
        if hasattr(self._engine.pool, 'overflow'):
            status['overflow'] = self._engine.pool.overflow()
        if hasattr(self._engine.pool, 'total'):
            status['total'] = self._engine.pool.total()
        
        status['active_connections'] = len(self._active_connections)
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            # Calculate connection statistics
            total_connections = len(self._stats)
            avg_use_time = 0.0
            total_uses = 0
            total_errors = 0
            
            for stats in self._stats.values():
                total_uses += stats.use_count
                total_errors += stats.errors
                if stats.use_count > 0:
                    avg_use_time += stats.average_time
            
            if total_connections > 0:
                avg_use_time /= total_connections
            
            return {
                **self._pool_stats,
                'pool_status': self.get_pool_status(),
                'total_connections': total_connections,
                'average_use_time': avg_use_time,
                'total_uses': total_uses,
                'total_errors': total_errors,
                'error_rate': total_errors / max(1, total_uses)
            }
    
    def reset_pool(self) -> None:
        """Reset the connection pool."""
        logger.info("Resetting connection pool")
        
        with self._lock:
            # Dispose of current engine
            if self._engine:
                self._engine.dispose()
            
            # Clear statistics
            self._stats.clear()
            self._active_connections.clear()
            
            # Reinitialize
            self._initialize_engine()
    
    def close(self) -> None:
        """Close the connection pool."""
        logger.info("Closing connection pool")
        
        with self._lock:
            if self._engine:
                self._engine.dispose()
                self._engine = None


# Specialized connection pools

class ReadWritePool:
    """Separate pools for read and write operations."""
    
    def __init__(self,
                 write_connection_string: str,
                 read_connection_strings: List[str],
                 config: Optional[PoolConfig] = None):
        """Initialize read/write pool.
        
        Args:
            write_connection_string: Connection string for writes
            read_connection_strings: List of read replica connection strings
            config: Pool configuration
        """
        self.write_pool = ConnectionPool(write_connection_string, config)
        self.read_pools = [
            ConnectionPool(conn_str, config) 
            for conn_str in read_connection_strings
        ]
        self._read_index = 0
        self._lock = threading.Lock()
    
    @contextmanager
    def get_write_connection(self):
        """Get connection for write operations."""
        with self.write_pool.get_connection() as conn:
            yield conn
    
    @contextmanager
    def get_read_connection(self):
        """Get connection for read operations (load balanced)."""
        with self._lock:
            # Round-robin selection
            pool = self.read_pools[self._read_index]
            self._read_index = (self._read_index + 1) % len(self.read_pools)
        
        with pool.get_connection() as conn:
            yield conn
    
    def close(self) -> None:
        """Close all pools."""
        self.write_pool.close()
        for pool in self.read_pools:
            pool.close()


class PoolManager:
    """Manages multiple connection pools."""
    
    def __init__(self, default_config: Optional[PoolConfig] = None):
        """Initialize pool manager.
        
        Args:
            default_config: Default configuration for new pools
        """
        self.default_config = default_config or PoolConfig()
        self._pools: Dict[str, ConnectionPool] = {}
        self._lock = threading.RLock()
    
    def get_pool(self, 
                 name: str,
                 connection_string: Optional[str] = None,
                 config: Optional[PoolConfig] = None) -> ConnectionPool:
        """Get or create a named pool.
        
        Args:
            name: Pool name
            connection_string: Connection string (required for new pools)
            config: Pool configuration
            
        Returns:
            Connection pool
        """
        with self._lock:
            if name not in self._pools:
                if not connection_string:
                    raise ValueError(f"Connection string required for new pool '{name}'")
                
                pool_config = config or self.default_config
                self._pools[name] = ConnectionPool(connection_string, pool_config)
                logger.info(f"Created new pool: {name}")
            
            return self._pools[name]
    
    def close_pool(self, name: str) -> None:
        """Close a specific pool."""
        with self._lock:
            if name in self._pools:
                self._pools[name].close()
                del self._pools[name]
                logger.info(f"Closed pool: {name}")
    
    def close_all(self) -> None:
        """Close all pools."""
        with self._lock:
            for name, pool in list(self._pools.items()):
                pool.close()
            self._pools.clear()
            logger.info("Closed all pools")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all pools."""
        with self._lock:
            return {
                name: pool.get_statistics()
                for name, pool in self._pools.items()
            }