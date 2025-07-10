#!/usr/bin/env python3
"""Simple test for storage backend components."""
import sqlite3
import time
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
import threading
import queue


class SimpleConnectionPool:
    """Simplified connection pool for testing"""
    
    def __init__(self, max_size=5):
        self.max_size = max_size
        self._pool = queue.Queue(maxsize=max_size)
        self._created_count = 0
        self._lock = threading.RLock()
        
        # Pre-create some connections
        for _ in range(2):
            conn = self._create_connection()
            self._pool.put(conn)
    
    def _create_connection(self):
        """Create a new SQLite connection"""
        with self._lock:
            if self._created_count >= self.max_size:
                raise RuntimeError(f"Pool exhausted (max={self.max_size})")
            
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._created_count += 1
            return conn
    
    def get_connection(self):
        """Get a connection from pool"""
        try:
            # Try to get existing connection
            conn = self._pool.get(timeout=0.1)
            return conn
        except queue.Empty:
            # Create new one if pool not full
            return self._create_connection()
    
    def return_connection(self, conn):
        """Return connection to pool"""
        try:
            self._pool.put(conn, timeout=0.1)
        except queue.Full:
            # Pool is full, close the connection
            conn.close()
            with self._lock:
                self._created_count -= 1
    
    def get_stats(self):
        """Get pool statistics"""
        with self._lock:
            return {
                "total": self._created_count,
                "available": self._pool.qsize(),
                "in_use": self._created_count - self._pool.qsize()
            }


def test_basic_pool():
    """Test basic pool operations"""
    print("=== Testing Basic Pool Operations ===")
    
    pool = SimpleConnectionPool(max_size=3)
    
    # Get and use connection
    conn = pool.get_connection()
    cursor = conn.execute("SELECT 1")
    result = cursor.fetchone()
    assert result == (1,)
    pool.return_connection(conn)
    
    print("✅ Basic operations work!")
    
    # Check stats
    stats = pool.get_stats()
    print(f"✅ Pool stats: {stats}")
    
    return True


def test_concurrent_access():
    """Test concurrent pool access"""
    print("\n=== Testing Concurrent Access ===")
    
    pool = SimpleConnectionPool(max_size=5)
    results = []
    
    def worker(n):
        conn = pool.get_connection()
        try:
            cursor = conn.execute("SELECT ?", (n,))
            result = cursor.fetchone()[0]
            time.sleep(0.01)  # Simulate work
            return result
        finally:
            pool.return_connection(conn)
    
    # Run concurrent workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(20)]
        results = [f.result() for f in futures]
    
    assert len(results) == 20
    assert results == list(range(20))
    
    stats = pool.get_stats()
    print(f"✅ Concurrent access works! Final stats: {stats}")
    
    return True


def test_pool_exhaustion():
    """Test pool exhaustion"""
    print("\n=== Testing Pool Exhaustion ===")
    
    pool = SimpleConnectionPool(max_size=2)
    connections = []
    exhausted = False
    
    try:
        # Get all available connections
        for i in range(3):
            conn = pool.get_connection()
            connections.append(conn)
    except RuntimeError as e:
        if "exhausted" in str(e):
            exhausted = True
            print("✅ Pool exhaustion handled correctly!")
    
    # Return connections
    for conn in connections:
        pool.return_connection(conn)
    
    return exhausted


def test_performance_improvement():
    """Test performance with pooling"""
    print("\n=== Testing Performance ===")
    
    # Without pooling
    start = time.perf_counter()
    for i in range(100):
        conn = sqlite3.connect(":memory:")
        conn.execute("SELECT 1")
        conn.close()
    no_pool_time = time.perf_counter() - start
    
    # With pooling
    pool = SimpleConnectionPool(max_size=5)
    start = time.perf_counter()
    for i in range(100):
        conn = pool.get_connection()
        conn.execute("SELECT 1")
        pool.return_connection(conn)
    pool_time = time.perf_counter() - start
    
    speedup = no_pool_time / pool_time
    print(f"✅ No pooling: {no_pool_time:.3f}s")
    print(f"✅ With pooling: {pool_time:.3f}s")
    print(f"✅ Speedup: {speedup:.2f}x")
    
    return speedup > 1.5  # Expect at least 1.5x speedup


def test_stateless_pattern():
    """Test stateless backend pattern"""
    print("\n=== Testing Stateless Pattern ===")
    
    class StatelessBackend:
        def __init__(self):
            self.pool = SimpleConnectionPool(max_size=5)
        
        def execute_query(self, query, params=None):
            conn = self.pool.get_connection()
            try:
                cursor = conn.execute(query, params or ())
                return cursor.fetchall()
            finally:
                self.pool.return_connection(conn)
    
    backend = StatelessBackend()
    
    # Concurrent operations
    def operation(n):
        result = backend.execute_query("SELECT ?", (n,))
        return result[0][0]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(operation, range(50)))
    
    assert results == list(range(50))
    print("✅ Stateless pattern works correctly!")
    
    return True


def main():
    """Run all tests"""
    print("🔍 Testing Storage Backend Concepts\n")
    
    tests = [
        test_basic_pool,
        test_concurrent_access,
        test_pool_exhaustion,
        test_performance_improvement,
        test_stateless_pattern
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("\n📋 Storage Migration Concepts Validated:")
        print("- Connection pooling reduces overhead")
        print("- Thread-safe concurrent access")
        print("- Proper exhaustion handling")
        print("- Stateless pattern enables scaling")
        print("\n🎉 Core concepts are working correctly!")
    else:
        print(f"❌ {total - passed} out of {total} tests failed!")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())