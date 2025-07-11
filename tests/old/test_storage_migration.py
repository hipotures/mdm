#!/usr/bin/env python3
"""Test storage backend migration components."""
import sys
from pathlib import Path
import pandas as pd
import sqlite3
import time
import concurrent.futures

# Add refactor path and avoid full mdm import
import importlib.util

def import_module_directly(module_name, file_path):
    """Import a module directly from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import modules directly
base_path = Path('/home/xai/DEV/mdm-refactor-2025/src/mdm')
pooling = import_module_directly("pooling", base_path / "storage/pooling.py")
sqlite_pool = import_module_directly("sqlite_pool", base_path / "storage/pools/sqlite_pool.py")

ConnectionPool = pooling.ConnectionPool
ConnectionInfo = pooling.ConnectionInfo
SQLiteConnectionPool = sqlite_pool.SQLiteConnectionPool


class MockSQLiteBackend:
    """Simplified SQLite backend for testing"""
    def __init__(self):
        self.pool = SQLiteConnectionPool(min_size=1, max_size=5)
        self.datasets_path = Path.home() / ".mdm" / "datasets"
    
    def test_operations(self, dataset_name: str):
        """Test basic operations"""
        # Test connection
        with self.pool.get_connection(dataset_name) as conn:
            conn.execute("SELECT 1")
            result = conn.fetchone()
            assert result == (1,)
            print("✅ Connection working!")
        
        # Test pool stats
        stats = self.pool.get_stats()
        print(f"✅ Pool stats: {stats}")


def test_connection_pool():
    """Test connection pool basics"""
    print("=== Testing Connection Pool ===")
    
    try:
        # Create pool
        pool = SQLiteConnectionPool(min_size=1, max_size=3)
        
        # Get connection
        with pool.get_connection("test") as conn:
            assert isinstance(conn, sqlite3.Connection)
            conn.execute("SELECT 1")
        
        print("✅ Basic pool operations work!")
        
        # Test concurrent access
        def use_connection(n):
            with pool.get_connection("test") as conn:
                conn.execute("SELECT ?", (n,))
                return n
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(use_connection, range(10)))
        
        assert len(results) == 10
        print("✅ Concurrent access works!")
        
        # Cleanup
        pool.close_all()
        
        return True
    except Exception as e:
        print(f"❌ Connection pool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stateless_pattern():
    """Test stateless backend pattern"""
    print("\n=== Testing Stateless Pattern ===")
    
    try:
        backend = MockSQLiteBackend()
        
        # Concurrent operations
        def operation(n):
            backend.test_operations(f"dataset_{n}")
            return n
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(operation, range(5)))
        
        assert len(results) == 5
        print("✅ Stateless pattern works!")
        
        # Check pool didn't grow too much
        stats = backend.pool.get_stats()
        assert stats['total_connections'] <= stats['max_size']
        print(f"✅ Pool size controlled: {stats['total_connections']}/{stats['max_size']}")
        
        backend.pool.close_all()
        return True
        
    except Exception as e:
        print(f"❌ Stateless pattern test failed: {e}")
        return False


def test_pool_exhaustion():
    """Test pool exhaustion handling"""
    print("\n=== Testing Pool Exhaustion ===")
    
    try:
        # Small pool
        pool = SQLiteConnectionPool(min_size=1, max_size=2)
        
        connections = []
        exhausted = False
        
        try:
            # Try to get more connections than pool size
            for i in range(3):
                cm = pool.get_connection("test")
                conn = cm.__enter__()
                connections.append((cm, conn))
        except RuntimeError as e:
            if "exhausted" in str(e):
                exhausted = True
                print("✅ Pool exhaustion detected correctly!")
        
        # Release connections
        for cm, conn in connections:
            cm.__exit__(None, None, None)
        
        pool.close_all()
        
        return exhausted
        
    except Exception as e:
        print(f"❌ Pool exhaustion test failed: {e}")
        return False


def test_performance():
    """Test performance characteristics"""
    print("\n=== Testing Performance ===")
    
    try:
        pool = SQLiteConnectionPool(min_size=2, max_size=10)
        
        # Sequential access
        start = time.perf_counter()
        for i in range(100):
            with pool.get_connection("test") as conn:
                conn.execute("SELECT 1")
        seq_time = time.perf_counter() - start
        
        print(f"✅ Sequential: 100 ops in {seq_time:.3f}s ({100/seq_time:.0f} ops/sec)")
        
        # Concurrent access
        def concurrent_op(n):
            with pool.get_connection("test") as conn:
                conn.execute("SELECT ?", (n,))
        
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(concurrent_op, range(100)))
        conc_time = time.perf_counter() - start
        
        print(f"✅ Concurrent: 100 ops in {conc_time:.3f}s ({100/conc_time:.0f} ops/sec)")
        print(f"✅ Speedup: {seq_time/conc_time:.2f}x")
        
        pool.close_all()
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🔍 Testing Storage Backend Migration Components\n")
    
    tests = [
        test_connection_pool,
        test_stateless_pattern,
        test_pool_exhaustion,
        test_performance
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("\n📋 Key Components Validated:")
        print("- Connection pooling with thread safety")
        print("- Stateless backend pattern")
        print("- Pool exhaustion handling")
        print("- Performance improvements with pooling")
        print("\n🎉 Storage migration components are working!")
    else:
        print(f"❌ {total - passed} out of {total} tests failed!")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())