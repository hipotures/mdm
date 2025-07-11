"""
Test suite to ensure backend compatibility.

This test verifies that all storage backends (old and new) implement
the complete API surface identified in the usage analysis.
"""
import pytest
import inspect
from typing import List, Set
import pandas as pd
from sqlalchemy import Engine

from mdm.storage.sqlite import SQLiteBackend
from mdm.storage.duckdb import DuckDBBackend
from mdm.storage.backends.stateless_sqlite import StatelessSQLiteBackend
from mdm.storage.backends.stateless_duckdb import StatelessDuckDBBackend


# Methods identified in API analysis
REQUIRED_METHODS = {
    'get_engine': {'params': ['database_path'], 'return': Engine},
    'create_table_from_dataframe': {'params': ['df', 'table_name', 'engine', 'if_exists'], 'return': None},
    'query': {'params': ['query'], 'return': pd.DataFrame},
    'read_table_to_dataframe': {'params': ['table_name', 'engine', 'limit'], 'return': pd.DataFrame},
    'close_connections': {'params': [], 'return': None},
    'read_table': {'params': ['table_name', 'columns', 'where', 'limit'], 'return': pd.DataFrame},
    'write_table': {'params': ['table_name', 'df', 'if_exists'], 'return': None},
    'get_table_info': {'params': ['table_name', 'engine'], 'return': dict},
    'execute_query': {'params': ['query', 'engine'], 'return': None},
    'get_connection': {'params': [], 'return': None},
    'get_columns': {'params': ['table_name'], 'return': list},
    'analyze_column': {'params': ['table_name', 'column_name'], 'return': dict},
    'database_exists': {'params': ['database_path'], 'return': bool},
    'create_database': {'params': ['database_path'], 'return': None},
}

# Backend classes to test
BACKEND_CLASSES = [
    SQLiteBackend,
    DuckDBBackend,
    StatelessSQLiteBackend,
    StatelessDuckDBBackend,
]


class TestBackendCompatibility:
    """Test that all backends implement required methods."""
    
    @pytest.mark.parametrize("backend_class", BACKEND_CLASSES)
    def test_all_required_methods_exist(self, backend_class):
        """Verify that backend has all required methods."""
        missing_methods = []
        
        for method_name in REQUIRED_METHODS:
            if not hasattr(backend_class, method_name):
                missing_methods.append(method_name)
        
        assert not missing_methods, (
            f"{backend_class.__name__} is missing methods: {missing_methods}\n"
            f"This will cause AttributeError at runtime!"
        )
    
    @pytest.mark.parametrize("backend_class", BACKEND_CLASSES)
    def test_method_signatures_compatible(self, backend_class):
        """Verify method signatures are compatible."""
        incompatible = []
        
        for method_name, expected in REQUIRED_METHODS.items():
            if hasattr(backend_class, method_name):
                method = getattr(backend_class, method_name)
                if callable(method):
                    # Get method signature
                    sig = inspect.signature(method)
                    params = list(sig.parameters.keys())
                    
                    # Remove 'self' from instance methods
                    if 'self' in params:
                        params.remove('self')
                    
                    # Check if expected parameters are present
                    expected_params = expected['params']
                    for param in expected_params:
                        if param not in params and not any(
                            p.kind == inspect.Parameter.VAR_KEYWORD 
                            for p in sig.parameters.values()
                        ):
                            incompatible.append(
                                f"{method_name}: missing parameter '{param}'"
                            )
        
        assert not incompatible, (
            f"{backend_class.__name__} has incompatible signatures:\n" + 
            "\n".join(incompatible)
        )
    
    @pytest.mark.parametrize("backend_class", BACKEND_CLASSES)
    def test_backend_instantiation(self, backend_class):
        """Test that backend can be instantiated."""
        try:
            # Old backends require config
            if backend_class in [SQLiteBackend, DuckDBBackend]:
                backend = backend_class({})
            else:
                # New backends don't require config
                backend = backend_class()
            
            assert backend is not None
            assert hasattr(backend, 'backend_type')
            
        except Exception as e:
            pytest.fail(f"Failed to instantiate {backend_class.__name__}: {e}")
    
    def test_stateless_backends_have_mixin(self):
        """Verify stateless backends inherit from compatibility mixin."""
        from mdm.storage.backends.compatibility_mixin import BackendCompatibilityMixin
        
        assert issubclass(StatelessSQLiteBackend, BackendCompatibilityMixin)
        assert issubclass(StatelessDuckDBBackend, BackendCompatibilityMixin)
    
    def test_method_count_matches_analysis(self):
        """Verify we're testing all 14 methods from analysis."""
        assert len(REQUIRED_METHODS) == 14, (
            f"Expected 14 methods from analysis, but testing {len(REQUIRED_METHODS)}"
        )


class TestBackendBehaviorCompatibility:
    """Test that old and new backends behave identically."""
    
    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database path."""
        db_path = tmp_path / "test.db"
        return str(db_path)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.0, 30.5]
        })
    
    def test_query_compatibility(self, temp_db_path, sample_dataframe):
        """Test that query() works identically on old and new backends."""
        # Test with old backend
        old_backend = SQLiteBackend({})
        engine = old_backend.get_engine(temp_db_path)
        old_backend.create_table_from_dataframe(
            sample_dataframe, 'test_table', engine
        )
        old_result = old_backend.query("SELECT * FROM test_table")
        old_backend.close_connections()
        
        # Test with new backend
        new_backend = StatelessSQLiteBackend()
        engine = new_backend.get_engine(temp_db_path)
        new_result = new_backend.query("SELECT * FROM test_table")
        new_backend.close_connections()
        
        # Results should be identical
        pd.testing.assert_frame_equal(old_result, new_result)
    
    def test_create_table_compatibility(self, temp_db_path, sample_dataframe):
        """Test table creation works identically."""
        # Create with old backend
        old_backend = SQLiteBackend({})
        engine = old_backend.get_engine(temp_db_path)
        old_backend.create_table_from_dataframe(
            sample_dataframe, 'test_old', engine, if_exists='replace'
        )
        
        # Create with new backend
        new_backend = StatelessSQLiteBackend()
        new_backend.create_table_from_dataframe(
            sample_dataframe, 'test_new', engine, if_exists='replace'
        )
        
        # Read both tables
        old_data = old_backend.read_table_to_dataframe('test_old', engine)
        new_data = new_backend.read_table_to_dataframe('test_new', engine)
        
        # Cleanup
        old_backend.close_connections()
        new_backend.close_connections()
        
        # Data should be identical
        pd.testing.assert_frame_equal(
            old_data.sort_values('id').reset_index(drop=True),
            new_data.sort_values('id').reset_index(drop=True)
        )
    
    def test_resource_cleanup(self, temp_db_path):
        """Test that close_connections() properly cleans up resources."""
        backend = StatelessSQLiteBackend()
        
        # Create engine and cache it
        engine = backend.get_engine(temp_db_path)
        assert backend._engine is not None
        
        # Close connections
        backend.close_connections()
        
        # Verify cleanup
        assert backend._engine is None
        assert backend._session_factory is None


class TestCompatibilityMixinBehavior:
    """Test specific behaviors of the compatibility mixin."""
    
    def test_mixin_logs_deprecation(self, caplog, tmp_path):
        """Test that mixin methods log usage for migration tracking."""
        backend = StatelessSQLiteBackend()
        db_path = str(tmp_path / "test.db")
        engine = backend.get_engine(db_path)
        
        # Create a simple table
        df = pd.DataFrame({'x': [1, 2, 3]})
        backend.create_table_from_dataframe(df, 'test', engine)
        
        # Use compatibility method
        result = backend.query("SELECT * FROM test")
        
        # Check that debug message was logged
        assert any(
            "Using compatibility method 'query'" in record.message
            for record in caplog.records
        )
    
    def test_mixin_handles_missing_engine(self):
        """Test mixin methods handle missing engine gracefully."""
        backend = StatelessSQLiteBackend()
        
        # Try to query without engine
        with pytest.raises(Exception) as exc_info:
            backend.query("SELECT 1")
        
        assert "No engine available" in str(exc_info.value)


class TestAnalysisCompleteness:
    """Verify our analysis and implementation are complete."""
    
    def test_all_analyzed_methods_implemented(self):
        """Ensure all methods from analysis are in REQUIRED_METHODS."""
        # Methods from the analysis document
        analyzed_methods = {
            'get_engine',  # 11 calls
            'create_table_from_dataframe',  # 10 calls
            'query',  # 9 calls
            'read_table_to_dataframe',  # 7 calls
            'close_connections',  # 7 calls
            'read_table',  # 7 calls
            'write_table',  # 3 calls
            'get_table_info',  # 2 calls
            'execute_query',  # 1 call
            'get_connection',  # 1 call
            'get_columns',  # 1 call
            'analyze_column',  # 1 call
            'database_exists',  # 1 call
            'create_database',  # 1 call
        }
        
        # All analyzed methods should be in our required list
        missing = analyzed_methods - set(REQUIRED_METHODS.keys())
        assert not missing, f"Methods from analysis not in tests: {missing}"
        
        # All required methods should be from analysis
        extra = set(REQUIRED_METHODS.keys()) - analyzed_methods
        assert not extra, f"Extra methods not in analysis: {extra}"