"""Unit tests for BackendFactory.

NOTE: These tests are for the legacy backend API where backends accepted
configuration in their constructors. The new stateless backends do not
accept configuration parameters.

These tests are marked as skip - see tests/unit/storage/test_stateless_backends.py 
for new stateless backend tests.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from mdm.storage.factory import BackendFactory
from mdm.storage.base import StorageBackend
from mdm.storage.backends.stateless_sqlite import StatelessSQLiteBackend as SQLiteBackend
from mdm.storage.backends.stateless_duckdb import StatelessDuckDBBackend as DuckDBBackend
from mdm.storage.postgresql import PostgreSQLBackend
from mdm.core.exceptions import BackendError


@pytest.mark.skip(reason="Legacy backend API - backends no longer accept config in constructor, already tested in test_stateless_backends.py")
class TestBackendFactory:
    """Test cases for BackendFactory."""

    def test_create_sqlite_backend(self):
        """Test creating SQLite backend."""
        config = {"backend": "sqlite", "path": "/test/path"}
        
        backend_instance = Mock(spec=StorageBackend)
        mock_backend_class = Mock(return_value=backend_instance)
        
        with patch.dict(BackendFactory._backends, {'sqlite': mock_backend_class}):
            backend = BackendFactory.create("sqlite", config)
            
            # Check correct backend was created
            mock_backend_class.assert_called_once_with(config)
            assert backend is backend_instance

    def test_create_duckdb_backend(self):
        """Test creating DuckDB backend."""
        config = {"backend": "duckdb", "path": "/test/path"}
        
        backend_instance = Mock(spec=StorageBackend)
        mock_backend_class = Mock(return_value=backend_instance)
        
        with patch.dict(BackendFactory._backends, {'duckdb': mock_backend_class}):
            backend = BackendFactory.create("duckdb", config)
            
            # Check correct backend was created
            mock_backend_class.assert_called_once_with(config)
            assert backend is backend_instance

    def test_create_postgresql_backend(self):
        """Test creating PostgreSQL backend."""
        config = {
            "backend": "postgresql",
            "host": "localhost",
            "port": 5432,
            "user": "test",
            "password": "test",
            "database": "test_db"
        }
        
        backend_instance = Mock(spec=StorageBackend)
        mock_backend_class = Mock(return_value=backend_instance)
        
        with patch.dict(BackendFactory._backends, {'postgresql': mock_backend_class}):
            backend = BackendFactory.create("postgresql", config)
            
            # Check correct backend was created
            mock_backend_class.assert_called_once_with(config)
            assert backend is backend_instance

    def test_create_unsupported_backend(self):
        """Test error when creating unsupported backend."""
        config = {"backend": "mongodb"}
        
        with pytest.raises(BackendError) as exc_info:
            BackendFactory.create("mongodb", config)
        
        assert "Unsupported backend type: mongodb" in str(exc_info.value)
        assert "sqlite" in str(exc_info.value)
        assert "duckdb" in str(exc_info.value)
        assert "postgresql" in str(exc_info.value)

    def test_get_supported_backends(self):
        """Test getting list of supported backends."""
        backends = BackendFactory.get_supported_backends()
        
        assert isinstance(backends, list)
        assert "sqlite" in backends
        assert "duckdb" in backends
        assert "postgresql" in backends
        assert len(backends) == 3

    def test_backend_creation_with_empty_config(self):
        """Test backend creation with empty config."""
        backend_instance = Mock(spec=StorageBackend)
        mock_backend_class = Mock(return_value=backend_instance)
        
        with patch.dict(BackendFactory._backends, {'sqlite': mock_backend_class}):
            backend = BackendFactory.create("sqlite", {})
            
            # Should still work with empty config
            mock_backend_class.assert_called_once_with({})
            assert backend is backend_instance

    def test_backend_creation_error_propagation(self):
        """Test that backend initialization errors are propagated."""
        config = {"backend": "sqlite", "invalid": "config"}
        
        mock_backend_class = Mock(side_effect=ValueError("Invalid configuration"))
        
        with patch.dict(BackendFactory._backends, {'sqlite': mock_backend_class}):
            with pytest.raises(ValueError, match="Invalid configuration"):
                BackendFactory.create("sqlite", config)

    def test_case_sensitivity(self):
        """Test that backend type is case sensitive."""
        config = {"backend": "sqlite"}
        
        # Uppercase should fail
        with pytest.raises(BackendError, match="Unsupported backend type: SQLITE"):
            BackendFactory.create("SQLITE", config)
        
        # Mixed case should fail
        with pytest.raises(BackendError, match="Unsupported backend type: SqLiTe"):
            BackendFactory.create("SqLiTe", config)

    def test_backends_dictionary_integrity(self):
        """Test that backends dictionary contains correct types."""
        # Access private attribute for testing
        backends = BackendFactory._backends
        
        assert isinstance(backends, dict)
        assert backends["sqlite"] is SQLiteBackend
        assert backends["duckdb"] is DuckDBBackend
        assert backends["postgresql"] is PostgreSQLBackend

    def test_create_with_none_backend_type(self):
        """Test error when backend type is None."""
        with pytest.raises(BackendError) as exc_info:
            BackendFactory.create(None, {})
        
        assert "Unsupported backend type: None" in str(exc_info.value)

    def test_create_with_empty_string_backend_type(self):
        """Test error when backend type is empty string."""
        with pytest.raises(BackendError) as exc_info:
            BackendFactory.create("", {})
        
        assert "Unsupported backend type: " in str(exc_info.value)

    def test_supported_backends_returns_copy(self):
        """Test that get_supported_backends returns a copy of the list."""
        backends1 = BackendFactory.get_supported_backends()
        backends2 = BackendFactory.get_supported_backends()
        
        # Should be equal but not same object
        assert backends1 == backends2
        assert backends1 is not backends2
        
        # Modifying returned list shouldn't affect factory
        backends1.append("custom")
        backends3 = BackendFactory.get_supported_backends()
        assert "custom" not in backends3

    def test_backend_instance_types(self):
        """Test that created backends are of correct type."""
        # Test without mocking to ensure type relationships
        config = {"backend": "sqlite", "path": ":memory:"}
        
        # Mock the database connection part but not the class itself
        with patch('sqlalchemy.create_engine'):
            with patch('sqlalchemy.orm.sessionmaker'):
                backend = BackendFactory.create("sqlite", config)
                assert isinstance(backend, SQLiteBackend)
                assert isinstance(backend, StorageBackend)

    def test_config_passed_correctly(self):
        """Test that config is passed correctly to backend constructor."""
        complex_config = {
            "backend": "postgresql",
            "host": "localhost",
            "port": 5432,
            "user": "testuser",
            "password": "testpass",
            "database": "testdb",
            "pool_size": 10,
            "max_overflow": 20,
            "echo": True,
            "custom_param": "value"
        }
        
        backend_instance = Mock(spec=StorageBackend)
        mock_backend_class = Mock(return_value=backend_instance)
        
        with patch.dict(BackendFactory._backends, {'postgresql': mock_backend_class}):
            backend = BackendFactory.create("postgresql", complex_config)
            
            # Check exact config was passed
            mock_backend_class.assert_called_once()
            call_args = mock_backend_class.call_args[0][0]
            assert call_args == complex_config

    @patch.dict('mdm.storage.factory.BackendFactory._backends', clear=True)
    def test_empty_backends_registry(self):
        """Test behavior when backends registry is empty."""
        # Clear backends (using patch.dict)
        with pytest.raises(BackendError, match="Unsupported backend type: sqlite"):
            BackendFactory.create("sqlite", {})
        
        # get_supported_backends should return empty list
        assert BackendFactory.get_supported_backends() == []

    def test_thread_safety_considerations(self):
        """Test that factory methods are essentially thread-safe (no shared state)."""
        # BackendFactory uses class methods and class variables only
        # Creating backends should not affect each other
        
        configs = [
            {"backend": "sqlite", "path": f"/test{i}"}
            for i in range(3)
        ]
        
        instances = []
        for i, config in enumerate(configs):
            mock_instance = Mock(spec=StorageBackend, id=i)
            mock_backend_class = Mock(return_value=mock_instance)
            with patch.dict(BackendFactory._backends, {'sqlite': mock_backend_class}):
                instances.append(BackendFactory.create("sqlite", config))
        
        # Each call should create a new instance
        assert len(set(id(inst) for inst in instances)) == 3