"""Tests for storage backends."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import inspect, text

from mdm.core.exceptions import BackendError, StorageError
from mdm.models.base import ColumnInfo, DatasetMetadata, QualityMetrics
from mdm.storage import BackendFactory, SQLiteBackend, StorageBackend


class TestBackendFactory:
    """Test BackendFactory class."""

    def test_create_sqlite_backend(self):
        """Test creating SQLite backend."""
        config = {"sqlite": {"journal_mode": "WAL"}}
        backend = BackendFactory.create("sqlite", config)
        assert isinstance(backend, SQLiteBackend)
        assert backend.backend_type == "sqlite"

    def test_create_with_invalid_backend(self):
        """Test error on invalid backend type."""
        with pytest.raises(BackendError, match="Unsupported backend type"):
            BackendFactory.create("invalid_backend", {})

    def test_get_supported_backends(self):
        """Test getting list of supported backends."""
        backends = BackendFactory.get_supported_backends()
        assert "sqlite" in backends
        assert "duckdb" in backends
        assert "postgresql" in backends
        assert len(backends) == 3


class TestSQLiteBackend:
    """Test SQLite storage backend."""

    @pytest.fixture
    def backend(self):
        """Create SQLite backend instance."""
        config = {
            "sqlalchemy": {"echo": False},
            "sqlite": {
                "journal_mode": "WAL",
                "synchronous": "NORMAL",
            },
        }
        return SQLiteBackend(config)

    def test_backend_type(self, backend):
        """Test backend type property."""
        assert backend.backend_type == "sqlite"

    def test_get_database_path(self, backend):
        """Test database path generation."""
        base_path = Path("/data/mdm/datasets")
        db_path = backend.get_database_path("test_dataset", base_path)
        assert db_path == "/data/mdm/datasets/test_dataset/dataset.sqlite"

    def test_create_and_check_database(self, backend):
        """Test creating and checking database existence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"

            # Database should not exist initially
            assert not backend.database_exists(str(db_path))

            # Create database
            backend.create_database(str(db_path))

            # Database should exist now
            assert backend.database_exists(str(db_path))
            assert db_path.exists()

    def test_create_engine(self, backend):
        """Test creating SQLAlchemy engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            backend.create_database(str(db_path))

            engine = backend.create_engine(str(db_path))
            assert engine is not None

            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

            engine.dispose()

    def test_initialize_database(self, backend):
        """Test database initialization with tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            backend.create_database(str(db_path))

            engine = backend.create_engine(str(db_path))
            backend.initialize_database(engine)

            # Check that metadata tables were created
            inspector = inspect(engine)
            tables = inspector.get_table_names()

            assert "_metadata" in tables
            assert "_columns" in tables
            assert "_quality_metrics" in tables

            engine.dispose()

    def test_session_context_manager(self, backend):
        """Test session context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            backend.create_database(str(db_path))

            engine = backend.create_engine(str(db_path))
            backend.initialize_database(engine)

            # Test session usage
            with backend.session(str(db_path)) as session:
                # Add metadata entry
                metadata = DatasetMetadata(
                    key="dataset_name", value="test_dataset"
                )
                session.add(metadata)

            # Verify data was committed
            with backend.session(str(db_path)) as session:
                result = session.query(DatasetMetadata).filter_by(
                    key="dataset_name"
                ).first()
                assert result is not None
                assert result.value == "test_dataset"

            engine.dispose()

    def test_create_table_from_dataframe(self, backend):
        """Test creating table from pandas DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            backend.create_database(str(db_path))

            engine = backend.create_engine(str(db_path))

            # Create test DataFrame
            df = pd.DataFrame({
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
            })

            # Create table
            backend.create_table_from_dataframe(df, "test_table", engine)

            # Verify table exists
            assert backend.table_exists(engine, "test_table")

            # Verify data
            result_df = backend.read_table_to_dataframe("test_table", engine)
            pd.testing.assert_frame_equal(df, result_df)

            engine.dispose()

    def test_get_table_info(self, backend):
        """Test getting table information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            backend.create_database(str(db_path))

            engine = backend.create_engine(str(db_path))

            # Create test table
            df = pd.DataFrame({
                "id": range(10),
                "value": range(10, 20),
            })
            backend.create_table_from_dataframe(df, "test_table", engine)

            # Get table info
            info = backend.get_table_info("test_table", engine)

            assert info["name"] == "test_table"
            assert info["row_count"] == 10
            assert info["column_count"] == 2
            assert len(info["columns"]) == 2

            engine.dispose()

    def test_drop_database(self, backend):
        """Test dropping database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"

            # Create and verify database exists
            backend.create_database(str(db_path))
            assert backend.database_exists(str(db_path))

            # Drop database
            backend.drop_database(str(db_path))

            # Verify database no longer exists
            assert not backend.database_exists(str(db_path))
            assert not Path(db_path).exists()

    def test_execute_query(self, backend):
        """Test executing arbitrary SQL query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            backend.create_database(str(db_path))

            engine = backend.create_engine(str(db_path))

            # Create test table
            df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
            backend.create_table_from_dataframe(df, "test_data", engine)

            # Execute query
            result = backend.execute_query(
                "SELECT SUM(x), AVG(y) FROM test_data", engine
            )
            row = result.fetchone()

            assert row[0] == 6  # sum(x)
            assert row[1] == 5.0  # avg(y)

            engine.dispose()


class TestStorageBackendAbstract:
    """Test abstract StorageBackend methods."""

    def test_table_exists(self):
        """Test table_exists method."""
        backend = SQLiteBackend({})

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            backend.create_database(str(db_path))

            engine = backend.create_engine(str(db_path))

            # Table should not exist
            assert not backend.table_exists(engine, "nonexistent")

            # Create table
            df = pd.DataFrame({"col": [1, 2, 3]})
            backend.create_table_from_dataframe(df, "existing_table", engine)

            # Table should exist
            assert backend.table_exists(engine, "existing_table")

            engine.dispose()

    def test_get_table_names(self):
        """Test getting list of table names."""
        backend = SQLiteBackend({})

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            backend.create_database(str(db_path))

            engine = backend.create_engine(str(db_path))
            backend.initialize_database(engine)

            # Create additional tables
            df1 = pd.DataFrame({"a": [1, 2]})
            df2 = pd.DataFrame({"b": [3, 4]})
            backend.create_table_from_dataframe(df1, "table1", engine)
            backend.create_table_from_dataframe(df2, "table2", engine)

            # Get table names
            tables = backend.get_table_names(engine)

            # Should include both user tables and metadata tables
            assert "table1" in tables
            assert "table2" in tables
            assert "_metadata" in tables
            assert "_columns" in tables
            assert "_quality_metrics" in tables

            engine.dispose()

    def test_read_table_with_limit(self):
        """Test reading table with row limit."""
        backend = SQLiteBackend({})

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            backend.create_database(str(db_path))

            engine = backend.create_engine(str(db_path))

            # Create table with 100 rows
            df = pd.DataFrame({"id": range(100), "value": range(100, 200)})
            backend.create_table_from_dataframe(df, "large_table", engine)

            # Read with limit
            result_df = backend.read_table_to_dataframe(
                "large_table", engine, limit=10
            )

            assert len(result_df) == 10
            assert list(result_df["id"]) == list(range(10))

            engine.dispose()

    def test_close_connections(self):
        """Test closing database connections."""
        backend = SQLiteBackend({})

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            backend.create_database(str(db_path))

            # Create engine
            engine = backend.get_engine(str(db_path))
            assert backend._engine is not None

            # Close connections
            backend.close_connections()

            # Engine should be None
            assert backend._engine is None
            assert backend._session_factory is None