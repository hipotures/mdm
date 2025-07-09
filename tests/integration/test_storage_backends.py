"""Integration tests for storage backends."""

import pandas as pd
import pytest
from sqlalchemy import text

from mdm.storage.factory import BackendFactory


class TestStorageBackends:
    """Test different storage backend implementations."""

    @pytest.mark.parametrize("backend_type", ["duckdb", "sqlite"])
    def test_backend_operations(self, backend_type, temp_dir):
        """Test basic operations for each backend."""
        # Create backend
        config = {
            "database_path": str(temp_dir / f"test.{backend_type}"),
            "dataset_name": "test_dataset",
        }
        backend = BackendFactory.create(backend_type, config)
        
        # Create database and get engine
        db_path = config["database_path"]
        backend.create_database(db_path)
        engine = backend.create_engine(db_path)
        
        # Create test data
        df = pd.DataFrame({
            'id': range(100),
            'value': [i * 2 for i in range(100)],
            'category': ['A', 'B'] * 50,
        })
        
        # Create table
        table_name = "test_dataset_test_table"
        backend.create_table_from_dataframe(df, table_name, engine)
        assert backend.table_exists(engine, table_name)
        
        # Read table
        df_read = backend.read_table_to_dataframe(table_name, engine)
        assert len(df_read) == 100
        assert list(df_read.columns) == ['id', 'value', 'category']
        
        # Read with limit
        df_limited = backend.read_table_to_dataframe(table_name, engine, limit=10)
        assert len(df_limited) == 10
        
        # Get table info
        info = backend.get_table_info(table_name, engine)
        assert info['row_count'] == 100
        assert len(info['columns']) == 3
        
        # Execute query
        with backend.session(db_path) as session:
            result = session.execute(
                text(f"SELECT COUNT(*) as cnt FROM {table_name} WHERE category = 'A'")
            ).fetchone()
            assert result[0] == 50
        
        # List tables
        tables = backend.get_table_names(engine)
        assert table_name in tables
        
        # Table exists
        assert backend.table_exists(engine, table_name) is True
        assert backend.table_exists(engine, "nonexistent") is False
        
        # Drop table
        with backend.session(db_path) as session:
            session.execute(text(f"DROP TABLE {table_name}"))
        assert backend.table_exists(engine, table_name) is False

    def test_backend_with_special_characters(self, temp_dir):
        """Test handling of special characters in data."""
        config = {
            "database_path": str(temp_dir / "test_special.duckdb"),
            "dataset_name": "test_special",
        }
        backend = BackendFactory.create("duckdb", config)
        
        # Create database and get engine
        db_path = config["database_path"]
        backend.create_database(db_path)
        engine = backend.create_engine(db_path)
        
        # Create data with special characters
        df = pd.DataFrame({
            'text': ["Hello, World!", "Test 'quotes'", 'Test "double"', "Test\nnewline"],
            'unicode': ["café", "naïve", "résumé", "🚀"],
        })
        
        # Create and read table
        table_name = "special_chars"
        backend.create_table_from_dataframe(df, table_name, engine)
        df_read = backend.read_table_to_dataframe(table_name, engine)
        
        # Verify data integrity
        assert df_read['text'].iloc[0] == "Hello, World!"
        assert df_read['text'].iloc[1] == "Test 'quotes'"
        assert df_read['unicode'].iloc[0] == "café"
        assert df_read['unicode'].iloc[3] == "🚀"

    def test_backend_performance(self, temp_dir):
        """Test backend performance with larger dataset."""
        config = {
            "database_path": str(temp_dir / "test_perf.duckdb"),
            "dataset_name": "test_perf",
        }
        backend = BackendFactory.create("duckdb", config)
        
        # Create database and get engine
        db_path = config["database_path"]
        backend.create_database(db_path)
        engine = backend.create_engine(db_path)
        
        # Create larger dataset
        n_rows = 10000
        df = pd.DataFrame({
            'id': range(n_rows),
            'value1': [i * 1.5 for i in range(n_rows)],
            'value2': [i ** 2 for i in range(n_rows)],
            'category': [f"cat_{i % 100}" for i in range(n_rows)],
            'text': [f"Text description {i}" for i in range(n_rows)],
        })
        
        # Test table creation
        import time
        start = time.time()
        table_name = "perf_test"
        backend.create_table_from_dataframe(df, table_name, engine)
        create_time = time.time() - start
        assert create_time < 5.0  # Should complete within 5 seconds
        
        # Test aggregation query
        start = time.time()
        with backend.session(db_path) as session:
            result = session.execute(text(f"""
                SELECT 
                    category,
                    COUNT(*) as count,
                    AVG(value1) as avg_value1,
                    MAX(value2) as max_value2
                FROM {table_name}
                GROUP BY category
                ORDER BY count DESC
            """)).fetchall()
        query_time = time.time() - start
        assert query_time < 1.0  # Should complete within 1 second
        assert len(result) == 100  # 100 categories