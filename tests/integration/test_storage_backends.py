"""Integration tests for storage backends."""

import pandas as pd
import pytest

from mdm.storage.factory import BackendFactory


class TestStorageBackends:
    """Test different storage backend implementations."""

    @pytest.mark.parametrize("backend_type", ["duckdb", "sqlite"])
    def test_backend_operations(self, backend_type, temp_dir):
        """Test basic operations for each backend."""
        # Create backend
        backend = BackendFactory.create_backend(
            backend_type,
            database_path=temp_dir / f"test.{backend_type}",
            dataset_name="test_dataset",
        )
        
        # Create test data
        df = pd.DataFrame({
            'id': range(100),
            'value': [i * 2 for i in range(100)],
            'category': ['A', 'B'] * 50,
        })
        
        # Create table
        table_name = backend.create_table("test_table", df)
        assert table_name == "test_dataset_test_table"
        
        # Read table
        df_read = backend.read_table(table_name)
        assert len(df_read) == 100
        assert list(df_read.columns) == ['id', 'value', 'category']
        
        # Read with limit
        df_limited = backend.read_table(table_name, limit=10)
        assert len(df_limited) == 10
        
        # Get table info
        info = backend.get_table_info(table_name)
        assert info['row_count'] == 100
        assert len(info['columns']) == 3
        
        # Execute query
        result = backend.execute_query(
            f"SELECT COUNT(*) as cnt FROM {table_name} WHERE category = 'A'"
        )
        assert result.iloc[0]['cnt'] == 50
        
        # List tables
        tables = backend.list_tables()
        assert table_name in tables
        
        # Table exists
        assert backend.table_exists(table_name) is True
        assert backend.table_exists("nonexistent") is False
        
        # Drop table
        backend.drop_table(table_name)
        assert backend.table_exists(table_name) is False
        
        # Close connection
        backend.close()

    def test_backend_with_special_characters(self, temp_dir):
        """Test handling of special characters in data."""
        backend = BackendFactory.create_backend(
            "duckdb",
            database_path=temp_dir / "test_special.duckdb",
            dataset_name="test_special",
        )
        
        # Create data with special characters
        df = pd.DataFrame({
            'text': ["Hello, World!", "Test 'quotes'", 'Test "double"', "Test\nnewline"],
            'unicode': ["café", "naïve", "résumé", "🚀"],
        })
        
        # Create and read table
        table_name = backend.create_table("special_chars", df)
        df_read = backend.read_table(table_name)
        
        # Verify data integrity
        assert df_read['text'].iloc[0] == "Hello, World!"
        assert df_read['text'].iloc[1] == "Test 'quotes'"
        assert df_read['unicode'].iloc[0] == "café"
        assert df_read['unicode'].iloc[3] == "🚀"
        
        backend.close()

    def test_backend_performance(self, temp_dir):
        """Test backend performance with larger dataset."""
        backend = BackendFactory.create_backend(
            "duckdb",
            database_path=temp_dir / "test_perf.duckdb",
            dataset_name="test_perf",
        )
        
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
        table_name = backend.create_table("perf_test", df)
        create_time = time.time() - start
        assert create_time < 5.0  # Should complete within 5 seconds
        
        # Test aggregation query
        start = time.time()
        result = backend.execute_query(f"""
            SELECT 
                category,
                COUNT(*) as count,
                AVG(value1) as avg_value1,
                MAX(value2) as max_value2
            FROM {table_name}
            GROUP BY category
            ORDER BY count DESC
        """)
        query_time = time.time() - start
        assert query_time < 1.0  # Should complete within 1 second
        assert len(result) == 100  # 100 categories
        
        backend.close()