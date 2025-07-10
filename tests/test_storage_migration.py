"""Test storage backend migration functionality."""
import pytest
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

from mdm.core import feature_flags
from mdm.adapters import get_storage_backend, clear_storage_cache
from mdm.migration import StorageMigrator, StorageValidator
from mdm.testing import StorageComparisonTester


class TestStorageBackendMigration:
    """Test storage backend migration features."""
    
    def setup_method(self):
        """Set up test environment."""
        # Save original feature flags
        self.original_flags = dict(feature_flags._flags)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="mdm_storage_test_")
        
        # Clear any cached backends
        clear_storage_cache()
    
    def teardown_method(self):
        """Clean up test environment."""
        # Restore original feature flags
        feature_flags._flags = self.original_flags
        
        # Clear cache again
        clear_storage_cache()
    
    def test_backend_adapter_interface(self):
        """Test that storage adapters implement the interface correctly."""
        # Test with legacy backend
        feature_flags.set("use_new_storage", False)
        legacy_backend = get_storage_backend("sqlite")
        
        # Check interface methods exist
        assert hasattr(legacy_backend, "backend_type")
        assert hasattr(legacy_backend, "get_engine")
        assert hasattr(legacy_backend, "create_table_from_dataframe")
        assert hasattr(legacy_backend, "query")
        assert hasattr(legacy_backend, "dataset_exists")
        assert hasattr(legacy_backend, "create_dataset")
        
        # Test with new backend
        feature_flags.set("use_new_storage", True)
        new_backend = get_storage_backend("sqlite")
        
        # Check same interface
        assert hasattr(new_backend, "backend_type")
        assert hasattr(new_backend, "get_engine")
        assert hasattr(new_backend, "create_table_from_dataframe")
        assert hasattr(new_backend, "query")
        assert hasattr(new_backend, "dataset_exists")
        assert hasattr(new_backend, "create_dataset")
    
    def test_feature_flag_switching(self):
        """Test switching between legacy and new backends."""
        # Get legacy backend
        feature_flags.set("use_new_storage", False)
        legacy = get_storage_backend("sqlite")
        assert "Adapter" in legacy.__class__.__name__
        
        # Get new backend
        feature_flags.set("use_new_storage", True)
        new = get_storage_backend("sqlite")
        assert "New" in new.__class__.__name__
    
    def test_dataset_operations_compatibility(self):
        """Test that dataset operations work with both backends."""
        test_dataset = "test_compat_dataset"
        test_data = pd.DataFrame({
            "id": range(100),
            "value": np.random.rand(100)
        })
        
        # Test with legacy backend
        feature_flags.set("use_new_storage", False)
        legacy_backend = get_storage_backend("sqlite")
        
        # Create and verify
        legacy_backend.create_dataset(test_dataset + "_legacy", {})
        assert legacy_backend.dataset_exists(test_dataset + "_legacy")
        
        # Save and load data
        legacy_backend.save_data(test_dataset + "_legacy", test_data)
        loaded = legacy_backend.load_data(test_dataset + "_legacy")
        assert len(loaded) == 100
        
        # Clean up
        legacy_backend.drop_dataset(test_dataset + "_legacy")
        assert not legacy_backend.dataset_exists(test_dataset + "_legacy")
        
        # Test with new backend
        feature_flags.set("use_new_storage", True)
        new_backend = get_storage_backend("sqlite")
        
        # Same operations
        new_backend.create_dataset(test_dataset + "_new", {})
        assert new_backend.dataset_exists(test_dataset + "_new")
        
        new_backend.save_data(test_dataset + "_new", test_data)
        loaded = new_backend.load_data(test_dataset + "_new")
        assert len(loaded) == 100
        
        new_backend.drop_dataset(test_dataset + "_new")
        assert not new_backend.dataset_exists(test_dataset + "_new")
    
    def test_storage_migrator(self):
        """Test storage migration between backends."""
        # Create test dataset with SQLite
        feature_flags.set("use_new_storage", False)
        sqlite_backend = get_storage_backend("sqlite")
        
        test_dataset = "test_migration"
        test_data = pd.DataFrame({
            "id": range(500),
            "name": [f"item_{i}" for i in range(500)],
            "value": np.random.rand(500)
        })
        
        # Create source dataset
        sqlite_backend.create_dataset(test_dataset, {})
        sqlite_backend.save_data(test_dataset, test_data)
        sqlite_backend.update_metadata(test_dataset, {"source": "test"})
        
        # Migrate to DuckDB
        migrator = StorageMigrator("sqlite", "duckdb")
        result = migrator.migrate_dataset(test_dataset, verify=True)
        
        assert result["success"]
        assert result["rows_migrated"] == 500
        assert "data" in result["tables_migrated"]
        
        # Verify target dataset exists
        duckdb_backend = get_storage_backend("duckdb")
        assert duckdb_backend.dataset_exists(test_dataset)
        
        # Verify data integrity
        migrated_data = duckdb_backend.load_data(test_dataset)
        assert len(migrated_data) == 500
        assert list(migrated_data.columns) == list(test_data.columns)
        
        # Clean up
        sqlite_backend.drop_dataset(test_dataset)
        duckdb_backend.drop_dataset(test_dataset)
    
    def test_storage_validator(self):
        """Test storage backend validation."""
        feature_flags.set("use_new_storage", True)
        backend = get_storage_backend("sqlite")
        
        validator = StorageValidator(backend)
        results = validator.validate_all()
        
        # Should have some passed tests
        assert len(results["passed"]) > 0
        
        # Basic operations should pass
        assert "backend_type property" in results["passed"]
        assert "get_engine" in results["passed"]
    
    def test_backend_specific_features(self):
        """Test backend-specific features are preserved."""
        # Test SQLite-specific features
        feature_flags.set("use_new_storage", True)
        sqlite_backend = get_storage_backend("sqlite", {
            "journal_mode": "WAL",
            "synchronous": "NORMAL"
        })
        
        # Create test database
        test_db = str(Path(self.temp_dir) / "test_features.db")
        engine = sqlite_backend.get_engine(test_db)
        
        # Verify SQLite pragmas were set
        with engine.connect() as conn:
            result = conn.execute("PRAGMA journal_mode")
            journal_mode = result.scalar()
            assert journal_mode.upper() == "WAL"
    
    def test_concurrent_backend_usage(self):
        """Test using both backends concurrently."""
        # Create backends with different feature flags
        feature_flags.set("use_new_storage", False)
        legacy_backend = get_storage_backend("sqlite")
        
        feature_flags.set("use_new_storage", True)
        new_backend = get_storage_backend("sqlite")
        
        # Both should work independently
        test_data = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        
        # Legacy operations
        legacy_backend.create_dataset("legacy_test", {})
        legacy_backend.save_data("legacy_test", test_data)
        
        # New operations
        new_backend.create_dataset("new_test", {})
        new_backend.save_data("new_test", test_data)
        
        # Verify both exist
        assert legacy_backend.dataset_exists("legacy_test")
        assert new_backend.dataset_exists("new_test")
        
        # Clean up
        legacy_backend.drop_dataset("legacy_test")
        new_backend.drop_dataset("new_test")
    
    def test_comparison_tester(self):
        """Test storage comparison framework."""
        tester = StorageComparisonTester("sqlite")
        
        # Run subset of tests
        result = tester.test_engine_creation()
        assert result[0] is True  # Legacy works
        assert result[1] is True  # New works
        
        # Test table operations
        result = tester.test_table_operations()
        assert result[0] == result[1]  # Same shape
    
    def test_error_handling_compatibility(self):
        """Test that error handling is consistent."""
        feature_flags.set("use_new_storage", False)
        legacy_backend = get_storage_backend("sqlite")
        
        feature_flags.set("use_new_storage", True)
        new_backend = get_storage_backend("sqlite")
        
        # Both should raise errors for non-existent datasets
        with pytest.raises(Exception):
            legacy_backend.load_data("non_existent")
        
        with pytest.raises(Exception):
            new_backend.load_data("non_existent")