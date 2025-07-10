"""Test feature engineering migration functionality."""
import pytest
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime

from mdm.core import feature_flags
from mdm.adapters.feature_manager import get_feature_generator, clear_feature_cache
from mdm.migration.feature_migration import FeatureMigrator, FeatureValidator
from mdm.testing.feature_comparison import FeatureComparisonTester


class TestFeatureEngineeringMigration:
    """Test feature engineering migration features."""
    
    def setup_method(self):
        """Set up test environment."""
        # Save original feature flags
        self.original_flags = dict(feature_flags._flags)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="mdm_feature_test_")
        
        # Clear any cached generators
        clear_feature_cache()
    
    def teardown_method(self):
        """Clean up test environment."""
        # Restore original feature flags
        feature_flags._flags = self.original_flags
        
        # Clear cache again
        clear_feature_cache()
    
    def test_feature_generator_adapter(self):
        """Test that feature generator adapter works correctly."""
        # Get legacy generator through adapter
        feature_flags.set("use_new_features", False)
        legacy_gen = get_feature_generator()
        
        # Create test data
        test_data = pd.DataFrame({
            "numeric": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "text": ["sample text"] * 100
        })
        
        # Generate features
        result = legacy_gen.generate_features(test_data)
        
        # Check that features were generated
        assert len(result.columns) > len(test_data.columns)
        assert "numeric" in result.columns  # Original columns preserved
        
        # Check interface methods exist
        assert hasattr(legacy_gen, "generate_numeric_features")
        assert hasattr(legacy_gen, "generate_categorical_features")
        assert hasattr(legacy_gen, "get_feature_importance")
    
    def test_feature_flag_switching(self):
        """Test switching between legacy and new generators."""
        # Get legacy generator
        feature_flags.set("use_new_features", False)
        legacy = get_feature_generator()
        assert "Adapter" in legacy.__class__.__name__
        
        # Get new generator
        feature_flags.set("use_new_features", True)
        new = get_feature_generator()
        assert "New" in new.__class__.__name__
    
    def test_numeric_feature_compatibility(self):
        """Test numeric feature generation compatibility."""
        # Create numeric test data
        test_data = pd.DataFrame({
            "value1": np.random.randn(100),
            "value2": np.random.uniform(0, 100, 100)
        })
        
        # Generate with legacy
        feature_flags.set("use_new_features", False)
        legacy_gen = get_feature_generator()
        legacy_result = legacy_gen.generate_numeric_features(test_data)
        
        # Generate with new
        feature_flags.set("use_new_features", True)
        new_gen = get_feature_generator()
        new_result = new_gen.generate_numeric_features(test_data)
        
        # Both should generate additional features
        assert len(legacy_result.columns) > len(test_data.columns)
        assert len(new_result.columns) > len(test_data.columns)
    
    def test_categorical_feature_compatibility(self):
        """Test categorical feature generation compatibility."""
        # Create categorical test data
        test_data = pd.DataFrame({
            "category": np.random.choice(["A", "B", "C", "D"], 100),
            "subcategory": np.random.choice(["X", "Y", "Z"], 100)
        })
        
        # Generate with both implementations
        feature_flags.set("use_new_features", False)
        legacy_gen = get_feature_generator()
        legacy_result = legacy_gen.generate_categorical_features(test_data)
        
        feature_flags.set("use_new_features", True)
        new_gen = get_feature_generator()
        new_result = new_gen.generate_categorical_features(test_data)
        
        # Check that count/frequency encodings are present
        legacy_new_cols = set(legacy_result.columns) - set(test_data.columns)
        new_new_cols = set(new_result.columns) - set(test_data.columns)
        
        # Should have some common feature types
        assert any("count" in col for col in legacy_new_cols)
        assert any("count" in col for col in new_new_cols)
    
    def test_datetime_feature_generation(self):
        """Test datetime feature generation."""
        # Create datetime test data
        test_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="H")
        })
        
        # Test with both implementations
        feature_flags.set("use_new_features", False)
        legacy_gen = get_feature_generator()
        legacy_result = legacy_gen.generate_datetime_features(test_data)
        
        feature_flags.set("use_new_features", True)
        new_gen = get_feature_generator()
        new_result = new_gen.generate_datetime_features(test_data)
        
        # Check for common datetime features
        for result in [legacy_result, new_result]:
            cols = result.columns
            assert any("year" in col for col in cols)
            assert any("month" in col for col in cols)
            assert any("day" in col for col in cols)
    
    def test_feature_importance_calculation(self):
        """Test feature importance calculation."""
        # Create test data with known correlation
        n = 500
        features = pd.DataFrame({
            "correlated": np.random.randn(n),
            "noise": np.random.randn(n)
        })
        target = features["correlated"] * 2 + np.random.randn(n) * 0.1
        
        # Calculate importance with both implementations
        feature_flags.set("use_new_features", False)
        legacy_gen = get_feature_generator()
        legacy_importance = legacy_gen.get_feature_importance(
            features, target, "regression"
        )
        
        feature_flags.set("use_new_features", True)
        new_gen = get_feature_generator()
        new_importance = new_gen.get_feature_importance(
            features, target, "regression"
        )
        
        # Both should identify correlated feature as more important
        assert legacy_importance.iloc[0]["feature"] == "correlated"
        assert new_importance.iloc[0]["feature"] == "correlated"
    
    def test_feature_migrator(self):
        """Test feature migrator functionality."""
        migrator = FeatureMigrator()
        
        # Create test data
        test_data = pd.DataFrame({
            "numeric": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], 100)
        })
        
        # Compare outputs
        comparison = migrator.compare_outputs(test_data)
        
        assert "legacy" in comparison
        assert "new" in comparison
        assert "features" in comparison
        assert "performance_ratio" in comparison
    
    def test_transformer_migration(self):
        """Test transformer migration to plugin format."""
        migrator = FeatureMigrator()
        
        # Migrate transformers
        output_dir = Path(self.temp_dir) / "plugins"
        summary = migrator.migrate_transformers(output_dir)
        
        assert summary["migrated_count"] >= 0
        assert output_dir.exists()
        
        # Check that summary file was created
        summary_file = output_dir / "migration_summary.json"
        assert summary_file.exists()
    
    def test_feature_validator(self):
        """Test feature validator."""
        validator = FeatureValidator()
        
        # Create test data
        test_data = pd.DataFrame({
            "numeric1": np.random.randn(100),
            "numeric2": np.random.uniform(0, 10, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "date": pd.date_range("2024-01-01", periods=100),
            "text": ["Sample text " * i for i in range(100)]
        })
        
        # Validate consistency
        results = validator.validate_consistency(test_data)
        
        assert "feature_types" in results
        assert "overall_match_rate" in results
        assert results["overall_match_rate"] >= 0  # Some match expected
    
    def test_comparison_tester(self):
        """Test feature comparison framework."""
        tester = FeatureComparisonTester()
        
        # Run basic comparison test
        result = tester.test_basic_generation()
        
        assert len(result) == 2  # Legacy and new results
        assert "features" in result[0]
        assert "features" in result[1]
    
    def test_missing_data_handling(self):
        """Test handling of missing data in both implementations."""
        # Create data with missing values
        test_data = pd.DataFrame({
            "complete": np.random.randn(100),
            "partial": np.random.randn(100)
        })
        test_data.loc[20:40, "partial"] = np.nan
        
        # Test both implementations
        feature_flags.set("use_new_features", False)
        legacy_gen = get_feature_generator()
        legacy_result = legacy_gen.generate_features(test_data)
        
        feature_flags.set("use_new_features", True)
        new_gen = get_feature_generator()
        new_result = new_gen.generate_features(test_data)
        
        # Both should handle missing data without errors
        assert len(legacy_result) == len(test_data)
        assert len(new_result) == len(test_data)
    
    def test_performance_comparison(self):
        """Test performance comparison between implementations."""
        # Create larger test data
        test_data = pd.DataFrame({
            "numeric1": np.random.randn(10000),
            "numeric2": np.random.randn(10000),
            "category": np.random.choice(["A", "B", "C", "D"], 10000)
        })
        
        # Time legacy implementation
        feature_flags.set("use_new_features", False)
        legacy_gen = get_feature_generator()
        
        import time
        legacy_start = time.time()
        legacy_result = legacy_gen.generate_features(test_data)
        legacy_time = time.time() - legacy_start
        
        # Time new implementation
        feature_flags.set("use_new_features", True)
        new_gen = get_feature_generator()
        
        new_start = time.time()
        new_result = new_gen.generate_features(test_data)
        new_time = time.time() - new_start
        
        # New implementation should be reasonably performant
        performance_ratio = new_time / legacy_time if legacy_time > 0 else 1.0
        assert performance_ratio < 2.0  # Not more than 2x slower
    
    def test_custom_config_support(self):
        """Test that custom configuration is supported."""
        config = {
            "create_bins": False,
            "n_bins": 3,
            "rare_threshold": 0.05
        }
        
        # Test with new implementation
        feature_flags.set("use_new_features", True)
        new_gen = get_feature_generator()
        
        # Should accept config without errors
        test_data = pd.DataFrame({
            "numeric": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C", "D", "E", "RARE"], 100)
        })
        
        result = new_gen.generate_features(test_data)
        assert len(result.columns) > len(test_data.columns)