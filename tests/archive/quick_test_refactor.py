#!/usr/bin/env python3
"""Quick test of refactoring components."""
import sys
import os

# Add the refactoring path to Python path
sys.path.insert(0, '/home/xai/DEV/mdm-refactor-2025/src')

def test_feature_flags():
    """Test feature flags."""
    print("Testing Feature Flags...")
    try:
        from mdm.core.feature_flags import FeatureFlags
        import tempfile
        from pathlib import Path
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        # Remove the file so FeatureFlags creates it fresh
        temp_path.unlink()
        
        # Test
        flags = FeatureFlags(config_path=temp_path)
        flags.set("test", True)
        assert flags.get("test") is True
        print("✅ Feature Flags working!")
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
            
    except Exception as e:
        print(f"❌ Feature Flags failed: {e}")
        import traceback
        traceback.print_exc()

def test_metrics():
    """Test metrics."""
    print("\nTesting Metrics...")
    try:
        from mdm.core.metrics import MetricsCollector
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=Path(tmpdir))
            collector.increment("test", 5)
            summary = collector.get_summary()
            assert summary["counters"]["test"] == 5
            print("✅ Metrics working!")
            
    except Exception as e:
        print(f"❌ Metrics failed: {e}")
        import traceback
        traceback.print_exc()

def test_ab_testing():
    """Test A/B testing."""
    print("\nTesting A/B Testing...")
    try:
        from mdm.core.ab_testing import ABTestConfig, ABTestRouter
        
        router = ABTestRouter()
        config = ABTestConfig(
            test_name="test",
            control_impl=lambda: "control",
            treatment_impl=lambda: "treatment",
            traffic_percentage=50.0
        )
        router.register_test(config)
        
        # Test routing
        result = router.route("test", "user1")
        assert result in ["control", "treatment"]
        print("✅ A/B Testing working!")
        
    except Exception as e:
        print(f"❌ A/B Testing failed: {e}")
        import traceback
        traceback.print_exc()

def test_comparison():
    """Test comparison framework."""
    print("\nTesting Comparison Framework...")
    try:
        from mdm.testing.comparison import ComparisonTester
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tester = ComparisonTester(results_dir=Path(tmpdir))
            
            result = tester.compare(
                test_name="test",
                old_impl=lambda x: x * 2,
                new_impl=lambda x: x * 2,
                args=(5,)
            )
            
            assert result.passed is True
            print("✅ Comparison Framework working!")
            
    except Exception as e:
        print(f"❌ Comparison Framework failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Quick Test of Refactoring Components ===\n")
    test_feature_flags()
    test_metrics()
    test_ab_testing()
    test_comparison()
    print("\n=== Tests Complete ===")