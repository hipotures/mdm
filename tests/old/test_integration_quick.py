#!/usr/bin/env python3
"""Quick integration test of refactoring components."""
import sys
import os

# Add the refactoring path to Python path
sys.path.insert(0, '/home/xai/DEV/mdm-refactor-2025/src')

def test_integration():
    """Test integration scenario."""
    print("Testing Integration Scenario...")
    
    from mdm.core.feature_flags import FeatureFlags
    from mdm.core.metrics import MetricsCollector
    from mdm.core.ab_testing import ABTestConfig, ABTestRouter
    from mdm.testing.comparison import ComparisonTester
    import tempfile
    from pathlib import Path
    import time
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        flags = FeatureFlags(config_path=Path(tmpdir) / "flags.json")
        metrics = MetricsCollector(output_dir=Path(tmpdir) / "metrics")
        router = ABTestRouter()
        tester = ComparisonTester(results_dir=Path(tmpdir) / "comparisons")
        
        # Test 1: Feature flags with metrics
        print("\n1. Testing feature flags with metrics...")
        
        def process_old(data):
            metrics.increment("process.old.calls")
            time.sleep(0.01)
            return f"old_{data}"
        
        def process_new(data):
            metrics.increment("process.new.calls")
            time.sleep(0.005)
            return f"new_{data}"
        
        # Feature flagged function
        def process(data):
            if flags.get("use_new_processor"):
                return process_new(data)
            else:
                return process_old(data)
        
        # Test with flag off
        flags.set("use_new_processor", False)
        result1 = process("test1")
        assert result1 == "old_test1"
        
        # Test with flag on
        flags.set("use_new_processor", True)
        result2 = process("test2")
        assert result2 == "new_test2"
        
        summary = metrics.get_summary()
        assert summary["counters"]["process.old.calls"] == 1
        assert summary["counters"]["process.new.calls"] == 1
        print("✅ Feature flags with metrics working!")
        
        # Test 2: Comparison
        print("\n2. Testing comparison framework...")
        
        # Use functions that return same result
        def compute_old():
            time.sleep(0.01)
            return 42
        
        def compute_new():
            time.sleep(0.005)
            return 42
        
        comparison = tester.compare(
            test_name="compute_comparison",
            old_impl=compute_old,
            new_impl=compute_new
        )
        
        assert comparison.passed  # Results are identical
        assert comparison.new_duration < comparison.old_duration  # New is faster
        print(f"✅ Comparison working! Performance improvement: {comparison.performance_delta:.1f}%")
        
        # Test 3: A/B testing
        print("\n3. Testing A/B testing...")
        
        config = ABTestConfig(
            test_name="process_ab",
            control_impl=process_old,
            treatment_impl=process_new,
            traffic_percentage=50.0
        )
        router.register_test(config)
        
        # Track variants
        control_count = 0
        treatment_count = 0
        
        for i in range(100):
            identifier = f"user_{i}"
            if config.should_use_treatment(identifier):
                treatment_count += 1
                metrics.increment("ab.treatment")
            else:
                control_count += 1
                metrics.increment("ab.control")
        
        print(f"Control: {control_count}, Treatment: {treatment_count}")
        assert 30 < control_count < 70  # Should be roughly 50/50
        assert 30 < treatment_count < 70
        print("✅ A/B testing working!")
        
        # Final summary
        print("\n4. Final metrics summary...")
        final_summary = metrics.get_summary()
        print(f"Total operations: {sum(final_summary['counters'].values())}")
        print(f"Metrics collected: {len(final_summary['counters'])} counters")
        
        print("\n✅ All integration tests passed!")

if __name__ == "__main__":
    test_integration()