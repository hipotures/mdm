#!/usr/bin/env python3
"""Test metrics fixes."""
import sys
sys.path.insert(0, '/home/xai/DEV/mdm-refactor-2025/src')

def test_metrics_export():
    """Test metrics export functionality."""
    print("Testing metrics export...")
    
    from mdm.core.metrics import MetricsCollector
    import tempfile
    from pathlib import Path
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = MetricsCollector(output_dir=Path(tmpdir))
        
        # Add some metrics
        collector.increment("test.counter", 5)
        collector.gauge("test.gauge", 42.0)
        collector.record_time("test.timer", 1.5)
        
        # Export
        filepath = collector.export("test.json")
        
        print(f"Export file: {filepath}")
        print(f"File exists: {filepath.exists()}")
        print(f"File size: {filepath.stat().st_size} bytes")
        
        # Read and verify
        with open(filepath) as f:
            data = json.load(f)
        
        print(f"Data keys: {list(data.keys())}")
        print(f"Counters: {data['summary']['counters']}")
        print("✅ Export test passed!")

def test_decorator():
    """Test decorator with mocking."""
    print("\nTesting decorator...")
    
    from mdm.core.metrics import track_metrics
    from unittest.mock import patch, MagicMock
    
    with patch('mdm.core.metrics.metrics_collector') as mock_collector:
        # Create mock timer
        mock_timer = MagicMock()
        mock_timer.__enter__ = MagicMock(return_value=mock_timer)
        mock_timer.__exit__ = MagicMock(return_value=None)
        mock_collector.timer.return_value = mock_timer
        
        @track_metrics("test.func", implementation="test")
        def test_function(x):
            return x * 2
        
        result = test_function(5)
        
        assert result == 10
        mock_collector.increment.assert_called()
        mock_collector.timer.assert_called_with("test.func.test")
        print("✅ Decorator test passed!")

if __name__ == "__main__":
    test_metrics_export()
    test_decorator()
    print("\n✅ All tests passed!")