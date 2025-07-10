#!/usr/bin/env python3
"""Simple metrics test."""
import sys
sys.path.insert(0, '/home/xai/DEV/mdm-refactor-2025/src')

from mdm.core.metrics import MetricsCollector
import tempfile
from pathlib import Path

print("Creating collector...")
with tempfile.TemporaryDirectory() as tmpdir:
    collector = MetricsCollector(output_dir=Path(tmpdir))
    
    print("Adding metrics...")
    collector.increment("test", 1)
    
    print("Getting summary...")
    summary = collector.get_summary()
    print(f"Summary: {summary}")
    
    print("✅ Simple test passed!")