#!/usr/bin/env python3
"""Debug test isolation issue."""

import os
import subprocess
import sys
from pathlib import Path

# Test 1: Direct config test
print("=== Test 1: Direct config test ===")
os.environ.clear()
os.environ["MDM_HOME_DIR"] = "/tmp/mdm_debug_test"

from mdm.core.config import reset_config, get_config
reset_config()
config = get_config()
print(f"Config home_dir: {config.home_dir}")
print(f"Config datasets_dir: {config.datasets_dir}")

# Test 2: Run MDM list in subprocess with env
print("\n=== Test 2: MDM list with MDM_HOME_DIR ===")
env = os.environ.copy()
env["MDM_HOME_DIR"] = "/tmp/mdm_debug_test2"
result = subprocess.run(
    [sys.executable, "-m", "mdm.cli.main", "dataset", "list"],
    capture_output=True,
    text=True,
    env=env
)
print(f"Return code: {result.returncode}")
print(f"Output preview: {result.stdout[:200]}")

# Test 3: Check where datasets are
print("\n=== Test 3: Where are test_yaml datasets? ===")
# Check various locations
locations = [
    Path.home() / ".mdm/datasets",
    Path.home() / ".mdm/config/datasets",
    Path("/tmp"),
]

for loc in locations:
    if loc.exists():
        # Look for test_yaml
        for item in loc.rglob("*test_yaml*"):
            print(f"Found: {item}")
            
# Test 4: List datasets with fresh config
print("\n=== Test 4: Fresh config dataset list ===")
os.environ["MDM_HOME_DIR"] = "/tmp/mdm_fresh_test"
reset_config()

# Create the directories
test_dir = Path("/tmp/mdm_fresh_test")
test_dir.mkdir(exist_ok=True)
(test_dir / "datasets").mkdir(exist_ok=True)
(test_dir / "config" / "datasets").mkdir(parents=True, exist_ok=True)

result = subprocess.run(
    [sys.executable, "-m", "mdm.cli.main", "dataset", "list"],
    capture_output=True,
    text=True,
    env={"MDM_HOME_DIR": "/tmp/mdm_fresh_test"}
)
print(f"Fresh list output: {result.stdout[:200]}")