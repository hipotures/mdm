#!/usr/bin/env python3
"""Quick test to check MDM availability and basic functionality."""

import subprocess
import sys
from pathlib import Path

def check_mdm_available():
    """Check if MDM command is available."""
    result = subprocess.run(["which", "mdm"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ MDM found at: {result.stdout.strip()}")
        return True
    else:
        print("✗ MDM command not found in PATH")
        return False

def check_mdm_runs():
    """Check if MDM command runs."""
    result = subprocess.run(["mdm", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ MDM version: {result.stdout.strip()}")
        return True
    else:
        print(f"✗ MDM command failed: {result.stderr}")
        return False

def check_test_environment():
    """Check test environment setup."""
    test_dir = Path("/tmp/mdm_test_quick")
    test_dir.mkdir(exist_ok=True)
    
    import os
    os.environ["MDM_HOME"] = str(test_dir)
    
    # Try to list datasets
    result = subprocess.run(["mdm", "dataset", "list"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ MDM dataset list works")
        return True
    else:
        print(f"✗ MDM dataset list failed: {result.stderr}")
        return False

def run_single_test():
    """Run a single simple test."""
    print("\nRunning single test...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/e2e/test_01_config/test_11_yaml.py::TestYAMLConfiguration::test_create_yaml_with_custom_settings",
        "-v"
    ], capture_output=True, text=True)
    
    print(f"Return code: {result.returncode}")
    print(f"Output: {result.stdout[:500]}...")
    if result.stderr:
        print(f"Errors: {result.stderr[:500]}...")

if __name__ == "__main__":
    print("Checking MDM test environment...")
    print("=" * 50)
    
    check_mdm_available()
    check_mdm_runs()
    check_test_environment()
    run_single_test()