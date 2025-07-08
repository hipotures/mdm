#!/usr/bin/env python3
"""Run and analyze a subset of tests to understand issues."""

import subprocess
import sys
from pathlib import Path
import os

# Set test environment
test_home = Path("/tmp/mdm_test_analysis")
test_home.mkdir(exist_ok=True)
os.environ["MDM_HOME"] = str(test_home)

# Test cases to analyze
test_cases = [
    ("tests/e2e/test_01_config/test_11_yaml.py::TestYAMLConfiguration::test_create_yaml_with_custom_settings", "1.1.1"),
    ("tests/e2e/test_01_config/test_11_yaml.py::TestYAMLConfiguration::test_verify_yaml_settings_applied", "1.1.2"),
    ("tests/e2e/test_01_config/test_12_env.py::TestEnvironmentVariables::test_mdm_log_level_debug", "1.2.1"),
    ("tests/e2e/test_02_dataset/test_21_register.py::TestDatasetRegistration::test_register_single_csv", "2.1.1.1"),
]

print("Running subset of tests for analysis...")
print("=" * 80)

results = []

for test_path, test_id in test_cases:
    print(f"\nTest {test_id}: {test_path}")
    print("-" * 40)
    
    # Run test
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-vv", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"Status: {status}")
    
    if result.returncode != 0:
        print("\nError output:")
        print(result.stdout[-1000:])  # Last 1000 chars
        if result.stderr:
            print("\nStderr:")
            print(result.stderr[-500:])
    
    results.append({
        "test_id": test_id,
        "status": status,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    })

# Analyze common issues
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

passed = sum(1 for r in results if r["status"] == "PASSED")
failed = sum(1 for r in results if r["status"] == "FAILED")

print(f"\nTotal: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

# Look for common error patterns
print("\nCommon issues found:")

for result in results:
    if result["status"] == "FAILED":
        output = result["stdout"] + result["stderr"]
        
        if "assert" in output and "exists()" in output:
            print(f"- Test {result['test_id']}: File/directory not created as expected")
        elif "not found" in output and "result.stdout" in output:
            print(f"- Test {result['test_id']}: Expected output not found in command result")
        elif "MDM_" in output and "environ" in output:
            print(f"- Test {result['test_id']}: Environment variable not properly set")
        elif "subprocess.CalledProcessError" in output:
            print(f"- Test {result['test_id']}: MDM command failed to execute")