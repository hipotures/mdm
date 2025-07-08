#!/usr/bin/env python3
"""Simple test runner that can run a single test by ID."""

import sys
import subprocess
from pathlib import Path


def find_test_by_id(test_id):
    """Find test file and function for given test ID."""
    test_dir = Path(__file__).parent
    
    for test_file in test_dir.rglob("test_*.py"):
        if "__pycache__" in str(test_file):
            continue
            
        with open(test_file) as f:
            content = f.read()
            
        if f'@pytest.mark.mdm_id("{test_id}")' in content:
            # Find the test function
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if f'@pytest.mark.mdm_id("{test_id}")' in line:
                    # Look for function name
                    for j in range(i, min(i + 5, len(lines))):
                        if lines[j].strip().startswith("def test_"):
                            func_name = lines[j].split("(")[0].replace("def ", "").strip()
                            # Find class name if any
                            class_name = None
                            for k in range(i, -1, -1):
                                if lines[k].startswith("class "):
                                    class_name = lines[k].split("(")[0].replace("class ", "").strip()
                                    break
                            
                            return test_file, class_name, func_name
    
    return None, None, None


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_single_test.py <test_id>")
        sys.exit(1)
    
    test_id = sys.argv[1]
    test_file, class_name, func_name = find_test_by_id(test_id)
    
    if not test_file:
        print(f"Test {test_id} not found")
        sys.exit(1)
    
    # Build test path
    if class_name:
        test_path = f"{test_file}::{class_name}::{func_name}"
    else:
        test_path = f"{test_file}::{func_name}"
    
    print(f"Running test {test_id}: {test_path}")
    
    # Run the test
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-v"],
        cwd=Path(__file__).parent.parent.parent  # Run from project root
    )
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()