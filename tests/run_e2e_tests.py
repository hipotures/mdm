#!/usr/bin/env python3
"""
Main entry point for running MDM end-to-end tests.

Examples:
    # Run all tests
    python tests/run_e2e_tests.py
    
    # Run specific test by ID
    python tests/run_e2e_tests.py 1.1.1
    
    # Run category
    python tests/run_e2e_tests.py 1.1
    
    # Run top-level group
    python tests/run_e2e_tests.py 1
    
    # Generate report
    python tests/run_e2e_tests.py --output report.md
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.e2e.runner import main

if __name__ == "__main__":
    main()