#!/usr/bin/env python3
"""Generate a summary of implemented E2E tests."""

from pathlib import Path
import re


def scan_tests(test_dir: Path):
    """Scan test directory and extract test information."""
    tests = {}
    
    for test_file in test_dir.rglob("test_*.py"):
        if "__pycache__" in str(test_file):
            continue
            
        with open(test_file) as f:
            content = f.read()
            
        # Extract test IDs and descriptions
        for match in re.finditer(r'@pytest\.mark\.mdm_id\("([^"]+)"\)\s*\n\s*def\s+(\w+)\(.*?\):\s*\n\s*"""([^"]+)"""', content, re.MULTILINE):
            test_id = match.group(1)
            test_name = match.group(2)
            description = match.group(3)
            
            category = test_id.split('.')[0]
            if category not in tests:
                tests[category] = []
            
            tests[category].append({
                'id': test_id,
                'name': test_name,
                'description': description,
                'file': test_file.relative_to(test_dir)
            })
    
    return tests


def generate_summary():
    """Generate summary of implemented tests."""
    test_dir = Path(__file__).parent
    tests = scan_tests(test_dir)
    
    print("# MDM E2E Tests Implementation Summary\n")
    print("## Implemented Tests by Category\n")
    
    total_tests = 0
    
    for category in sorted(tests.keys()):
        category_tests = tests[category]
        total_tests += len(category_tests)
        
        if category == '1':
            print("### 1. Configuration System")
        elif category == '2':
            print("### 2. Dataset Operations")
        
        print(f"\n**Implemented:** {len(category_tests)} tests\n")
        
        for test in sorted(category_tests, key=lambda x: x['id']):
            print(f"- **{test['id']}**: {test['description']}")
            print(f"  - File: `{test['file']}`")
            print(f"  - Function: `{test['name']}`")
        print()
    
    print(f"\n## Total Implemented Tests: {total_tests}")
    
    print("\n## Test Structure")
    print("```")
    print("tests/e2e/")
    print("├── conftest.py              # Shared fixtures")
    print("├── runner.py                # Test runner with hierarchical selection")
    print("├── test_01_config/          # Configuration tests")
    print("│   ├── test_11_yaml.py     # YAML configuration (6 tests)")
    print("│   └── test_12_env.py      # Environment variables (10 tests)")
    print("└── test_02_dataset/         # Dataset operation tests")
    print("    └── test_21_register.py # Dataset registration (20 tests)")
    print("```")
    
    print("\n## Running Tests Examples")
    print("```bash")
    print("# Run all tests")
    print("python tests/run_e2e_tests.py")
    print()
    print("# Run specific test")
    print("python tests/run_e2e_tests.py 1.1.1")
    print()
    print("# Run category")
    print("python tests/run_e2e_tests.py 1.1")
    print()
    print("# Generate report")
    print("python tests/run_e2e_tests.py --output report.md")
    print("```")
    
    print("\n## Notes")
    print("- Tests run in isolated environment in `/tmp`")
    print("- Each test gets a clean MDM installation")
    print("- Tests marked with `@pytest.mark.skip` are for unimplemented features")
    print("- Use `-v` flag for verbose output")


if __name__ == "__main__":
    generate_summary()