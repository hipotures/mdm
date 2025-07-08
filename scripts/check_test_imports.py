#!/usr/bin/env python3
"""Check that test files use correct patch imports."""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


def extract_patch_paths(file_path: Path) -> List[Tuple[int, str]]:
    """Extract all patch paths from a test file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    patch_paths = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Str):  # Python 3.7 compatibility
            if 'mdm.' in node.s and '.patch' not in node.s:
                # Check if this looks like a patch path
                if any(pattern in node.s for pattern in ['mdm.dataset.operations', 'mdm.cli.', 'mdm.dataset.manager']):
                    patch_paths.append((node.lineno, node.s))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):  # Python 3.8+
            if 'mdm.' in node.value and '.patch' not in node.value:
                # Check if this looks like a patch path
                if any(pattern in node.value for pattern in ['mdm.dataset.operations', 'mdm.cli.', 'mdm.dataset.manager']):
                    patch_paths.append((node.lineno, node.value))
    
    return patch_paths


def check_patch_imports(file_path: Path) -> List[str]:
    """Check if patch imports are using the correct paths."""
    errors = []
    patch_paths = extract_patch_paths(file_path)
    
    # Known mappings of old paths to new paths
    mappings = {
        'mdm.dataset.operations.BatchOperation': 'mdm.cli.batch.BatchOperation',
        'mdm.dataset.operations.BatchRegisterOperation': 'mdm.cli.batch.BatchRegisterOperation',
        'mdm.dataset.operations.BatchStatsOperation': 'mdm.cli.batch.BatchStatsOperation',
        'mdm.dataset.operations.BatchExportOperation': 'mdm.cli.batch.BatchExportOperation',
        'mdm.dataset.operations.BatchRemoveOperation': 'mdm.cli.batch.BatchRemoveOperation',
        'mdm.dataset.operations.BatchUpdateOperation': 'mdm.cli.batch.BatchUpdateOperation',
        'mdm.dataset.operations.BatchSearchOperation': 'mdm.cli.batch.BatchSearchOperation',
    }
    
    for line_no, path in patch_paths:
        for old_path, new_path in mappings.items():
            if path == old_path:
                errors.append(f"{file_path}:{line_no}: Found old import path '{old_path}', should be '{new_path}'")
    
    return errors


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: check_test_imports.py <file1> [file2] ...")
        sys.exit(1)
    
    all_errors = []
    
    for file_path_str in sys.argv[1:]:
        file_path = Path(file_path_str)
        if file_path.suffix == '.py' and file_path.exists():
            errors = check_patch_imports(file_path)
            all_errors.extend(errors)
    
    if all_errors:
        print("Test import errors found:")
        for error in all_errors:
            print(f"  {error}")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()