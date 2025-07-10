#!/usr/bin/env python3
"""
Analyze actual backend API usage in MDM codebase.
This tool should have been created BEFORE refactoring!
"""

import ast
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple
import json

class BackendAPIAnalyzer(ast.NodeVisitor):
    """AST visitor to find all backend method/attribute usage"""
    
    def __init__(self):
        self.method_calls = defaultdict(list)  # method -> [(file, line)]
        self.attribute_access = defaultdict(list)  # attr -> [(file, line)]
        self.current_file = None
        
        # Common backend variable names
        self.backend_names = {
            'backend', 'self.backend', 'self._backend', 
            'storage_backend', 'self.storage_backend',
            'self._storage_backend', 'db_backend'
        }
    
    def visit_Call(self, node):
        """Track method calls like backend.query()"""
        if isinstance(node.func, ast.Attribute):
            obj_name = self._get_object_name(node.func.value)
            if obj_name in self.backend_names:
                method = node.func.attr
                self.method_calls[method].append((self.current_file, node.lineno))
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Track attribute access like backend.engine"""
        obj_name = self._get_object_name(node.value)
        if obj_name in self.backend_names:
            attr = node.attr
            # Don't count method calls as attribute access
            if not isinstance(node.ctx, ast.Load) or attr not in self.method_calls:
                self.attribute_access[attr].append((self.current_file, node.lineno))
        
        self.generic_visit(node)
    
    def _get_object_name(self, node) -> str:
        """Extract object name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Handle self.backend
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
        return ""
    
    def analyze_file(self, filepath: Path):
        """Analyze a single Python file"""
        self.current_file = str(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(filepath))
                self.visit(tree)
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
    
    def analyze_directory(self, root_dir: Path):
        """Analyze all Python files in directory"""
        for filepath in root_dir.rglob("*.py"):
            # Skip test files and migrations
            if 'test' in filepath.parts or '__pycache__' in str(filepath):
                continue
            self.analyze_file(filepath)
    
    def get_summary(self) -> Dict[str, any]:
        """Get analysis summary"""
        return {
            "total_methods": len(self.method_calls),
            "total_attributes": len(self.attribute_access),
            "method_usage": {
                method: len(locations) 
                for method, locations in self.method_calls.items()
            },
            "attribute_usage": {
                attr: len(locations)
                for attr, locations in self.attribute_access.items()
            }
        }
    
    def print_report(self):
        """Print detailed usage report"""
        print("=" * 80)
        print("STORAGE BACKEND API USAGE ANALYSIS")
        print("=" * 80)
        
        # Method calls
        print("\n📊 METHOD CALLS (sorted by frequency):")
        print("-" * 50)
        sorted_methods = sorted(
            self.method_calls.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        for method, locations in sorted_methods:
            print(f"\n{method}() - {len(locations)} calls")
            # Show first 3 locations as examples
            for file, line in locations[:3]:
                short_file = file.replace(str(Path.cwd()), ".")
                print(f"  {short_file}:{line}")
            if len(locations) > 3:
                print(f"  ... and {len(locations) - 3} more")
        
        # Attributes
        print("\n\n📊 ATTRIBUTE ACCESS (sorted by frequency):")
        print("-" * 50)
        sorted_attrs = sorted(
            self.attribute_access.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for attr, locations in sorted_attrs:
            print(f"\n{attr} - {len(locations)} accesses")
            for file, line in locations[:3]:
                short_file = file.replace(str(Path.cwd()), ".")
                print(f"  {short_file}:{line}")
            if len(locations) > 3:
                print(f"  ... and {len(locations) - 3} more")
        
        # Summary
        print("\n\n📊 SUMMARY:")
        print("-" * 50)
        print(f"Total unique methods called: {len(self.method_calls)}")
        print(f"Total unique attributes accessed: {len(self.attribute_access)}")
        print(f"Total method calls: {sum(len(locs) for locs in self.method_calls.values())}")
        print(f"Total attribute accesses: {sum(len(locs) for locs in self.attribute_access.values())}")
    
    def generate_interface(self) -> str:
        """Generate Protocol interface based on usage"""
        interface = '''"""Auto-generated interface based on actual usage"""
from typing import Protocol, Any, Dict, List, Optional
import pandas as pd
from sqlalchemy.engine import Engine

class IStorageBackendComplete(Protocol):
    """Complete storage backend interface based on actual usage"""
    
'''
        # Add methods
        for method in sorted(self.method_calls.keys()):
            interface += f"    def {method}(self, *args, **kwargs) -> Any:\n"
            interface += f'        """Used {len(self.method_calls[method])} times in codebase"""\n'
            interface += "        ...\n\n"
        
        # Add properties/attributes
        if self.attribute_access:
            interface += "    # Attributes accessed:\n"
            for attr in sorted(self.attribute_access.keys()):
                interface += f"    # {attr} - accessed {len(self.attribute_access[attr])} times\n"
        
        return interface
    
    def save_results(self, output_file: str):
        """Save analysis results to JSON"""
        results = {
            "methods": {
                method: {
                    "count": len(locations),
                    "locations": [{"file": f, "line": l} for f, l in locations]
                }
                for method, locations in self.method_calls.items()
            },
            "attributes": {
                attr: {
                    "count": len(locations),
                    "locations": [{"file": f, "line": l} for f, l in locations]
                }
                for attr, locations in self.attribute_access.items()
            },
            "summary": self.get_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_backend_api_usage.py <directory>")
        print("Example: python analyze_backend_api_usage.py src/mdm")
        sys.exit(1)
    
    target_dir = Path(sys.argv[1])
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist")
        sys.exit(1)
    
    print(f"Analyzing backend API usage in: {target_dir}")
    
    analyzer = BackendAPIAnalyzer()
    analyzer.analyze_directory(target_dir)
    analyzer.print_report()
    
    # Generate interface
    print("\n\n📝 GENERATED INTERFACE:")
    print("-" * 80)
    interface_code = analyzer.generate_interface()
    print(interface_code)
    
    # Save interface to file
    interface_file = "generated_storage_interface.py"
    with open(interface_file, 'w') as f:
        f.write(interface_code)
    print(f"\nInterface saved to: {interface_file}")
    
    # Save detailed results
    analyzer.save_results("backend_api_usage.json")
    
    print("\n⚠️  IMPORTANT: This analysis should be done BEFORE refactoring!")
    print("The interface should include ALL methods found above.")


if __name__ == "__main__":
    main()