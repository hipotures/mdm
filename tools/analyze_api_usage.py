#!/usr/bin/env python3
"""
Analyze actual API usage across the MDM codebase.

This tool uses AST analysis to find all method calls and attribute access
on target objects (e.g., backend, registrar, generator).
"""

import ast
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys


class APIUsageAnalyzer(ast.NodeVisitor):
    """Find all method calls and attribute access on target objects."""
    
    def __init__(self, target_class: str):
        self.target_class = target_class
        self.target_vars = self._get_common_var_names(target_class)
        self.method_calls = defaultdict(list)
        self.attr_access = defaultdict(list)
        self.current_file = None
        self.current_line = None
        
    def _get_common_var_names(self, class_name: str) -> Set[str]:
        """Get common variable names for a class."""
        base_name = class_name.lower().replace('backend', '').replace('generator', '').replace('registrar', '')
        
        # Common patterns for backend objects
        if 'backend' in class_name.lower():
            return {
                'backend', 'self.backend', 'self._backend',
                'storage_backend', 'self.storage_backend',
                'db_backend', 'self.db_backend',
                'sqlite_backend', 'duckdb_backend', 'postgresql_backend'
            }
        
        # Common patterns for registrar objects
        elif 'registrar' in class_name.lower():
            return {
                'registrar', 'self.registrar', 'self._registrar',
                'dataset_registrar', 'self.dataset_registrar'
            }
            
        # Common patterns for generator objects
        elif 'generator' in class_name.lower():
            return {
                'generator', 'self.generator', 'self._generator',
                'feature_generator', 'self.feature_generator',
                'feat_gen', 'self.feat_gen'
            }
            
        # Generic patterns
        return {
            base_name, f'self.{base_name}', f'self._{base_name}',
            class_name.lower(), f'self.{class_name.lower()}'
        }
    
    def visit_Call(self, node):
        """Track method calls."""
        if isinstance(node.func, ast.Attribute):
            # Check if it's a method call on our target object
            obj_name = self._get_object_name(node.func.value)
            if obj_name in self.target_vars:
                method_name = node.func.attr
                location = f"{self.current_file}:{node.lineno}"
                
                # Extract call details
                call_info = {
                    'file': self.current_file,
                    'line': node.lineno,
                    'object': obj_name,
                    'args_count': len(node.args),
                    'has_kwargs': bool(node.keywords)
                }
                
                self.method_calls[method_name].append(call_info)
                
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Track attribute access."""
        # Only track if not part of a call (those are handled by visit_Call)
        if not isinstance(node.ctx, ast.Store):
            obj_name = self._get_object_name(node.value)
            if obj_name in self.target_vars:
                attr_name = node.attr
                
                # Skip if this is part of a method call
                parent = getattr(node, '_parent', None)
                if not isinstance(parent, ast.Call):
                    location = f"{self.current_file}:{node.lineno}"
                    
                    attr_info = {
                        'file': self.current_file,
                        'line': node.lineno,
                        'object': obj_name,
                        'context': type(node.ctx).__name__
                    }
                    
                    self.attr_access[attr_name].append(attr_info)
                    
        self.generic_visit(node)
    
    def _get_object_name(self, node) -> str:
        """Extract object name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Handle self.backend, self._backend etc.
            base = self._get_object_name(node.value)
            if base:
                return f"{base}.{node.attr}"
        return ""
    
    def analyze_file(self, filepath: Path):
        """Analyze a single Python file."""
        # Try to make path relative, but use absolute if that fails
        try:
            self.current_file = str(filepath.relative_to(Path.cwd()))
        except ValueError:
            self.current_file = str(filepath)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(filepath))
            
            # Add parent references for context
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child._parent = node
                    
            self.visit(tree)
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}", file=sys.stderr)
    
    def analyze_codebase(self, root_path: Path) -> Dict:
        """Analyze all Python files in the codebase."""
        py_files = list(root_path.rglob("*.py"))
        
        # Filter out test files and __pycache__
        py_files = [
            f for f in py_files 
            if '__pycache__' not in str(f) and 'test_' not in f.name
        ]
        
        print(f"Analyzing {len(py_files)} Python files for {self.target_class} usage...")
        
        for py_file in py_files:
            self.analyze_file(py_file)
            
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate usage report."""
        # Sort methods by usage count
        method_summary = {}
        for method, calls in self.method_calls.items():
            method_summary[method] = {
                'count': len(calls),
                'locations': calls[:10]  # First 10 locations
            }
            
        attr_summary = {}
        for attr, accesses in self.attr_access.items():
            attr_summary[attr] = {
                'count': len(accesses),
                'locations': accesses[:10]
            }
            
        # Sort by usage count
        sorted_methods = dict(sorted(
            method_summary.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        ))
        
        sorted_attrs = dict(sorted(
            attr_summary.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ))
        
        return {
            'target_class': self.target_class,
            'total_method_calls': sum(m['count'] for m in method_summary.values()),
            'unique_methods': len(method_summary),
            'total_attr_access': sum(a['count'] for a in attr_summary.values()),
            'unique_attributes': len(attr_summary),
            'methods': sorted_methods,
            'attributes': sorted_attrs
        }


def main():
    parser = argparse.ArgumentParser(description='Analyze API usage in MDM codebase')
    parser.add_argument(
        '--target-class',
        required=True,
        help='Target class to analyze (e.g., StorageBackend, FeatureGenerator, DatasetRegistrar)'
    )
    parser.add_argument(
        '--scan-dir',
        default='src/mdm',
        help='Directory to scan (default: src/mdm)'
    )
    parser.add_argument(
        '--output',
        help='Output file for JSON report (default: print to stdout)'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'markdown', 'summary'],
        default='summary',
        help='Output format'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = APIUsageAnalyzer(args.target_class)
    
    # Run analysis
    scan_path = Path(args.scan_dir)
    if not scan_path.exists():
        print(f"Error: Directory {scan_path} does not exist", file=sys.stderr)
        sys.exit(1)
        
    report = analyzer.analyze_codebase(scan_path)
    
    # Output results
    if args.format == 'json':
        output = json.dumps(report, indent=2)
    elif args.format == 'markdown':
        output = generate_markdown_report(report)
    else:  # summary
        output = generate_summary_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)


def generate_summary_report(report: Dict) -> str:
    """Generate human-readable summary report."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"API Usage Analysis: {report['target_class']}")
    lines.append(f"{'='*60}")
    lines.append(f"\nSummary:")
    lines.append(f"  Total method calls: {report['total_method_calls']}")
    lines.append(f"  Unique methods: {report['unique_methods']}")
    lines.append(f"  Total attribute access: {report['total_attr_access']}")
    lines.append(f"  Unique attributes: {report['unique_attributes']}")
    
    lines.append(f"\nTop Methods by Usage:")
    lines.append(f"{'Method':<40} {'Count':>10}")
    lines.append(f"{'-'*40} {'-'*10}")
    
    for method, info in list(report['methods'].items())[:20]:
        lines.append(f"{method:<40} {info['count']:>10}")
    
    if report['attributes']:
        lines.append(f"\nTop Attributes by Usage:")
        lines.append(f"{'Attribute':<40} {'Count':>10}")
        lines.append(f"{'-'*40} {'-'*10}")
        
        for attr, info in list(report['attributes'].items())[:10]:
            lines.append(f"{attr:<40} {info['count']:>10}")
    
    return '\n'.join(lines)


def generate_markdown_report(report: Dict) -> str:
    """Generate markdown report."""
    lines = []
    lines.append(f"# API Usage Report: {report['target_class']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total method calls**: {report['total_method_calls']}")
    lines.append(f"- **Unique methods**: {report['unique_methods']}")
    lines.append(f"- **Total attribute access**: {report['total_attr_access']}")
    lines.append(f"- **Unique attributes**: {report['unique_attributes']}")
    lines.append("")
    
    lines.append("## Methods by Usage")
    lines.append("")
    lines.append("| Method | Count | Example Locations |")
    lines.append("|--------|-------|------------------|")
    
    for method, info in report['methods'].items():
        locations = ', '.join([loc['file'] for loc in info['locations'][:3]])
        lines.append(f"| `{method}()` | {info['count']} | {locations} |")
    
    if report['attributes']:
        lines.append("")
        lines.append("## Attributes by Usage")
        lines.append("")
        lines.append("| Attribute | Count | Example Locations |")
        lines.append("|-----------|-------|------------------|")
        
        for attr, info in report['attributes'].items():
            locations = ', '.join([loc['file'] for loc in info['locations'][:3]])
            lines.append(f"| `{attr}` | {info['count']} | {locations} |")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    main()