"""Error categorization and analysis utilities."""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ErrorPattern:
    """Pattern for matching and categorizing errors."""
    name: str
    patterns: List[str]
    category: str
    priority: int = 0
    
    def matches(self, error_text: str) -> bool:
        """Check if error text matches any pattern."""
        for pattern in self.patterns:
            if re.search(pattern, error_text, re.IGNORECASE):
                return True
        return False


# Common error patterns for categorization
ERROR_PATTERNS = [
    # File/Path errors
    ErrorPattern(
        name="file_not_found",
        patterns=[
            r"FileNotFoundError",
            r"No such file or directory",
            r"does not exist",
            r"assert.*\.exists\(\)",
            r"Path.*not found"
        ],
        category="file-system",
        priority=10
    ),
    
    # Import errors
    ErrorPattern(
        name="import_error",
        patterns=[
            r"ModuleNotFoundError",
            r"ImportError",
            r"No module named",
            r"cannot import name"
        ],
        category="import",
        priority=10
    ),
    
    # Type errors
    ErrorPattern(
        name="type_error",
        patterns=[
            r"TypeError",
            r"type.*expected.*got",
            r"argument.*must be",
            r"unsupported operand type"
        ],
        category="type",
        priority=8
    ),
    
    # Attribute errors
    ErrorPattern(
        name="attribute_error",
        patterns=[
            r"AttributeError",
            r"has no attribute",
            r"object has no attribute",
            r"Mock.*has no attribute"
        ],
        category="attribute",
        priority=8
    ),
    
    # Key errors
    ErrorPattern(
        name="key_error",
        patterns=[
            r"KeyError",
            r"key.*not found",
            r"missing.*key"
        ],
        category="key",
        priority=7
    ),
    
    # Value errors
    ErrorPattern(
        name="value_error",
        patterns=[
            r"ValueError",
            r"invalid.*value",
            r"expected.*got"
        ],
        category="value",
        priority=7
    ),
    
    # Mock/Test setup errors
    ErrorPattern(
        name="mock_error",
        patterns=[
            r"Mock.*called.*times",
            r"Expected.*calls.*Actual",
            r"assert.*called",
            r"mock.*not.*called"
        ],
        category="mock",
        priority=6
    ),
    
    # Output mismatch
    ErrorPattern(
        name="output_mismatch",
        patterns=[
            r"assert.*in.*stdout",
            r"not in result\.stdout",
            r"expected output",
            r"assertion.*failed.*output"
        ],
        category="output",
        priority=5
    ),
    
    # Command execution errors
    ErrorPattern(
        name="command_error",
        patterns=[
            r"CalledProcessError",
            r"Command.*failed",
            r"returned non-zero exit status",
            r"subprocess.*error"
        ],
        category="command",
        priority=9
    ),
    
    # Database errors
    ErrorPattern(
        name="database_error",
        patterns=[
            r"DatabaseError",
            r"OperationalError",
            r"sqlite3\.Error",
            r"database.*locked",
            r"no such table"
        ],
        category="database",
        priority=9
    ),
    
    # Configuration errors
    ErrorPattern(
        name="config_error",
        patterns=[
            r"ConfigError",
            r"configuration.*error",
            r"invalid.*config",
            r"missing.*configuration"
        ],
        category="config",
        priority=8
    ),
    
    # Generic assertion errors
    ErrorPattern(
        name="assertion_error",
        patterns=[
            r"AssertionError",
            r"assert.*==",
            r"assertion failed"
        ],
        category="assertion",
        priority=3
    )
]


class ErrorAnalyzer:
    """Analyzes and categorizes test errors."""
    
    def __init__(self, patterns: Optional[List[ErrorPattern]] = None):
        """Initialize with error patterns."""
        self.patterns = patterns or ERROR_PATTERNS
        # Sort by priority (higher priority first)
        self.patterns.sort(key=lambda p: p.priority, reverse=True)
    
    def categorize_error(self, error_text: str) -> Tuple[str, str]:
        """Categorize error and return (error_type, category)."""
        for pattern in self.patterns:
            if pattern.matches(error_text):
                return pattern.name, pattern.category
        
        # Default categorization based on common error types
        if "Error" in error_text:
            # Extract error type from text
            match = re.search(r'(\w+Error)', error_text)
            if match:
                error_type = match.group(1)
                return error_type.lower(), "other"
        
        return "unknown", "other"
    
    def analyze_failures(self, failures: List[Dict]) -> Dict[str, List[Dict]]:
        """Analyze and group failures by error category."""
        grouped = defaultdict(list)
        
        for failure in failures:
            error_text = failure.get("error_message", "") + " " + failure.get("error_info", "")
            error_type, category = self.categorize_error(error_text)
            
            failure["error_type"] = error_type
            failure["error_category"] = category
            
            grouped[category].append(failure)
        
        return dict(grouped)
    
    def get_category_summary(self, failures: List[Dict]) -> Dict[str, int]:
        """Get count of failures by category."""
        grouped = self.analyze_failures(failures)
        return {category: len(items) for category, items in grouped.items()}
    
    def suggest_fixes(self, error_type: str, error_text: str) -> List[str]:
        """Suggest potential fixes based on error type and text."""
        suggestions = []
        
        # File not found errors
        if error_type == "file_not_found":
            suggestions.extend([
                "Check if test fixtures are properly created in setup",
                "Verify path construction is correct",
                "Ensure test data files are included in repository",
                "Check if temporary directories are created before use"
            ])
        
        # Import errors
        elif error_type == "import_error":
            module_match = re.search(r"No module named ['\"](\w+)['\"]", error_text)
            if module_match:
                module = module_match.group(1)
                suggestions.append(f"Install missing module: uv pip install {module}")
            suggestions.extend([
                "Check if module is in requirements.txt",
                "Verify PYTHONPATH is set correctly",
                "Ensure __init__.py files exist in package directories"
            ])
        
        # Type errors
        elif error_type == "type_error":
            suggestions.extend([
                "Check function signatures match expected types",
                "Verify mock return values have correct types",
                "Ensure proper type conversions are applied",
                "Check if None is being passed where object expected"
            ])
        
        # Attribute errors
        elif error_type == "attribute_error":
            suggestions.extend([
                "Verify mock objects have all required attributes",
                "Check if API has changed recently",
                "Ensure objects are properly initialized",
                "Add missing attributes to mock specifications"
            ])
        
        # Mock errors
        elif error_type == "mock_error":
            suggestions.extend([
                "Verify mock is called with correct arguments",
                "Check call count expectations match actual usage",
                "Ensure mock is properly reset between tests",
                "Review test setup and teardown methods"
            ])
        
        # Output mismatch
        elif error_type == "output_mismatch":
            suggestions.extend([
                "Update expected output to match current behavior",
                "Check if output format has changed",
                "Verify string matching is case-sensitive if needed",
                "Consider using regex for flexible matching"
            ])
        
        # Database errors
        elif error_type == "database_error":
            suggestions.extend([
                "Ensure database is properly initialized",
                "Check database permissions",
                "Verify table creation scripts are run",
                "Consider using in-memory database for tests"
            ])
        
        # Command errors
        elif error_type == "command_error":
            suggestions.extend([
                "Verify command is available in PATH",
                "Check command syntax and arguments",
                "Ensure required environment variables are set",
                "Consider mocking external command calls"
            ])
        
        # Generic suggestions
        suggestions.extend([
            "Review recent changes to affected code",
            "Check test isolation and dependencies",
            "Verify test environment matches requirements"
        ])
        
        return suggestions


def group_failures_by_pattern(failures: List[Dict]) -> Dict[str, List[Dict]]:
    """Group failures by common patterns for issue creation."""
    analyzer = ErrorAnalyzer()
    
    # First group by error category
    categorized = analyzer.analyze_failures(failures)
    
    # Then create logical groups for issue creation
    groups = {}
    
    # Group file system errors together
    if "file-system" in categorized:
        groups["file-system-errors"] = {
            "title": "File System Related Test Failures",
            "failures": categorized["file-system"],
            "description": "Tests failing due to missing files or directories"
        }
    
    # Group import errors together
    if "import" in categorized:
        groups["import-errors"] = {
            "title": "Import and Module Related Test Failures",
            "failures": categorized["import"],
            "description": "Tests failing due to import or module issues"
        }
    
    # Group mock errors together
    if "mock" in categorized:
        groups["mock-errors"] = {
            "title": "Mock Configuration Test Failures",
            "failures": categorized["mock"],
            "description": "Tests failing due to incorrect mock setup"
        }
    
    # Group type-related errors (type, attribute, key, value)
    type_related = []
    for cat in ["type", "attribute", "key", "value"]:
        if cat in categorized:
            type_related.extend(categorized[cat])
    
    if type_related:
        groups["type-errors"] = {
            "title": "Type and Attribute Related Test Failures",
            "failures": type_related,
            "description": "Tests failing due to type mismatches or missing attributes"
        }
    
    # Group output and assertion errors
    output_related = []
    for cat in ["output", "assertion"]:
        if cat in categorized:
            output_related.extend(categorized[cat])
    
    if output_related:
        groups["output-errors"] = {
            "title": "Output and Assertion Test Failures",
            "failures": output_related,
            "description": "Tests failing due to output mismatches or assertion errors"
        }
    
    # Group database errors
    if "database" in categorized:
        groups["database-errors"] = {
            "title": "Database Related Test Failures",
            "failures": categorized["database"],
            "description": "Tests failing due to database issues"
        }
    
    # Group remaining errors
    other_errors = []
    for cat in ["command", "config", "other"]:
        if cat in categorized:
            other_errors.extend(categorized[cat])
    
    if other_errors:
        groups["other-errors"] = {
            "title": "Other Test Failures",
            "failures": other_errors,
            "description": "Miscellaneous test failures"
        }
    
    return groups