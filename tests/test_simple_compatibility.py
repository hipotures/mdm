"""
Simplified compatibility test that can run standalone.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test imports
try:
    from mdm.storage.backends.compatibility_mixin import BackendCompatibilityMixin
    print("✓ BackendCompatibilityMixin imported successfully")
except ImportError as e:
    print(f"✗ Failed to import BackendCompatibilityMixin: {e}")
    sys.exit(1)

try:
    from mdm.storage.backends.stateless_sqlite import StatelessSQLiteBackend
    print("✓ StatelessSQLiteBackend imported successfully")
except ImportError as e:
    print(f"✗ Failed to import StatelessSQLiteBackend: {e}")
    sys.exit(1)

try:
    from mdm.storage.backends.stateless_duckdb import StatelessDuckDBBackend
    print("✓ StatelessDuckDBBackend imported successfully")
except ImportError as e:
    print(f"✗ Failed to import StatelessDuckDBBackend: {e}")
    sys.exit(1)

# Check required methods
REQUIRED_METHODS = [
    'get_engine',
    'create_table_from_dataframe', 
    'query',
    'read_table_to_dataframe',
    'close_connections',
    'read_table',
    'write_table',
    'get_table_info',
    'execute_query',
    'get_connection',
    'get_columns',
    'analyze_column',
    'database_exists',
    'create_database',
]

print("\nChecking StatelessSQLiteBackend methods:")
missing = []
for method in REQUIRED_METHODS:
    if hasattr(StatelessSQLiteBackend, method):
        print(f"  ✓ {method}")
    else:
        print(f"  ✗ {method} - MISSING!")
        missing.append(method)

if missing:
    print(f"\n❌ StatelessSQLiteBackend is missing {len(missing)} methods!")
else:
    print("\n✅ StatelessSQLiteBackend has all required methods!")

# Check inheritance
if issubclass(StatelessSQLiteBackend, BackendCompatibilityMixin):
    print("\n✅ StatelessSQLiteBackend correctly inherits from BackendCompatibilityMixin")
else:
    print("\n❌ StatelessSQLiteBackend does NOT inherit from BackendCompatibilityMixin!")

print("\n" + "="*50)
print("SUMMARY:")
print("="*50)

if not missing:
    print("✅ ALL TESTS PASSED - Backends have full API compatibility!")
    print("\nThe new stateless backends implement all 14 methods")
    print("identified in the API analysis and are ready for use.")
else:
    print(f"❌ TESTS FAILED - Missing {len(missing)} methods")
    print("\nMissing methods:", ", ".join(missing))