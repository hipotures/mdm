# MDM Performance Optimization Guide

## CLI Startup Optimization

As of version 0.2.0, MDM includes significant performance optimizations that reduce CLI startup time by up to 65x for simple commands.

### Background

Previously, running `mdm version` took ~6.5 seconds due to eager loading of all modules including heavy dependencies like:
- SQLAlchemy
- Pandas
- NumPy
- DuckDB
- ydata-profiling

### Optimization Techniques

#### 1. Lazy Loading of Subcommands

The main CLI entry point (`src/mdm/cli/main.py`) now uses lazy loading:

```python
# Only import and add subcommands when actually needed
if len(sys.argv) > 1:
    cmd = sys.argv[1]
    
    if cmd == 'dataset':
        from mdm.cli.dataset import dataset_app
        app.add_typer(dataset_app, name="dataset", help="Dataset management commands")
    # ... similar for other subcommands
```

#### 2. Fast Path for Version Command

The version command has a special fast path that bypasses Typer entirely:

```python
def main():
    # Special fast path for version command
    if len(sys.argv) == 2 and sys.argv[1] == 'version':
        from mdm import __version__
        console.print(f"[bold green]MDM[/bold green] version {__version__}")
        sys.exit(0)
```

#### 3. Conditional Logging Setup

Logging setup is skipped for simple commands that don't need it:

```python
@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    # Skip setup for simple commands that don't need it
    if ctx.invoked_subcommand in ['version', None]:
        return
    setup_logging()
```

#### 4. Lazy Module Imports

The main `mdm/__init__.py` uses `__getattr__` for lazy loading:

```python
def __getattr__(name):
    """Lazy import mechanism for heavy modules."""
    if name == "MDMClient":
        from mdm.api import MDMClient
        return MDMClient
    # ... similar for other exports
```

### Performance Results

| Command | Before | After | Improvement |
|---------|--------|-------|-------------|
| `mdm version` | 6.55s | 0.10s | 65x faster |
| `mdm --help` | 6.5s | 0.2s | 32x faster |
| `mdm dataset list` | 6.8s | 2.3s | 3x faster |

### Best Practices for Developers

When adding new features to MDM, follow these guidelines to maintain fast startup:

1. **Defer Heavy Imports**: Only import heavy libraries when actually needed
   ```python
   # Bad
   import pandas as pd  # At module level
   
   # Good
   def process_data():
       import pandas as pd  # Import when needed
   ```

2. **Use Function-Level Imports**: For CLI commands, import dependencies inside the function
   ```python
   @app.command()
   def analyze():
       # Import here, not at module level
       from mdm.analysis import Analyzer
       analyzer = Analyzer()
   ```

3. **Create Lightweight Entry Points**: Keep the main CLI module minimal
   ```python
   # Keep main.py lightweight
   # Move heavy logic to separate modules
   ```

4. **Test Startup Performance**: Regularly check startup time
   ```bash
   time mdm version
   ```

### Implementation Details

The optimization is implemented across several files:

1. **`src/mdm/cli/main.py`**: Main CLI entry point with lazy loading
2. **`src/mdm/__init__.py`**: Package initialization with `__getattr__`
3. Individual subcommand modules remain unchanged

### Debugging Slow Imports

To identify which imports are slow:

```bash
# Show import times
python -X importtime -m mdm version 2>&1 | grep -E "^\s*[0-9]+" | sort -k1 -n | tail -20

# Profile with cProfile
python -m cProfile -s cumulative -m mdm version | head -30
```

### Future Improvements

Potential areas for further optimization:

1. **Lazy Loading in Submodules**: Apply similar techniques to dataset, storage, and feature modules
2. **Import Caching**: Cache imported modules between commands in the same session
3. **Compiled Extensions**: Use Cython for performance-critical paths
4. **Progressive Loading**: Load only necessary parts of large modules

### Backward Compatibility

All optimizations maintain full backward compatibility:
- All CLI commands work exactly as before
- Python API remains unchanged
- No changes to configuration or data formats