# Contributing to MDM

Thank you for your interest in contributing to MDM! This guide will help you get started.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style](#code-style)
4. [Testing Guidelines](#testing-guidelines)
5. [Submitting Changes](#submitting-changes)
6. [Architecture Overview](#architecture-overview)
7. [Common Tasks](#common-tasks)

## Getting Started

### Prerequisites
- Python 3.9 or higher
- uv (recommended) or pip
- Git
- SQLite3 (included with Python)

### Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/mdm.git
cd mdm
git remote add upstream https://github.com/ORIGINAL_OWNER/mdm.git
```

## Development Setup

### Using uv (Recommended)
```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"

# Generate lock file if missing
uv lock
```

### Using pip
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Verify Installation
```bash
# Run tests
./scripts/run_tests.sh --unit-only

# Check CLI
mdm --version
```

## Code Style

### Formatting
We use Black for code formatting with a line length of 100:
```bash
black src/ tests/ --line-length 100
```

### Linting
We use Ruff for linting:
```bash
ruff check src/
```

### Type Checking
We use mypy for type checking:
```bash
mypy src/mdm
```

### Pre-commit Checks
Run all checks before committing:
```bash
# Format
black src/ tests/ --line-length 100

# Lint
ruff check src/

# Type check
mypy src/mdm

# Check test imports
./scripts/check_test_imports.py

# Run tests
./scripts/run_tests.sh
```

### Code Style Guidelines

#### Imports
```python
# Standard library imports first
import os
from pathlib import Path

# Third-party imports
import pandas as pd
from rich.console import Console

# Local imports
from mdm.config import MDMConfig
from mdm.core.exceptions import DatasetError
```

#### Type Hints
Always use type hints for public functions:
```python
def register_dataset(
    name: str,
    path: Union[str, Path],
    target_column: Optional[str] = None
) -> DatasetInfo:
    """Register a new dataset."""
    pass
```

#### Docstrings
Use Google style docstrings:
```python
def process_data(df: pd.DataFrame, batch_size: int = 10000) -> pd.DataFrame:
    """Process DataFrame in batches.
    
    Args:
        df: Input DataFrame to process
        batch_size: Size of each processing batch
        
    Returns:
        Processed DataFrame with additional features
        
    Raises:
        ValueError: If DataFrame is empty
        MemoryError: If batch_size is too large
    """
    pass
```

#### Error Handling
Use specific exceptions and helpful messages:
```python
# Good
if not path.exists():
    raise DatasetError(f"Dataset path does not exist: {path}")

# Bad
if not path.exists():
    raise Exception("Error")
```

## Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Unit tests (fast, isolated)
├── integration/    # Integration tests (test components together)
├── e2e/           # End-to-end tests (full workflows)
└── fixtures/      # Test data and fixtures
```

### Writing Tests

#### Unit Tests
```python
# tests/unit/dataset/test_manager.py
import pytest
from unittest.mock import Mock, patch

class TestDatasetManager:
    @pytest.fixture
    def manager(self):
        """Create manager with mocked config."""
        with patch('mdm.config.get_config_manager') as mock:
            mock.return_value.config = Mock()
            return DatasetManager()
    
    def test_list_datasets_empty(self, manager):
        """Test listing when no datasets exist."""
        with patch.object(manager, '_load_dataset_configs') as mock:
            mock.return_value = []
            
            result = manager.list_datasets()
            
            assert result == []
            mock.assert_called_once()
```

#### Integration Tests
```python
# tests/integration/test_dataset_registration.py
def test_register_and_load(tmp_path):
    """Test full registration and loading flow."""
    # Create test data
    data_file = tmp_path / "data.csv"
    df = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
    df.to_csv(data_file, index=False)
    
    # Register dataset
    client = MDMClient()
    info = client.register_dataset("test", data_file)
    
    # Load and verify
    loaded_df = client.load_dataset("test")
    assert len(loaded_df) == 2
```

### Test Best Practices

1. **Use tmp_path for file operations**
   ```python
   def test_file_operation(tmp_path):
       test_file = tmp_path / "test.csv"
       # Use test_file, not real filesystem
   ```

2. **Mock external dependencies**
   ```python
   @patch('mdm.storage.sqlite.create_engine')
   def test_database_operation(mock_engine):
       # Test without real database
   ```

3. **Test error cases**
   ```python
   def test_invalid_input():
       with pytest.raises(ValueError, match="Invalid dataset name"):
           register_dataset("", "path.csv")
   ```

4. **Use descriptive test names**
   ```python
   # Good
   def test_register_dataset_with_duplicate_name_raises_error():
   
   # Bad
   def test_register():
   ```

### Running Tests

```bash
# All tests
./scripts/run_tests.sh

# Specific test suite
./scripts/run_tests.sh --unit-only
./scripts/run_tests.sh --integration-only
./scripts/run_tests.sh --e2e-only

# With coverage
./scripts/run_tests.sh --coverage

# Specific test file
pytest tests/unit/test_config.py -v

# Specific test
pytest tests/unit/test_config.py::test_function_name -v

# With debugging
pytest tests/unit/test_config.py -v -s --pdb
```

## Submitting Changes

### Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Write code
   - Add tests
   - Update documentation

3. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "feat: add support for PostgreSQL backend
   
   - Implement PostgreSQLBackend class
   - Add configuration options
   - Include integration tests"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format
Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code refactoring
- `style:` Code style changes
- `perf:` Performance improvements
- `chore:` Build/tooling changes

### Pull Request Guidelines

1. **Clear description**: Explain what and why
2. **Link issues**: Reference any related issues
3. **Include tests**: All new features need tests
4. **Update docs**: Document new features
5. **Pass CI**: Ensure all checks pass

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Added new tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

## Architecture Overview

### Key Principles
1. **Single Backend**: All datasets use same storage backend
2. **Decentralized**: Each dataset is self-contained
3. **Batch Processing**: Handle large datasets efficiently
4. **Rich CLI**: User-friendly command-line interface

### Core Components

#### Storage Layer
```python
# Base interface all backends implement
class StorageBackend(ABC):
    @abstractmethod
    def create_table_from_dataframe(self, df, table_name):
        pass
    
    @abstractmethod
    def read_table_to_dataframe(self, table_name):
        pass
```

#### Dataset Management
```python
# Registration flow
DatasetRegistrar:
  1. Validate inputs
  2. Auto-detect structure
  3. Create storage
  4. Load data in batches
  5. Detect types
  6. Generate features
  7. Save configuration
```

#### Feature Engineering
```python
# Two-tier system
1. Generic features (automatic)
   - Statistical
   - Temporal
   - Text
   
2. Custom features (user-defined)
   - Domain-specific
   - Business logic
```

## Common Tasks

### Adding a New CLI Command
```python
# src/mdm/cli/dataset.py
@app.command()
def analyze(
    name: str = Argument(..., help="Dataset name"),
    output: Optional[Path] = Option(None, help="Output file")
):
    """Analyze dataset quality."""
    client = MDMClient()
    # Implementation
```

### Adding a Storage Backend
```python
# src/mdm/storage/mongodb.py
class MongoDBBackend(StorageBackend):
    def __init__(self, config):
        # Initialize connection
        
    def create_table_from_dataframe(self, df, table_name):
        # Implementation
```

### Adding a Feature Transformer
```python
# src/mdm/features/transformers/seasonal.py
class SeasonalTransformer(BaseTransformer):
    def transform(self, df):
        # Add seasonal features
        return df
```

### Debugging Tips

1. **Enable debug logging**
   ```bash
   export MDM_LOGGING_LEVEL=DEBUG
   mdm dataset register test data.csv
   ```

2. **Check configuration**
   ```bash
   mdm info
   ```

3. **Inspect database**
   ```bash
   sqlite3 ~/.mdm/datasets/test/dataset.sqlite
   .tables
   .schema
   ```

## Need Help?

- Check existing issues on GitHub
- Read the documentation in `/docs`
- Ask questions in discussions
- Tag maintainers in complex PRs

Thank you for contributing to MDM!