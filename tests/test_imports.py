"""Test basic imports and setup."""

import pytest


def test_mdm_imports():
    """Test that MDM can be imported."""
    import mdm
    
    # Version should be dynamically loaded from pyproject.toml
    assert mdm.__version__  # Just check it exists
    assert isinstance(mdm.__version__, str)
    # Check it's a valid version format
    assert "." in mdm.__version__  # Has version separators
    assert hasattr(mdm, "MDMError")


def test_core_exceptions():
    """Test that exception classes are properly defined."""
    from mdm.core.exceptions import (
        MDMError,
        ConfigError,
        StorageError,
        DatasetError,
        ValidationError,
        FeatureEngineeringError,
        ExportError,
        BackendError,
    )
    
    # Test inheritance
    assert issubclass(ConfigError, MDMError)
    assert issubclass(StorageError, MDMError)
    assert issubclass(DatasetError, MDMError)
    assert issubclass(ValidationError, MDMError)
    assert issubclass(FeatureEngineeringError, MDMError)
    assert issubclass(ExportError, MDMError)
    assert issubclass(BackendError, MDMError)