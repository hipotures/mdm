"""Test basic imports and setup."""

import pytest


def test_mdm_imports():
    """Test that MDM can be imported."""
    import mdm
    
    assert mdm.__version__ == "0.1.0"
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