"""MDM Exception classes."""


class MDMError(Exception):
    """Base exception for all MDM errors."""
    pass


class ConfigError(MDMError):
    """Configuration related errors."""
    pass


class StorageError(MDMError):
    """Storage backend related errors."""
    pass


class DatasetError(MDMError):
    """Dataset operation related errors."""
    pass


class DatasetExistsError(DatasetError):
    """Raised when dataset already exists."""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when dataset is not found."""
    pass


class DatasetRegistrationError(DatasetError):
    """Raised when dataset registration fails."""
    pass


class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails."""
    pass


class ValidationError(MDMError):
    """Data validation related errors."""
    pass


class FeatureEngineeringError(MDMError):
    """Feature engineering related errors."""
    pass


class ExportError(MDMError):
    """Export operation related errors."""
    pass


class BackendError(MDMError):
    """Database backend related errors."""
    pass

