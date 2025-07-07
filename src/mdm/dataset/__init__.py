"""Dataset registration and management module."""

from mdm.dataset.auto_detect import (
    detect_delimiter,
    detect_id_columns,
    detect_kaggle_structure,
    discover_data_files,
    extract_target_from_sample_submission,
    infer_problem_type,
    is_id_column,
    is_id_pattern,
    validate_kaggle_submission_format,
)
from mdm.dataset.manager import DatasetManager
from mdm.dataset.registrar import DatasetRegistrar

__all__ = [
    # Core classes
    'DatasetManager',
    'DatasetRegistrar',

    # Auto-detection functions
    'detect_kaggle_structure',
    'extract_target_from_sample_submission',
    'detect_id_columns',
    'infer_problem_type',
    'discover_data_files',
    'is_id_column',
    'is_id_pattern',
    'validate_kaggle_submission_format',
    'detect_delimiter'
]
