"""Enums for MDM models."""

from enum import Enum


class ProblemType(str, Enum):
    """Machine learning problem type."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTICLASS = "multiclass"
    CLUSTERING = "clustering"
    TIME_SERIES = "time-series"


class FileType(str, Enum):
    """Data file type."""

    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
    SUBMISSION = "submission"
    DATA = "data"


class ColumnType(str, Enum):
    """Column data type."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    ID = "id"
    TARGET = "target"
