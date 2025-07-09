"""Unit tests for MDM models."""

from datetime import datetime, timezone

import pytest

from mdm.models.dataset import DatasetInfo
from mdm.models.enums import ColumnType, FileType, ProblemType


class TestDatasetInfo:
    """Test DatasetInfo model."""

    def test_dataset_info_creation(self):
        """Test creating a DatasetInfo instance."""
        dataset_info = DatasetInfo(
            name="test_dataset",
            database={"backend": "duckdb", "path": "/tmp/test.db"},
            tables={"train": "test_dataset_train", "test": "test_dataset_test"},
            target_column="target",
            id_columns=["id"],
            problem_type="binary_classification",
            description="Test dataset",
            tags=["test", "unit"],
        )
        
        assert dataset_info.name == "test_dataset"
        assert dataset_info.display_name is None
        assert dataset_info.target_column == "target"
        assert dataset_info.id_columns == ["id"]
        assert dataset_info.problem_type == "binary_classification"
        assert dataset_info.has_train_test_split is True
        assert dataset_info.has_validation is False
        assert "test" in dataset_info.tags

    def test_dataset_name_validation(self):
        """Test dataset name validation."""
        # Valid names
        valid_names = ["dataset1", "my_dataset", "data-123", "MyDataset"]
        for name in valid_names:
            dataset = DatasetInfo(
                name=name,
                database={"backend": "duckdb", "path": "/tmp/test.db"}
            )
            assert dataset.name == name.lower()
        
        # Invalid names
        invalid_names = ["", "data set", "data@set", "data/set"]
        for name in invalid_names:
            with pytest.raises(ValueError):
                DatasetInfo(
                    name=name,
                    database={"backend": "duckdb", "path": "/tmp/test.db"}
                )

    def test_time_series_fields(self):
        """Test time series specific fields."""
        dataset_info = DatasetInfo(
            name="timeseries_data",
            database={"backend": "duckdb", "path": "/tmp/test.db"},
            problem_type="time_series",
            time_column="date",
            group_column="store_id",
        )
        
        assert dataset_info.problem_type == "time_series"
        assert dataset_info.time_column == "date"
        assert dataset_info.group_column == "store_id"

    def test_feature_tables(self):
        """Test feature table tracking."""
        dataset_info = DatasetInfo(
            name="test_dataset",
            database={"backend": "duckdb", "path": "/tmp/test.db"},
            tables={"train": "test_dataset_train"},
            feature_tables={"train": "test_dataset_train_features"},
        )
        
        assert "train" in dataset_info.feature_tables
        assert dataset_info.feature_tables["train"] == "test_dataset_train_features"

    def test_metadata_field(self):
        """Test metadata storage."""
        metadata = {
            "source": "kaggle",
            "competition": "titanic",
            "custom_field": 123,
        }
        
        dataset_info = DatasetInfo(
            name="test_dataset",
            database={"backend": "duckdb", "path": "/tmp/test.db"},
            metadata=metadata,
        )
        
        assert dataset_info.metadata["source"] == "kaggle"
        assert dataset_info.metadata["competition"] == "titanic"
        assert dataset_info.metadata["custom_field"] == 123


class TestEnums:
    """Test enum definitions."""

    def test_problem_type_enum(self):
        """Test ProblemType enum values."""
        assert ProblemType.BINARY_CLASSIFICATION.value == "binary_classification"
        assert ProblemType.MULTICLASS_CLASSIFICATION.value == "multiclass_classification"
        assert ProblemType.REGRESSION.value == "regression"

    def test_column_type_enum(self):
        """Test ColumnType enum values."""
        assert ColumnType.NUMERIC.value == "numeric"
        assert ColumnType.CATEGORICAL.value == "categorical"
        assert ColumnType.TEXT.value == "text"
        assert ColumnType.DATETIME.value == "datetime"
        assert ColumnType.BINARY.value == "binary"

    def test_file_type_enum(self):
        """Test FileType enum values."""
        assert FileType.CSV.value == "csv"
        assert FileType.PARQUET.value == "parquet"
        assert FileType.JSON.value == "json"
        assert FileType.EXCEL.value == "excel"